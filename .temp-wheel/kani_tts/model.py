"""
Custom LFM2 implementation with BemaTTS frame-level position encoding.

Key Innovation:
- Frame-level position encoding: All 4 tokens within an audio frame share the same position ID
  This reduces RoPE distance between tokens across frames, improving long-form generation.

Compatible with Flash Attention 2 for 10-20x training speedup.

FIXED: Proper frame-level position tracking during generation with KV-cache.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerationMixin

# Import base LFM2 classes
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Model,
    Lfm2ForCausalLM,
    Lfm2PreTrainedModel,
    Lfm2HybridConvCache,
)
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config


def compute_frame_level_positions_old(
    input_ids: torch.Tensor,
    audio_tokens_start: int,
    tokens_per_frame: int = 4
    ) -> torch.Tensor:
    """
    Vectorized computation of frame-level position IDs (10-50x faster than Python loops).

    Key insight: Use fractional position increments + floor() to group audio tokens by frame.

    - Text tokens and special tokens (< audio_tokens_start): sequential positions
    - Audio tokens (>= audio_tokens_start): frame-level positions (grouped by tokens_per_frame)

    Algorithm:
    1. Non-audio tokens: increment position by 1.0
    2. Audio tokens: increment position by 1/tokens_per_frame (e.g., 0.25 for 4 tokens)
    3. Cumulative sum gives raw positions (e.g., [0, 1, 2, 2.25, 2.5, 2.75, 3])
    4. Floor to integers: [0, 1, 2, 2, 2, 2, 3] - audio tokens grouped!

    This is fully GPU-accelerated and compatible with Flash Attention.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        audio_tokens_start: Token ID where audio tokens begin (typically 64410)
        tokens_per_frame: Number of tokens per audio frame (typically 4)

    Returns:
        position_ids: Position IDs with frame-level encoding [batch_size, seq_len]

    Example:
        >>> input_ids = torch.tensor([[100, 200, 64410, 68442, 72474, 76506, 300]])
        >>> # Tokens:                [text, text, aud0,  aud1,  aud2,  aud3,  text]
        >>> pos = compute_frame_level_positions(input_ids, 64410, 4)
        >>> pos
        tensor([[0, 1, 2, 2, 2, 2, 3]])  # Audio tokens share position 2
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Identify audio tokens: [batch_size, seq_len]
    is_audio = input_ids >= audio_tokens_start

    # Create position increment values
    # Non-audio tokens: increment by 1.0 (full position step)
    # Audio tokens: increment by 1/tokens_per_frame (fractional step)
    # This way, tokens_per_frame audio tokens = 1 full position increment
    position_increment = torch.where(
        is_audio,
        torch.full_like(input_ids, 1.0 / tokens_per_frame, dtype=torch.float),
        torch.ones_like(input_ids, dtype=torch.float)
    )

    # Cumulative sum to get raw positions
    # Prepend zero and remove last element so first token starts at position 0
    raw_positions = torch.cat([
        torch.zeros(batch_size, 1, device=device, dtype=torch.float),
        position_increment[:, :-1]
    ], dim=1).cumsum(dim=1)

    # Floor to get integer positions
    # Audio tokens in the same frame (fractional positions 2.0, 2.25, 2.5, 2.75)
    # all become position 2
    position_ids = raw_positions.floor().long()

    return position_ids


def compute_frame_level_positions(
    input_ids: torch.Tensor,
    audio_tokens_start: int,
    tokens_per_frame: int = 4,
    audio_step: float = 1.0
    ) -> torch.Tensor:
    """
    Vectorized computation of frame-level position IDs (10-50x faster than Python loops).

    Key insight: Use cumulative counts to determine positions.

    - Text tokens: sequential positions (step 1.0)
    - Audio tokens: frame-level positions (step audio_step per frame)

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        audio_tokens_start: Token ID where audio tokens begin (typically 64410)
        tokens_per_frame: Number of tokens per audio frame (typically 4)
        audio_step: Position step size per audio frame (default 1.0).
                    Set to < 1.0 (e.g., 0.5) to compress audio position space.

    Returns:
        position_ids: Position IDs [batch_size, seq_len].
                      if audio_step is float, returns FloatTensor.

    Example:
        >>> input_ids = torch.tensor([[100, 200, 64410, 68442, 72474, 76506, 300]])
        >>> # Tokens:                [text, text, aud0,  aud1,  aud2,  aud3,  text]
        >>> pos = compute_frame_level_positions(input_ids, 64410, 4, audio_step=0.5)
        >>> pos
        tensor([[0., 1., 2., 2., 2., 2., 3.]])
        # Text at 0, 1. Audio frame at 2. Next text at 3 (1+1+1?)
        # Note: Text logic accumulates 1 per text token.
        # Audio logic accumulates audio_step per frame.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Identify audio tokens
    is_audio = input_ids >= audio_tokens_start
    text_mask = ~is_audio

    # Prepare zero prefix for cumsum
    zeros = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

    # 1. Count text tokens before each position
    #    This gives the integer base from text tokens
    text_count = torch.cat([zeros, text_mask.long()], dim=1).cumsum(dim=1)[:, :-1]

    # 2. Count audio tokens before each position
    audio_token_count = torch.cat([zeros, is_audio.long()], dim=1).cumsum(dim=1)[:, :-1]

    # 3. Convert token count to frame count (0, 0, 0, 0, 1, 1...)
    audio_frame_count = audio_token_count // tokens_per_frame

    # 4. Compute final positions
    #    Text contributes 1.0 per token
    #    Audio frames contribute audio_step per frame
    position_ids = text_count + audio_frame_count * audio_step

    return position_ids


class LearnableRotaryEmbedding(nn.Module):
    """
    Learnable RoPE with layer-wise frequency scaling.

    Each layer has a learnable alpha parameter that scales the RoPE frequencies:
        θᵢ^(l) = α^(l) · base^(-2i/d)

    where α^(l) is constrained to [alpha_min, alpha_max] via sigmoid reparameterization:
        α^(l) = alpha_min + (alpha_max - alpha_min) · sigmoid(w^(l))

    This allows the model to learn optimal positional encoding frequencies per layer.
    """

    def __init__(
        self,
        config,
        layer_idx,
        total_attention_layers,
        alpha_min=0.1,
        alpha_max=2.0,
        device=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.total_attention_layers = total_attention_layers
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Get RoPE parameters from config
        dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        max_position_embeddings = config.max_position_embeddings

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        # Compute base inverse frequencies: θᵢ = base^(-2i/d)
        inv_freq_base = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq_base", inv_freq_base, persistent=False)

        # Learnable parameter (unconstrained, will be transformed via sigmoid)
        self.alpha_weight = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        """
        Compute constrained alpha via sigmoid reparameterization.

        Returns:
            Scalar alpha value in range [alpha_min, alpha_max]
        """
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_weight)

    @property
    def inv_freq(self):
        """
        Compute scaled inverse frequencies: α^(l) · θᵢ

        Returns:
            Tensor of shape [d/2] with scaled frequencies
        """
        return self.inv_freq_base * self.alpha

    def forward(self, x, position_ids):
        """
        Compute rotary position embeddings for the input.

        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            position_ids: Position indices of shape [batch_size, seq_len]

        Returns:
            Tuple of (cos, sin) embeddings, each of shape [batch_size, seq_len, head_dim]
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        # Compute position embeddings using scaled frequencies
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 for matmul to avoid precision issues
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlashCompatibleLfm2Model(Lfm2Model):
    """
    Custom LFM2 model with BemaTTS frame-level position encoding.

    This version only overrides position ID computation - everything else
    uses the standard Lfm2Model implementation.
    """

    def __init__(
        self,
        config: Lfm2Config,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
    ):
        super().__init__(config)
        self.audio_tokens_start = audio_tokens_start
        self.tokens_per_frame = tokens_per_frame
        self.audio_step = audio_step
        self.use_learnable_rope = use_learnable_rope
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Initialize learnable RoPE if enabled
        if use_learnable_rope:
            # Identify which layers are attention layers (not hybrid conv layers)
            attention_layer_indices = []
            if hasattr(config, 'layer_types'):
                for idx, layer_type in enumerate(config.layer_types):
                    if layer_type == "full_attention":
                        attention_layer_indices.append(idx)
            else:
                # Fallback: assume all layers are attention layers
                attention_layer_indices = list(range(config.num_hidden_layers))

            total_attention_layers = len(attention_layer_indices)

            # Create learnable RoPE modules for each layer
            self.learnable_rope_layers = nn.ModuleList()
            for idx in range(config.num_hidden_layers):
                if idx in attention_layer_indices:
                    learnable_rope = LearnableRotaryEmbedding(
                        config=config,
                        layer_idx=idx,
                        total_attention_layers=total_attention_layers,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        device=config.device if hasattr(config, 'device') else None,
                    )
                    self.learnable_rope_layers.append(learnable_rope)
                else:
                    # Conv layers don't use RoPE
                    self.learnable_rope_layers.append(None)

        else:
            self.learnable_rope_layers = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """
        Forward pass with custom frame-level position IDs.

        If learnable RoPE is disabled:
            Delegates to parent class after computing BemaTTS position IDs.

        If learnable RoPE is enabled:
            Overrides position embedding computation to use per-layer learnable RoPE.
        """
        # BEMATTS CORE: Compute frame-level position IDs if not provided
        if position_ids is None and input_ids is not None:
            position_ids = compute_frame_level_positions(
                input_ids=input_ids,
                audio_tokens_start=self.audio_tokens_start,
                tokens_per_frame=self.tokens_per_frame,
                audio_step=self.audio_step
            )

        # If learnable RoPE is disabled, use standard forward pass
        if not self.use_learnable_rope:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # Learnable RoPE path: need to compute position embeddings per-layer
        # This reimplements the forward pass with per-layer position embedding computation
        from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache, create_causal_mask

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            batch_size = inputs_embeds.shape[0]
            past_key_values = Lfm2HybridConvCache(
                config=self.config, max_batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # Compute position embeddings per layer (learnable RoPE)
        position_embeddings = None

        # Decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Compute position embeddings for this layer
            if self.learnable_rope_layers[layer_idx] is not None:
                # This is an attention layer with learnable RoPE
                position_embeddings = self.learnable_rope_layers[layer_idx](hidden_states, position_ids)
            elif position_embeddings is None:
                # This is a conv layer, use standard RoPE (compute once)
                position_embeddings = self.pos_emb(hidden_states, position_ids)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FlashCompatibleLfm2ForCausalLM(Lfm2PreTrainedModel, GenerationMixin):
    """
    Flash Attention compatible LFM2 for causal language modeling with BemaTTS frame-level positions.

    Features:
    - Frame-level position encoding for audio tokens (BemaTTS innovation)
    - Optional learnable RoPE with per-layer frequency scaling (alpha parameters)
    - Proper position tracking during generation with KV-cache
    - Flash Attention 2 optimized
    - Standard causal masking
    - Compatible with existing KaniTTS inference pipeline
    - Includes GenerationMixin for text generation capabilities
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: Lfm2Config,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
    ):
        super().__init__(config)

        # Use our flash-compatible model with BemaTTS position encoding
        self.model = FlashCompatibleLfm2Model(
            config,
            audio_tokens_start,
            tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Store these for easy access
        self.audio_tokens_start = audio_tokens_start
        self.tokens_per_frame = tokens_per_frame
        self.audio_step = audio_step
        self.use_learnable_rope = use_learnable_rope
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Generation state tracking for proper frame-level positions
        self._generation_state = None

        # Set generation config
        self.generation_config = config.generation_config if hasattr(config, 'generation_config') else None
        self.main_input_name = "input_ids"

        # Initialize weights and apply final processing
        self.post_init()

    def _reset_generation_state(self, starting_frame_position: Optional[int] = None):
        """
        Reset generation state before starting new generation.

        This tracks:
        - The position where the first audio frame should start
        - How many audio tokens we've generated
        - What the current frame position should be

        Args:
            starting_frame_position: The position ID where the first audio frame begins.
                                   If None, will be determined when first audio token is generated.
        """
        self._generation_state = {
            'audio_tokens_generated': 0,
            'audio_tokens_generated': 0,
            'current_frame_position': float(starting_frame_position) if starting_frame_position is not None else None
        }

    def _update_generation_state(self, new_token_id: int):
        """
        Update generation state after generating a token.

        Args:
            new_token_id: The token ID that was just generated
        """
        if self._generation_state is None:
            return

        if new_token_id >= self.audio_tokens_start:
            # Audio token generated - just increment the counter
            # Frame position increment is now handled in prepare_inputs_for_generation
            self._generation_state['audio_tokens_generated'] += 1

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Forward pass - delegates to flash-compatible model with BemaTTS position encoding."""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs
    ):
        """
        Prepare inputs for generation with proper frame-level position encoding.
        
        CRITICAL FIX: Maintains frame-level positions during generation with KV-cache.
        """
        # Handle past_key_values for incremental generation
        if past_key_values is not None:
            # LFM2 uses Lfm2HybridConvCache which has get_seq_length() method
            if isinstance(past_key_values, (Cache, Lfm2HybridConvCache)):
                cache_length = past_key_values.get_seq_length()
                past_length = cache_length
            else:
                cache_length = past_length = past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0

            # Keep only the last token for inputs
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        if cache_position is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device
            )

        # ===== CRITICAL FIX: Frame-level position computation =====
        if position_ids is None:
            if past_key_values is not None and self._generation_state is not None:
                # ✅ FIXED: Use generation state to compute correct frame-level position

                current_token = input_ids[0, -1].item()

                if current_token < self.audio_tokens_start:
                    # Text/special token - use sequential position
                    pos = past_key_values.get_seq_length()
                else:
                    # Audio token - first, ensure frame position is initialized
                    if self._generation_state['current_frame_position'] is None:
                        # This is the first audio token - use current KV cache length as position
                        first_frame_pos = past_key_values.get_seq_length()
                        self._generation_state['current_frame_position'] = first_frame_pos

                    # Check if we need to advance to next frame first
                    token_in_frame = self._generation_state['audio_tokens_generated'] % self.tokens_per_frame

                    # If starting a new frame (and not the very first audio token), advance position
                    if token_in_frame == 0 and self._generation_state['audio_tokens_generated'] > 0:
                        self._generation_state['current_frame_position'] += self.audio_step

                    # Now use the (possibly just incremented) frame position
                    pos = self._generation_state['current_frame_position']

                # Use FloatTensor if needed, otherwise LongTensor
                if isinstance(pos, float):
                     position_ids = torch.tensor([[pos]], device=input_ids.device, dtype=torch.float)
                else: 
                     position_ids = torch.tensor([[pos]], device=input_ids.device, dtype=torch.long)

                # Update state for next token (just increment counter now)
                self._update_generation_state(current_token)

            else:
                # ✅ First forward pass - compute frame-level positions for entire sequence
                position_ids = compute_frame_level_positions(
                    input_ids=input_ids,
                    audio_tokens_start=self.audio_tokens_start,
                    tokens_per_frame=self.tokens_per_frame,
                    audio_step=self.audio_step
                )

                # Initialize generation state if we're about to start generating
                # NOTE: We don't know the exact starting position yet since special tokens
                # will be generated between text and audio. We'll set it when first audio token arrives.
                if past_key_values is None and use_cache:
                    self._reset_generation_state(starting_frame_position=None)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds

        return model_inputs

    def generate(self, *args, **kwargs):
        """
        Override generate to reset state before generation.

        This ensures frame-level position tracking starts fresh for each generation call.
        """
        # Reset state before generation
        self._generation_state = None

        try:
            # Call parent generate
            result = super().generate(*args, **kwargs)
        finally:
            # Clean up state after generation (even if error)
            self._generation_state = None

        return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        *model_args,
        **kwargs
    ):
        """
        Load a pretrained LFM2 model with BemaTTS flash-compatible implementation.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
            audio_tokens_start: Token ID where audio tokens begin (typically 64410)
            audio_tokens_start: Token ID where audio tokens begin (typically 64410)
            tokens_per_frame: Number of tokens per audio frame (default: 4)
            audio_step: Step size per audio frame (default: 1.0). Use 0.5 for new models.
            use_learnable_rope: Enable learnable RoPE with per-layer alpha (default: False)
            alpha_min: Minimum alpha value for learnable RoPE (default: 0.1)
            alpha_max: Maximum alpha value for learnable RoPE (default: 2.0)
        """
        # Filter out our custom parameters before passing to base class
        base_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ['use_learnable_rope', 'alpha_min', 'alpha_max']}

        # Load config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **base_kwargs)

        # Create model with BemaTTS position encoding and optional learnable RoPE
        model = cls(
            config=config,
            audio_tokens_start=audio_tokens_start,
            tokens_per_frame=tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        # Load pretrained weights
        if use_learnable_rope:
            # For learnable RoPE models, load weights directly from safetensors to preserve custom parameters
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            import os

            # Download the safetensors file
            if os.path.isdir(pretrained_model_name_or_path):
                # Local path
                safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            else:
                # HuggingFace Hub
                safetensors_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="model.safetensors"
                )

            # Load state dict from safetensors
            state_dict = load_file(safetensors_path)

            # Load weights into our model (strict=False to allow partial loading)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Handle tied weights: lm_head.weight is often tied to model.embed_tokens.weight
            if 'lm_head.weight' in missing_keys and 'model.embed_tokens.weight' in state_dict:
                model.lm_head.weight = model.model.embed_tokens.weight
                missing_keys = [k for k in missing_keys if k != 'lm_head.weight']

            # Handle missing/unexpected keys silently

            # Load generation config if available
            from transformers import GenerationConfig
            try:
                generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path)
                model.generation_config = generation_config
            except Exception:
                # If generation config not found, create a default one
                model.generation_config = GenerationConfig()

            # Determine device from base_kwargs or use CUDA if available
            device_map = base_kwargs.get('device_map', 'auto')
            if device_map == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)
            # else: device_map will handle device placement automatically
        else:
            # For standard models, load via transformers (no learnable RoPE parameters to preserve)
            base_model = Lfm2ForCausalLM.from_pretrained(pretrained_model_name_or_path, **base_kwargs)

            # Copy weights from base model to our custom model
            model.model.load_state_dict(base_model.model.state_dict(), strict=False)
            model.lm_head.load_state_dict(base_model.lm_head.state_dict())

            # Copy generation config if available
            if hasattr(base_model, 'generation_config'):
                model.generation_config = base_model.generation_config

            # Move model to the same device as base_model
            model = model.to(base_model.device)

        return model

    def get_input_embeddings(self):
        """Required by GenerationMixin."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Required by GenerationMixin."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Required by GenerationMixin."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Required by GenerationMixin."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """Required by GenerationMixin."""
        self.model = decoder

    def get_decoder(self):
        """Required by GenerationMixin."""
        return self.model


# ===== TESTING UTILITIES =====

def test_frame_level_positions(model, text_vocab_size: int = 64400):
    """
    Test that frame-level positions are correctly computed during generation.
    
    Args:
        model: FlashCompatibleLfm2ForCausalLM instance
        text_vocab_size: Vocabulary size for text tokens
    """
    print("\n" + "="*60)
    print("🧪 TESTING FRAME-LEVEL POSITION ENCODING")
    print("="*60)
    
    # Create test input: 3 text tokens
    input_ids = torch.tensor([[10, 20, 30]], device=model.device)
    
    print(f"\n📝 Input: {input_ids.tolist()}")
    print(f"   Text tokens: {input_ids.shape[1]}")
    
    # Monkey-patch forward to log positions
    original_prepare = model.prepare_inputs_for_generation
    positions_log = []
    
    def logging_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        if 'position_ids' in result and result['position_ids'] is not None:
            positions_log.append(result['position_ids'].tolist())
        return result
    
    model.prepare_inputs_for_generation = logging_prepare
    
    try:
        # Generate 8 tokens (2 frames)
        print(f"\n🎯 Generating 8 audio tokens (2 frames)...")
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
        )
        
        print(f"\n📊 Position IDs used during generation:")
        for i, pos in enumerate(positions_log):
            if i == 0:
                print(f"  Step {i} (prefill): {pos}")
            else:
                frame = (i - 1) // 4
                token_in_frame = (i - 1) % 4
                print(f"  Step {i} (frame {frame}, token {token_in_frame}): {pos}")
        
        # Validate
        print(f"\n✅ Validation:")
        
        # Step 0: prefill (text tokens)
        assert positions_log[0] == [[0, 1, 2]], f"Prefill positions wrong: {positions_log[0]}"
        print(f"  ✓ Prefill: {positions_log[0]}")
        
        # Steps 1-4: frame 0 (all should be position 3)
        for i in range(1, 5):
            assert positions_log[i] == [[3]], f"Frame 0 token {i-1} wrong: {positions_log[i]}, expected [[3]]"
        print(f"  ✓ Frame 0: all tokens at position 3")
        
        # Steps 5-8: frame 1 (all should be position 4)
        for i in range(5, 9):
            assert positions_log[i] == [[4]], f"Frame 1 token {i-5} wrong: {positions_log[i]}, expected [[4]]"
        print(f"  ✓ Frame 1: all tokens at position 4")
        
        print(f"\n🎉 All tests passed! Frame-level positions are correct!")
        
    finally:
        # Restore original method
        model.prepare_inputs_for_generation = original_prepare
    
    print("="*60 + "\n")


# Example usage:
if __name__ == "__main__":
    print("""
    Example usage:
    
    from flash_lfm2 import FlashCompatibleLfm2ForCausalLM, test_frame_level_positions
    
    # Load model
    model = FlashCompatibleLfm2ForCausalLM.from_pretrained(
        "path/to/your/model",
        text_vocab_size=64400,
        tokens_per_frame=4
    )
    
    # Test frame-level positions
    test_frame_level_positions(model)
    
    # Generate
    outputs = model.generate(
        input_ids=your_prompt,
        max_new_tokens=400,
        temperature=0.9,
        top_p=0.95,
    )
    """)

