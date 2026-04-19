"""Core components for Kani-TTS audio generation."""
import torch
from nemo.collections.tts.models import AudioCodecModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import os

# Import BemaTTS custom model
from .model import FlashCompatibleLfm2ForCausalLM


@dataclass
class TTSConfig:
    """Configuration for TTS model."""
    device_map: str = "auto"
    tokeniser_length: int = 64400
    start_of_text: int = 1
    end_of_text: int = 2
    max_new_tokens: int = 3000
    nanocodec_model:str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
    sample_rate = 22050

    # BemaTTS configuration
    use_bematts: bool = False  # Enable BemaTTS frame-level position encoding
    text_vocab_size: int = 64400  # Text vocabulary size (for position encoding)
    tokens_per_frame: int = 4  # Number of audio tokens per frame
    audio_step: float = 1  # Position step size per audio frame

    # Learnable RoPE configuration
    use_learnable_rope: bool = False  # Enable learnable RoPE with per-layer alpha
    alpha_min: float = 0.1  # Minimum value for alpha (frequency scaling)
    alpha_max: float = 2.0  # Maximum value for alpha (frequency scaling)


class NemoAudioPlayer:
    """Handles audio codec operations using NVIDIA NeMo."""

    def __init__(self, config: TTSConfig, text_tokenizer_name: Optional[str] = None) -> None:
        self.conf = config
        self.nemo_codec_model = AudioCodecModel\
                .from_pretrained(self.conf.nanocodec_model).eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nemo_codec_model.to(self.device)
        self.text_tokenizer_name = text_tokenizer_name
        if self.text_tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_name)

        self.tokeniser_length = self.conf.tokeniser_length
        self.start_of_text = self.conf.start_of_text
        self.end_of_text = self.conf.end_of_text
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

    def output_validation(self, out_ids: torch.Tensor) -> None:
        """Validate that output contains required speech markers."""
        start_of_speech_flag = self.start_of_speech in out_ids
        end_of_speech_flag = self.end_of_speech in out_ids
        if not (start_of_speech_flag and end_of_speech_flag):
            raise ValueError('Special speech tokens not exist!')

    def get_nano_codes(self, out_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and process audio codes from model output."""
        start_a_idx = (out_ids == self.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_a_idx   = (out_ids == self.end_of_speech).nonzero(as_tuple=True)[0].item()
        if start_a_idx >= end_a_idx:
            raise ValueError('Invalid audio codes sequence!')

        audio_codes = out_ids[start_a_idx+1 : end_a_idx]
        if len(audio_codes) % 4:
            raise ValueError('The length of the sequence must be a multiple of 4!')
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - torch.tensor([self.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError('Invalid audio tokens!')

        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]])
        return audio_codes, len_

    def get_text(self, out_ids: torch.Tensor) -> str:
        """Extract text from token sequence."""
        start_t_idx = (out_ids == self.start_of_text).nonzero(as_tuple=True)[0].item()
        end_t_idx   = (out_ids == self.end_of_text).nonzero(as_tuple=True)[0].item()
        txt_tokens = out_ids[start_t_idx : end_t_idx+1]
        text = self.tokenizer.decode(txt_tokens, skip_special_tokens=True)
        return text

    def get_waveform(self, out_ids: torch.Tensor) -> Tuple[np.ndarray, Optional[str]]:
        """Convert model output tokens to audio waveform."""
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes, len_ = audio_codes.to(self.device), len_.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()

        if self.text_tokenizer_name:
            text = self.get_text(out_ids)
            return output_audio, text
        else:
            return output_audio, None


class KaniModel:
    """Text-to-speech model using causal language model."""

    def __init__(self, config: TTSConfig, model_name: str, player: NemoAudioPlayer) -> None:
        self.conf = config
        self.player = player
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model with BemaTTS if enabled
        if self.conf.use_bematts:
            if self.conf.use_learnable_rope:
                print("🚀 Loading model with BemaTTS + Learnable RoPE...")
            else:
                print("🚀 Loading model with BemaTTS frame-level position encoding...")
            self.model = FlashCompatibleLfm2ForCausalLM.from_pretrained(
                model_name,
                audio_tokens_start=self.player.audio_tokens_start,
                tokens_per_frame=self.conf.tokens_per_frame,
                audio_step=self.conf.audio_step,
                use_learnable_rope=self.conf.use_learnable_rope,
                alpha_min=self.conf.alpha_min,
                alpha_max=self.conf.alpha_max,
                torch_dtype=torch.bfloat16,
                device_map=self.conf.device_map,
            )
        else:
            print("📦 Loading standard model (original KaniTTS)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.conf.device_map,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.speaker_settings = getattr(self.model.config, 'speaker_settings', None)
        self.status = 'singlspeaker'
        self.speaker_list = []
        if self.speaker_settings is not None:
            self.status = self.speaker_settings.get('status')
            self.speaker_list = self.speaker_settings.get('speaker_list', [])

    def get_input_ids(self, text_prompt: str, speaker_id: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tokens with special markers."""
        if speaker_id is not None:
            text_prompt = f"{speaker_id.strip()}: {text_prompt}"

        START_OF_HUMAN = self.player.start_of_human
        END_OF_TEXT = self.player.end_of_text
        END_OF_HUMAN = self.player.end_of_human

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        return modified_input_ids, attention_mask

    def model_request(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Generate audio tokens from text tokens."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.conf.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.player.end_of_speech,
            )
        return generated_ids.to('cpu')

    def run_model(
        self,
        text: str,
        speaker_id: str = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, str]:
        """Generate audio from text."""

        if (self.status == 'multispeaker') and (speaker_id is None):
            print('='*30)
            print('!!! YOU CAN CHOOSE A SPEAKER ID !!!')
            print(f'Speakers available:')
            print(print(*self.speaker_list, sep='\n'))
            print('='*30)
        elif (self.status == 'singlspeaker') and (speaker_id is not None):
            print('='*30)
            print('!!! This model does not support speaker selection !!!')
            print('='*30)

        input_ids, attention_mask = self.get_input_ids(text, speaker_id)
        model_output = self.model_request(input_ids, attention_mask, temperature, top_p, repetition_penalty)
        audio, _ = self.player.get_waveform(model_output)
        return audio, text
