"""Simple API for Kani-TTS."""
from typing import Tuple, Optional
import numpy as np
import logging
import warnings
from .core import TTSConfig, NemoAudioPlayer, KaniModel


def suppress_all_logs():
    """
    Suppress all logging output from transformers, NeMo, PyTorch, and other libraries.
    Only print() statements from user code will be visible.
    """
    # Suppress Python warnings
    warnings.filterwarnings('ignore')

    # Suppress transformers logs
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except ImportError:
        pass

    # Suppress NeMo logs
    logging.getLogger('nemo').setLevel(logging.ERROR)
    logging.getLogger('nemo_logger').setLevel(logging.ERROR)

    # Suppress PyTorch logs
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('pytorch').setLevel(logging.ERROR)

    # Suppress other common loggers
    logging.getLogger('numba').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

    # Set root logger to ERROR level
    logging.getLogger().setLevel(logging.ERROR)


class KaniTTS:
    """
    Simple interface for Kani text-to-speech model.

    Example:
        >>> model = KaniTTS('your-model-name')
        >>> audio, text = model("Hello, world!")
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        max_new_tokens: int = 3000,
        tokeniser_length: int = 64400,
        suppress_logs: bool = True,
        show_info: bool = True,
        use_bematts: bool = False,
        text_vocab_size: int = 64400,
        tokens_per_frame: int = 4,
        audio_step: float = 1,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
    ):
        """
        Initialize Kani-TTS model.

        Args:
            model_name: Hugging Face model ID or path to local model
            device_map: Device mapping for model (default: "auto")
            max_new_tokens: Maximum number of tokens to generate (default: 3000)
            tokeniser_length: Length of text tokenizer vocabulary (default: 64400)
            suppress_logs: Whether to suppress library logs (default: True)
            show_info: Whether to display model info on initialization (default: True)
            use_bematts: Enable BemaTTS frame-level position encoding (default: False)
            text_vocab_size: Text vocabulary size for position encoding (default: 64400)
            tokens_per_frame: Number of audio tokens per frame (default: 4)
            audio_step: Position step size per audio frame (default: 1)
            use_learnable_rope: Enable learnable RoPE with per-layer alpha (default: False)
            alpha_min: Minimum alpha value for learnable RoPE (default: 0.1)
            alpha_max: Maximum alpha value for learnable RoPE (default: 2.0)
        """
        if suppress_logs:
            suppress_all_logs()

        self.config = TTSConfig(
            device_map=device_map,
            tokeniser_length=tokeniser_length,
            max_new_tokens=max_new_tokens,
            use_bematts=use_bematts,
            text_vocab_size=text_vocab_size,
            tokens_per_frame=tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        self.model_name = model_name

        self.player = NemoAudioPlayer(self.config)
        self.model = KaniModel(self.config, model_name, self.player)
        self.status = self.model.status
        self.speaker_list = self.model.speaker_list
        self.sample_rate = self.config.sample_rate

        if show_info:
            self.show_model_info()

    def __call__(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate audio from text.

        Args:
            text: Input text to convert to speech
            speaker_id: Optional speaker ID for multi-speaker models
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 0.95)
            repetition_penalty: Repetition penalty (default: 1.1)

        Returns:
            Tuple of (audio_waveform, text) where audio_waveform is a numpy array
            containing the audio samples and text is the input text.
        """
        return self.generate(text, speaker_id, temperature, top_p, repetition_penalty)

    def generate(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Tuple[np.ndarray, str]:
        """
        Generate audio from text.

        Args:
            text: Input text to convert to speech
            speaker_id: Optional speaker ID for multi-speaker models
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p sampling parameter (default: 0.95)
            repetition_penalty: Repetition penalty (default: 1.1)

        Returns:
            Tuple of (audio_waveform, text) where audio_waveform is a numpy array
            containing the audio samples and text is the input text.
        """
        return self.model.run_model(text, speaker_id, temperature, top_p, repetition_penalty)

    def save_audio(self, audio: np.ndarray, output_path: str):
        """
        Save audio waveform to file.

        Args:
            audio: Audio waveform as numpy array
            output_path: Path to save audio file (e.g., "output.wav")
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        except ImportError:
            raise ImportError(
                "soundfile is required to save audio. "
                "Install it with: pip install soundfile"
            )

    def show_model_info(self):
        """
        Display beautiful model information banner.
        """
        print()
        print("╔════════════════════════════════════════════════════════════╗")
        print("║                                                            ║")
        print("║                   N I N E N I N E S I X  😼                ║")
        print("║                                                            ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()
        print("              /\\_/\\  ")
        print("             ( o.o )")
        print("              > ^ <")
        print()
        print("─" * 62)

        # Model name
        model_display = self.model_name
        if len(model_display) > 50:
            model_display = "..." + model_display[-47:]
        print(f"  Model: {model_display}")

        # Device info
        import torch
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        print(f"  Device: {device}")

        # Speaker info
        if self.status == 'multispeaker':
            print(f"  Mode: Multi-speaker ({len(self.speaker_list)} speakers)")
            if self.speaker_list and len(self.speaker_list) <= 5:
                speakers_str = ", ".join(self.speaker_list)
                print(f"  Speakers: {speakers_str}")
            elif self.speaker_list:
                print(f"  Speakers: {self.speaker_list[0]}, {self.speaker_list[1]}, ... (use .show_speakers() to see all)")
        else:
            print(f"  Mode: Single-speaker")

        print()
        print("  Configuration:")
        print(f"    • Sample Rate: {self.sample_rate} Hz")
        print(f"    • Max Tokens: {self.config.max_new_tokens}")
        if self.config.use_bematts:
            print(f"    • BemaTTS: Enabled (frame-level position encoding)")
            print(f"    • Text Vocab Size: {self.config.text_vocab_size}")
            print(f"    • Tokens per Frame: {self.config.tokens_per_frame}")
            print(f"    • Audio Step: {self.config.audio_step}")
            if self.config.use_learnable_rope:
                print(f"    • Learnable RoPE: Enabled (per-layer frequency scaling)")
                print(f"    • Alpha Range: [{self.config.alpha_min}, {self.config.alpha_max}]")
            else:
                print(f"    • Learnable RoPE: Disabled (standard RoPE)")
        else:
            print(f"    • BemaTTS: Disabled (standard position encoding)")

        print("─" * 62)
        print()
        print("  Ready to generate speech! 🎵")
        print()

    def show_speakers(self):
        """
        Display available speakers for multi-speaker models.

        For single-speaker models, displays a message that speaker selection
        is not available.
        """
        print("=" * 50)
        if self.status == 'multispeaker':
            print("Available Speakers:")
            print("-" * 50)
            if self.speaker_list:
                for i, speaker in enumerate(self.speaker_list, 1):
                    print(f"  {i}. {speaker}")
            else:
                print("  No speakers configured")
        else:
            print("Single-speaker model")
            print("Speaker selection is not available for this model")
        print("=" * 50)
