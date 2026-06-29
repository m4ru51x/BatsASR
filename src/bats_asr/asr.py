from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class BatsbiWhisperASR:
    """Fine-tuned Whisper ASR wrapper for Batsbi / Tsova-Tush."""

    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
        language: str = "Georgian",
        task: str = "transcribe",
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(str(self.model_dir))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        # Do not suppress uncommon Georgian/Batsbi symbols.
        self.model.config.suppress_tokens = []
        self.model.generation_config.suppress_tokens = []

        try:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language,
                task=task,
            )
            self.model.generation_config.forced_decoder_ids = forced_decoder_ids
        except Exception:
            # Some saved models already include generation settings.
            pass

    def transcribe(
        self,
        audio_path: str | Path,
        max_new_tokens: int = 225,
        sampling_rate: int = 16000,
    ) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        speech, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)

        inputs = self.processor.feature_extractor(
            speech,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            pred_ids = self.model.generate(
                input_features,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        text = self.processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
        )[0]

        return " ".join(str(text).split())


# Short alias for the project name.
BatsASR = BatsbiWhisperASR
