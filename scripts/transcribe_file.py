from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bats_asr import BatsASR


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with BATS ASR.")
    parser.add_argument("--model", default="models/final_batsbi_whisper", help="Path to model folder")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--max_new_tokens", type=int, default=225)
    args = parser.parse_args()

    asr = BatsASR(args.model)
    text = asr.transcribe(args.audio, max_new_tokens=args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
