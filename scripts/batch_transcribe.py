from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bats_asr import BatsASR

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch transcribe audio files with BATS ASR.")
    parser.add_argument("--model", default="models/final_batsbi_whisper", help="Path to model folder")
    parser.add_argument("--audio_dir", required=True, help="Folder with audio files")
    parser.add_argument("--out", default="transcriptions.tsv", help="Output TSV")
    parser.add_argument("--max_new_tokens", type=int, default=225)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    files = sorted([p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS])
    if not files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    asr = BatsASR(args.model)
    rows = []

    for path in tqdm(files):
        try:
            text = asr.transcribe(path, max_new_tokens=args.max_new_tokens)
            rows.append({"audio_path": str(path), "transcription": text, "error": ""})
        except Exception as e:
            rows.append({"audio_path": str(path), "transcription": "", "error": repr(e)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False, encoding="utf-8")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
