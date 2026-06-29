from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bats_asr import BatsASR, cer, character_accuracy


def resolve_audio_path(path_value: str, clips_dir: Path) -> Path:
    p = Path(str(path_value))
    if p.is_absolute() and p.exists():
        return p
    candidate = clips_dir / p
    if candidate.exists():
        return candidate
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BATS ASR on Common Voice-style TSV.")
    parser.add_argument("--model", default="models/final_batsbi_whisper", help="Path to model folder")
    parser.add_argument("--tsv", required=True, help="Common Voice-style TSV with path and sentence columns")
    parser.add_argument("--clips", required=True, help="Path to clips folder")
    parser.add_argument("--out", default="eval.tsv", help="Output TSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows")
    parser.add_argument("--max_new_tokens", type=int, default=225)
    args = parser.parse_args()

    df = pd.read_csv(args.tsv, sep="\t", dtype=str, keep_default_na=False)
    if "path" not in df.columns or "sentence" not in df.columns:
        raise ValueError("TSV must contain 'path' and 'sentence' columns.")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    clips_dir = Path(args.clips)
    asr = BatsASR(args.model)
    rows = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = resolve_audio_path(row["path"], clips_dir)
        ref = str(row["sentence"])
        try:
            hyp = asr.transcribe(audio_path, max_new_tokens=args.max_new_tokens)
            rows.append({
                "idx": i,
                "audio_path": str(audio_path),
                "reference": ref,
                "hypothesis": hyp,
                "cer": cer(ref, hyp),
                "character_accuracy": character_accuracy(ref, hyp),
                "error": "",
            })
        except Exception as e:
            rows.append({
                "idx": i,
                "audio_path": str(audio_path),
                "reference": ref,
                "hypothesis": "",
                "cer": None,
                "character_accuracy": None,
                "error": repr(e),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False, encoding="utf-8")

    ok = out[out["cer"].notna()]
    print("Saved:", args.out)
    print("N:", len(ok))
    if len(ok):
        print("Mean CER:", ok["cer"].mean())
        print("Mean character accuracy:", ok["character_accuracy"].mean())


if __name__ == "__main__":
    main()
