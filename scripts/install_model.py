from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


REQUIRED_ANY = ["model.safetensors", "pytorch_model.bin"]
REQUIRED_BASIC = ["config.json", "preprocessor_config.json"]


def find_model_root(path: Path) -> Path | None:
    if (path / "config.json").exists():
        return path
    matches = list(path.rglob("config.json"))
    if not matches:
        return None
    return matches[0].parent


def copy_folder_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def install_from_source(source: Path, out: Path) -> None:
    tmp = out.parent / f".{out.name}_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    if source.is_file() and source.suffix.lower() == ".zip":
        print(f"Unzipping {source} ...")
        with zipfile.ZipFile(source, "r") as z:
            z.extractall(tmp)
    elif source.is_dir():
        print(f"Copying {source} ...")
        copy_folder_contents(source, tmp)
    else:
        raise FileNotFoundError(f"Unsupported model source: {source}")

    model_root = find_model_root(tmp)
    if model_root is None:
        raise FileNotFoundError("config.json was not found in the supplied model source.")

    out.mkdir(parents=True, exist_ok=True)
    copy_folder_contents(model_root, out)
    shutil.rmtree(tmp)


def check_model(model_dir: Path) -> None:
    model_root = find_model_root(model_dir)
    if model_root is None:
        raise FileNotFoundError(
            f"config.json was not found under {model_dir}. "
            "Put the Hugging Face Whisper model files in this folder."
        )

    missing = [name for name in REQUIRED_BASIC if not (model_root / name).exists()]
    has_weights = any((model_root / name).exists() for name in REQUIRED_ANY) or bool(
        list(model_root.glob("*.safetensors")) or list(model_root.glob("pytorch_model*.bin"))
    )

    if missing:
        print("Warning: missing expected files:", ", ".join(missing))
    if not has_weights:
        print("Warning: no model weights found (*.safetensors or pytorch_model*.bin).")

    print("Model folder is ready:", model_root)



def main() -> None:
    parser = argparse.ArgumentParser(description="Install/check the BATS ASR Whisper model.")
    parser.add_argument("--model", default="models/final_batsbi_whisper", help="Target model folder")
    parser.add_argument("--source", default=None, help="Optional model zip/folder to copy into --model")
    args = parser.parse_args()

    model_dir = Path(args.model)

    if args.source is not None:
        install_from_source(Path(args.source), model_dir)

    check_model(model_dir)


if __name__ == "__main__":
    main()
