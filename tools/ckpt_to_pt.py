from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location="cpu")


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "model."
    if any(key.startswith(prefix) for key in state_dict):
        stripped = {}
        for key, value in state_dict.items():
            while key.startswith(prefix):
                key = key[len(prefix) :]
            stripped[key] = value
        return stripped
    return state_dict


def _filter_model_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Only keep MLP weights (drop pos_weight and loss_fn.*).
    return {
        key: value
        for key, value in state_dict.items()
        if key.startswith("net.") or key.startswith("net")
    }


def main() -> int:
    """Command-line tool to convert a PyTorch Lightning .ckpt to a .pt state_dict for the API model."""
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch Lightning .ckpt to a .pt state_dict for the API model."
    )
    parser.add_argument("ckpt_path", type=Path, help="Path to the Lightning checkpoint (.ckpt).")
    parser.add_argument("output_dir", type=Path, help="Directory to write the .pt file into.")
    parser.add_argument(
        "--output-name",
        default="best_model.pt",
        help="Name of the output .pt file (default: best_model.pt).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    args = parser.parse_args()

    ckpt_path = args.ckpt_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / args.output_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_path} (use --overwrite to replace)")

    checkpoint = _load_checkpoint(ckpt_path)
    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain a state_dict.")

    state_dict = _normalize_state_dict(checkpoint["state_dict"])
    state_dict = _filter_model_weights(state_dict)
    if not state_dict:
        raise ValueError("No model weights found after filtering; check checkpoint format.")
    torch.save(state_dict, output_path)

    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
