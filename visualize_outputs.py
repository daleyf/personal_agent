#!/usr/bin/env python3
"""Summarise agent artifacts into a concise highlights markdown file.

This script reads all artifact files produced by ``foundry_agent.py`` and
uses the same local LLM via Ollama to extract only the key insights.  The
result is written to ``highlights.md`` within the given output directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from foundry_agent import (
    ARTIFACT_FILENAMES,
    MAX_INPUT_CHARS,
    call_ollama,
    now_iso,
    read_text,
    write_text,
    DEFAULT_OPTIONS,
)


def gather_artifacts(out_root: Path) -> str:
    """Return concatenated text of all artifact files under ``out_root``."""
    artifacts_dir = out_root / "artifacts"
    pieces = []
    for fname in ARTIFACT_FILENAMES.values():
        path = artifacts_dir / fname
        text = read_text(path)
        if text:
            pieces.append(f"## {fname}\n\n{text}\n")
    return "\n".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize and summarise agent outputs into highlights.md"
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Path to outputs/<basename> directory",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b-instruct",
        help="Ollama model tag",
    )
    parser.add_argument(
        "--timeout", type=int, default=240, help="Per-call timeout seconds"
    )
    parser.add_argument(
        "--ctx", type=int, default=DEFAULT_OPTIONS["num_ctx"], help="Model context window"
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_OPTIONS["temperature"], help="Temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_OPTIONS["top_p"], dest="top_p", help="Top-p"
    )
    args = parser.parse_args()

    args.options = {
        "temperature": args.temp,
        "num_ctx": args.ctx,
        "top_p": args.top_p,
    }

    out_root = Path(args.base).expanduser().resolve()
    if not out_root.exists():
        raise SystemExit(f"Output directory not found: {out_root}")

    text = gather_artifacts(out_root)
    if not text:
        raise SystemExit("No artifacts found to summarise")

    prompt = (
        "Extract only the most relevant and surprising insights from the agent "
        "artifacts below. Provide concise bullet points and avoid repetition.\n\n"
        + text[:MAX_INPUT_CHARS]
    )

    resp = call_ollama(args.model, prompt, args.options, args.timeout)

    out_file = out_root / "highlights.md"
    header = f"# Key Insights (generated {now_iso()})\n\n"
    write_text(out_file, header + resp.strip() + "\n")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
