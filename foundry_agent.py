"""Minimal local LLM agent that maps over a large text file and reduces
results into readable Markdown artifacts.

The script targets small laptops and uses the Ollama HTTP API to run
quantised models such as `llama3.2:3b-instruct`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import requests

# -------------------- Config --------------------
DEFAULT_SLEEP_SECONDS = 120          # pause between iterations
CHUNK_BYTES = 48_000                 # ~48KB per map chunk
CHUNK_OVERLAP = 2_000                # overlap to avoid cutting thoughts
MAX_INPUT_CHARS = 8_000              # safety cap for prompt content
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_OPTIONS = {
    "temperature": 0.4,
    "num_ctx": 4096,
    "top_p": 0.9,
}

LENSES = [
    "outline",
    "summary",
    "insights",
    "todos",
    "business_plan",
    "questions",
]

ARTIFACT_FILENAMES = {
    "outline": "outline.md",
    "summary": "summary.md",
    "insights": "insights.md",
    "todos": "users_TODOs.md",
    "business_plan": "business_plan.md",
    "questions": "open_questions.md",
}

# -------------------- Helpers --------------------

def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")

def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_chunk(fp: Path, cursor: int, size: int) -> str:
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(cursor)
        data = f.read(size)
    return data

def append_text(path: Path, text: str) -> None:
    ensure_dirs(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)

def write_text(path: Path, text: str) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)

def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")

def load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}

def save_state(path: Path, state: Dict[str, Any]) -> None:
    ensure_dirs(path.parent)
    path.write_text(json.dumps(state, indent=2))

def call_ollama(model: str, prompt: str, options: Dict[str, Any], timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "options": options,
    }
    url = f"{OLLAMA_BASE_URL}/api/generate"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        # Ollama streams responses line by line; concatenate all outputs
        text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            part = json.loads(line.decode("utf-8"))
            text += part.get("response", "")
            if part.get("done"):
                break
        return text
    except Exception as e:
        return f"ERROR: {e}"

# -------------------- Map / Reduce --------------------

def process_map_step(args: argparse.Namespace, lens: str, state: Dict[str, Any],
                     file_path: Path, file_size: int, out_root: Path) -> str:
    lens_state = state["lenses"].setdefault(lens, {"cursor": 0, "phase": "map", "maps": 0})
    cursor = lens_state.get("cursor", 0)
    if cursor >= file_size:
        lens_state["phase"] = "reduce"
        lens_state["cursor"] = 0
        return "done"

    chunk = read_chunk(file_path, cursor, CHUNK_BYTES)
    prompt = f"You are analysing a document for the '{lens}' lens. Extract the relevant points from the following text:\n\n{chunk[:MAX_INPUT_CHARS]}"
    resp = call_ollama(args.model, prompt, args.options, args.timeout)
    maps_file = out_root / "maps" / f"{lens}.md"
    append_text(maps_file, f"## chunk @ {cursor}\n\n{resp.strip()}\n\n")
    lens_state["cursor"] = min(file_size, cursor + CHUNK_BYTES - CHUNK_OVERLAP)
    lens_state["maps"] = lens_state.get("maps", 0) + 1
    state["last_action"] = f"map:{lens}"
    return "ok"

def process_reduce_step(args: argparse.Namespace, lens: str, state: Dict[str, Any], out_root: Path) -> str:
    maps_file = out_root / "maps" / f"{lens}.md"
    text = read_text(maps_file)
    if not text:
        return "skip"
    prompt = f"Summarise the following notes into a coherent {lens.replace('_', ' ')}:\n\n{text[:MAX_INPUT_CHARS]}"
    resp = call_ollama(args.model, prompt, args.options, args.timeout)
    art_file = out_root / "artifacts" / ARTIFACT_FILENAMES[lens]
    header = f"# {lens.replace('_', ' ').title()} (updated {now_iso()})\n\n"
    write_text(art_file, header + resp.strip() + "\n")
    lens_state = state["lenses"][lens]
    lens_state["phase"] = "map"
    lens_state["cursor"] = 0
    lens_state["maps"] = 0
    state["last_action"] = f"reduce:{lens}"
    return "ok"

# -------------------- Status --------------------

def write_status(out_root: Path, state: Dict[str, Any]) -> None:
    lines = [
        f"status updated: {now_iso()}",
        f"iteration: {state.get('iteration', 0)}",
        f"model: {state.get('model', '')}",
        f"file_size_bytes: {state.get('file_size', 0)}",
        "",
        "lenses:",
    ]
    for l, s in state.get("lenses", {}).items():
        lines.append(f"- {l}: phase={s.get('phase')} cursor={s.get('cursor')} maps={s.get('maps')}")
    write_text(out_root / "status.md", "\n".join(lines) + "\n")

# -------------------- Main loop --------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Local looping LLM agent for large .txt (map/reduce to artifacts).")
    parser.add_argument("--file", required=True, help="Path to large .txt")
    parser.add_argument("--model", default="llama3.2:3b-instruct", help="Ollama model tag")
    parser.add_argument("--sleep", type=int, default=DEFAULT_SLEEP_SECONDS, help="Seconds to sleep between iterations")
    parser.add_argument("--timeout", type=int, default=240, help="Per-call timeout seconds")
    parser.add_argument("--ctx", type=int, default=DEFAULT_OPTIONS["num_ctx"], help="Model context window")
    parser.add_argument("--temp", type=float, default=DEFAULT_OPTIONS["temperature"], help="Temperature")
    parser.add_argument("--top-p", type=float, default=DEFAULT_OPTIONS["top_p"], dest="top_p", help="Top-p")
    parser.add_argument("--once", action="store_true", help="Run a single iteration then exit")
    args = parser.parse_args()

    args.options = {
        "temperature": args.temp,
        "num_ctx": args.ctx,
        "top_p": args.top_p,
    }

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")
    file_size = file_path.stat().st_size
    file_mtime = file_path.stat().st_mtime

    base = file_path.stem
    out_root = Path("outputs") / base
    ensure_dirs(out_root)

    state_path = out_root / "state.json"
    state = load_state(state_path)
    state.setdefault("file", str(file_path))
    state["model"] = args.model
    state.setdefault("iteration", 0)
    state.setdefault("lenses", {})
    state["file_size"] = file_size
    state["file_mtime"] = file_mtime
    state.setdefault("last_lens_idx", -1)

    append_text(out_root / "run.log", f"[{now_iso()}] START agent on {file_path} with model={args.model}\n")

    try:
        while True:
            cur_size = file_path.stat().st_size
            cur_mtime = file_path.stat().st_mtime
            if cur_size != state.get("file_size") or cur_mtime != state.get("file_mtime"):
                append_text(out_root / "run.log", f"[{now_iso()}] File changed; resetting map cursors.\n")
                for l in LENSES:
                    state["lenses"][l] = {"cursor": 0, "phase": "map", "maps": 0}
                state["file_size"] = cur_size
                state["file_mtime"] = cur_mtime

            lens_idx = (state["last_lens_idx"] + 1) % len(LENSES)
            lens = LENSES[lens_idx]
            state["last_lens_idx"] = lens_idx

            lens_state = state["lenses"].setdefault(lens, {"cursor": 0, "phase": "map", "maps": 0})
            phase = lens_state.get("phase", "map")

            if phase == "map":
                process_map_step(args, lens, state, file_path, state["file_size"], out_root)
            else:
                process_reduce_step(args, lens, state, out_root)

            save_state(state_path, state)
            write_status(out_root, state)

            if args.once:
                break

            if lens_idx == len(LENSES) - 1:
                state["iteration"] += 1

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        append_text(out_root / "run.log", f"[{now_iso()}] STOP (KeyboardInterrupt)\n")
        save_state(state_path, state)
        write_status(out_root, state)

if __name__ == "__main__":
    main()

