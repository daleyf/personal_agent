# Foundry Agent

Foundry Agent is a tiny Python script that chews on a large `.txt` file using a local LLM and keeps dropping fresh Markdown artifacts.  It is designed for laptops with modest specs (e.g. M2 Air or 16 GB RAM) and relies on [Ollama](https://ollama.com) to run small quantised models such as `llama3.2:3b-instruct`.

## Features

* runs entirely locally – no API keys or cloud calls
* round‑robin "lenses" that examine the file from different angles
* incremental state so the agent can be stopped and resumed
* generates Markdown artifacts:
  * `outline.md`
  * `summary.md`
  * `insights.md`
  * `users_TODOs.md`
  * `business_plan.md`
  * `open_questions.md`

## Installation

1. Install Python 3.9+.
2. Install [Ollama](https://ollama.com/download) and pull a small model:
   ```bash
   ollama pull llama3.2:3b-instruct
   ```
3. Install the Python dependency:
   ```bash
   pip install requests
   ```

## Usage

```bash
python foundry_agent.py --file /path/to/notes.txt --model llama3.2:3b-instruct
```

The agent creates an `outputs/<basename>/` directory next to the script and keeps iterating forever, mapping and reducing each lens in turn.  Press `Ctrl+C` to stop; the next run resumes from the last state.

Useful flags:

* `--sleep SECONDS` – pause between iterations (default 120)
* `--once` – run a single iteration then exit (debug)

## Notes

The model is called through Ollama's HTTP API (`http://localhost:11434`).  Set `OLLAMA_BASE_URL` if your server runs elsewhere.  The code intentionally keeps prompts small and temperature low for faster, more deterministic output.

## License

MIT

