# Claude Agent Story Teller

A short bedtime-story loop built with the Claude Agent SDK. One Python file defines three subagents and calls them in order:

1. **storyteller** writes a story for ages 5–10.
2. **judge** returns a score from 1 to 10 as JSON.
3. **rewriter** revises the story when the score is 7 or lower.

Python stops after two rewrites, or as soon as a draft scores above 7. The model is the same Ollama model used by the research agent, `gemma4:31b-cloud` at `http://localhost:11434`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Run

```powershell
python agent.py
```

One topic:

```powershell
python agent.py "a shy fox who helps a lost duck"
```

Type `exit` to leave the prompt.

## License

MIT. See [LICENSE](LICENSE).
