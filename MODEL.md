# Local Model Setup

KnowledgeShard answers from cited SQLite facts by default. To let a local LLM
write the final response from those retrieved citations, enable one model
backend.

## Ollama backend

This is the simplest local path because the Python app stays dependency-light.

```powershell
ollama pull mistral
Copy-Item .env.example .env
python -m knowledgeshard.cli model-status
python -m knowledgeshard.cli ask "Why is Funky Kong on Flame Runner valued?"
```

The prompt sent to the model includes only the top retrieved facts and asks the
model to answer from that evidence. If Ollama is not running or the model is not
installed, the app falls back to the deterministic cited answer.

`.env` is ignored by git, so it can hold your local runtime settings. Real
environment variables still win over values in `.env`.

## Transformers backend

Use this for the Mistral/LoRA path when the machine has suitable ML dependencies
and hardware.

```powershell
python -m pip install -r requirements-ml.txt
$env:KS_ENABLE_MODEL="1"
$env:KS_MODEL_BACKEND="transformers"
$env:KS_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
python -m knowledgeshard.cli model-status
```

LoRA training remains separate:

```powershell
python -m knowledgeshard.cli train-lora
```
