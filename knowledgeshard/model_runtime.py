"""Optional Mistral/LoRA runtime integration.

The core package must keep importing on machines without ML dependencies. This
module therefore loads transformers/peft lazily and reports clear fallback
reasons to callers.
"""

from __future__ import annotations

import os
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
ENV_FILE = ".env"


def load_dotenv(path: str | Path = ENV_FILE) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    model_path: str | None = None
    lora_path: str | None = None
    device: str = "auto"
    enabled: bool = False
    backend: str = "ollama"
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    @classmethod
    def from_env(cls) -> "ModelConfig":
        load_dotenv()
        return cls(
            model_id=os.getenv("KS_MODEL_ID", DEFAULT_MODEL_ID),
            model_path=os.getenv("KS_MODEL_PATH") or None,
            lora_path=os.getenv("KS_LORA_PATH") or None,
            device=os.getenv("KS_DEVICE", "auto"),
            enabled=os.getenv("KS_ENABLE_MODEL", "").lower() in {"1", "true", "yes"},
            backend=os.getenv("KS_MODEL_BACKEND", "ollama").lower(),
            ollama_url=os.getenv("KS_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/"),
            ollama_model=os.getenv("KS_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        )


class OptionalModelRuntime:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self._pipeline = None
        self.error: str | None = None
        self.backend_loaded = False

    @property
    def available(self) -> bool:
        if not self.config.enabled:
            self.error = "model runtime disabled"
            return False
        if self.config.backend == "ollama":
            return self._ollama_available()
        if self.config.backend == "transformers":
            return self._load_transformers()
        self.error = f"unsupported model backend: {self.config.backend}"
        return False

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str | None:
        if not self.available:
            return None
        if self.config.backend == "ollama":
            return self._generate_ollama(prompt)
        if self._pipeline is None:
            return None
        output = self._pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = output[0]["generated_text"]
        return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()

    def status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "backend": self.config.backend,
            "model": self.config.ollama_model if self.config.backend == "ollama" else self.config.model_path or self.config.model_id,
            "available": self.available,
            "loaded": self.backend_loaded or self._pipeline is not None,
            "error": self.error,
        }

    def _ollama_available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.config.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self.error = f"ollama unavailable at {self.config.ollama_url}: {exc}"
            return False
        models = {item.get("name", "").split(":")[0] for item in payload.get("models", [])}
        if self.config.ollama_model.split(":")[0] not in models:
            self.error = f"ollama model '{self.config.ollama_model}' is not installed"
            return False
        self.error = None
        self.backend_loaded = True
        return True

    def _generate_ollama(self, prompt: str) -> str | None:
        body = json.dumps(
            {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.ollama_url}/api/generate",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self.error = f"ollama generation failed: {exc}"
            return None
        self.error = None
        return str(payload.get("response", "")).strip() or None

    def _load_transformers(self) -> bool:
        if self._pipeline is not None:
            return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            try:
                from peft import PeftModel
            except ImportError:
                PeftModel = None
        except ImportError as exc:
            self.error = f"optional ML dependency missing: {exc}"
            return False

        model_name = self.config.model_path or self.config.model_id
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.config.device)
        if self.config.lora_path:
            if PeftModel is None:
                self.error = "peft is required to load LoRA adapters"
                return False
            model = PeftModel.from_pretrained(model, self.config.lora_path)
        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.error = None
        self.backend_loaded = True
        return True


def train_lora(
    domain: str,
    db_path: str | Path,
    output_dir: str | Path = "weights/lora",
    model_id: str | None = None,
) -> dict:
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        return {"trained": False, "reason": f"optional ML dependency missing: {exc}"}

    from .storage import KnowledgeStore

    store = KnowledgeStore(db_path)
    facts = store.list_facts(domain)
    if not facts:
        return {"trained": False, "reason": f"no approved facts found for domain {domain}"}

    examples = [
        {
            "text": (
                "### Instruction\nAnswer with cited Mario Kart Wii knowledge.\n"
                f"### Input\nWhat should I know about {fact.subject}?\n"
                f"### Response\n{fact.text}. Source: {fact.source}."
            )
        }
        for fact in facts
    ]
    model_name = model_id or os.getenv("KS_MODEL_ID", DEFAULT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(examples).map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=os.getenv("KS_DEVICE", "auto"))
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    target = Path(output_dir) / domain
    args = TrainingArguments(
        output_dir=str(target),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    return {"trained": True, "domain": domain, "facts": len(facts), "adapter_path": str(target)}
