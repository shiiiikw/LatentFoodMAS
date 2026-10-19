"""Frozen Qwen3-8B backbone: vLLM sampling client + HF hidden-state encoder.

The vLLM worker (rclup/scripts/worker*.sbatch) publishes host/port in
RUNS/worker_<job>.json. Generation goes through the OpenAI-compatible chat
API with Qwen thinking disabled. Hidden states for the representative
generation of each candidate are recovered by a teacher-forced HF prefill of
the identical prompt+completion tokens (Sec. 4.1: final-layer response states).
"""
from __future__ import annotations

import glob
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .common import MODEL_DIR, RUNS


@dataclass
class Sample:
    text: str
    tokens: list[str]
    logprobs: list[float]
    finish_reason: str
    greedy: bool = False

    @property
    def mean_logprob(self) -> float:
        return sum(self.logprobs) / max(1, len(self.logprobs))


@dataclass
class GenUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    by_purpose: dict = field(default_factory=dict)


def discover_worker(job_id: str | None = None) -> dict:
    job_id = job_id or os.environ.get("RCLUP_WORKER_JOB")
    paths = [RUNS / f"worker_{job_id}.json"] if job_id else sorted(glob.glob(str(RUNS / "worker_*.json")), key=os.path.getmtime)
    for path in reversed(list(paths)):
        try:
            status = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") == "ready":
            return status
        if status.get("status") == "starting" and status.get("host"):
            # The sbatch readiness loop may have timed out while weights were still loading; probe directly.
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://{status['host']}:{status['port']}/v1/models", timeout=5) as r:
                    if b"qwen3-8b" in r.read():
                        status["status"] = "ready"
                        Path(path).write_text(json.dumps(status))
                        return status
            except Exception:
                continue
    raise RuntimeError("No ready vLLM worker found; submit rclup/scripts/worker_h100.sbatch")


class VLLMClient:
    """Thread-safe sampling client with usage accounting."""

    def __init__(self, base_url: str | None = None, model: str = "qwen3-8b", timeout: float = 600.0):
        from openai import OpenAI
        if base_url is None:
            status = discover_worker()
            base_url = f"http://{status['host']}:{status['port']}/v1"
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout, max_retries=3)
        self.usage = GenUsage()
        self._lock = threading.Lock()
        from transformers import AutoTokenizer  # eager: lazy imports race across threads
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    def render(self, messages: list[dict], tools=None) -> str:
        """Client-side chat-template rendering (identical to the HF encoder's rendering)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        if tools:
            kwargs["tools"] = tools
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def chat(self, messages: list[dict], *, n: int = 1, temperature: float = 0.0, top_p: float = 1.0,
             max_tokens: int = 512, seed: int | None = None, tools: list[dict] | None = None,
             purpose: str = "gen", stop: list[str] | None = None) -> list[Sample]:
        prompt = self.render(messages, tools)
        kwargs = dict(model=self.model, prompt=prompt, n=n, temperature=temperature, top_p=top_p,
                      max_tokens=max_tokens, logprobs=1, seed=seed, extra_body={"skip_special_tokens": True})
        if stop:
            kwargs["stop"] = stop
        start = time.monotonic()
        response = self.client.completions.create(**kwargs)
        elapsed = time.monotonic() - start
        samples = []
        for choice in response.choices:
            lp = choice.logprobs
            toks = list(lp.tokens) if lp and lp.tokens else []
            lps = [float(x) if x is not None else 0.0 for x in (lp.token_logprobs if lp and lp.token_logprobs else [])]
            samples.append(Sample(text=(choice.text or "").strip(), tokens=toks, logprobs=lps,
                                  finish_reason=choice.finish_reason or "", greedy=temperature == 0.0))
        with self._lock:
            u = self.usage
            u.calls += 1
            u.prompt_tokens += int(getattr(response.usage, "prompt_tokens", 0) or 0)
            u.completion_tokens += int(getattr(response.usage, "completion_tokens", 0) or 0)
            u.wall_seconds += elapsed
            p = u.by_purpose.setdefault(purpose, {"calls": 0, "completion_tokens": 0})
            p["calls"] += 1
            p["completion_tokens"] += int(getattr(response.usage, "completion_tokens", 0) or 0)
        return samples

    def propose(self, messages: list[dict], *, s: int = 8, temperature: float = 0.7, top_p: float = 0.9,
                max_tokens: int = 512, seed: int = 0, tools=None, purpose: str = "propose") -> list[Sample]:
        """Appendix A.1: S stochastic proposals plus one greedy proposal."""
        stochastic = self.chat(messages, n=s, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                               seed=seed, tools=tools, purpose=purpose)
        greedy = self.chat(messages, n=1, temperature=0.0, max_tokens=max_tokens, seed=seed, tools=tools, purpose=purpose + ":greedy")
        return greedy + stochastic

    def usage_dict(self) -> dict:
        return {"calls": self.usage.calls, "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens, "wall_seconds": self.usage.wall_seconds,
                "by_purpose": self.usage.by_purpose}


class HiddenEncoder:
    """Teacher-forced final-layer states for a completion given its prompt."""

    def __init__(self, model_dir: str | None = None, device: str = "cuda", dtype: str = "bfloat16", max_tokens: int = 16384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if model_dir is None:
            local = Path("/local/qwen3-8b")
            model_dir = str(local) if (local / "config.json").exists() else MODEL_DIR
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=getattr(torch, dtype),
                                                          attn_implementation="sdpa", low_cpu_mem_usage=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_tokens = max_tokens
        self.hidden_size = int(self.model.config.hidden_size)
        self.calls = 0
        self.tokens = 0
        self.wall = 0.0

    def render(self, messages: list[dict], tools=None) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _mean_pool(h):
        return h.float().mean(0)

    def encode(self, messages: list[dict], completion: str, tools=None, pool_prompt: bool = True):
        """Return (completion_states [L_c, d], prompt_mean [d] or None)."""
        prompt = self.render(messages, tools)
        p_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        c_ids = self.tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(c_ids) == 0:
            c_ids = self.tokenizer("<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = self.torch.cat([p_ids, c_ids])[: self.max_tokens]
        start = time.monotonic()
        with self.torch.inference_mode():
            out = self.model.model(input_ids=ids.unsqueeze(0).to(self.device), use_cache=False)
            h = out.last_hidden_state[0]
            comp = h[len(p_ids):].to(self.torch.float16).cpu()
            prompt_mean = self._mean_pool(h[: len(p_ids)]).to(self.torch.float16).cpu() if pool_prompt else None
        self.wall += time.monotonic() - start
        self.calls += 1
        self.tokens += int(len(ids))
        return comp, prompt_mean

    def encode_text(self, text: str):
        """Mean-pooled final-layer state of a bare text (for enc(b), enc(q), task state)."""
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0][: self.max_tokens]
        if len(ids) == 0:
            return self.torch.zeros(self.hidden_size, dtype=self.torch.float16)
        start = time.monotonic()
        with self.torch.inference_mode():
            h = self.model.model(input_ids=ids.unsqueeze(0).to(self.device), use_cache=False).last_hidden_state[0]
            pooled = self._mean_pool(h).to(self.torch.float16).cpu()
        self.wall += time.monotonic() - start
        self.calls += 1
        self.tokens += int(len(ids))
        return pooled

    def usage_dict(self) -> dict:
        return {"calls": self.calls, "tokens": self.tokens, "wall_seconds": self.wall}
