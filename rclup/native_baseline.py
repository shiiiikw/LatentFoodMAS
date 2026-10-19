"""Official HoosierHelp single-agent baseline (vendor runner, unchanged agent prompt/loop) with the same Qwen3-8B
backbone served by our vLLM worker and the gpt-4.1-mini simulated user.

  python -m rclup.native_baseline --out baselines/native_single_agent --jobs 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .backbone import discover_worker
from .common import RUNS, load_openai_key, ensure_vendor_on_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="baselines/native_single_agent")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--user-model", default="gpt-4.1-mini")
    p.add_argument("--behaviors", default="normal,rambling,impatience,self_contradictory,unsupported_request")
    a = p.parse_args()
    status = discover_worker()
    os.environ["LLAMA_CPP_BASE_URL"] = f"http://{status['host']}:{status['port']}/v1"
    os.environ["LLAMA_CPP_API_KEY"] = "vllm"
    os.environ["OPENAI_API_KEY"] = load_openai_key()
    vendor = Path(__file__).resolve().parent.parent / "vendor" / "hoosierhelp"
    ensure_vendor_on_path()
    os.chdir(vendor)   # the runner resolves data paths relative to the repo root
    from eval.runner import EvalConfig, run as run_eval
    cfg = EvalConfig()
    cfg.backend, cfg.provider, cfg.model = "llama_cpp", "openai", "qwen3-8b"
    cfg.agent_generation_token_limit, cfg.agent_enable_thinking, cfg.agent_temperature = 2048, False, 0.0
    cfg.user_provider, cfg.user_model, cfg.user_temperature, cfg.user_enable_thinking = "openai", a.user_model, 0.0, False
    cfg.sample_count, cfg.sample_seed, cfg.user_seed, cfg.max_agent_turns = 48, 1, 7, 8
    cfg.user_behaviors = a.behaviors.split(",")
    cfg.jobs = a.jobs
    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    cfg.output_dir = out
    run_eval(cfg)


if __name__ == "__main__":
    main()
