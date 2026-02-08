#!/usr/bin/env python3
"""Run live SFT+GRPO smoke checks across model families.

This script is intended for CI/ops validation with real Tinker credentials.
It runs tiny, low-cost training steps to verify harness compatibility across
multiple model families and catches runtime regressions early.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODEL_FAMILY_DEFAULTS = {
    "base": "Qwen/Qwen3-30B-A3B-Base",
    "instruction": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "hybrid": "Qwen/Qwen3-30B-A3B",
    "reasoning": "openai/gpt-oss-20b",
}

SFT_EXAMPLES = [
    {"prompt": "Return exactly the token OK.", "response": "OK"},
    {"prompt": "Return exactly the token READY.", "response": "READY"},
]

GRPO_PROMPTS = [
    {"prompt": "Write one short sentence about apples.", "ground_truth": "apples"},
    {"prompt": "Write one short sentence about oceans.", "ground_truth": "oceans"},
]

GRPO_REWARD_FUNCTION = """
def compute_reward(completion: str, ground_truth: str) -> float:
    text = (completion or "").strip().lower()
    if not text:
        return 0.0
    coverage = 0.8 if ground_truth.lower() in text else 0.2
    # Deterministic shaping to reduce all-tie groups in smoke mode.
    shape = (sum(ord(c) for c in text[:64]) % 21) / 100.0
    score = coverage + shape
    return float(max(0.0, min(1.0, score)))
""".strip()

USED_FOR_TRAINING_RE = re.compile(r"Used for training:\s*(\d+)")


@dataclass
class MethodResult:
    ok: bool
    duration_sec: float
    summary: str
    error: str | None = None
    used_for_training: int | None = None


@dataclass
class FamilyResult:
    family: str
    model: str
    sft: MethodResult | None = None
    grpo: MethodResult | None = None
    started_at: float = field(default_factory=time.time)


def _is_error_response(response: str) -> bool:
    text = (response or "").lstrip()
    lowered = text.lower()
    # Tinker MCP tools are human-readable; errors are conventionally prefixed with "Error"
    # but some older paths omit the colon (e.g., "Error initializing ...").
    return lowered.startswith("error") or lowered.startswith("critical") or "error in " in lowered


def _extract_summary(response: str, max_lines: int = 8) -> str:
    lines = [line.rstrip() for line in (response or "").strip().splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def _require_api_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY is required for live smoke tests. "
            "Set it in your environment or CI secret configuration."
        )


def _parse_used_for_training(report: str) -> int | None:
    match = USED_FOR_TRAINING_RE.search(report or "")
    if not match:
        return None
    return int(match.group(1))


async def _call_with_timeout(coro, timeout_sec: float, operation: str) -> str:
    """Run a server coroutine with a bounded timeout and clear error."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError:
        return f"Error: Timed out after {timeout_sec:.0f}s during {operation}."


async def _run_sft(
    server,
    model: str,
    *,
    lora_rank: int,
    debug: bool,
    method_timeout_sec: float,
) -> MethodResult:
    started = time.time()
    response = await _call_with_timeout(
        server.init_sft(model, lora_rank=lora_rank, debug=debug),
        method_timeout_sec,
        f"SFT init ({model})",
    )
    if _is_error_response(response):
        return MethodResult(
            ok=False,
            duration_sec=time.time() - started,
            summary=_extract_summary(response),
            error=response,
        )
    try:
        response = await _call_with_timeout(
            server.train_sft_step(
                examples=json.dumps(SFT_EXAMPLES),
                num_epochs=1,
                learning_rate=5e-5,
                batch_size=2,
                warmup_ratio=0.1,
                lr_scheduler="constant",
                validation_split=0.0,
                debug=debug,
            ),
            method_timeout_sec,
            f"SFT train ({model})",
        )
        if _is_error_response(response):
            return MethodResult(
                ok=False,
                duration_sec=time.time() - started,
                summary=_extract_summary(response),
                error=response,
            )
        return MethodResult(
            ok=True,
            duration_sec=time.time() - started,
            summary=_extract_summary(response),
        )
    finally:
        await _call_with_timeout(server.finish(debug=debug), method_timeout_sec, f"SFT finish ({model})")


async def _run_grpo(
    server,
    model: str,
    *,
    lora_rank: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    debug: bool,
    method_timeout_sec: float,
) -> MethodResult:
    started = time.time()
    response = await _call_with_timeout(
        server.init_grpo(model, lora_rank=lora_rank, group_size=group_size, debug=debug),
        method_timeout_sec,
        f"GRPO init ({model})",
    )
    if _is_error_response(response):
        return MethodResult(
            ok=False,
            duration_sec=time.time() - started,
            summary=_extract_summary(response),
            error=response,
        )
    try:
        response = await _call_with_timeout(
            server.train_grpo_step(
                prompts=json.dumps(GRPO_PROMPTS),
                reward_function=GRPO_REWARD_FUNCTION,
                num_iterations=1,
                learning_rate=1e-5,
                warmup_ratio=0.1,
                lr_scheduler="constant",
                temperature=temperature,
                max_tokens=max_tokens,
                auto_checkpoint=False,
                sampling_debug_prompt_limit=0,
                debug=debug,
            ),
            method_timeout_sec,
            f"GRPO train ({model})",
        )
        used_for_training = _parse_used_for_training(response)
        if _is_error_response(response):
            return MethodResult(
                ok=False,
                duration_sec=time.time() - started,
                summary=_extract_summary(response),
                error=response,
                used_for_training=used_for_training,
            )
        # Note: Instruction-tuned models may produce uniform rewards (all samples contain
        # ground truth), resulting in zero variance and no training samples. This is expected
        # behavior - GRPO correctly skips uniform-reward groups. Consider using --allow-failures=1
        # for smoke tests if this occurs.
        if used_for_training is None or used_for_training <= 0:
            return MethodResult(
                ok=False,
                duration_sec=time.time() - started,
                summary=_extract_summary(response),
                error="GRPO completed but reported no training samples used.",
                used_for_training=used_for_training,
            )
        return MethodResult(
            ok=True,
            duration_sec=time.time() - started,
            summary=_extract_summary(response),
            used_for_training=used_for_training,
        )
    finally:
        await _call_with_timeout(server.finish(debug=debug), method_timeout_sec, f"GRPO finish ({model})")


async def _run_family(
    server,
    *,
    family: str,
    methods: Sequence[str],
    lora_rank: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    debug: bool,
    method_timeout_sec: float,
) -> FamilyResult:
    model = MODEL_FAMILY_DEFAULTS[family]
    result = FamilyResult(family=family, model=model)
    if "sft" in methods:
        result.sft = await _run_sft(
            server,
            model,
            lora_rank=lora_rank,
            debug=debug,
            method_timeout_sec=method_timeout_sec,
        )
    if "grpo" in methods:
        result.grpo = await _run_grpo(
            server,
            model,
            lora_rank=lora_rank,
            group_size=group_size,
            max_tokens=max_tokens,
            temperature=temperature,
            debug=debug,
            method_timeout_sec=method_timeout_sec,
        )
    return result


def _print_report(results: list[FamilyResult]) -> None:
    print("\n=== Live Model Matrix Smoke Report ===")
    for family_result in results:
        print(f"\n[{family_result.family}] {family_result.model}")
        if family_result.sft is not None:
            status = "PASS" if family_result.sft.ok else "FAIL"
            print(f"  SFT : {status} ({family_result.sft.duration_sec:.1f}s)")
            if family_result.sft.error:
                print(f"    error: {family_result.sft.error.splitlines()[0]}")
        if family_result.grpo is not None:
            status = "PASS" if family_result.grpo.ok else "FAIL"
            used = (
                f", used_for_training={family_result.grpo.used_for_training}"
                if family_result.grpo.used_for_training is not None
                else ""
            )
            print(f"  GRPO: {status} ({family_result.grpo.duration_sec:.1f}s{used})")
            if family_result.grpo.error:
                print(f"    error: {family_result.grpo.error.splitlines()[0]}")


def _to_json(results: list[FamilyResult]) -> list[dict]:
    payload: list[dict] = []
    for item in results:
        row = asdict(item)
        payload.append(row)
    return payload


async def _main_async(args: argparse.Namespace) -> int:
    _require_api_key()
    if not args.enable_wandb:
        os.environ["WANDB_API_KEY"] = ""

    import tinker_mcp.server as server

    families = [family.strip() for family in args.families.split(",") if family.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    invalid_families = [family for family in families if family not in MODEL_FAMILY_DEFAULTS]
    if invalid_families:
        raise ValueError(f"Unknown families: {', '.join(invalid_families)}")
    invalid_methods = [method for method in methods if method not in {"sft", "grpo"}]
    if invalid_methods:
        raise ValueError(f"Unknown methods: {', '.join(invalid_methods)}")

    results: list[FamilyResult] = []
    for family in families:
        print(f"\n>>> Running {family} smoke checks")
        family_result = await _run_family(
            server,
            family=family,
            methods=methods,
            lora_rank=args.lora_rank,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            debug=args.debug,
            method_timeout_sec=args.method_timeout_sec,
        )
        results.append(family_result)

    _print_report(results)
    failed_checks = 0
    for family_result in results:
        for method_result in (family_result.sft, family_result.grpo):
            if method_result is not None and not method_result.ok:
                failed_checks += 1

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_to_json(results), indent=2))
        print(f"\nReport written to {report_path}")

    if failed_checks > args.allow_failures:
        print(f"\nFAIL: {failed_checks} checks failed (allow_failures={args.allow_failures}).")
        return 1
    print(f"\nPASS: failed_checks={failed_checks}, allow_failures={args.allow_failures}.")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live model-family smoke matrix (SFT+GRPO).")
    parser.add_argument(
        "--families",
        default="base,instruction,hybrid,reasoning",
        help=f"Comma-separated families (available: {', '.join(MODEL_FAMILY_DEFAULTS)})",
    )
    parser.add_argument("--methods", default="sft,grpo", help="Comma-separated methods: sft,grpo")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for smoke runs")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size")
    parser.add_argument("--max-tokens", type=int, default=96, help="GRPO max_tokens for smoke runs")
    parser.add_argument("--temperature", type=float, default=1.0, help="GRPO temperature for smoke runs")
    parser.add_argument(
        "--method-timeout-sec",
        type=float,
        default=1200.0,
        help="Per-tool timeout in seconds (init/train/finish each).",
    )
    parser.add_argument("--allow-failures", type=int, default=0, help="How many failed checks are allowed")
    parser.add_argument("--report-path", default="", help="Optional JSON report output path")
    parser.add_argument("--debug", action="store_true", help="Enable MCP debug logs")
    parser.add_argument("--enable-wandb", action="store_true", help="Keep WANDB_API_KEY enabled during smoke runs")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
