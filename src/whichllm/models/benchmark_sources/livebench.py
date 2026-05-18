"""LiveBench (livebench.ai) source.

LiveBench is a monthly-refreshed benchmark covering Reasoning, Math, Coding,
Language, IF, and Data Analysis. The leaderboard is published as a JSON
payload embedded in the Next.js site; the same defensive parsing strategy as
:mod:`whichllm.models.benchmark_sources.aa_index` is used.

We map LiveBench display names to HuggingFace IDs through a small alias table.
Anything we cannot map is silently dropped.
"""

from __future__ import annotations

import json
import logging

import httpx

from whichllm.models.benchmark_sources.constants import _NEXT_DATA_RE
from whichllm.models.benchmark_sources.types import ExtractionFailed
from whichllm.models.benchmark_sources.utils import _walk

logger = logging.getLogger(__name__)

LIVEBENCH_URL = "https://livebench.ai/"

# Curated LiveBench global-average snapshot from 2026-04 (the last refresh
# verifiable at research time). Values are raw 0-100; the same normalizer is
# applied as the live path.
LIVEBENCH_FALLBACK_2026_04: dict[str, float] = {
    "deepseek-ai/DeepSeek-R1-0528": 71.0,
    "deepseek-ai/DeepSeek-R1": 65.0,
    "deepseek-ai/DeepSeek-V3.2": 64.0,
    "deepseek-ai/DeepSeek-V3-0324": 57.0,
    "deepseek-ai/DeepSeek-V4-Flash": 66.0,
    "deepseek-ai/DeepSeek-V4-Pro": 72.0,
    "Qwen/Qwen3-235B-A22B": 65.0,
    "Qwen/Qwen3-32B": 60.0,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 63.0,
    "Qwen/Qwen3.6-27B": 66.0,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": 58.0,
    "Qwen/QwQ-32B": 57.0,
    "Qwen/Qwen3-4B-Thinking-2507": 50.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 56.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 50.0,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 42.0,
    "meta-llama/Llama-3.3-70B-Instruct": 48.0,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 54.0,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 49.0,
    "google/gemma-3-27b-it": 50.0,
    "google/gemma-4-31b-it": 56.0,
    "google/gemma-4-26b-a4b-it": 54.0,
    "microsoft/phi-4": 53.0,
    "mistralai/Mistral-Large-Instruct-2411": 58.0,
    "mistralai/Devstral-Small-2505": 50.0,
    "openai/gpt-oss-20b": 52.0,
    "openai/gpt-oss-120b": 60.0,
    "zai-org/GLM-4.5": 58.0,
    "zai-org/GLM-4.5-Air": 52.0,
    "zai-org/GLM-5": 67.0,
    "zai-org/GLM-5.1": 68.0,
    "moonshotai/Kimi-K2-Instruct": 62.0,
    "XiaomiMiMo/MiMo-V2.5-Pro": 70.0,
    # 8B-class entries to anchor the smaller-model scoring
    "Qwen/Qwen3-8B": 50.0,
    "Qwen/Qwen3-14B": 56.0,
    "Qwen/Qwen3-4B-Instruct-2507": 45.0,
    "Qwen/Qwen3-4B": 42.0,
    "Qwen/Qwen3-30B-A3B": 58.0,
    "Qwen/Qwen2.5-7B-Instruct": 38.0,
    "Qwen/Qwen2.5-14B-Instruct": 42.0,
    "Qwen/Qwen2.5-32B-Instruct": 50.0,
    "meta-llama/Llama-3.1-8B-Instruct": 36.0,
    "google/gemma-2-9b-it": 38.0,
    "google/gemma-3-12b-it": 44.0,
    "microsoft/Phi-4-mini-instruct": 40.0,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 50.0,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 48.0,
}

# LiveBench global-average tops out around 72 for current frontier models
# (DeepSeek V4 Pro 72, Kimi K2.6 71, Qwen3.6-27B 66) with 8B-class around 35.
# Anchored by a two-point fit: top frontier (72) → 95, mid/8B (35) → 30 — so
# 8B models get a meaningful but not dominant share and frontier MoE clears
# the OLLB cap.
_LB_MIN = 18.0
_LB_MAX = 75.0

LIVEBENCH_NAME_TO_HF_IDS: dict[str, list[str]] = {
    "DeepSeek-R1": ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1-0528"],
    "DeepSeek-V3": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324"],
    "DeepSeek-V3.1": ["deepseek-ai/DeepSeek-V3.1"],
    "DeepSeek-V3.2": ["deepseek-ai/DeepSeek-V3.2"],
    "DeepSeek-V4-Pro": ["deepseek-ai/DeepSeek-V4-Pro"],
    "DeepSeek-V4-Flash": ["deepseek-ai/DeepSeek-V4-Flash"],
    "Qwen3-235B-A22B": ["Qwen/Qwen3-235B-A22B"],
    "Qwen3-32B": ["Qwen/Qwen3-32B"],
    "Qwen3-Next-80B-A3B": ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    "Qwen3.6-27B": ["Qwen/Qwen3.6-27B"],
    "Llama-3.3-70B": ["meta-llama/Llama-3.3-70B-Instruct"],
    "Llama-4-Maverick": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
    "Gemma-3-27B-it": ["google/gemma-3-27b-it"],
    "Gemma-4-31B": ["google/gemma-4-31b-it"],
    "Phi-4": ["microsoft/phi-4"],
    "Mistral-Large-2": ["mistralai/Mistral-Large-Instruct-2411"],
    "gpt-oss-120b": ["openai/gpt-oss-120b"],
    "gpt-oss-20b": ["openai/gpt-oss-20b"],
    "GLM-4.5": ["zai-org/GLM-4.5"],
    "GLM-5": ["zai-org/GLM-5"],
    "GLM-5.1": ["zai-org/GLM-5.1"],
    "Kimi-K2": ["moonshotai/Kimi-K2-Instruct"],
    "MiMo-V2.5-Pro": ["XiaomiMiMo/MiMo-V2.5-Pro"],
}


def _extract_lb_pairs(payload: dict) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for node in _walk(payload):
        name = None
        score = None
        for k in ("model", "Model", "model_name", "modelName", "name"):
            v = node.get(k)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break
        for k in (
            "global_average",
            "Average",
            "average",
            "global_score",
            "globalScore",
            "score",
            "livebench_score",
            "live_bench_score",
        ):
            v = node.get(k)
            if isinstance(v, (int, float)):
                score = float(v)
                break
        if name and score is not None and score > 0:
            pairs.append((name, score))
    return pairs


def _normalize_livebench(score: float) -> float:
    if not isinstance(score, (int, float)):
        return 0.0
    # LiveBench is usually reported as 0-100 already; treat values <= 1 as
    # fractions (some pages use 0-1) and rescale.
    if score <= 1.0:
        score *= 100.0
    span = _LB_MAX - _LB_MIN
    normalized = (score - _LB_MIN) / span * 100.0
    return max(0.0, min(100.0, round(normalized, 1)))


async def fetch_livebench_scores(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch LiveBench scores. Raises on HTTP / parse failure."""
    scores: dict[str, float] = {}
    resp = await client.get(LIVEBENCH_URL)
    resp.raise_for_status()
    match = _NEXT_DATA_RE.search(resp.text)
    if not match:
        raise ExtractionFailed("__NEXT_DATA__ payload not found")
    payload = json.loads(match.group("json"))
    pairs = _extract_lb_pairs(payload)
    if not pairs:
        raise ExtractionFailed("no (name, score) pairs extracted, using fallback")
    best_by_name: dict[str, float] = {}
    for name, score in pairs:
        cur = best_by_name.get(name)
        if cur is None or score > cur:
            best_by_name[name] = score
    for name, score in best_by_name.items():
        ids = LIVEBENCH_NAME_TO_HF_IDS.get(name)
        if not ids:
            continue
        normalized = _normalize_livebench(score)
        if normalized <= 0:
            continue
        for hf_id in ids:
            if scores.get(hf_id, 0.0) < normalized:
                scores[hf_id] = normalized
    if not scores:
        raise ExtractionFailed("live fetch returned 0 mapped scores")
    logger.debug(f"LiveBench: {len(scores)} mapped scores")
    return scores


def get_livebench_curated_fallback() -> dict[str, float]:
    """Return the 2026-04 curated snapshot, normalized to the 0-100 scale.

    Used when the live HTML scrape cannot extract data. This is the same
    pattern as :func:`whichllm.models.benchmark_sources.aa_index._curated_fallback`.
    """
    return {
        hf_id: _normalize_livebench(raw)
        for hf_id, raw in LIVEBENCH_FALLBACK_2026_04.items()
        if _normalize_livebench(raw) > 0
    }
