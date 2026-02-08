"""Model information registry for the Tinker MCP Server.

This module contains the MODEL_INFO dictionary with metadata about
all supported models for fine-tuning. Each entry includes:
- training_type: Vision, Instruction, Hybrid, Base, Reasoning
- architecture: MoE (Mixture of Experts) or Dense
- size: Compact, Small, Medium, Large
- total_params: Total model parameters
- active_params: Parameters active per forward pass (key for MoE!)
- recommended_group_size: Optimal group_size for GRPO training
- timeout_risk: low, medium, high - likelihood of timeout
- warning: Optional warning about model tradeoffs

MoE models are RECOMMENDED because you only pay for active params.
A 30B MoE with 3B active is faster than an 8B Dense model.
"""

from typing import Iterable

# Model information registry with architecture and timeout risk
MODEL_INFO = {
    # ============================================
    # MoE MODELS - RECOMMENDED (sparse activation = fast)
    # ============================================
    # Vision MoE
    "Qwen/Qwen3-VL-235B-A22B-Instruct": {
        "training_type": "Vision",
        "architecture": "MoE",
        "size": "Large",
        "total_params": "235B",
        "active_params": "22B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {
        "training_type": "Vision",
        "architecture": "MoE",
        "size": "Medium",
        "total_params": "30B",
        "active_params": "3B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    # Instruction MoE
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "training_type": "Instruction",
        "architecture": "MoE",
        "size": "Large",
        "total_params": "235B",
        "active_params": "22B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "training_type": "Instruction",
        "architecture": "MoE",
        "size": "Medium",
        "total_params": "30B",
        "active_params": "3B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    # Hybrid MoE
    "Qwen/Qwen3-30B-A3B": {
        "training_type": "Hybrid",
        "architecture": "MoE",
        "size": "Medium",
        "total_params": "30B",
        "active_params": "3B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "deepseek-ai/DeepSeek-V3.1": {
        "training_type": "Hybrid",
        "architecture": "MoE",
        "size": "Large",
        "total_params": "671B",
        "active_params": "37B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    # Base MoE
    "Qwen/Qwen3-30B-A3B-Base": {
        "training_type": "Base",
        "architecture": "MoE",
        "size": "Medium",
        "total_params": "30B",
        "active_params": "3B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
        # Keep aligned with the recommended renderer mapping for Base checkpoints.
        # Qwen3 Base uses role_colon, while Instruct/Hybrid variants use qwen3* renderers.
        "renderer_name_override": "role_colon",
    },
    "deepseek-ai/DeepSeek-V3.1-Base": {
        "training_type": "Base",
        "architecture": "MoE",
        "size": "Large",
        "total_params": "671B",
        "active_params": "37B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    # Reasoning MoE
    "openai/gpt-oss-120b": {
        "training_type": "Reasoning",
        "architecture": "MoE",
        "size": "Medium",
        "total_params": "120B",
        "active_params": "~20B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "openai/gpt-oss-20b": {
        "training_type": "Reasoning",
        "architecture": "MoE",
        "size": "Small",
        "total_params": "20B",
        "active_params": "~5B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "moonshotai/Kimi-K2-Thinking": {
        "training_type": "Reasoning",
        "architecture": "MoE",
        "size": "Large",
        "total_params": "1T+",
        "active_params": "~32B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "moonshotai/Kimi-K2.5": {
        "training_type": "Reasoning",
        "architecture": "MoE",
        "size": "Large",
        "total_params": "1T+",
        "active_params": "~32B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
        # This model may not yet be covered by the pinned renderer mapping.
        # Kimi K2 variants share the same chat template, so we can safely use
        # the existing renderer.
        "renderer_name_override": "kimi_k2",
    },
    # ============================================
    # DENSE MODELS - USE WITH CAUTION (all params active)
    # ============================================
    # Dense Large (70B+) - SLOW (use MoE for faster training)
    "meta-llama/Llama-3.1-70B": {
        "training_type": "Base",
        "architecture": "Dense",
        "size": "Large",
        "total_params": "70B",
        "active_params": "70B",
        "recommended_group_size": 16,
        "timeout_risk": "medium",
        "warning": "Dense 70B is slower than MoE. For faster training, use DeepSeek-V3.1 (37B active, same quality).",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "training_type": "Instruction",
        "architecture": "Dense",
        "size": "Large",
        "total_params": "70B",
        "active_params": "70B",
        "recommended_group_size": 16,
        "timeout_risk": "medium",
        "warning": "Dense 70B is slower than MoE. For faster training, use Qwen3-235B-A22B-Instruct-2507 (22B active).",
    },
    # Dense Medium (30-32B) - SLOW (use MoE for faster training)
    "Qwen/Qwen3-32B": {
        "training_type": "Hybrid",
        "architecture": "Dense",
        "size": "Medium",
        "total_params": "32B",
        "active_params": "32B",
        "recommended_group_size": 16,
        "timeout_risk": "medium",
        "warning": "Dense 32B is slower than MoE. For faster training, use Qwen3-30B-A3B (3B active, same quality).",
    },
    # Dense Small (8B) - SLOW (use MoE for faster training)
    "Qwen/Qwen3-8B": {
        "training_type": "Hybrid",
        "architecture": "Dense",
        "size": "Small",
        "total_params": "8B",
        "active_params": "8B",
        "recommended_group_size": 16,
        "timeout_risk": "medium",
        "warning": "Dense 8B is slower than MoE. For faster training, use Qwen3-30B-A3B (3B active, same quality).",
    },
    "Qwen/Qwen3-8B-Base": {
        "training_type": "Base",
        "architecture": "Dense",
        "size": "Small",
        "total_params": "8B",
        "active_params": "8B",
        "recommended_group_size": 16,
        "timeout_risk": "medium",
        "warning": "Dense 8B is slower than MoE. For faster training, use Qwen3-30B-A3B-Base (3B active).",
    },
    "meta-llama/Llama-3.1-8B": {
        "training_type": "Base",
        "architecture": "Dense",
        "size": "Small",
        "total_params": "8B",
        "active_params": "8B",
        "recommended_group_size": 16,
        "timeout_risk": "medium",
        "warning": "Dense 8B is slower than MoE alternatives.",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "training_type": "Instruction",
        "architecture": "Dense",
        "size": "Small",
        "total_params": "8B",
        "active_params": "8B",
        "recommended_group_size": 16,
        "timeout_risk": "medium",
        "warning": "Dense 8B is slower than MoE. For faster training, use Qwen3-30B-A3B-Instruct-2507 (3B active).",
    },
    # Dense Compact (1-4B) - LOW RISK
    "Qwen/Qwen3-4B-Instruct-2507": {
        "training_type": "Instruction",
        "architecture": "Dense",
        "size": "Compact",
        "total_params": "4B",
        "active_params": "4B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "meta-llama/Llama-3.2-3B": {
        "training_type": "Base",
        "architecture": "Dense",
        "size": "Compact",
        "total_params": "3B",
        "active_params": "3B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
    "meta-llama/Llama-3.2-1B": {
        "training_type": "Base",
        "architecture": "Dense",
        "size": "Compact",
        "total_params": "1B",
        "active_params": "1B",
        "recommended_group_size": 4,
        "timeout_risk": "low",
    },
}


def _resolve_renderer_name(model_name: str) -> str:
    """Resolve the recommended renderer name for a model using the Tinker
    renderer mapping.

    Raises:
        ImportError: If the renderer mapping dependency is not installed.
        ValueError: If the model is not recognized by the renderer mapping.
    """
    try:
        from tinker_cookbook import model_info as tinker_model_info
    except ImportError as exc:
        raise ImportError(
            "A renderer mapping dependency is required to determine renderer names. "
            "Install the pinned dependencies to use renderer-aware training."
        ) from exc
    return tinker_model_info.get_recommended_renderer_name(model_name)


def _attach_renderer_names() -> None:
    for name, info in MODEL_INFO.items():
        override = info.get("renderer_name_override")
        info["renderer_name"] = override if override else None


_attach_renderer_names()

# Supported models for fine-tuning (derived from MODEL_INFO)
SUPPORTED_MODELS = list(MODEL_INFO.keys())


def get_model_list_formatted() -> str:
    """Get a formatted list of supported models for display.

    Returns:
        Newline-separated list of model names
    """
    return "\n  ".join(SUPPORTED_MODELS)


def get_base_models() -> list[str]:
    """Return a list of Base models from MODEL_INFO.

    Base models are the right starting point when you want to do research or
    run a full post-training pipeline. For product-style fine-tuning, starting
    from an existing post-trained model (Instruction/Hybrid/Reasoning) is often
    a better default.
    """
    return [name for name, info in MODEL_INFO.items() if info.get("training_type") == "Base"]


def get_renderer_name(model_name: str) -> str:
    """Return the recommended renderer name for a model from MODEL_INFO.

    Raises:
        ValueError: If renderer_name is missing.
    """
    model_info = MODEL_INFO.get(model_name)
    if model_info is None:
        # Not in our registry: fall back to the canonical renderer mapping.
        # This keeps the harness low-lift: users can try models without first
        # editing MODEL_INFO, as long as the renderer mapping can be resolved.
        try:
            return _resolve_renderer_name(model_name)
        except Exception as exc:
            raise ValueError(
                f"Renderer mapping not found for model '{model_name}'.\n"
                "Fix:\n"
                "- Use a model that the renderer mapping can resolve, OR\n"
                "- Add the model to MODEL_INFO with renderer_name_override.\n"
            ) from exc
    renderer_name = model_info.get("renderer_name")
    if not renderer_name:
        try:
            renderer_name = _resolve_renderer_name(model_name)
        except Exception as exc:
            raise ValueError(
                f"Renderer mapping not found for model '{model_name}'. "
                "Add renderer_name_override in MODEL_INFO or upgrade pinned dependencies."
            ) from exc
        model_info["renderer_name"] = renderer_name
    return renderer_name


def create_renderer_for_model(model_name: str, tokenizer):
    """Create the renderer + optional image processor for a model."""
    from tinker_cookbook import renderers as renderer_helpers
    from tinker_cookbook.image_processing_utils import get_image_processor

    renderer_name = get_renderer_name(model_name)
    image_processor = None
    if renderer_name in ("qwen3_vl", "qwen3_vl_instruct"):
        image_processor = get_image_processor(model_name)

    renderer = renderer_helpers.get_renderer(
        renderer_name,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    return renderer, renderer_name, image_processor


def validate_renderer_mappings(models: Iterable[str] | None = None) -> list[str]:
    """Return models missing renderer mappings."""
    target_models = list(models) if models is not None else SUPPORTED_MODELS
    failures: list[str] = []
    for model_name in target_models:
        try:
            get_renderer_name(model_name)
        except Exception:
            failures.append(model_name)
    return failures


def validate_renderer_override_consistency(
    models: Iterable[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return override mismatches against the recommended renderer mapping.

    Each mismatch is returned as:
        (model_name, configured_override, recommended_renderer)
    """
    target_models = list(models) if models is not None else SUPPORTED_MODELS
    mismatches: list[tuple[str, str, str]] = []
    for model_name in target_models:
        model_info = MODEL_INFO.get(model_name, {})
        configured_override = model_info.get("renderer_name_override")
        if not configured_override:
            continue
        try:
            recommended = _resolve_renderer_name(model_name)
        except Exception:
            # Defer unknown-mapping cases to validate_renderer_mappings.
            continue
        if configured_override != recommended:
            mismatches.append((model_name, configured_override, recommended))
    return mismatches
