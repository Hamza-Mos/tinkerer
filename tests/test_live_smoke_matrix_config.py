"""Static validation for live model-matrix smoke configuration."""


def _imports():
    from tinker_mcp.models import MODEL_INFO
    from scripts.live_model_matrix_smoke import (
        GRPO_PROMPTS,
        MODEL_FAMILY_DEFAULTS,
        SFT_EXAMPLES,
    )

    return MODEL_INFO, GRPO_PROMPTS, MODEL_FAMILY_DEFAULTS, SFT_EXAMPLES


def test_matrix_families_are_known():
    """Smoke matrix should cover expected family keys."""
    _, _, family_defaults, _ = _imports()
    assert sorted(family_defaults.keys()) == ["base", "hybrid", "instruction", "reasoning"]


def test_matrix_models_exist_in_registry():
    """Configured smoke models should exist in MODEL_INFO."""
    model_info, _, family_defaults, _ = _imports()
    for model in family_defaults.values():
        assert model in model_info


def test_smoke_payloads_have_minimum_examples():
    """Smoke payloads should include at least two examples/prompts."""
    _, grpo_prompts, _, sft_examples = _imports()
    assert len(sft_examples) >= 2
    assert len(grpo_prompts) >= 2
