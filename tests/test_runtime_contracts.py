"""Runtime contract tests for renderer/SDK alignment.

These tests are pure import/shape checks and do not require API credentials.
"""


def test_renderer_apis_present():
    """Renderer API used by the harness should exist."""
    from tinker_cookbook import model_info as tinker_model_info
    from tinker_cookbook import renderers as renderer_helpers

    assert hasattr(tinker_model_info, "get_recommended_renderer_name")
    assert hasattr(renderer_helpers, "get_renderer")
    assert hasattr(renderer_helpers, "get_text_content")


def test_sample_response_has_sequences_field():
    """Tinker SDK response contract should expose SampleResponse.sequences."""
    from tinker import types as tinker_types

    fields = getattr(tinker_types.SampleResponse, "model_fields", {})
    assert "sequences" in fields


def test_all_supported_models_have_renderer_mapping():
    """Every model in MODEL_INFO must resolve to a renderer mapping."""
    from tinker_mcp.models import validate_renderer_mappings

    unmapped = validate_renderer_mappings()
    assert unmapped == []


def test_renderer_overrides_match_recommended_mapping():
    """Manual renderer overrides should stay aligned with recommended mappings."""
    from tinker_mcp.models import validate_renderer_override_consistency

    assert validate_renderer_override_consistency() == []


def test_version_parser_and_floor_check_helpers():
    """Version helper functions should handle common runtime version formats."""
    from tinker_mcp.utils import _parse_version_tuple, _version_at_least

    assert _parse_version_tuple("2.10.0") == (2, 10, 0)
    assert _parse_version_tuple("2.10.0+cu121") == (2, 10, 0)
    assert _parse_version_tuple("4.56.2rc1") == (4, 56, 2)
    assert _version_at_least("2.10.0", (2, 10, 0)) is True
    assert _version_at_least("2.10.1", (2, 10, 0)) is True
    assert _version_at_least("2.9.9", (2, 10, 0)) is False
