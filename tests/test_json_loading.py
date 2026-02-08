import json

from tinker_mcp.training.common import load_json_from_file_or_string


def test_load_json_array_string():
    data = json.dumps([{"prompt": "hi", "response": "yo"}])
    parsed, err = load_json_from_file_or_string(data, "examples")
    assert err is None
    assert parsed == [{"prompt": "hi", "response": "yo"}]


def test_load_jsonl_string():
    data = '{"prompt":"a","response":"b"}\n{"prompt":"c","response":"d"}\n'
    parsed, err = load_json_from_file_or_string(data, "examples")
    assert err is None
    assert parsed == [{"prompt": "a", "response": "b"}, {"prompt": "c", "response": "d"}]


def test_load_jsonl_file(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text('{"prompt":"a","response":"b"}\n{"prompt":"c","response":"d"}\n', encoding="utf-8")
    parsed, err = load_json_from_file_or_string(str(path), "examples")
    assert err is None
    assert parsed == [{"prompt": "a", "response": "b"}, {"prompt": "c", "response": "d"}]


def test_invalid_json_returns_error_prefix():
    parsed, err = load_json_from_file_or_string('{"prompt": "a",}', "examples")
    assert parsed is None
    assert err is not None
    assert err.startswith("Error:")

