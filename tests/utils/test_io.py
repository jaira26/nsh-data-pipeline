from nsh_pipeline.utils.io import read_jsonl, write_jsonl


def test_write_then_read_round_trip(tmp_path):
    records = [{"b": 2, "a": 1}, {"a": "x", "nested": {"k": [1, 2]}}]
    path = write_jsonl(records, tmp_path / "sub" / "data.jsonl")  # creates missing folder
    assert path.exists()
    assert read_jsonl(path) == records


def test_write_uses_sorted_keys_and_lf_line_endings(tmp_path):
    path = write_jsonl([{"b": 2, "a": 1}], tmp_path / "data.jsonl")
    assert path.read_bytes() == b'{"a": 1, "b": 2}\n'


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]