import json
from pathlib import Path

from fas2h.data.pairs import load_pairs_jsonl


def test_pairs_loader_reads_toy_jsonl() -> None:
    pair_file = Path("configs/data/toy_pairs.jsonl")
    pairs = load_pairs_jsonl(pair_file)
    assert len(pairs) >= 1
    assert pairs[0]["pair_id"] == "toy_0001"
    assert "target_keywords" in pairs[0]


def test_pairs_loader_allows_optional_target_keywords(tmp_path: Path) -> None:
    pair_file = tmp_path / "pairs.jsonl"
    payload = {
        "pair_id": "tmp_0001",
        "source_path": "tests/assets/smoke_source.png",
        "target_path": "tests/assets/smoke_target.png",
    }
    pair_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    pairs = load_pairs_jsonl(pair_file)
    assert pairs[0]["pair_id"] == "tmp_0001"
    assert "target_keywords" not in pairs[0]
