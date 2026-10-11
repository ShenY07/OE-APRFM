"""Keep the collected manuscript complete and detect modified artifacts."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_paper import check_paper


def test_paper_snapshot():
    paper = Path(__file__).resolve().parents[1] / "paper"
    assert len(check_paper(paper)) == 31
    assert (paper / "tables.tex").read_text().count(r"\begin{table}") == 26


def test_changed_asset(tmp_path):
    asset = tmp_path / "panel.pdf"
    asset.write_bytes(b"changed figure")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "panel.pdf",
                        "sha256": hashlib.sha256(
                            b"original figure"
                        ).hexdigest(),
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Paper snapshot changed"):
        check_paper(tmp_path)
