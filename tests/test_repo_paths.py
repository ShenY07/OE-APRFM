"""Keep historical artifact records usable after moving their directories."""

import json

from scripts import paper_paths, repo_paths
from scripts.source_compat import layout_source
from training.train import runner_path


def test_recorded_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_paths, "ROOT", tmp_path)
    old = "results/table_refresh_20261010/raw/result.json"
    new = tmp_path / "results/runs/feature_study/raw/result.json"
    new.parent.mkdir(parents=True)
    new.write_text('{"epsilon": 0.001}')
    assert repo_paths.resolve_path(old).read_bytes() == new.read_bytes()
    assert repo_paths.resolve_path(tmp_path / old) == new
    assert repo_paths.resolve_path(new) == new


def test_paper_paths(tmp_path, monkeypatch):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source": "results/figure/new/panel.pdf",
                        "path": "panel.pdf",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(paper_paths, "ROOT", tmp_path)
    monkeypatch.setattr(paper_paths, "PAPER", paper)
    for folder in ("figure", "figures"):
        source = f"results/{folder}/new/panel.pdf"
        assert paper_paths.paper_file(source) == paper / "panel.pdf"


def test_path_mapping():
    current = '"src/baselines/oe_apnn/train.py"'
    assert layout_source(current) == current
    assert layout_source('"baselines/oe_apnn/train.py"') == current
    assert layout_source('"results/figures/new"') == '"results/figures/new"'


def test_training_paths():
    for method, problems in {
        "oe_aprfm": range(1, 7),
        "oe_apnn": range(1, 6),
        "mm_aprfm": (1, 2, 3, 4, 6),
    }.items():
        for problem in problems:
            assert runner_path(method, f"p{problem}").is_file()
