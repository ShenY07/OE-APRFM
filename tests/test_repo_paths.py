"""Verify the executable selected for every supported method/problem."""

from training.train import runner_path


def test_training_paths():
    for method, problems in {
        "oe_aprfm": range(1, 7),
        "oe_apnn": range(1, 6),
        "mm_aprfm": (1, 2, 3, 4, 6),
    }.items():
        for problem in problems:
            assert runner_path(method, f"p{problem}").is_file()
