"""Keep historical source checks strict about executable changes."""

from scripts.source_compat import rename_source, source_ast


def test_source_style():
    original = '''"""Old module description."""
from modules.solution_construction import OddEvenDecompositionConstructor2D

def field(x):
    """Old field description."""
    # Old explanation.
    return OddEvenDecompositionConstructor2D(x * 0.5)
'''
    current = '''"""Reconstruct the transport field."""
from oeaprfm.features.reconstruction import OEField2D

def field(x):
    return OEField2D(x * 0.5)
'''
    assert source_ast(original) == source_ast(current)
    assert source_ast(original) != source_ast(current.replace("0.5", "1.0"))


def test_source_path():
    assert rename_source("src/modules/solution_construction.py") == (
        "src/oeaprfm/features/reconstruction.py"
    )
    assert rename_source("src/oeaprfm/features/function_space.py") == (
        "src/oeaprfm/features/function_space.py"
    )
