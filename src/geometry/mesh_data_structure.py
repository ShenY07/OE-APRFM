"""Provide mesh data structure functionality for the geometry layer.

Distinction: This file is the canonical implementation in this module family; numerical logic is preserved.
"""

from abc import ABCMeta


class MeshDataStructure(metaclass=ABCMeta):
    """Abstract MeshDataStructure class.
    Any mesh data structure should inherit from this class.
    """

    def _number_of_cells(self) -> int:
        return -1

    def _number_of_nodes(self) -> int:
        return -1
