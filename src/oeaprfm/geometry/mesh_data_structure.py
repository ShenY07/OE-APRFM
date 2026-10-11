"""Store mesh dimensions and cell connectivity."""

from abc import ABCMeta


class MeshDataStructure(metaclass=ABCMeta):
    """Define cell and node counts for concrete mesh types."""

    def _number_of_cells(self) -> int:
        return -1

    def _number_of_nodes(self) -> int:
        return -1
