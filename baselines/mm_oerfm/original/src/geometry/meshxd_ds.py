import jax.numpy as jnp
from typing import Dict, List, Tuple, Iterable
from geometry.mesh_ds import MeshDataStructure
from utils.coord_generator import cartesian_product


class UniformMeshX(MeshDataStructure):
    def __init__(
        self,
        domain: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
        strides: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
    ):
        self._domain = domain
        self._strides = strides
        self._xmin, self._xmax = self._domain["x"]
        self._nx = int((self._xmax - self._xmin) / self._strides["x"])
        self._dx = self._strides["x"]
        self.number_of_cells = self._number_of_cells()
        self.number_of_nodes = self._number_of_nodes()
        self.cell = self._cell()
        self.node = self._node()
        self.radius_of_cell = self._radius_of_cell()

    def neighbor_of_cell(self, query_index: int) -> list[Iterable[int]]:
        cell_indices = jnp.arange(self.number_of_cells)
        i = query_index
        indices = [i - 1, i + 1]
        for k in range(len(indices)):
            if indices[k] < 0 or indices[k] >= self._nx:
                indices[k] = i
        return list(map(int, cell_indices))

    def interior_cell_indices(self) -> set[int]:
        cell_indices = {i for i in range(self.number_of_cells)}
        bdy_indices = self.bdy_cell_indices()
        int_indices = cell_indices - bdy_indices
        return int_indices

    def interior_node_indices(self) -> set[int]:
        node_indices = {i for i in range(self.number_of_nodes)}
        bdy_indices = self.bdy_node_indices()
        int_indices = node_indices - bdy_indices
        return int_indices

    def bdy_cell_indices(self) -> set[int]:
        cell_indices = jnp.arange(self.number_of_cells)
        bdy_indices = [cell_indices[0], cell_indices[-1]]
        return set(map(int, bdy_indices))

    def bdy_node_indices(self) -> set[int]:
        node_indices = jnp.arange(self.number_of_nodes)
        bdy_indices = [node_indices[0], node_indices[-1]]
        return set(map(int, bdy_indices))

    def center_of_cell(self, query_index: int) -> jnp.array:
        node_indices = self.cell[query_index]
        nodes = self.node[node_indices]
        return jnp.mean(nodes, axis=0, keepdims=True).squeeze()

    def _radius_of_cell(self) -> jnp.array:
        rx = self._dx / 2
        return jnp.array([rx])

    def _cell(self) -> jnp.array:
        node_list = []
        for i in range(self.number_of_cells):
            node_indices = jnp.array([i, i + 1])
            node_list.append(node_indices)
        return jnp.asarray(node_list)

    def _node(self) -> jnp.array:
        nodes1d = jnp.linspace(self._xmin, self._xmax, self._nx + 1)[:, None]
        return nodes1d

    def _number_of_cells(self) -> int:
        return self._nx

    def _number_of_nodes(self) -> int:
        return self._nx + 1


class UniformMeshXY(MeshDataStructure):
    def __init__(
        self,
        domain: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
        strides: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
    ):
        self._domain = domain
        self._strides = strides
        self._xmin, self._xmax = self._domain["x"]
        self._ymin, self._ymax = self._domain["y"]
        self._dx, self._dy = self._strides["x"], self._strides["y"]
        self._nx = int((self._xmax - self._xmin) / self._dx)
        self._ny = int((self._ymax - self._ymin) / self._dy)

        self.number_of_cells = self._number_of_cells()
        self.number_of_nodes = self._number_of_nodes()
        self.cell = self._cell()
        self.node = self._node()
        self.radius_of_cell = self._radius_of_cell()

    # NOTE: This method is implemented for the pou function \psi_b(x)
    def neighbor_of_cell(self, query_index: int) -> list[Iterable[int]]:
        cell_indices = jnp.arange(self.number_of_cells)
        indices_matrix = cell_indices.reshape(self._nx, self._ny)
        i, j = divmod(query_index, self._ny)
        indices = [
            [i, j - 1],
            [i, j + 1],
            [i - 1, j],
            [i + 1, j],
            [i + 1, j - 1],
            [i - 1, j + 1],
            [i - 1, j - 1],
            [i + 1, j + 1],
        ]
        for k in range(len(indices)):
            if indices[k][0] < 0 or indices[k][0] >= self._nx:
                indices[k] = [i, j]
            if indices[k][1] < 0 or indices[k][1] >= self._ny:
                indices[k] = [i, j]

        cell_indices = [indices_matrix[i, j] for i, j in indices]

        # return set(map(int, cell_indices)) - {query_index}
        return list(map(int, cell_indices))

    def interior_cell_indices(self) -> set[int]:
        cell_indices = {i for i in range(self.number_of_cells)}
        bdy_indices = self.bdy_cell_indices()
        int_indices = cell_indices - bdy_indices
        return int_indices

    def interior_node_indices(self) -> set[int]:
        node_indices = {i for i in range(self.number_of_nodes)}
        bdy_indices = self.bdy_node_indices()
        int_indices = node_indices - bdy_indices
        return int_indices

    def bdy_cell_indices(self) -> set[int]:
        cell_indices = jnp.arange(self.number_of_cells)
        indices_matrix = cell_indices.reshape(self._nx, self._ny)
        bdy_indices = (
            list(indices_matrix[0, :])
            + list(indices_matrix[-1, :])
            + list(indices_matrix[:, 0])
            + list(indices_matrix[:, -1])
        )
        return set(map(int, bdy_indices))

    def bdy_node_indices(self) -> set[int]:
        node_indices = jnp.arange(self.number_of_nodes)
        indices_matrix = node_indices.reshape(self._nx + 1, self._ny + 1)
        bdy_indices = (
            list(indices_matrix[0, :])
            + list(indices_matrix[-1, :])
            + list(indices_matrix[:, 0])
            + list(indices_matrix[:, -1])
        )
        return set(map(int, bdy_indices))

    def center_of_cell(self, query_index: int) -> jnp.array:
        node_indices = self.cell[query_index]
        nodes = self.node[node_indices]
        return jnp.mean(nodes, axis=0, keepdims=True).squeeze()

    def _radius_of_cell(self) -> jnp.array:
        rx = self._dx / 2
        ry = self._dy / 2
        return jnp.array([rx, ry])

    def _cell(self) -> jnp.array:
        node_list = []
        cnt = 1
        for _ in range(self.number_of_cells):
            node_indices = jnp.array(
                [
                    cnt - 1,
                    cnt,
                    cnt + self._ny,
                    cnt + self._ny + 1,
                ]
            )
            node_list.append(node_indices)

            if (cnt + 1) % (self._ny + 1) == 0:
                cnt += 2
            else:
                cnt += 1
        return jnp.asarray(node_list)

    def _node(self) -> jnp.array:
        x = jnp.linspace(self._xmin, self._xmax, self._nx + 1)[:, None]
        y = jnp.linspace(self._ymin, self._ymax, self._ny + 1)[:, None]
        nodes2d = cartesian_product([x, y]).reshape(self.number_of_nodes, 2)
        return nodes2d

    def _number_of_cells(self) -> int:
        return self._nx * self._ny

    def _number_of_nodes(self) -> int:
        return (self._nx + 1) * (self._ny + 1)


class UniformMeshXV(MeshDataStructure):
    def __init__(
        self,
        domain: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
        strides: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
    ):
        self._domain = domain
        self._strides = strides
        self._xmin, self._xmax = self._domain["x"]
        self._vmin, self._vmax = self._domain["v"]
        self._dx, self._dv = self._strides["x"], self._strides["v"]
        self._nx = int((self._xmax - self._xmin) / self._dx)
        self._nv = int((self._vmax - self._vmin) / self._dv)

        self.number_of_cells = self._number_of_cells()
        self.number_of_nodes = self._number_of_nodes()
        self.cell = self._cell()
        self.node = self._node()
        self.radius_of_cell = self._radius_of_cell()

    # # NOTE: This method is implemented for the pou function \psi_a(x)
    # def neighbor_of_cell(self, query_index: int) -> list[Iterable[int]]:
    #     cell_indices = jnp.arange(self.number_of_cells)
    #     indices_matrix = cell_indices.reshape(self._nx, self._nv)
    #     i, j = divmod(query_index, self._nv)
    #     indices = [
    #         [i, j - 1],
    #         [i, j + 1],
    #         [i - 1, j],
    #         [i + 1, j],
    #     ]
    #     for k in range(len(indices)):
    #         if indices[k][0] < 0 or indices[k][0] >= self._nx:
    #             indices[k] = [i, j]
    #         if indices[k][1] < 0 or indices[k][1] >= self._nv:
    #             indices[k] = [i, j]

    #     cell_indices = [indices_matrix[i, j] for i, j in indices]

    #     # return set(map(int, cell_indices)) - {query_index}
    #     return list(map(int, cell_indices))

    # NOTE: This method is implemented for the pou function \psi_b(x)
    def neighbor_of_cell(self, query_index: int) -> list[Iterable[int]]:
        cell_indices = jnp.arange(self.number_of_cells)
        indices_matrix = cell_indices.reshape(self._nx, self._nv)
        i, j = divmod(query_index, self._nv)
        indices = [
            [i, j - 1],
            [i, j + 1],
            [i - 1, j],
            [i + 1, j],
            [i + 1, j - 1],
            [i - 1, j + 1],
            [i - 1, j - 1],
            [i + 1, j + 1],
        ]
        for k in range(len(indices)):
            if indices[k][0] < 0 or indices[k][0] >= self._nx:
                indices[k] = [i, j]
            if indices[k][1] < 0 or indices[k][1] >= self._nv:
                indices[k] = [i, j]

        cell_indices = [indices_matrix[i, j] for i, j in indices]

        # return set(map(int, cell_indices)) - {query_index}
        return list(map(int, cell_indices))

    def interior_cell_indices(self) -> set[int]:
        cell_indices = {i for i in range(self.number_of_cells)}
        bdy_indices = self.bdy_cell_indices()
        int_indices = cell_indices - bdy_indices
        return int_indices

    def interior_node_indices(self) -> set[int]:
        node_indices = {i for i in range(self.number_of_nodes)}
        bdy_indices = self.bdy_node_indices()
        int_indices = node_indices - bdy_indices
        return int_indices

    def bdy_cell_indices(self) -> set[int]:
        cell_indices = jnp.arange(self.number_of_cells)
        indices_matrix = cell_indices.reshape(self._nx, self._nv)
        bdy_indices = (
            list(indices_matrix[0, :])
            + list(indices_matrix[-1, :])
            + list(indices_matrix[:, 0])
            + list(indices_matrix[:, -1])
        )
        return set(map(int, bdy_indices))

    def bdy_node_indices(self) -> set[int]:
        node_indices = jnp.arange(self.number_of_nodes)
        indices_matrix = node_indices.reshape(self._nx + 1, self._nv + 1)
        bdy_indices = (
            list(indices_matrix[0, :])
            + list(indices_matrix[-1, :])
            + list(indices_matrix[:, 0])
            + list(indices_matrix[:, -1])
        )
        return set(map(int, bdy_indices))

    def center_of_cell(self, query_index: int) -> jnp.array:
        node_indices = self.cell[query_index]
        nodes = self.node[node_indices]
        return jnp.mean(nodes, axis=0, keepdims=True).squeeze()

    def _radius_of_cell(self) -> jnp.array:
        rx = self._dx / 2
        ry = self._dv / 2
        return jnp.array([rx, ry])

    def _cell(self) -> jnp.array:
        node_list = []
        cnt = 1
        for _ in range(self.number_of_cells):
            node_indices = jnp.array(
                [
                    cnt - 1,
                    cnt,
                    cnt + self._nv,
                    cnt + self._nv + 1,
                ]
            )
            node_list.append(node_indices)

            if (cnt + 1) % (self._nv + 1) == 0:
                cnt += 2
            else:
                cnt += 1
        return jnp.asarray(node_list)

    def _node(self) -> jnp.array:
        x = jnp.linspace(self._xmin, self._xmax, self._nx + 1)[:, None]
        y = jnp.linspace(self._vmin, self._vmax, self._nv + 1)[:, None]
        nodes2d = cartesian_product([x, y]).reshape(self.number_of_nodes, 2)
        return nodes2d

    def _number_of_cells(self) -> int:
        return self._nx * self._nv

    def _number_of_nodes(self) -> int:
        return (self._nx + 1) * (self._nv + 1)


class UniformMeshXYV(MeshDataStructure):
    def __init__(
        self,
        domain: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
        strides: Dict[str, Tuple[Iterable[float]] | List[Iterable[float]]],
    ):
        self._domain = domain
        self._strides = strides
        self._xmin, self._xmax = self._domain["x"]
        self._ymin, self._ymax = self._domain["y"]
        self._vmin, self._vmax = self._domain["theta"]
        self._dx, self._dy, self._dv = (
            self._strides["x"],
            self._strides["y"],
            self._strides["theta"],
        )
        self._nx = int((self._xmax - self._xmin) / self._dx)
        self._ny = int((self._ymax - self._ymin) / self._dy)
        self._nv = int((self._vmax - self._vmin) / self._dv)

        self.number_of_cells = self._number_of_cells()
        self.number_of_nodes = self._number_of_nodes()
        self.cell = self._cell()
        self.node = self._node()
        self.radius_of_cell = self._radius_of_cell()

    def neighbor_of_cell(self, query_index: int) -> list[Iterable[int]]:
        cell_indices = jnp.arange(self.number_of_cells)
        indices_matrix = cell_indices.reshape(self._nx, self._ny, self._nv)
        i, jk = divmod(query_index, self._ny * self._nv)
        j, k = divmod(jk, self._nv)
        indices = [
            [i, j, k - 1],
            [i, j, k + 1],
            [i, j - 1, k],
            [i, j + 1, k],
            [i - 1, j, k],
            [i + 1, j, k],
        ]
        for _l in range(len(indices)):
            if indices[_l][0] < 0 or indices[_l][0] >= self._nx:
                indices[_l] = [i, j, k]
            if indices[_l][1] < 0 or indices[_l][1] >= self._ny:
                indices[_l] = [i, j, k]
            if indices[_l][2] < 0 or indices[_l][2] >= self._nv:
                indices[_l] = [i, j, k]

        cell_indices = [indices_matrix[i, j, k] for i, j, k in indices]

        # return set(map(int, cell_indices)) - {query_index}
        return list(map(int, cell_indices))

    def interior_cell_indices(self) -> set[int]:
        cell_indices = {i for i in range(self.number_of_cells)}
        bdy_indices = self.bdy_cell_indices()
        int_indices = cell_indices - bdy_indices
        return int_indices

    def interior_node_indices(self) -> set[int]:
        node_indices = {i for i in range(self.number_of_nodes)}
        bdy_indices = self.bdy_node_indices()
        int_indices = node_indices - bdy_indices
        return int_indices

    def bdy_cell_indices(self) -> set[int]:
        cell_indices = jnp.arange(self.number_of_cells)
        indices_matrix = cell_indices.reshape(self._nx, self._ny, self._nv)
        bdy_indices = (
            list(indices_matrix[0, :, :].flatten())
            + list(indices_matrix[-1, :, :].flatten())
            + list(indices_matrix[:, 0, :].flatten())
            + list(indices_matrix[:, -1, :].flatten())
            + list(indices_matrix[:, :, 0].flatten())
            + list(indices_matrix[:, :, -1].flatten())
        )
        return set(map(int, bdy_indices))

    def bdy_node_indices(self) -> set[int]:
        node_indices = jnp.arange(self.number_of_nodes)
        indices_matrix = node_indices.reshape(
            self._nx + 1, self._ny + 1, self._nv + 1
        )
        bdy_indices = (
            list(indices_matrix[0, :, :].flatten())
            + list(indices_matrix[-1, :, :].flatten())
            + list(indices_matrix[:, 0, :].flatten())
            + list(indices_matrix[:, -1, :].flatten())
            + list(indices_matrix[:, :, 0].flatten())
            + list(indices_matrix[:, :, -1].flatten())
        )
        return set(map(int, bdy_indices))

    def center_of_cell(self, query_index: int) -> jnp.array:
        node_indices = self.cell[query_index]
        nodes = self.node[node_indices]
        return jnp.mean(nodes, axis=0, keepdims=True).squeeze()

    def _radius_of_cell(self) -> jnp.array:
        rx = self._dx / 2
        ry = self._dy / 2
        rz = self._dv / 2
        return jnp.array([rx, ry, rz])

    def _cell(self) -> jnp.array:
        node_list = []
        cnt = 1
        for _ in range(self.number_of_cells):
            node_indices = jnp.array(
                [
                    cnt - 1,
                    cnt,
                    cnt - 1 + (self._nv + 1),
                    cnt + (self._nv + 1),
                    cnt - 1 + (self._ny + 1) * (self._nv + 1),
                    cnt + (self._ny + 1) * (self._nv + 1),
                    cnt - 1 + (self._ny + 1) * (self._nv + 1) + (self._nv + 1),
                    cnt + (self._ny + 1) * (self._nv + 1) + (self._nv + 1),
                ]
            )
            node_list.append(node_indices)

            if (cnt + 1 + self._nv + 1) % (
                (self._ny + 1) * (self._nv + 1)
            ) != 0:
                if (cnt + 1) % (self._nv + 1) == 0:
                    new_cnt = cnt + 2
                else:
                    new_cnt = cnt + 1
            else:
                new_cnt = cnt + 2 + self._nv + 1
            cnt = new_cnt

        return jnp.asarray(node_list)

    def _node(self) -> jnp.array:
        x = jnp.linspace(self._xmin, self._xmax, self._nx + 1)[:, None]
        y = jnp.linspace(self._ymin, self._ymax, self._ny + 1)[:, None]
        v = jnp.linspace(self._vmin, self._vmax, self._nv + 1)[:, None]
        nodes3d = cartesian_product([x, y, v]).reshape(self.number_of_nodes, 3)
        return nodes3d

    def _number_of_cells(self) -> int:
        return self._nx * self._ny * self._nv

    def _number_of_nodes(self) -> int:
        return (self._nx + 1) * (self._ny + 1) * (self._nv + 1)
