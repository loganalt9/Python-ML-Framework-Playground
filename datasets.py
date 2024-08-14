import typing as t
from pathlib import Path

import numpy as np
import torch

# flight types
Node: t.TypeAlias = t.Any
@t.runtime_checkable
class DataLoadable(t.Protocol):
    def load(self, node: Node) -> np.ndarray | t.Sequence[torch.Tensor]: #TODO: find a use for 'node' or get rid of it.
        """Loads data onto a node/endpoint.

        Args:
            node (Node): The current node/endpoint.

        Returns:
            np.ndarray | t.Sequence[torch.Tensor]: The sequence of data that will be loaded.
        """
        pass


class LocalTorchLoadable:
    def __init__(self, path: Path) -> None:
        """Initalization of a real world DataLoadable object.

        Args:
            path (Path): The path to data stored on the node/endpoint.
        """
        self.path = path

    def load(self, node: Node) -> np.ndarray | t.Sequence[torch.Tensor]:
        """Loads data from disc onto a given node/endpoint.

        Args:
            node (Node): The node to load data onto.

        Returns:
            np.ndarray | t.Sequence[torch.Tensor]: The sequence of data that will be loaded.
        """
        import torch

        with open(self.path, "r") as disk:
            raw_data = disk.read()
        lines = raw_data.strip().splitlines('\n')
        data = [list(map(torch.tensor,line.split(','))) for line in lines]
        return data


class SimulatedTorchLoadable:
    def __init__(self, data, batch_size) -> None:
        """Initalization of a simulation based DataLoadable object.

        Args:
            data (_type_): The data to be be loaded amongst all nodes/endpoints.
            batch_size (_type_): The portion of the total dataset that each node/endpoint will carry.
        """
        self.data = data
        self.batch_size = batch_size
        self.batch_start_point = 0

    def load(self, node: Node) -> np.ndarray | t.Sequence[torch.Tensor]:
        """Loads data from the given data set onto the given node/endpoint.

        Args:
            node (Node): The node to load data onto.

        Returns:
            np.ndarray | t.Sequence[torch.Tensor]: The sequence of data that will be loaded.
        """
        import torch

        subset = []
        for i in range(self.batch_start_point, self.batch_size):
            subset.append(torch.tensor(self.data[i]))

        self.batch_start_point += self.batch_size
        return subset
