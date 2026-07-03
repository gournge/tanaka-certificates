import numpy as np


class Facet:
    def __init__(self, vertices: np.ndarray, values: np.ndarray):
        assert (
            vertices.shape == values.shape
        ), "Vertices and values must have the same shape."
        self.vertices = vertices
        self.values = values

    def __repr__(self):
        return f"Facet(vertices={self.vertices}, values={self.values})"


class Breakpoint(Facet):
    def __init__(self, vertices: np.ndarray, values: np.ndarray):
        assert (
            vertices.shape[0] == 1 and values.shape[0] == 1
        ), "Breakpoint must have exactly one vertex and one value."
        super().__init__(vertices, values)

    @property
    def get_breakpoint(self) -> float:
        return self.vertices[0]

    @property
    def get_value(self) -> float:
        return self.values[0]

    def __repr__(self):
        return f"Breakpoint(breakpoint={self.get_breakpoint}, value={self.get_value})"
