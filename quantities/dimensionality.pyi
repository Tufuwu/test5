class Dimensionality(dict):
    @property
    def ndims(self) -> int:
        ...

    @property
    def simplified(self) -> Dimensionality:
        ...

    @property
    def string(self) -> str:
        ...

    @property
    def unicode(self) -> str:
        ...

    @property
    def latex(self) -> str:
        ...

    @property
    def html(self) -> str:
        ...


    def __hash__(self) -> int: # type: ignore[override]
        ...

    def __add__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __iadd__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __sub__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __isub__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __mul__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __imul__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __truediv__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __itruediv__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __pow__(self, other : Dimensionality) -> Dimensionality:
        ...

    def __ipow__(self, other: Dimensionality) -> Dimensionality:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __gt__(self, other: Dimensionality) -> bool:
        ...

    def __ge__(self, other: Dimensionality) -> bool:
        ...

    def __lt__(self, other: Dimensionality) -> bool:
        ...

    def __le__(self, other: Dimensionality)-> bool:
        ...

    def copy(self) -> Dimensionality:
        ...
