"""Bounded, discoverable input contracts for the optional MCP tools."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PageSize = Annotated[int, Field(ge=1, le=200)]
Offset = Annotated[int, Field(ge=0)]
PreviewSize = Annotated[int, Field(ge=0, le=100)]
Coordinate = Annotated[float, Field(allow_inf_nan=False)]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Interval(Input):
    """Half-open selection in the stated axis units."""

    lower: Coordinate
    upper: Coordinate

    @model_validator(mode="after")
    def ordered(self) -> "Interval":
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")
        return self

    def contains(self, value: float) -> bool:
        return self.lower <= value < self.upper


class Predicate(Input):
    column: str
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge", "is_null"] = "eq"
    value: str | int | Coordinate | None = None


class Operation(Input):
    """Named built-in configuration. Discover names and fields with get_processing_options."""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict, max_length=30)


class Processing(Input):
    """All noise filtering happens before centroiding. Defaults match the Python API."""

    mode: Literal["raw", "centroid"] = "centroid"
    centroider: Operation | None = None
    noise: list[Operation] = Field(default_factory=list, max_length=8)
    smoothing: Operation | None = None
    exclusion: Operation | None = None
    ion_mobility_type: Literal["ook0", "voltage"] = "ook0"

    @model_validator(mode="after")
    def raw_has_no_centroider(self) -> "Processing":
        if self.mode == "raw" and self.centroider is not None:
            raise ValueError("A raw extraction cannot specify a centroider")
        return self


class SpectrumSelection(Input):
    """Use IDs returned by query tools. Window indices refer to the acquisition's ordered lookup."""

    kind: Literal["frame", "precursor", "dia_window", "prm_transition"]
    id: Annotated[int, Field(ge=0)]
    scan_begin: Annotated[int, Field(ge=0)] | None = None
    scan_end: Annotated[int, Field(ge=0)] | None = None
    mz_range: Interval | None = Field(
        default=None,
        description="Select output m/z after processing, with half-open bounds",
    )
    mobility_range: Interval | None = Field(
        default=None,
        description="Select output mobility after processing, in the requested mobility units",
    )

    @model_validator(mode="after")
    def scans(self) -> "SpectrumSelection":
        if self.kind == "precursor" and self.mobility_range is not None:
            raise ValueError(
                "Mobility-collapsed precursor spectra have no mobility axis"
            )
        if (self.scan_begin is None) != (self.scan_end is None):
            raise ValueError("Specify both scan_begin and scan_end")
        if self.scan_begin is not None:
            if self.kind != "frame":
                raise ValueError("Explicit scan bounds apply only to frame selections")
            if self.scan_end is None or self.scan_begin > self.scan_end:
                raise ValueError("scan_begin must not exceed scan_end")
        return self
