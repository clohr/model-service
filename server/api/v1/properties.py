"""
Standard units are taken from
https://www.ewh.ieee.org/soc/ias/pub-dept/abbreviation.pdf
"""

from dataclasses import dataclass
from typing import ClassVar, List, Optional

import marshmallow
from auth_middleware.models import DatasetPermission  # type: ignore
from marshmallow import fields

from core.json import CamelCaseSchema, Serializable
from server.auth import permission_required
from server.db import PartitionedDatabase
from server.models import JsonDict, datatypes


class UnitSchema(CamelCaseSchema):
    name = fields.String()
    display_name = fields.String()
    description = fields.String()


@dataclass(frozen=True)
class Unit(Serializable):
    __schema__: ClassVar[UnitSchema] = UnitSchema(unknown=marshmallow.EXCLUDE)

    name: str
    display_name: str
    description: str

    @staticmethod
    def find_display_name(name: str) -> Optional[str]:
        """
        Given a string, look up all units from the UNITS constant.
        If one is found, give back the display_name of that Unit
        Else return None
        """
        for dimension in UNITS:
            for unit in dimension.units:
                if unit.name == name:
                    return unit.display_name
        return None


class UnitDimensionSchema(CamelCaseSchema):

    dimension = fields.String()
    units = fields.Nested(Unit.schema(), many=True)


@dataclass(frozen=True)
class UnitDimension(Serializable):

    __schema__: ClassVar[UnitDimensionSchema] = UnitDimensionSchema(
        unknown=marshmallow.EXCLUDE
    )

    dimension: str
    units: List[Unit]


# fmt: off
UNITS = [
    UnitDimension(
        "Time",
        [
            Unit("us", "μs", "microsecond"),
            Unit("ms", "ms", "millisecond"),
            Unit("s", "s", "second"),
            Unit("min", "min", "minute"),
            Unit("h", "h", "hour"),
        ],
    ),
    UnitDimension(
        "Length",
        [
            Unit("um", "μm", "micrometer"),
            Unit("mm", "mm", "millimeter"),
            Unit("cm", "cm", "centimeter"),
            Unit("m", "m", "meter"),
            Unit("km", "km", "kilometer"),
        ],
    ),
    UnitDimension(
        "Mass",
        [
            Unit("ug", "μg", "microgram"),
            Unit("mg", "mg", "milligram"),
            Unit("g", "g", "gram"),
            Unit("kg", "kg", "kilogram"),
        ],
    ),
    UnitDimension(
        "Electric Current",
        [
            Unit("uA", "μA", "microampere"),
            Unit("mA", "mA", "milliampere"),
            Unit("A", "A", "ampere"),
        ],
    ),
    UnitDimension(
        "Resistance",
        [
            Unit("ohm", "Ω", "ohm"),
        ]
    ),
    UnitDimension(
        "Temperature",
        [
            Unit("C", "°C", "degree Celsius"),
            Unit("F", "°F", "degree Fahrenheit"),
        ],
    ),
    UnitDimension(
        "Capacity",
        [
            Unit("uL", "μL", "microliter"),
            Unit("mL", "mL", "milliliter"),
            Unit("L", "L", "liter"),
        ],
    ),
    UnitDimension(
        "Luminous Intensity",
        [
            Unit("cd", "cd", "candela"),
        ]
    ),
    UnitDimension(
        "Energy",
        [
            Unit("J", "J", "joule"),
        ]
    ),
    UnitDimension(
        "Angle",
        [
            Unit("rad", "rad", "radian"),
            Unit("deg", "°", "degree"),
        ]
    ),
    UnitDimension(
        "Frequency",
        [
            Unit("Hz", "Hz", "hertz"),
            Unit("kHz", "kHz", "kilohertz"),
            Unit("MHz", "MHz", "megahertz"),
            Unit("GHz", "GHz", "gigahertz"),
        ],
    ),
    UnitDimension(
        "Force",
        [
            Unit("N", "N", "newton"),
        ]
    ),
    UnitDimension(
        "Pressure",
        [
            Unit("Pa", "Pa", "pascal"),
        ]
    ),
    UnitDimension(
        "Velocity",
        [
            Unit("m/s", "m/s", "meter per second"),
        ]
    ),
    UnitDimension(
        "Area",
        [
            Unit("mm^2", "mm²", "square milimeter"),
            Unit("cm^2", "cm²", "square centimeter"),
            Unit("m^2", "m²", "square meter"),
            Unit("km^2", "km²", "square kilometer"),
        ],
    ),
    UnitDimension(
        "Volume",
        [
            Unit("mm^3", "mm³", "cubic milimeter"),
            Unit("cm^3", "cm³", "cubic centimeter"),
            Unit("m^3", "m³", "cubic meter"),
        ],
    ),
    UnitDimension(
        "Acceleration",
        [
            Unit("m/s^2", "m/s²", "meter per second squared"),
        ]
    ),
]
# fmt: on


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_units(db: PartitionedDatabase) -> List[JsonDict]:
    return [unit.to_dict() for unit in UNITS]


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_string_subtypes(db: PartitionedDatabase) -> JsonDict:
    return datatypes.STRING_SUBTYPES
