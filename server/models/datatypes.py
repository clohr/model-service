import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from email.utils import parseaddr
from enum import Enum
from re import match
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union, cast
from uuid import UUID

from core.util import normalize_datetime

T = TypeVar("T")


class StringDateFormat(str, Enum):
    DATE = "Date"
    DATETIME = "DateTime"
    TIME = "Time"


class StringSubtypeFormat(str, Enum):
    URL = "URL"
    EMAIL = "Email"


STRING_SUBTYPES = {
    StringSubtypeFormat.URL: {
        "label": "URL",
        "regex": r"^(ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-/]))?$",
    },
    StringSubtypeFormat.EMAIL: {
        "label": "E-Mail",
        "regex": r"^\w[\w.-]*@([\w-]+\.)+[\w-]+$",
    },
}


def as_list(t: Any) -> List[Any]:
    try:
        iter(t)
        return t
    except:  # noqa: E722
        return [t]


def is_uuid(maybe_id: Any) -> bool:
    """
    Check if the given ID is an UUID.
    """
    if isinstance(maybe_id, UUID):
        return True
    try:
        UUID(maybe_id)
        return True
    except:  # noqa: E77
        return False


def into_uuid(maybe_id: object) -> UUID:
    """
    Attempts to convert a value into a UUID, or error out if not possible.

    Parameters
    ----------
    maybe_id: Any

    Raises
    ------
    ValueError

    Returns
    -------
    UUID
    """
    if isinstance(maybe_id, UUID):
        return maybe_id
    try:
        if isinstance(maybe_id, str):
            return UUID(maybe_id)
    except:  # noqa: E722
        pass
    raise ValueError(f"Not a UUID: {maybe_id}")


def is_email_address(thing: str) -> bool:
    result = match(STRING_SUBTYPES[StringSubtypeFormat.EMAIL]["regex"], thing)
    if result:
        return True
    else:
        return False


# TODO: WE CAN ADD VALIDATION OF FORMAT HERE WHEN WE WANT TO IMPLEMENT THIS FEATURE
def is_date(thing: str) -> bool:
    return True


def is_date_time(thing: str) -> bool:
    return True


def is_time(thing: str) -> bool:
    return True


def is_url(thing: str) -> bool:
    result = match(STRING_SUBTYPES[StringSubtypeFormat.URL]["regex"], thing)
    if result:
        return True
    else:
        return False


class DataType:
    """
    A datatype that can interpret a value as the ascribed type.
    """

    type: str

    def is_a(self, data: Any) -> bool:
        """
        Tests if `data` is of the ascribed type.

        Parameters
        ----------
        data : Any

        Returns
        -------
        bool
        """
        raise NotImplementedError

    def into(self, data: Any) -> Any:
        """
        Attempt to parse data into this type

        Parameters
        ----------
        data : Any
        """
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls, t: Dict[str, Any], raise_on_fail: bool = True
    ) -> Optional["DataType"]:
        """
        Parse a data type from a dict.
        """
        typename = cls.type.lower()
        if isinstance(t, dict) and "type" in t and t["type"].lower() == typename:
            try:
                t.pop("type")
                args = cls.pre_load(**t)
                return cls(**args)  # type: ignore
            except:  # noqa: E722
                pass
        if raise_on_fail:
            raise ValueError(f"Not a type: {t}")
        return None

    @classmethod
    def from_json(cls, s: str, raise_on_fail: bool = True) -> Optional["DataType"]:
        """
        Parse a data type from a string.
        """
        typename = cls.type.lower()
        if s.lower() == typename:
            return cls()
        else:
            try:
                t = json.loads(s)
                if isinstance(t, str) and t.lower() in (typename, f'"{typename}"'):
                    return cls()
                if "type" in t and t["type"].lower() == typename:
                    t.pop("type")
                    args = cls.pre_load(**t)
                    return cls(**args)  # type: ignore
            except Exception as e:  # noqa: E722
                pass
        if raise_on_fail:
            raise ValueError(f"Not a {typename}: {s}")
        return None

    @classmethod
    def pre_load(cls, **data: Any) -> Any:
        """
        By default, keys appearing in `data` that are not part of the
        datatype data class `cls` will be discared prior to `data` being
        returned.

        Raises
        ------
        ValueError
            Raises an error if `data` is not of the implementing type.
        """
        supported_fields = fields(cls)
        field_names = set([f.name for f in supported_fields])
        return {k: v for k, v in data.items() if k in field_names}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dict using dataclass.asdict()
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Convert to JSON.
        """
        return json.dumps(self.to_dict())

    def validate(self, data: Any) -> None:
        """
        Raises
        ------
        ValueError
            Raises an error if `data` is not of the implementing type.
        """
        if not self.is_a(data):
            raise ValueError(f"Not a {str(self)}: {data}")

    @property
    def is_simple(self) -> bool:
        """
        Tests if the datatype can be "simply" represented by a string, i.e.
        it is not parameterized by a unit/format/etc.
        """
        d = self.to_dict()
        d.pop("type")
        return all(v is None for v in d.values())

    def into_simple(self) -> Optional[str]:
        """
        Convert the data type into a simple, string representation based on
        the value of its [type] property.
        """
        return self.type if self.is_simple else None

    def __str__(self):
        return self.type or self.__class__.__name__

    @staticmethod
    def _only_unit_changed_number(original: "DataType", other: "DataType") -> bool:
        return (isinstance(original, Long) and isinstance(other, Long)) or (
            isinstance(original, Double) and isinstance(other, Double)
        )

    # intended for use when we already know self and "other" are not equivalent
    # signals a change to a Numeric data type that is only the "unit" field
    def only_unit_changed(self, other: "DataType") -> bool:
        if DataType._only_unit_changed_number(self, other):
            return True
        if (isinstance(self, Array) and isinstance(other, Array)) or (
            isinstance(self, Enumeration) and isinstance(other, Enumeration)
        ):
            return DataType._only_unit_changed_number(self.items, other.items)
        return False

    @staticmethod
    def is_date(dt: "DataType") -> bool:
        """
        Tests if the provided datatype is strictly a Date or an Array/Enumeration of Date
        """
        if isinstance(dt, Array) or isinstance(dt, Enumeration):
            return DataType.is_date(dt.items)
        if isinstance(dt, Date):
            return True
        return False

    @staticmethod
    def is_date_like(dt: "DataType") -> bool:
        """
        Tests if the datatype is a Date, or an Array/Enumeration of Date,
        based on being a Date instance or a string with a format of "Date" or "DateTime"
        """
        if isinstance(dt, Array) or isinstance(dt, Enumeration):
            return DataType.is_date_like(dt.items)
        if isinstance(dt, String) and (
            dt.format == StringDateFormat.DATE or dt.format == StringDateFormat.DATETIME
        ):
            return True
        if isinstance(dt, Date):
            return True
        return False

    @staticmethod
    def get_unit(dt: "DataType") -> Optional[str]:
        """
        Returns the scientific unit of the DataType.
        If there is none or sci units are not relevant, returns None
        """
        if isinstance(dt, Array) or isinstance(dt, Enumeration):
            return DataType.get_unit(dt.items)
        if isinstance(dt, Long) or isinstance(dt, Double):
            return dt.unit
        return None


@dataclass(frozen=True)
class Boolean(DataType):
    """
    Datatype: Boolean
    Decode: "Boolean" | { "type": "Boolean" }
    Encode: { "type": "Boolean" }
    """

    type: str = field(default="Boolean", init=False)

    def is_a(self, data: Any) -> bool:
        return isinstance(data, bool)

    def into(self, data: Any) -> bool:
        if isinstance(data, str):
            d = data.lower().strip()
            if d == "false":
                return False
            elif d == "true":
                return True
            raise ValueError(
                f"Cannot convert [{data}] into a {self.__class__.__name__} value"
            )
        return bool(data)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class String(DataType):
    """
    Datatype: String
    Decode: "String" | { "type": "String", "format": null|"<format>" }
    Encode: { "type": "String", "format": "<format>" }
    """

    FORMAT_VALIDATORS = {
        StringDateFormat.DATE: is_date,
        StringDateFormat.DATETIME: is_date_time,
        StringDateFormat.TIME: is_time,
        StringSubtypeFormat.URL: is_url,
        StringSubtypeFormat.EMAIL: is_email_address,
    }

    type: str = field(default="String", init=False)
    format: Optional[Union[StringDateFormat, StringSubtypeFormat]] = None

    def is_a(self, data: Any) -> bool:
        is_string = isinstance(data, str)
        if is_string and self.format is not None:
            return self.FORMAT_VALIDATORS[self.format](data)
        return is_string

    def into(self, data: Any) -> str:
        return str(data)

    @classmethod
    def pre_load(cls, **data: Any) -> Any:
        dict = {}
        if data.get("format"):
            key = data["format"].lower().replace("-", "").upper()
            val = (
                StringSubtypeFormat[key]
                if key in StringSubtypeFormat.__members__
                else StringDateFormat[key]
            )
            if val is None:
                raise ValueError(f"expected a valid string format but got {key}")
            dict["format"] = val
        return dict

    def __post_init__(self):
        if self.format is not None:
            if not isinstance(self.format, str):
                raise ValueError(f"string: format must be a String: {self.format}")
            if self.format not in self.FORMAT_VALIDATORS:
                raise ValueError(f"string: invalid format [{self.format}]")

    def __str__(self) -> str:
        return (
            f"{str(super().__str__())}({self.format})"
            if self.format
            else super().__str__()
        )


@dataclass(frozen=True)
class Long(DataType):
    """
    Datatype: Long
    Decode: "Long" | { "type": "Long", "unit": null|"<unit>" }
    Encode: { "type": "Long", "unit": null|"<unit>" }
    """

    type: str = field(default="Long", init=False)
    unit: Optional[str] = None

    def is_a(self, data: Any) -> bool:
        return isinstance(data, int)

    def into(self, data: Any) -> int:
        return int(data)

    def __post_init__(self):
        if self.unit is not None and not isinstance(self.unit, str):
            raise ValueError(f"long: unit must be a string: {self.unit}")
        elif self.unit == "":
            object.__setattr__(self, "unit", None)

    def __str__(self) -> str:
        return (
            f"{str(super().__str__())}({self.unit})" if self.unit else super().__str__()
        )


@dataclass(frozen=True)
class Double(DataType):
    """
    Datatype: Double
    Decode: "Double" | { "type": "Double", "unit": null|"<format>" }
    Encode: { "type": "Double", "unit": null|"<format>" }
    """

    type: str = field(default="Double", init=False)
    unit: Optional[str] = None

    def is_a(self, data: Any) -> bool:
        # integers can be safely coerced to a float as decimal extension
        # preserves data:
        return isinstance(data, (int, float))

    def into(self, data: Any) -> float:
        return float(data)

    def __post_init__(self):
        if self.unit is not None and not isinstance(self.unit, str):
            raise ValueError(f"double: unit must be a string: {self.unit}")
        elif self.unit == "":
            object.__setattr__(self, "unit", None)

    def __str__(self) -> str:
        return (
            f"{str(super().__str__())}({self.unit})" if self.unit else super().__str__()
        )


@dataclass(frozen=True)
class Date(DataType):
    """
    Datatype: Date
    Decode: "Date" | { "type": "Date" }
    Encode: { "type": "Date" }
    """

    type: str = field(default="Date", init=False)

    def is_a(self, data: Any) -> bool:
        if isinstance(data, datetime):
            return True
        try:
            self.into(data)
            return True
        except Exception:
            pass
        return False

    def into(self, data: Any) -> datetime:
        return normalize_datetime(data)

    def to_dict(self):
        return asdict(self)


class ArrayEnumMixin:
    ENUM_REQUIRED: bool

    @classmethod
    def pre_load(cls, **data):
        """
        Run before a datatype instance is constructed

        Raises
        ------
        ValueError

            Raised if properties `items` or `items.enum` is missing.
        """
        data = super().pre_load(**data)

        items = data.pop("items", None)
        if items is None:
            raise ValueError(
                f"collection: {cls.type.lower()} property .[items] missing"
            )

        enum = items.pop("enum", None)

        if cls.ENUM_REQUIRED and enum is None:
            raise ValueError(
                f"collection: {cls.type.lower()} property .[items][enum] missing"
            )

        inner_type = deserialize(deepcopy(items))

        if enum is not None:
            enum = [inner_type.into(e) for e in enum]

        return dict(items=inner_type, enum=enum)

    def to_dict(self) -> Dict[str, Any]:
        """
        Add special handling for Arrays and Enumerations:
        """
        d = cast("DataType", super()).to_dict()

        enum = d.pop("enum", None)
        if enum is not None:
            d["items"]["enum"] = enum
        return d


@dataclass(frozen=True)
class Array(ArrayEnumMixin, DataType):
    """
    Datatype: Array
    Decode/Encode: { "type": "array", "items": { "type": "<type>", "enum": [...] } }
    """

    ENUM_REQUIRED = False

    type: str = field(default="Array", init=False)
    # Despite the name, this is actually the type:
    items: Union[Boolean, Double, Long, String, Date]
    enum: Optional[List[Union[bool, int, str, float, datetime]]] = field(default=None)

    def __str__(self):
        return f"{super().__str__()}<{str(self.items)}>({str(self.enum)})"

    def into(self, data: List[Any]) -> List[Union[str, int, bool, float, datetime]]:
        values = [self.items.into(d) for d in data]
        self.validate(values)
        return values

    def validate(self, data: Sequence[Any]) -> None:
        """
        If enum is non-null, check if all items of data are in the enum.

        Raises
        -------
        ValueError
            - if all of subset is not of items
        """
        if self.enum:
            if not all(d in self.enum for d in data):
                raise ValueError(f"Items of {data} not all in enum {self.enum}")


@dataclass(frozen=True)
class Enumeration(ArrayEnumMixin, DataType):
    """
    Datatype: Enum
    Decode/Encode: { "type": "enum", "items": { "type": "<type>", "enum": [...] } }
    """

    ENUM_REQUIRED = True

    type: str = field(default="Enum", init=False)
    # Despite the name, this is actually the type:
    items: Union[Boolean, Double, Long, String, Date]
    enum: List[Union[bool, int, str, float, datetime]]

    def __str__(self):
        return f"{super().__str__()}<{str(self.items)}>({str(self.enum)})"

    def into(self, data: Any) -> Union[str, int, bool, float, datetime]:
        value = self.items.into(data)
        self.validate(value)
        return value

    def validate(self, data: Any) -> None:
        """
        Check if `data` is a valid value in this enum.

        Raises
        -------
        ValueError
            - if self.enum is None
            - if data is not of items
        """
        if self.enum is None:
            raise ValueError("Enumeration values are required for type `Enumeration`")

        if data not in self.enum:
            raise ValueError(f"[{data}] not in enum {self.enum}")


__types__: List[Type[DataType]] = [
    Boolean,
    String,
    Date,
    Long,
    Double,
    Array,
    Enumeration,
]


def deserialize(data: Union[str, Dict[str, Any]]) -> Optional[DataType]:
    """
    Deserialize a string to a datatype.
    """
    for t in __types__:
        result = None
        if isinstance(data, dict):
            result = t.from_dict(data, raise_on_fail=False)
        elif isinstance(data, str):
            result = t.from_json(data, raise_on_fail=False)
        if result is not None:
            return result
    raise ValueError(f"deserialize: can't decode: {data}")


def serialize(dt: DataType) -> str:
    """
    Serialize a datatype to a string.
    """
    return dt.to_json()
