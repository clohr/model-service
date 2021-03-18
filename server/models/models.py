import dataclasses
import keyword
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Union, cast
from uuid import UUID, uuid4

import humps  # type: ignore
import marshmallow  # type: ignore
from marshmallow import (  # type: ignore
    ValidationError,
    fields,
    post_load,
    pre_dump,
    validates,
)
from more_itertools import partition  # type: ignore

from core import types as t
from core.json import CamelCaseSchema, Serializable
from core.util import is_datetime, normalize_datetime

from .. import errors as err
from . import datatypes as dt
from .datatypes import into_uuid
from .util import (
    RESERVED_PREFIX,
    is_reserved_model_name,
    is_reserved_property_name,
    strip_reserved_prefix,
)


def get_organization_id(thing: object) -> t.OrganizationId:
    """
    Extract an organization ID from an input value.

    Raises
    ------
    InvalidOrganizationError
    """
    if isinstance(thing, int):
        return t.OrganizationId(thing)
    try:
        int_id = int(thing)  # type: ignore
        return t.OrganizationId(int_id)
    except ValueError:
        raise err.InvalidOrganizationError(id=str(thing))


def get_dataset_id(thing: object) -> t.DatasetId:
    """
    Extract an dataset ID from an input value.

    Raises
    ------
    InvalidDatasetError
    """
    if isinstance(thing, int):
        return t.DatasetId(thing)
    try:
        int_id = int(thing)  # type: ignore
        return t.DatasetId(int_id)
    except ValueError:
        raise err.InvalidDatasetError(id=str(thing))


def get_model_id(thing: Union["Model", t.ModelId, UUID, str]) -> t.ModelId:
    """
    Given a `Model` or an model ID, return a model ID.
    """
    if isinstance(thing, UUID):
        return t.ModelId(thing)
    elif isinstance(thing, Model):
        return thing.id
    return t.ModelId(UUID(thing))


is_model_id = dt.is_uuid


def get_model_property_id(
    thing: Union["ModelProperty", t.ModelPropertyId, UUID, str]
) -> t.ModelPropertyId:
    """
    Given a `ModelProperty` or an model property ID, return a model property ID.
    """
    if isinstance(thing, UUID):
        return t.ModelPropertyId(thing)
    elif isinstance(thing, ModelProperty):
        return thing.id
    return t.ModelPropertyId(UUID(thing))


is_model_property_id = dt.is_uuid


def get_record_id(thing: Union["Record", t.RecordId, UUID, str]) -> t.RecordId:
    """
    Given a `Record` or an record ID, return a record ID.
    """
    if isinstance(thing, UUID):
        return t.RecordId(thing)
    elif isinstance(thing, Record):
        return thing.id
    return t.RecordId(UUID(thing))


is_record_id = dt.is_uuid


def get_model_relationship_id(
    thing: Union["ModelRelationship", t.ModelRelationshipId, UUID, str]
) -> t.ModelRelationshipId:
    """
    Given a `ModelRelationship` or a relationship ID, return a relationship ID.
    """
    if isinstance(thing, UUID):
        return t.ModelRelationshipId(thing)
    elif isinstance(thing, ModelRelationship):
        return thing.id
    return t.ModelRelationshipId(UUID(thing))


is_model_relationship_id = dt.is_uuid

# UUIDs have 36 characters
UUID_LENGTH = 36


def normalize_relationship_type(relationship_name: str) -> str:
    """
    Normalizes a relationship name to upper-snake-case.

    If the relationship name has a UUID suffix added by the Python client
    or frontend, it is removed.

    This also helps dealing with the Neo4j relationship type limitation of max
    of 65K unique names.

    Examples
    --------
    - ""belongs_to_478e215d-04ec-4cdf-ac8b-d5289601c9f7" -> "BELONGS_TO"
    """
    from .validate import validate_relationship_name

    validate_relationship_name(relationship_name)

    if (
        len(relationship_name) > UUID_LENGTH + 1
        and relationship_name[-(UUID_LENGTH + 1)] == "_"
        and dt.is_uuid(relationship_name[-UUID_LENGTH:])
    ):
        relationship_name = relationship_name[: -(UUID_LENGTH + 1)]

    return relationship_name.replace("/", "_").replace(".", "_").upper().strip()


def get_relationship_type(
    r: Union["ModelRelationship", t.RelationshipType, t.RelationshipName, str]
) -> t.RelationshipType:
    """
    Transform and format a string into a relationship type.

    A relationship type is the canonical representation of a relationship
    in Neo4j: a typeful, an upper-snake-cased name.

    Examples
    --------
    - "foo"          -> "FOO"
    - "DoctorVisit"  -> "DOCTOR_VISIT"
    - "tHiS_IsATesT" -> "THIS_IS_A_TEST"
    """
    relationship_type = r.type if isinstance(r, ModelRelationship) else r
    return t.RelationshipType(normalize_relationship_type(relationship_type))


def get_record_relationship_id(
    thing: Union["RecordRelationship", t.RecordRelationshipId, UUID, str]
) -> t.RecordRelationshipId:
    """
    Given a `RecordRelationship` or a relationship ID, return a typed
    relationship ID.
    """
    if isinstance(thing, UUID):
        return t.RecordRelationshipId(thing)
    elif isinstance(thing, RecordRelationship):
        return thing.id
    return t.RecordRelationshipId(UUID(thing))


is_record_relationship_id = dt.is_uuid


def get_package_proxy_id(
    r: Union["PackageProxy", t.PackageProxyId, UUID, str]
) -> t.PackageProxyId:
    """
    Given a `PackageProxy` or an PackageProxy ID, return a PackageProxy ID.
    """
    if isinstance(r, UUID):
        return t.PackageProxyId(r)
    elif isinstance(r, PackageProxy):
        return t.PackageProxyId(r.id)
    return t.PackageProxyId(UUID(r))


is_package_proxy_id = dt.is_uuid

###############################################################################


class OrderDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"

    @classmethod
    def parse(cls, s: str) -> "OrderDirection":
        return OrderDirection(s.strip().lower())


# Order by


@dataclass(frozen=True)
class OrderBy:
    @classmethod
    def field(cls, name: str, ascending: bool = True) -> "OrderBy":
        return OrderByField(name=name, ascending=ascending)

    @property
    def is_field(self) -> bool:
        return False

    @classmethod
    def relationship(cls, type: str, ascending: bool = True) -> "OrderBy":
        return OrderByRelationship(type=type, ascending=ascending)

    @property
    def is_relationship(self) -> bool:
        return False


@dataclass(frozen=True)
class OrderByField(OrderBy):

    CREATED_AT_FIELDS: ClassVar[Set[str]] = set(
        [
            "~created_at",
            "created_at",
            "createdAt",
            "$created_at",
            "$createdAt",
            RESERVED_PREFIX + "created_at",
            RESERVED_PREFIX + "createdAt",
        ]
    )
    UPDATED_AT_FIELDS: ClassVar[Set[str]] = set(
        [
            "~updated_at",
            "updated_at",
            "updatedAt",
            "$updated_at",
            "$updatedAt",
            RESERVED_PREFIX + "updated_at",
            RESERVED_PREFIX + "updatedAt",
        ]
    )

    name: str
    ascending: bool = field(default=True)

    @property
    def is_field(self) -> bool:
        return True

    @property
    def is_created_at(self) -> bool:
        name = self.name.strip()
        return name in self.CREATED_AT_FIELDS or name.lower() in self.CREATED_AT_FIELDS

    @property
    def is_updated_at(self) -> bool:
        name = self.name.strip()
        return name in self.UPDATED_AT_FIELDS or name.lower() in self.UPDATED_AT_FIELDS

    @property
    def direction(self) -> OrderDirection:
        if self.ascending:
            return OrderDirection.ASC
        else:
            return OrderDirection.DESC


@dataclass(frozen=True)
class OrderByRelationship(OrderBy):

    SUPPORTED_LABELS: ClassVar[Set[str]] = set(
        [
            "~label",
            "label",
            "$label",
            RESERVED_PREFIX + "label",
            "type",
            "$type",
            RESERVED_PREFIX + "type",
        ]
    )

    type: str
    ascending: bool = field(default=True)

    @property
    def is_relationship(self) -> bool:
        return True

    @property
    def is_supported_type(self) -> bool:
        """
        Only relationship labels (types) are supported for sorting.
        """
        t = self.type.strip()
        return t in self.SUPPORTED_LABELS or t.lower() in self.SUPPORTED_LABELS


###############################################################################


class FromNodeMixin:
    @classmethod
    def _is_reserved(cls, t: Tuple[str, t.GraphValue]) -> bool:
        k, _ = t
        return is_reserved_property_name(k)

    @classmethod
    def from_node(cls, **data) -> object:
        defined_properties = set([f.name for f in dataclasses.fields(cls)])

        # Partition all reserved properties (those whose name begin with the
        # RESERVED_PREFIX character), and user-settable properties:
        user_props, reserved_props = partition(cls._is_reserved, data.items())

        props = {humps.decamelize(k): v for k, v in user_props}
        for k, v in reserved_props:
            kk = strip_reserved_prefix(humps.decamelize(k))
            if kk in defined_properties:
                props[kk] = v

        # Append '_' to any kwargs that are a reserved word:
        for k in props:
            if keyword.iskeyword(k):
                props[k + "_"] = props.pop(k)

        return cls(**props)  # type: ignore


class DatasetSchema(CamelCaseSchema):
    id = fields.Integer()
    node_id = fields.String(allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return PropertyValue(**data)


@dataclass(frozen=True)
class Dataset(Serializable):

    __schema__: ClassVar[DatasetSchema] = DatasetSchema(unknown=marshmallow.EXCLUDE)

    PUBLIC: ClassVar[Set[str]] = set(["id", "node_id"])

    id: t.DatasetId
    node_id: Optional[str] = field(default=None)

    @classmethod
    def from_node(cls, data) -> "Dataset":

        id = t.DatasetId(data["id"])
        node_id: Optional[str] = data.get("node_id")

        return Dataset(id=id, node_id=node_id)


@dataclass(frozen=True)
class Package(Serializable):
    id: int
    node_id: str


class ModelSchema(CamelCaseSchema):
    id = fields.UUID()
    name = fields.String()
    display_name = fields.String()
    description = fields.String()
    count = fields.Integer(default=0)
    created_at = fields.DateTime(format="iso")
    updated_at = fields.DateTime(format="iso")
    created_by = fields.String()
    updated_by = fields.String()
    template_id = fields.UUID(required=False, allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return Model(**data)


@dataclass(frozen=True)
class Model(FromNodeMixin, Serializable):

    __schema__: ClassVar[ModelSchema] = ModelSchema(unknown=marshmallow.EXCLUDE)

    PUBLIC: ClassVar[Set[str]] = set(
        ["id", "name", "display_name", "description", "template_id"]
    )

    id: t.ModelId
    name: str
    display_name: str
    description: str
    count: int
    created_at: datetime
    updated_at: datetime
    created_by: t.UserNodeId
    updated_by: t.UserNodeId
    template_id: Optional[UUID] = field(default=None)

    @validates("name")
    def validate_name(self, name):
        # HACK: this validation is defined as a method to work around a
        # circular import between `models.validation` and `models.types`
        from .validate import validate_model_name

        try:
            validate_model_name(name)
        except err.ModelNameValidationError as e:
            raise ValidationError from e

    def __post_init__(self):
        # Needed since neotime.DateTime does not work with Python copy.deepcopy
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))


class ModelPropertySchema(CamelCaseSchema):
    name = fields.String(required=True)
    display_name = fields.String(required=True)
    data_type = fields.Function(
        required=True,
        serialize=lambda o: o.data_type.to_dict(),
        deserialize=dt.deserialize,
    )
    description = fields.String(required=False)
    index = fields.Integer(required=False, default=0)
    locked = fields.Boolean(required=False, default=False)
    required = fields.Boolean(required=False, default=False)
    model_title = fields.Boolean(required=False, default=False)
    # If True, show this property as a column in tables of records
    default = fields.Boolean(required=False, default=True)
    default_value = fields.Raw(required=False, allow_none=True)
    created_at = fields.DateTime(required=False, format="iso")
    updated_at = fields.DateTime(required=False, format="iso")
    id = fields.UUID(required=False, allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return ModelProperty(**data)


@dataclass(frozen=True)
class ModelProperty(FromNodeMixin, Serializable):
    """
    A property on a model represented using a node and modelled as

      (m:Model)--[r:MODEL_RELATIONSHIP_TYPE]->(p:ModelProperty)
    """

    IMMUTABLE: ClassVar[Set[str]] = set(["name", "data_type"])
    PUBLIC: ClassVar[Set[str]] = set(
        [
            "id",
            "name",
            "display_name",
            "data_type",
            "description",
            "index",
            "locked",
            "required",
            "model_title",
            "default_value",
            "default",
        ]
    )

    __schema__: ClassVar[ModelPropertySchema] = ModelPropertySchema(
        unknown=marshmallow.EXCLUDE
    )

    name: str
    display_name: str
    data_type: dt.DataType
    description: str = field(default="")
    index: int = field(default=0)
    locked: bool = field(default=False)
    required: bool = field(default=False)
    model_title: bool = field(default=False)
    default: bool = field(default=True)
    default_value: Optional[t.GraphValue] = field(default=None)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = field(default="")
    updated_by: str = field(default="")
    id: t.ModelPropertyId = field(default_factory=lambda: t.ModelPropertyId(uuid4()))

    @validates("name")
    def validate_name(self, name):
        # HACK: this validation is defined as a method to work around a
        # circular import between `models.validation` and `models.types`
        from .validate import validate_property_name

        try:
            validate_property_name(name)
        except err.PropertyNameValidationError as e:
            raise ValidationError from e

    def to_dict_with_string_datatype(self, camel_case: bool = False):
        """
        Special method for serializing properties with the datatype represented
        as a serialized JSON dict.
        """
        d = self.to_dict(camel_case=camel_case)
        if camel_case:
            d["dataType"] = dt.serialize(self.data_type)
        else:
            d["data_type"] = dt.serialize(self.data_type)
        return d

    def __post_init__(self):
        if isinstance(self.data_type, str):
            # HACK: This is required to mutate frozen dataclasses
            object.__setattr__(self, "data_type", dt.deserialize(self.data_type))
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))


class ModelRelationshipSchema(CamelCaseSchema):
    id = fields.UUID()
    from_ = fields.UUID(data_key="from")
    to = fields.UUID()
    type = fields.String()
    name = fields.String()
    display_name = fields.String()
    description = fields.String()
    one_to_many = fields.Boolean()
    created_by = fields.String()
    updated_by = fields.String()
    created_at = fields.DateTime(required=False, format="iso")
    updated_at = fields.DateTime(required=False, format="iso")
    index = fields.Integer(required=False, allow_none=True, default=None)

    @post_load
    def make(self, data, **kwargs):
        return ModelRelationship(**data)


@dataclass(frozen=True)
class ModelRelationship(Serializable):
    """
    A representation of a relationship between two `Model` nodes at the
    schema level of the form:

        (m:Model)--[r:MODEL_RELATIONSHIP_TYPE]->(p:Model)

    The `one_to_many` property encodes how the relationship can applied at the
    record level:

        - one_to_many = True : The relationship can be used to link one record
            to many other records, or potentially none.

        - one_to_many = False : The relationship can only be used to link to a
            maximum of one other record (one-to-one). This is how
            "linked-property" functionality currently works in the
            concepts-service.

    The `index` property stores the display order of linked property
    relationships, like the `ModelProperty.index`.  This is returned as
    `position` in the legacy API.
    """

    __schema__: ClassVar[ModelRelationshipSchema] = ModelRelationshipSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(
        ["id", "name", "display_name", "one_to_many", "from_", "to"]
    )

    id: t.ModelRelationshipId
    one_to_many: bool
    name: t.RelationshipName
    display_name: str
    description: str
    from_: UUID
    to: UUID
    type: t.RelationshipType
    created_by: t.UserNodeId
    updated_by: t.UserNodeId
    created_at: datetime
    updated_at: datetime
    index: Optional[int]

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))


class PropertyValueSchema(CamelCaseSchema):
    name = fields.String()
    value = fields.Raw()

    @post_load
    def make(self, data, **kwargs):
        return PropertyValue(**data)


@dataclass(frozen=True)
class PropertyValue(Serializable):
    """
    Key value structure used to specify property to update for a record

    NOTE: this is currently unused but will be needed when we build the legacy/
    backwards-compatible API.
    """

    __schema__: ClassVar[PropertyValueSchema] = PropertyValueSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["name", "value"])

    name: str
    value: t.GraphValue


class RecordStubSchema(CamelCaseSchema):
    id = fields.UUID()
    title = fields.String(required=False, allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return RecordStub(**data)


@dataclass(frozen=True)
class RecordStub(Serializable):
    """
    A stub of a record which contains partial information about it.

    A stub is a reference to a full record containing the ID of the actual
    record along with a string describing the contents of the record itself.
    """

    __schema__: ClassVar[RecordStubSchema] = RecordStubSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["id", "title"])

    id: UUID
    title: Optional[str] = field(default=None)

    @classmethod
    def from_node(cls, data: List[Tuple[str, t.GraphValue]]) -> "RecordStub":
        values = dict(data)
        title: Optional[str] = cast(Optional[str], values.get("title")) or cast(
            Optional[str], values.get("name")
        )
        id = into_uuid(values["@id"])
        return RecordStub(id=id, title=title)


class RecordSchema(CamelCaseSchema):
    id = fields.UUID()
    values = fields.Dict(keys=fields.Str(), values=fields.Raw())
    created_at = fields.DateTime(required=False, format="iso")
    updated_at = fields.DateTime(required=False, format="iso")
    created_by = fields.String()
    updated_by = fields.String()

    @pre_dump
    def json_dump_safe(self, record, many=False):
        for k in record.values:
            v = record.values[k]
            if is_datetime(v):
                record.values[k] = normalize_datetime(v)
        return record

    @post_load
    def make(self, data, **kwargs):
        return Record(**data)


@dataclass(frozen=True, order=True)
class Record(Serializable):
    """
    A record and associated property values.
    """

    __schema__: ClassVar[RecordSchema] = RecordSchema(unknown=marshmallow.EXCLUDE)

    PUBLIC: ClassVar[Set[str]] = set(["id", "values"])

    id: t.RecordId
    values: Dict[str, Union[t.GraphValue, "Record", RecordStub]]
    created_at: datetime
    updated_at: datetime
    created_by: t.UserNodeId
    updated_by: t.UserNodeId
    name: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))

    @classmethod
    def from_node(
        cls,
        data: List[Tuple[str, Union[t.GraphValue, "Record", RecordStub]]],
        created_at: datetime,
        updated_at: datetime,
        created_by: str,
        updated_by: str,
        property_map: Optional[Dict[str, ModelProperty]] = None,
        fill_missing: bool = False,
    ) -> "Record":
        """
        Hydrate a `Record` from a Neo4j `Node`.

        This assumes that all non-id fields on the Node are property values.

        - If a property map is provided, missing default values will be used
          to fill in missing property entries from the `Record#values` member.

        - If `fill_missing` is `True` and a property map is provided, any
          property appearing in the property map that is either not present in
         `data` or that does not possess a default value will be set to `None`.

          If `fill_missing` is omitted or `False`, no entry will be filled in
          `Record#values`.
        """
        values = dict(data)

        id = t.RecordId(into_uuid(values.pop("@id")))

        # Pop keywords in a loop instead of a dict-comprehension to avoid
        # allocating a new dictionary
        reserved = [k for k in values if is_reserved_property_name(k)]
        for k in reserved:
            values.pop(k)

        record = cls(
            id=id,
            values=values,
            created_at=normalize_datetime(created_at),
            updated_at=normalize_datetime(updated_at),
            created_by=t.UserNodeId(created_by),
            updated_by=t.UserNodeId(updated_by),
            name=cls.compute_record_name(property_map, values),
        )

        if property_map is not None:
            record.fill_missing_values(property_map, fill_missing=fill_missing)

        return record

    def embed(self, with_key: str, other: Union["Record", RecordStub]) -> "Record":
        """
        Embeds the `other` record or stub to the `values` dict of this record
        under under the supplied key `with_key`.
        """
        key = humps.decamelize(with_key).lower()
        if key in self.values:
            raise Exception(f"violation: record linking name already taken: {key}")
        self.values[key] = other
        return self

    def fill_missing_values(
        self, property_map: Dict[str, ModelProperty], fill_missing: bool = False
    ) -> None:
        """
        Scan `records` and for each record `r`, populate any properties of `r`
        that have a default value defined for it.
        """
        for name, prop in property_map.items():
            # Does the property have a default value?
            if prop.default and (name not in self.values or self.values[name] is None):
                self.values[name] = prop.default_value
            elif fill_missing and name not in self.values:
                # Create an entry for the missing value:
                self.values[name] = None

    @staticmethod
    def compute_record_name(
        property_map: Optional[Dict[str, ModelProperty]],
        values: Dict[str, Union[t.GraphValue, "Record", RecordStub]],
    ) -> Optional[str]:
        """
        The name of the record is the value of the of the "title" property for
        the record's model.
        """
        if property_map is None:
            return None

        title_property = next((p for p in property_map.values() if p.model_title), None)
        if title_property is None:
            return None

        title_value = values.get(title_property.name)
        if title_value is None:
            return None

        return str(title_value)


class PagedResultSchema(CamelCaseSchema):
    results = fields.Nested(Record.schema(), many=True)
    next_page = fields.Integer(allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return PagedResult(**data)


@dataclass(frozen=True)
class PagedResult(Serializable):
    """
    A paged result
    """

    __schema__: ClassVar[PagedResultSchema] = PagedResultSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["results", "next_page"])

    results: List[Record]
    next_page: Optional[t.NextPageCursor]

    @property
    def empty(self):
        return len(self.results) == 0

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]


class RecordRelationshipSchema(CamelCaseSchema):
    id = fields.UUID()
    from_ = fields.UUID(data_key="from")
    to = fields.UUID()
    type = fields.String()
    model_relationship_id = fields.UUID()
    name = fields.String()
    display_name = fields.String()
    one_to_many = fields.Boolean()
    created_at = fields.DateTime(required=False, format="iso")
    updated_at = fields.DateTime(required=False, format="iso")
    created_by = fields.String()
    updated_by = fields.String()

    @post_load
    def make(self, data, **kwargs):
        return RecordRelationship(**data)


@dataclass(frozen=True)
class RecordRelationship(FromNodeMixin, Serializable):
    """
    A representation of a relationship between two `Record` nodes at the
    instance level of the form:

        (r1:Record)--[r:MODEL_RELATIONSHIP_TYPE]->(r2:Record)
    """

    __schema__: ClassVar[RecordRelationshipSchema] = RecordRelationshipSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(
        ["id", "from_", "to", "type", "name", "model_relationship_id", "display_name"]
    )

    from_: t.RecordId
    to: t.RecordId
    type: t.RelationshipType
    model_relationship_id: t.ModelRelationshipId
    name: t.RelationshipName
    display_name: str
    one_to_many: bool
    created_at: datetime
    updated_at: datetime
    created_by: t.UserNodeId
    updated_by: t.UserNodeId
    id: t.RecordRelationshipId

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))


class PackageProxySchema(CamelCaseSchema):
    id = fields.UUID()
    proxy_instance_id = fields.UUID()
    package_id = fields.Integer()
    package_node_id = fields.String()
    relationship_type = fields.String()
    created_at = fields.DateTime(required=False, format="iso")
    updated_at = fields.DateTime(required=False, format="iso")
    created_by = fields.String()
    updated_by = fields.String()

    @post_load
    def make(self, data, **kwargs):
        return PackageProxy(**data)


@dataclass(frozen=True)
class PackageProxy(FromNodeMixin, Serializable):

    __schema__: ClassVar[PackageProxySchema] = PackageProxySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(
        ["id", "proxy_instance_id", "package_id", "package_node_id"]
    )

    id: UUID
    proxy_instance_id: UUID
    package_id: int
    package_node_id: str
    relationship_type: t.RelationshipName
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))


class ProxyRelationshipCountSchema(CamelCaseSchema):
    name = fields.String()
    display_name = fields.String()
    count = fields.Integer()

    @post_load
    def make(self, data, **kwargs):
        return ProxyRelationshipCount(**data)


@dataclass(frozen=True)
class ProxyRelationshipCount(FromNodeMixin, Serializable):

    __schema__: ClassVar[ProxyRelationshipCountSchema] = ProxyRelationshipCountSchema(
        unknown=marshmallow.EXCLUDE
    )

    name: str
    display_name: str
    count: int


# --- Topology ----------------------------------------------------------------


class ModelTopologySchema(CamelCaseSchema):
    id = fields.UUID()
    name = fields.String()
    display_name = fields.String()
    description = fields.String()
    count = fields.Integer()
    created_at = fields.DateTime(required=False, format="iso")
    updated_at = fields.DateTime(required=False, format="iso")

    @post_load
    def make(self, data, **kwargs):
        return ModelTopology(**data)


@dataclass(frozen=True)
class ModelTopology(Serializable):

    __schema__: ClassVar[ModelTopologySchema] = ModelTopologySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(
        [
            "id",
            "name",
            "display_name",
            "description",
            "count",
            "created_at",
            "updated_at",
        ]
    )

    id: UUID
    name: str
    display_name: str
    description: str
    count: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", normalize_datetime(self.updated_at))


# --- Structure ---------------------------------------------------------------


@dataclass(frozen=True)
class GraphSchemaStructure:
    models: List[Model]
    relationships: List[ModelRelationship]


# --- Summary -----------------------------------------------------------------


class ModelSummarySchema(CamelCaseSchema):
    name = fields.String()
    count = fields.Integer()

    @post_load
    def make(self, data, **kwargs):
        return ModelSummary(**data)


@dataclass(frozen=True)
class ModelSummary(Serializable):

    __schema__: ClassVar[ModelSummarySchema] = ModelSummarySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["name", "count"])

    name: str
    count: int


class RelationshipSummarySchema(CamelCaseSchema):
    name = fields.String()
    from_ = fields.UUID(data_key="from")
    to = fields.UUID()
    count = fields.Integer()

    @post_load
    def make(self, data, **kwargs):
        return RelationshipSummary(**data)


@dataclass(frozen=True)
class RelationshipSummary(Serializable):

    __schema__: ClassVar[RelationshipSummarySchema] = RelationshipSummarySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["name", "from_", "to", "count"])

    name: str
    from_: UUID
    to: UUID
    count: int


class RelationshipTypeSummarySchema(CamelCaseSchema):
    name = fields.String()
    count = fields.Integer()

    @post_load
    def make(self, data, **kwargs):
        return RelationshipTypeSummary(**data)


class RecordSummarySchema(CamelCaseSchema):
    name = fields.String()
    display_name = fields.String()
    count = fields.Integer()

    @post_load
    def make(self, data, **kwargs):
        return RecordSummary(**data)


@dataclass(frozen=True)
class RecordSummary(Serializable):

    __schema__: ClassVar[RecordSummarySchema] = RecordSummarySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["name", "display_name", "count"])

    name: str
    display_name: str
    count: int


@dataclass(frozen=True)
class RelationshipTypeSummary(Serializable):

    __schema__: ClassVar[RelationshipTypeSummarySchema] = RelationshipTypeSummarySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["name", "count"])

    name: str
    count: int


class TopologySummarySchema(CamelCaseSchema):
    model_summary = fields.Nested(ModelSummary.schema(), many=True)
    relationship_summary = fields.Nested(RelationshipSummary.schema(), many=True)
    relationship_type_summary = fields.Nested(
        RelationshipTypeSummary.schema(), many=True
    )
    model_count = fields.Integer()
    model_record_count = fields.Integer()
    relationship_count = fields.Integer()
    relationship_record_count = fields.Integer()
    relationship_type_count = fields.Integer()

    @post_load
    def make(self, data, **kwargs):
        return TopologySummary(**data)


@dataclass(frozen=True)
class TopologySummary(Serializable):

    __schema__: ClassVar[TopologySummarySchema] = TopologySummarySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(
        [
            "model_summary",
            "relationship_summary",
            "relationship_type_summary",
            "model_count",
            "model_record_count",
            "relationship_count",
            "relationship_record_count",
            "relationship_type_count",
        ]
    )

    model_summary: ModelSummary
    relationship_summary: RelationshipSummary
    relationship_type_summary: RelationshipTypeSummary
    model_count: int
    model_record_count: int
    relationship_count: int
    relationship_record_count: int
    relationship_type_count: int


@dataclass(frozen=False)
class CreateRecordRelationship(Serializable):

    from .legacy import CreateModelRelationship

    from_: t.RecordId
    to: t.RecordId
    model_relationship_to_create: Optional[CreateModelRelationship] = None
    model_relationship: Optional[ModelRelationship] = None


class DatasetDeletionCountsSchema(CamelCaseSchema):
    models = fields.Integer()
    properties = fields.Integer()
    records = fields.Integer()
    packages = fields.Integer()
    relationship_stubs = fields.Integer()

    @post_load
    def make(self, data, **kwargs):
        return DatasetDeletionCounts(**data)


@dataclass(frozen=False)
class DatasetDeletionCounts(Serializable):

    __schema__: ClassVar[DatasetDeletionCountsSchema] = DatasetDeletionCountsSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(
        ["models", "properties", "records", "packages", "relationship_stubs"]
    )

    models: int
    properties: int
    records: int
    packages: int
    relationship_stubs: int

    @classmethod
    def empty(cls):
        return DatasetDeletionCounts(
            models=0, properties=0, records=0, packages=0, relationship_stubs=0
        )

    def __add__(self, other) -> "DatasetDeletionCounts":
        return DatasetDeletionCounts(
            models=self.models + other.models,
            properties=self.properties + other.properties,
            records=self.records + other.records,
            packages=self.packages + other.packages,
            relationship_stubs=self.relationship_stubs + other.relationship_stubs,
        )

    def update(self, counts: "DatasetDeletionCounts") -> "DatasetDeletionCounts":
        return self + counts


class DatasetDeletionSummarySchema(CamelCaseSchema):
    done = fields.Boolean()
    counts = fields.Nested(DatasetDeletionCounts.schema(), many=False)

    @post_load
    def make(self, data, **kwargs):
        return DatasetDeletionSummarySchema(**data)


@dataclass(frozen=False)
class DatasetDeletionSummary(Serializable):

    __schema__: ClassVar[DatasetDeletionSummarySchema] = DatasetDeletionSummarySchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["done", "counts"])

    done: bool
    counts: DatasetDeletionCounts

    def update_counts(self, counts: DatasetDeletionCounts) -> "DatasetDeletionSummary":
        return DatasetDeletionSummary(done=self.done, counts=self.counts.update(counts))
