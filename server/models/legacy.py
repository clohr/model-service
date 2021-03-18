from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, List, Optional, Set
from uuid import UUID, uuid4

import marshmallow  # type: ignore
from marshmallow import Schema, fields, post_load  # type: ignore

from core import types as t
from core import util as u
from core.json import CamelCaseSchema, Serializable

from .models import FromNodeMixin

# -----------------------------------------------------------------------------
# LEGACY MODELS
# -----------------------------------------------------------------------------


class ModelRelationshipStubSchema(CamelCaseSchema):
    id = fields.UUID()
    type = fields.String()
    name = fields.String()
    display_name = fields.String()
    description = fields.String()
    created_at = fields.DateTime(required=False, format="iso")
    updated_at = fields.DateTime(required=False, format="iso")
    created_by = fields.String()
    updated_by = fields.String()

    @post_load
    def make(self, data, **kwargs):
        return ModelRelationshipStub(**data)


@dataclass(frozen=True)
class ModelRelationshipStub(FromNodeMixin, Serializable):
    """
    A stub of a model relationship. This is used to create model relationships
    that do not specify a `from` and `to` target.
    """

    __schema__: ClassVar[ModelRelationshipStubSchema] = ModelRelationshipStubSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(["id", "type", "name", "display_name"])

    name: str
    display_name: str
    description: str
    type: str
    created_at: datetime
    updated_at: datetime
    created_by: t.UserNodeId
    updated_by: t.UserNodeId

    id: UUID = field(default_factory=uuid4)

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "created_at", u.normalize_datetime(self.created_at))
        object.__setattr__(self, "updated_at", u.normalize_datetime(self.updated_at))


class CreateRelationshipSchema(CamelCaseSchema):
    name = fields.String()
    display_name = fields.String()
    description = fields.String(required=False)
    from_ = fields.UUID(data_key="from")
    to = fields.UUID()

    @post_load
    def make(self, data, **kwargs):
        return CreateRelationship(**data)


@dataclass(frozen=True)
class CreateModelRelationship(FromNodeMixin, Serializable):
    """
    The request payload to create a model-level relationship in the legacy API.
    """

    __schema__: ClassVar[CreateRelationshipSchema] = CreateRelationshipSchema(
        unknown=marshmallow.EXCLUDE
    )

    PUBLIC: ClassVar[Set[str]] = set(
        ["name", "display_name", "description", "from_", "to"]
    )

    name: str
    display_name: str
    from_: UUID
    to: UUID
    description: str = field(default="")


# Root class for ProxyLinkTarget ADT used below
class ProxyLinkTarget:
    pass


# Used to encode the proxy link target ADT from the legacy concept/model
# service :
#
# object ProxyLinkTarget {
#   case class ConceptInstance(id: UUID) extends ProxyLinkTarget
#   case class ProxyInstance(`type`: ProxyType, externalId: ExternalId)
#       extends ProxyLinkTarget
# }
class ConceptInstanceSchema(CamelCaseSchema):
    id = fields.UUID()

    @post_load
    def make(self, data, **kwargs):
        return ConceptInstance(**data)


@dataclass(frozen=True)
class ConceptInstance(Serializable):

    __schema__: ClassVar[ConceptInstanceSchema] = ConceptInstanceSchema(
        unknown=marshmallow.EXCLUDE
    )

    id: UUID


# Use `Schema` vs `CamelCaseSchema` to preserve case:
class ProxyLinkConceptInstanceSchema(Schema):
    # Preserve the property case for "ConceptInstance" instead of "conceptInstance":
    ConceptInstance = fields.Nested(ConceptInstanceSchema())

    @post_load
    def make(self, data, **kwargs):
        return ProxyLinkConceptInstance(**data)


@dataclass(frozen=True)
class ProxyLinkConceptInstance(Serializable, ProxyLinkTarget):

    __schema__: ClassVar[
        ProxyLinkConceptInstanceSchema
    ] = ProxyLinkConceptInstanceSchema(unknown=marshmallow.EXCLUDE)

    ConceptInstance: ConceptInstance

    @property
    def id(self) -> UUID:
        return self.ConceptInstance.id


# -----------------------------------------------------------------------------
# Legacy graph query payload
# -----------------------------------------------------------------------------

# === Query type ==============================================================


class QueryType:
    @classmethod
    def read(cls, obj):
        if "concept" in obj:
            return ConceptQueryType(type=obj["concept"]["type"])
        elif "proxy" in obj:
            return ProxyQueryType()
        raise ValueError(f"[type]: bad type: [{str(obj)}]")

    @property
    def is_concept(self) -> bool:
        raise NotImplementedError

    @property
    def is_proxy(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ConceptQueryType(Serializable, QueryType):
    type: str

    @property
    def is_concept(self) -> bool:
        return True

    @property
    def is_proxy(self) -> bool:
        return False


@dataclass(frozen=True)
class ProxyQueryType(Serializable, QueryType):
    type: str = field(default="package")

    @property
    def is_concept(self) -> bool:
        return False

    @property
    def is_proxy(self) -> bool:
        return True


# === Order by ================================================================


@dataclass(frozen=True)
class OrderBy:
    field: str
    ascending: bool

    @classmethod
    def read(cls, obj):
        if "Ascending" in obj:
            return cls(field=obj["Ascending"]["field"], ascending=True)
        elif "ascending" in obj:
            return cls(field=obj["ascending"]["field"], ascending=True)
        elif "Descending" in obj:
            return cls(field=obj["Descending"]["field"], ascending=False)
        elif "descending" in obj:
            return cls(field=obj["descending"]["field"], ascending=False)
        else:
            raise ValueError(f"[order_by]: bad type: [{str(obj)}]")


# === Filters ===============================================================


class Predicate:
    @classmethod
    def read(cls, obj):
        # case 2-predicate:
        if "value1" in obj and "value2" in obj:
            return Predicate2(**obj)
        # case 1-predicate:
        elif "value" in obj:
            return Predicate1(**obj)
        else:
            raise ValueError(f"[filters]: malformed predicate: [{str(obj)}]")


@dataclass(frozen=True)
class Predicate1(Predicate):
    ALLOWED_OPERATIONS: ClassVar[Set[str]] = set(
        ["eq", "neq", "lt", "lte", "gt", "gte"]
    )

    operation: str
    value: t.GraphValue

    def __post_init__(self):
        if self.operation not in self.ALLOWED_OPERATIONS:
            raise ValueError(f"Predicate1: invalid operation [{self.operation}]")


@dataclass(frozen=True)
class Predicate2(Predicate):
    ALLOWED_OPERATIONS: ClassVar[Set[str]] = set(["between", "inside", "outside"])

    operation: str
    value1: t.GraphValue
    value2: t.GraphValue

    def __post_init__(self):
        if self.operation not in self.ALLOWED_OPERATIONS:
            raise ValueError(f"Predicate1: invalid operation [{self.operation}]")


@dataclass(frozen=True)
class KeyFilter:
    key: str
    predicate: Predicate

    @classmethod
    def read(cls, obj):
        return cls(key=obj["key"], predicate=Predicate.read(obj["predicate"]))


# === Joins ===================================================================


@dataclass(frozen=True)
class Join:
    relationship_type: Optional[str]
    target_type: QueryType
    filters: List[KeyFilter]
    key: Optional[str] = field(default=None)

    @classmethod
    def read(cls, obj):
        return cls(
            relationship_type=obj.get("relationshipType", obj.get("relationship_type")),
            target_type=QueryType.read(obj.get("targetType", obj.get("target_type"))),
            filters=[KeyFilter.read(o) for o in obj.get("filters", [])],
            key=obj.get("key", None),
        )


# === Select ==================================================================


class Select:
    @classmethod
    def read(cls, obj):
        if obj is None:
            return None
        if "GroupCount" in obj:
            return SelectGroupCount(**obj["GroupCount"])
        elif "Concepts" in obj:
            concepts_options = obj["Concepts"]
            join_keys = concepts_options.get(
                "join_keys", concepts_options.get("joinKeys", [])
            )
            return SelectConcepts(join_keys=join_keys)
        else:
            raise ValueError(f"[select]: malformed select: [{str(obj)}]")

    @property
    def is_group_count(self) -> bool:
        return False


@dataclass(frozen=True)
class SelectConcepts(Select):
    join_keys: List[str]


@dataclass(frozen=True)
class SelectGroupCount(Select):
    field: str
    key: Optional[str] = None  # field(default=None)

    @property
    def is_group_count(self) -> bool:
        return True


# =============================================================================


class GraphQuerySchema(marshmallow.Schema):
    type = fields.Method(
        data_key="type", serialize="_write_type", deserialize="_read_type"
    )
    filters = fields.Method(serialize="_write_filters", deserialize="_read_filters")
    joins = fields.Method(serialize="_write_joins", deserialize="_read_joins")
    order_by = fields.Method(
        serialize="_write_order_by",
        deserialize="_read_order_by",
        required=False,
        allow_none=True,
    )
    limit = fields.Integer(required=False, allow_none=True)
    offset = fields.Integer(required=False, allow_none=True)
    select = fields.Method(
        serialize="_write_select",
        deserialize="_read_select",
        required=False,
        allow_none=True,
        default=None,
    )

    @post_load
    def make(self, data, **kwargs):
        return GraphQuery(**data)

    def _read_type(self, obj):
        return QueryType.read(obj)

    def _write_type(self, value):
        raise NotImplementedError

    def _read_filters(self, obj):
        return [KeyFilter.read(o) for o in (obj or [])]

    def _write_filters(self, value):
        raise NotImplementedError

    def _read_joins(self, obj):
        return [Join.read(o) for o in (obj or [])]

    def _write_joins(self, value):
        raise NotImplementedError

    def _read_order_by(self, obj):
        return OrderBy.read(obj)

    def _write_order_by(self, value):
        raise NotImplementedError

    def _read_select(self, obj):
        return Select.read(obj)

    def _write_select(self, value):
        raise NotImplementedError


@dataclass(frozen=True)
class GraphQuery(Serializable):
    __schema__: ClassVar[GraphQuerySchema] = GraphQuerySchema(
        unknown=marshmallow.EXCLUDE
    )

    type: QueryType
    filters: List[KeyFilter]
    joins: List[Join]
    order_by: Optional[OrderBy] = field(default=None)
    limit: Optional[int] = field(default=None)
    offset: Optional[int] = field(default=None)
    select: Optional[Select] = field(default=None)
