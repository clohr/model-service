import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Set, Union

from marshmallow import fields, post_load  # type: ignore

from core.json import CamelCaseSchema, Serializable

from . import GraphValue, Model, ModelId

# --- Connections and predicates ----------------------------------------------


class RelationshipTargetSchema(CamelCaseSchema):
    """
    Represents a model that a relationship ship should be created when
    generating a query.

    A relationship target is basically a predicate (in that it may create
    relationship in q query), but doesn't generate a clause in the WHERE
    portion of the query.
    """

    model = fields.String()

    @post_load
    def make(self, data, **kwargs):
        return RelationshipTarget(**data)


@dataclass(frozen=True)
class RelationshipTarget(Serializable):

    __schema__: ClassVar[CamelCaseSchema] = RelationshipTargetSchema()

    """
    A representation of a node to produce a join to.
    """
    model: Union[str, ModelId, Model]  # The model `field` resides on

    def with_model(self, model: Model) -> "RelationshipTarget":
        """
        Replace the value of `model`, returning a new predicate.
        """
        return dataclasses.replace(self, model=model)


class Operator(str, Enum):
    IS = "IS"
    IS_NOT = "IS NOT"
    EQUALS = "="
    NOT_EQUALS = "<>"
    LESS_THAN = "<"
    LESS_THAN_EQUALS = "<="
    GREATER_THAN = ">"
    GREATER_THAN_EQUALS = ">="
    STARTS_WITH = "STARTS WITH"
    CONTAINS = "CONTAINS"


class PredicateSchema(CamelCaseSchema):
    model = fields.String()
    field = fields.String()
    op = fields.Function(
        serialize=lambda o: Operator[o], deserialize=lambda o: Operator(o)
    )
    argument = fields.Raw()
    negate = fields.Boolean(required=False, default=False)

    @post_load
    def make(self, data, **kwargs) -> "Predicate":
        return Predicate(**data)


# TODO: validate that STARTS_WITH, IN can only
# be used on string properties
@dataclass(frozen=True)
class Predicate(RelationshipTarget, Serializable):
    """
    Query predicates serve to generate a predicate clause appearing in the
    `WHERE` portion of the generated query.

    When including a predicate in the query, if `Predicate.model` does not
    refer to the source model provided to the query is executed,
    a relationship "join" will be generated to the model specified by
    `Predicate.model`.
    """

    __schema__: ClassVar[CamelCaseSchema] = PredicateSchema()

    field: str
    op: Operator
    argument: GraphValue
    negate: bool = dataclasses.field(default=False)

    def cql(self, i: int, model=None) -> str:
        if model is None:
            model_name = (
                self.model.name if isinstance(self.model, Model) else str(self.model)
            )
        else:
            model_name = model.name if isinstance(model, Model) else str(model)
        return f"{'NOT' if self.negate else ''} {model_name}[${self.predicate_field(i)}] {self.op.value} ${self.predicate_value(i)}"

    def parameters(self, i: int) -> Dict[str, GraphValue]:
        return {
            self.predicate_field(i): self.field,
            self.predicate_value(i): self.argument,
        }

    def predicate_field(self, i: int) -> str:
        return f"predicate_{i}_field"

    def predicate_value(self, i: int) -> str:
        return f"predicate_{i}_value"

    def __str__(self):
        return self.cql


# --- Order by ----------------------------------------------------------------


@dataclass(frozen=True)
class OrderBy:
    field: str
    ascending: bool = dataclasses.field(default=True)

    @classmethod
    def created_at(cls, ascending: bool = True) -> "OrderBy":
        return CreatedAt(ascending=ascending)

    @property
    def is_created_at(self) -> bool:
        return False

    @classmethod
    def updated_at(cls, ascending: bool = True) -> "OrderBy":
        return UpdatedAt(ascending=ascending)

    @property
    def is_updated_at(self) -> bool:
        return False


@dataclass(frozen=True)
class CreatedAt(OrderBy, Serializable):
    field: str = "created_at"
    ascending: bool = dataclasses.field(default=True)

    def is_created_at(self) -> bool:
        return True


@dataclass(frozen=True)
class UpdatedAt(OrderBy, Serializable):
    field: str = "updated_at"
    ascending: bool = dataclasses.field(default=True)

    def is_updated_at(self) -> bool:
        return True


# --- Aggregate ---------------------------------------------------------------


class Aggregate:
    @classmethod
    def group_count(cls, *args, **kwargs) -> "GroupCount":
        return GroupCount(*args, **kwargs)

    @property
    def is_group_count(self) -> bool:
        return False


@dataclass(frozen=True)
class GroupCount(Aggregate, Serializable):
    field: str
    model: Optional[Union[str, ModelId, Model]] = dataclasses.field(default=None)

    @property
    def is_group_count(self) -> bool:
        return True


# --- Query -------------------------------------------------------------------


class UserQuerySchema(CamelCaseSchema):
    connections = fields.Nested(RelationshipTarget.schema(), many=True)
    filters = fields.Nested(Predicate.schema(), many=True)

    @post_load
    def make(self, data, **kwargs):
        return UserQuery(**data)


# @dataclass(frozen=True)
@dataclass
class UserQuery(Serializable):

    __schema__: ClassVar[UserQuerySchema] = UserQuerySchema()

    connections: List[RelationshipTarget] = dataclasses.field(default_factory=list)
    filters: List[Predicate] = dataclasses.field(default_factory=list)
    ordering: Optional[OrderBy] = dataclasses.field(default=None)
    selection: Set[str] = dataclasses.field(default_factory=set)
    aggregation: Optional[Aggregate] = dataclasses.field(default=None)
    aliases: Dict[str, Union[Model, ModelId, str]] = dataclasses.field(
        default_factory=dict
    )

    @property
    def is_aggregating(self) -> bool:
        return self.aggregation is not None

    def connect_to(self, *args, **kwargs) -> "UserQuery":
        self.connections.append(RelationshipTarget(*args, **kwargs))
        return self

    def with_filter(self, *args, **kwargs) -> "UserQuery":
        self.filters.append(Predicate(*args, **kwargs))
        return self

    def order_by(self, order_by: OrderBy) -> "UserQuery":
        self.ordering = order_by
        return self

    def select_model(self, model_name: str, alias: Optional[str] = None) -> "UserQuery":
        self.selection.add(model_name)
        return self

    def alias(self, name: str, model: Union[Model, ModelId, str]) -> "UserQuery":
        self.aliases[name] = model
        return self

    def aggregate(self, operation: Aggregate) -> "UserQuery":
        self.aggregation = operation
        return self
