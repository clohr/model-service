from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Optional, Union

import marshmallow  # type: ignore
from marshmallow import post_load  # type: ignore
from marshmallow import fields

from core.json import CamelCaseSchema, Serializable
from core.types import DatasetId, GraphValue, ModelId, NativeScalar
from core.util import normalize_datetime

from .models import Dataset, Model, ModelProperty, Package, PackageProxy, Record
from .query import Operator


class ModelFilterSchema(CamelCaseSchema):
    name = fields.String(required=True)


@dataclass(frozen=True)
class ModelFilter(Serializable):
    __schema__: ClassVar[ModelFilterSchema] = ModelFilterSchema(
        unknown=marshmallow.EXCLUDE
    )

    name: str


class DatasetFilterSchema(CamelCaseSchema):
    id = fields.Integer(required=True)


# TODO allow node IDs?
@dataclass(frozen=True)
class DatasetFilter(Serializable):
    __schema__: ClassVar[DatasetFilterSchema] = DatasetFilterSchema(
        unknown=marshmallow.EXCLUDE
    )
    id: DatasetId


class PropertyFilterSchema(CamelCaseSchema):
    model = fields.String(required=True)
    property_ = fields.String(required=True, data_key="property")
    value = fields.Raw(required=True)
    operator = fields.Function(
        serialize=lambda o: str(o),
        deserialize=lambda s: Operator(s),
        default=Operator.EQUALS,
    )
    unit = fields.String(allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return PropertyFilter(**data)


@dataclass(frozen=True)
class PropertyFilter(Serializable):
    __schema__: ClassVar[PropertyFilterSchema] = PropertyFilterSchema(
        unknown=marshmallow.RAISE
    )
    model: str
    property_: str
    value: NativeScalar
    operator: Operator = field(default=Operator.EQUALS)
    unit: Optional[str] = field(default=None)

    def __post_init__(self):
        """
        If possible, coerce string filter values to UTC datetimes.
        """
        if isinstance(self.value, str):
            try:
                dt = normalize_datetime(self.value)
            except ValueError:
                pass
            else:
                object.__setattr__(self, "value", dt)


class SearchDownloadRequestSchema(CamelCaseSchema):
    model = fields.Function(
        serialize=lambda m: m.name, deserialize=lambda s: ModelFilter(s)
    )
    datasets = fields.Function(
        serialize=lambda ds: [d.id.id for d in ds],
        deserialize=lambda ss: [DatasetFilter(DatasetId(s)) for s in ss],
    )
    filters = fields.Nested(PropertyFilter.schema(), many=True)

    @post_load
    def make(self, data, **kwargs):
        return SearchDownloadRequest(**data)


@dataclass(frozen=True)
class SearchDownloadRequest(Serializable):
    __schema__: ClassVar[SearchDownloadRequestSchema] = SearchDownloadRequestSchema(
        unknown=marshmallow.RAISE
    )

    model: ModelFilter
    datasets: List[DatasetFilter]
    filters: List[PropertyFilterSchema]


@dataclass(frozen=True)
class SearchResult:
    model_id: ModelId
    properties: List[ModelProperty]
    record: Record
    dataset: Dataset


@dataclass(frozen=True)
class PackageSearchResult:
    package: Package
    dataset: Dataset


class SuggestedValuesSchema(CamelCaseSchema):
    property_ = fields.Nested(ModelProperty.schema(), data_key="property")
    operators = fields.Function(
        serialize=lambda o: str(o),
        deserialize=lambda s: Operator(s),
        default=Operator.EQUALS,
        many=True,
    )
    values = fields.Raw(many=True)


@dataclass(frozen=True)
class SuggestedValues(Serializable):
    __schema__: ClassVar[SuggestedValuesSchema] = SuggestedValuesSchema(
        unknown=marshmallow.RAISE
    )
    property_: ModelProperty
    operators: List[Operator]
    values: List[GraphValue]


class ModelSuggestionSchema(CamelCaseSchema):
    name = fields.String(required=True)
    display_name = fields.String(required=True)


@dataclass(frozen=True)
class ModelSuggestion(Serializable):
    __schema__: ClassVar[ModelSuggestionSchema] = ModelSuggestionSchema(
        unknown=marshmallow.EXCLUDE
    )

    name: str
    display_name: str
