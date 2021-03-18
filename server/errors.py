from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

from dataclasses_json import LetterCase, dataclass_json  # type: ignore

from core.errors import ExternalRequestError, ModelServiceError, PackagesNotFoundError

from .models import (
    GraphValue,
    Model,
    ModelId,
    ModelRelationshipId,
    PackageProxyId,
    RecordId,
    RecordRelationshipId,
)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class MissingTraceId(ModelServiceError):
    def __post_init__(self):
        self.message = f"missing trace ID"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NameEmptyError(ModelServiceError):
    def __post_init__(self):
        self.message = f"name empty"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NameTooLongError(ModelServiceError):
    name: str

    def __post_init__(self):
        self.message = f'name too long: ["{self.name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NameValidationError(ModelServiceError):
    name: str

    def __post_init__(self):
        self.message = f'name validation error: ["{self.name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InvalidOrganizationError(ModelServiceError):
    id: str

    def __post_init__(self):
        self.message = f"invalid organization: [{self.id}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InvalidDatasetError(ModelServiceError):
    id: str

    def __post_init__(self):
        self.message = f"invalid dataset: [{self.id}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class LockedDatasetError(ModelServiceError):
    id: str

    def __post_init__(self):
        self.message = f"either the dataset is locked or the state cannot be determined from the provided claim: [{self.id}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class OperationError(ModelServiceError):
    message: str
    cause: InitVar[Optional[Exception]] = None

    def __post_init__(self, cause):
        if cause:
            self.message = f"{self.message} (caused by {str(cause)})"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ExceededTimeLimitError(ModelServiceError):
    def __post_init__(self):
        self.message = f"Operation exceeded time limit. Try adjusting the batch size and/or duration time"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class BadProperty:
    name: str
    description: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DuplicateModelNameError(ModelServiceError):
    model_name: str

    def __post_init__(self):
        self.message = f"duplicate model name: [{self.model_name}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ReservedModelNameError(ModelServiceError):
    name: str

    def __post_init__(self):
        self.message = f'reserved name ["{self.name}"] not allowed for models'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelNameCountError(ModelServiceError):
    model_name: str
    count: int

    def __post_init__(self):
        self.message = f"model name count: [{self.model_name}] = {self.count}"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelNameEmptyError(NameEmptyError):
    def __post_init__(self):
        self.message = f"model name empty"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelNameTooLongError(NameTooLongError):
    def __post_init__(self):
        self.message = f'model name too long: ["{self.name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelNameValidationError(NameValidationError):
    def __post_init__(self):
        self.message = f'model name validation error: ["{self.name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DuplicateModelPropertyNameError(ModelServiceError):
    model_name: str
    property_name: str

    def __post_init__(self):
        self.message = (
            f"duplicate model property name: [{self.model_name}.{self.property_name}]"
        )


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RecordNotFoundError(ModelServiceError):
    record_id: RecordId

    def __post_init__(self):
        self.message = f"record does not exist: {self.record_id}"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RecordValidationError(ModelServiceError):
    with_errors: InitVar[List[Tuple[str, Exception]]] = None
    errors: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self, with_errors):
        self.message = f"invalid values"
        self.errors = [
            BadProperty(name=prop, description=str(e)) for (prop, e) in with_errors
        ]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PropertyNameEmptyError(NameEmptyError):
    def __post_init__(self):
        self.message = f"property name empty"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PropertyNameTooLongError(NameTooLongError):
    def __post_init__(self):
        self.message = f'property name too long: ["{self.name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PropertyNameValidationError(NameValidationError):
    def __post_init__(self):
        self.message = f'property validation error: ["{self.name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RelationshipValidationError(ModelServiceError):
    relationship_name: str

    def __post_init__(self):
        self.message = f'relationship validation error: ["{self.relationship_name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InvalidPropertyDefaultValue(ModelServiceError):
    property_name: str
    value: GraphValue

    def __post_init__(self):
        self.message = f'invalid default value [{self.value}] for property ["{self.property_name}"]'


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelNotFoundError(ModelServiceError):
    model: Union[str, ModelId, UUID, Model]

    def __post_init__(self):
        self.message = f"violation: model does not exist: {str(self.model)}"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelPropertyNotFoundError(ModelServiceError):
    model: Union[str, ModelId, UUID, Model]
    property_name: str

    def __post_init__(self):
        self.message = f"violation: model property not found: [{str(self.model)}.{self.property_name}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelPropertyInUseError(ModelServiceError):
    model: Union[str, ModelId, UUID, Model]
    property_name: str
    usage_count: int = field()

    def __post_init__(self):
        self.message = f"violation: model property in use: [{str(self.model)}.{self.property_name}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelInUseError(ModelServiceError):
    model: Union[str, ModelId, UUID, Model]

    def __post_init__(self):
        self.message = f"violation: model in use: {str(self.model)}"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DuplicateModelRelationshipError(ModelServiceError):
    relationship_name: str
    from_model: Union[str, ModelId, UUID, Model]
    to_model: Union[str, ModelId, UUID, Model]

    def __post_init__(self):
        self.message = f"duplicate model relationship name: ({self.from_model})-[{self.relationship_name}]->({self.to_model})"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RelationshipConstraintError(ModelServiceError):
    model_from: Union[str, ModelId, UUID]
    model_to: Union[str, ModelId, UUID]
    type: Optional[str]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelRelationshipNotFoundError(RelationshipConstraintError):

    id: InitVar[Optional[str]] = None

    def __post_init__(self, id):
        model_from = self.model_from or "<null>"
        model_to = self.model_to or "<null>"
        type_ = self.type or self.id
        self.message = f"violation: model relationship ({model_from})-[{type_}]->({model_to}) does not exist"

    @classmethod
    def from_id(cls, id: str) -> "ModelRelationshipNotFoundError":
        return cls(id=id, model_from="*", model_to="*", type="*")


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class LegacyModelRelationshipNotFoundError(ModelServiceError):
    name: str
    id: Optional[str] = field(default=None)

    def __post_init__(self):
        id_ = self.id or "<null>"
        name_ = self.name or "<null>"
        self.message = f"violation: legacy model relationship [{id_}] : (?)-[{name_}]->(?) does not exist"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RecordRelationshipNotFoundError(ModelServiceError):
    record_id: Optional[RecordId]
    relationship_id: RecordRelationshipId

    def __post_init__(self):
        self.message = f"violation: record relationship does not exist ({self.record_id or '*'})-[{self.relationship_id}]->"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class RecordRelationshipExistsError(RelationshipConstraintError):
    record_from: RecordId
    record_to: RecordId

    def __post_init__(self):
        model_from = self.model_from or "<null>"
        model_to = self.model_to or "<null>"
        record_from = self.record_from or "<null>"
        record_to = self.record_to or "<null>"
        type_ = self.type or "<null>"
        self.message = (
            f"violation: cannot delete model relationship ({model_from})-[{type_}]->({model_to}): "
            + f"record relationship exists ({record_from})-[{type_}]->({record_to})"
        )


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class MultipleRelationshipsViolationError(ModelServiceError):
    record_id: RecordId
    model_relationship_id: ModelRelationshipId
    model_relationship_type: str

    def __post_init__(self):
        self.message = (
            f"violation: a maximum of one relationship can exist from {self.record_id}"
            + f' for this type "{self.model_relationship_type}" {self.model_relationship_id}'
        )


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ImmutablePropertyError(ModelServiceError):
    source_model: str
    property_name: str

    def __post_init__(self):
        self.message = f"violation: cannot change immutable property [{self.source_model}.{self.property_name}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InvalidQueryModelError(ModelServiceError):
    model_name: str

    def __post_init__(self):
        self.message = f"violation: [{self.model_name}] not in search path"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InfeasibleQueryError(ModelServiceError):
    source_model: str
    target_models: List[str]

    def __post_init__(self):
        self.message = f"not possible to generate a query from source model [{self.source_model}] to target models [{', '.join(self.target_models)}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class MultiplePropertyTitleError(ModelServiceError):
    model_name: str
    property_names: List[str]

    def __post_init__(self):
        self.message = f"violation: only 1 model title property allowed: [{self.model_name}.{self.property_names}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class NoPropertyTitleError(ModelServiceError):
    model_name: str

    def __post_init__(self):
        self.message = (
            f"violation: at least 1 model title property required: [{self.model_name}]"
        )


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class MultiplePropertyNameError(ModelServiceError):
    model_name: str
    property_value: str

    def __post_init__(self):
        self.message = f"violation: multiple properties with name [{self.model_name}.{self.property_value}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class MultiplePropertyDisplayNameError(ModelServiceError):
    model_name: str
    property_value: str

    def __post_init__(self):
        self.message = f"violation: multiple properties with display name [{self.model_name}.{self.property_value}]"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PackageProxyNotFoundError(ModelServiceError):
    package_proxy_id: PackageProxyId

    def __post_init__(self):
        self.message = f"package proxy does not exist: {self.package_proxy_id}"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class InvalidPackageProxyLinkTargetError(ModelServiceError):
    link_target: str

    def __post_init__(self):
        self.message = f"package proxy does not exist: {self.link_target}"


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CannotSortRecordsError(ModelServiceError):
    count: int
    max_records: int

    def __post_init__(self):
        self.message = (
            f"model has too many records to sort: {self.count} > {self.max_records}"
        )
