import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import ClassVar, List, Optional, Type, Union
from uuid import UUID

import marshmallow
from marshmallow import post_load  # type: ignore
from marshmallow import fields

from core import types as t
from core.json import CamelCaseSchema, Serializable
from server.models import Model, ModelRelationship, PackageProxy
from server.models import datatypes as dt

# =============================================================================


@dataclass(frozen=True)
class OutputFile:
    file_name: str
    extension: str
    file_type: str
    key_prefix: str = ""

    @property
    def full_name(self) -> str:
        """
        Full name of the file within the dataset
        """
        return f"{self.file_name}.{self.extension}"

    @property
    def full_path(self) -> str:
        """
        Absolute path to file, including prefixes.
        """
        return os.path.join(self.key_prefix, self.full_name)

    @classmethod
    def csv(cls, file_name: str) -> "OutputFile":
        return cls(file_name=file_name, extension="csv", file_type="CSV")

    @classmethod
    def json(cls, file_name: str) -> "OutputFile":
        # Note: the file type is `Json` instead of `JSON` to match the
        # serialization used in `pennsieve-api`
        return cls(file_name=file_name, extension="json", file_type="Json")

    def __str__(self) -> str:
        return self.full_path

    def as_manifest(self, size) -> "FileManifest":
        return FileManifest(path=self.full_path, size=size, file_type=self.file_type)

    def with_prefix(self, key_prefix) -> "OutputFile":
        return replace(self, key_prefix=key_prefix)

    @classmethod
    def csv_for_model(cls, model: str) -> "OutputFile":
        """
        Helper to find the S3 destination for a model.
        """
        return cls.csv(f"records/{model}")

    @classmethod
    def csv_for_relationship(cls, relationship: str) -> "OutputFile":
        """
        Helper to find the S3 destination for a relationship.
        """
        return cls.csv(f"relationships/{relationship}")

    @classmethod
    def json_for_schema(cls) -> "OutputFile":
        return cls.json(f"schema")


# =============================================================================


class ExportProperty(Serializable):
    @classmethod
    def model_property(cls, **kwargs) -> "ExportProperty":
        return ExportModelProperty(**kwargs)

    @classmethod
    def linked_property(cls, **kwargs) -> "ExportProperty":
        return ExportLinkedProperty(**kwargs)


class ExportModelPropertySchema(CamelCaseSchema):
    id = fields.UUID()
    name = fields.String()
    display_name = fields.String()
    description = fields.String()
    data_type = fields.Function(
        required=True,
        serialize=lambda o: o.data_type.to_dict(),
        deserialize=dt.deserialize,
    )


@dataclass(frozen=True)
class ExportModelProperty(ExportProperty):

    __schema__: ClassVar[ExportModelPropertySchema] = ExportModelPropertySchema(
        unknown=marshmallow.EXCLUDE
    )

    name: str
    display_name: str
    description: str
    data_type: dt.DataType


class LinkedModelDataTypeSchema(CamelCaseSchema):
    to = fields.String()
    file = fields.String()
    type = fields.String()


@dataclass(frozen=True)
class LinkedModelDataType(Serializable):
    """
    Special-cased class used to export linked properties as a property of a
    model, instead of relationships.
    """

    __schema__: ClassVar[LinkedModelDataTypeSchema] = LinkedModelDataTypeSchema(
        unknown=marshmallow.EXCLUDE
    )

    to: str
    file: str
    type: str = field(default="Model")


class ExportLinkedPropertySchema(CamelCaseSchema):
    name = fields.String()
    display_name = fields.String()
    description = fields.String()
    data_type = fields.Nested(LinkedModelDataType.schema())


@dataclass(frozen=True)
class ExportLinkedProperty(ExportProperty):

    __schema__: ClassVar[ExportLinkedPropertySchema] = ExportLinkedPropertySchema(
        unknown=marshmallow.EXCLUDE
    )

    name: str
    display_name: str
    description: str
    data_type: LinkedModelDataType


# =============================================================================


class ExportModelSchema(CamelCaseSchema):
    name = fields.String()
    display_name = fields.String()
    description = fields.String()
    properties = fields.Nested(ExportModelProperty.schema(), many=True)
    file = fields.Function(serialize=lambda o: str(o.file))


@dataclass(frozen=True)
class ExportModel(Serializable):

    __schema__: ClassVar[ExportModelSchema] = ExportModelSchema(
        unknown=marshmallow.EXCLUDE
    )

    model: Optional[Model]

    name: str
    display_name: str
    description: str
    properties: List[ExportProperty]
    file: OutputFile = field(init=False)

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "file", OutputFile.csv_for_model(self.name))

    @post_load
    def make(self, data, **kwargs):
        return ExportModel(**data)

    @classmethod
    def package_proxy(cls) -> "ExportModel":
        return ExportModel(
            model=None,
            # TODO: use constant
            name="file",
            display_name="File",
            description="A file in the dataset",
            properties=[
                ExportProperty.model_property(
                    name="path",
                    display_name="Path",
                    description="The path to the file from the root of the dataset",
                    data_type=dt.String(),
                )
            ],
        )


# =============================================================================


class ExportModelRelationshipSchema(CamelCaseSchema):
    name = fields.String()
    from_ = fields.String(data_key="from")
    to = fields.String()
    file = fields.Function(serialize=lambda o: str(o.file))


@dataclass(frozen=True)
class ExportModelRelationship(Serializable):

    __schema__: ClassVar[ExportModelRelationshipSchema] = ExportModelRelationshipSchema(
        unknown=marshmallow.EXCLUDE
    )

    relationship: Optional[ModelRelationship]

    name: t.RelationshipName
    from_: str
    to: str
    file: OutputFile = field(init=False)

    def __post_init__(self):
        # HACK: This is required to mutate frozen dataclasses
        object.__setattr__(self, "file", OutputFile.csv_for_relationship(self.name))

    @post_load
    def make(self, data, **kwargs):
        return ExportModelRelationship(**data)

    def is_proxy_relationship(self):
        return self.relationship is None


# =============================================================================


class ExportGraphSchemaSchema(CamelCaseSchema):
    models = fields.Nested(ExportModel.schema(), many=True)
    relationships = fields.Nested(ExportModelRelationship.schema(), many=True)

    @post_load
    def make(self, data, **kwargs):
        return ExportGraphSchema(**data)


@dataclass(frozen=True)
class ExportGraphSchema(Serializable):

    __schema__: ClassVar[ExportGraphSchemaSchema] = ExportGraphSchemaSchema(
        unknown=marshmallow.EXCLUDE
    )

    models: List[ExportModel]
    relationships: List[ExportModelRelationship]


# =============================================================================


class FileManifestSchema(CamelCaseSchema):
    path = fields.String()
    size = fields.Integer()
    file_type = fields.String()
    id = fields.UUID(allow_none=True)
    source_package_id = fields.String(allow_none=True)

    @post_load
    def make(self, data, **kwargs):
        return FileManifest(**data)


@dataclass(frozen=True, order=True)
class FileManifest(Serializable):

    __schema__: ClassVar[FileManifestSchema] = FileManifestSchema(
        unknown=marshmallow.EXCLUDE
    )

    path: str
    size: int
    file_type: str
    id: Optional[UUID] = field(default=None)
    source_package_id: Optional[str] = field(default=None)


# =============================================================================


class ExportedGraphManifestsSchema(CamelCaseSchema):
    manifests = fields.Nested(FileManifestSchema, many=True)

    @post_load
    def make(self, data, **kwargs):
        return ExportedGraphManifests(**data)


@dataclass(frozen=True)
class ExportedGraphManifests(Serializable):

    __schema__: ClassVar[ExportedGraphManifestsSchema] = ExportedGraphManifestsSchema(
        unknown=marshmallow.EXCLUDE
    )

    manifests: List[FileManifest]


# =============================================================================


class ModelService(str, Enum):
    NEO4J = "neo4j"
    NEPTUNE = "neptune"


class ExportedDatasetLocationSchema(CamelCaseSchema):
    service = fields.Function(
        serialize=lambda o: o.service, deserialize=lambda s: ModelService(s)
    )

    @post_load
    def make(self, data, **kwargs):
        return ExportedDatasetLocation(**data)


@dataclass(frozen=True)
class ExportedDatasetLocation(Serializable):

    __schema__: ClassVar[ExportedDatasetLocationSchema] = ExportedDatasetLocationSchema(
        unknown=marshmallow.EXCLUDE
    )

    service: ModelService


# =============================================================================


@dataclass(frozen=True)
class PackageProxyRelationship:
    """
    Thin wrapper around a relationship between a record and a package proxy.
    """

    from_: UUID
    to: UUID
    relationship: t.RelationshipName
