from typing import Optional

ORGANIZATION_LABEL = "Organization"
DATASET_LABEL = "Dataset"
USER_LABEL = "User"
MODEL_LABEL = "Model"
MODEL_PROPERTY_LABEL = "ModelProperty"
RECORD_LABEL = "Record"
PACKAGE_LABEL = "Package"
PROXY_RELATIONSHIP_TYPE = "belongs_to"
MODEL_RELATIONSHIP_STUB = "ModelRelationshipStub"


"""
Node label types
"""


def organization(alias: Optional[str] = None) -> str:
    return f"{alias}:{ORGANIZATION_LABEL}" if alias else f":{ORGANIZATION_LABEL}"


def dataset(alias: Optional[str] = None) -> str:
    return f"{alias}:{DATASET_LABEL}" if alias else f":{DATASET_LABEL}"


def user(alias: Optional[str] = None) -> str:
    return f"{alias}:{USER_LABEL}" if alias else f":{USER_LABEL}"


def model(alias: Optional[str] = None) -> str:
    return f"{alias}:{MODEL_LABEL}" if alias else f":{MODEL_LABEL}"


def model_property(alias: Optional[str] = None) -> str:
    return f"{alias}:{MODEL_PROPERTY_LABEL}" if alias else f":{MODEL_PROPERTY_LABEL}"


def model_relationship_stub(alias: Optional[str] = None) -> str:
    return (
        f"{alias}:{MODEL_RELATIONSHIP_STUB}" if alias else f":{MODEL_RELATIONSHIP_STUB}"
    )


def record(alias: Optional[str] = None) -> str:
    return f"{alias}:{RECORD_LABEL}" if alias else f":{RECORD_LABEL}"


def package(alias: Optional[str] = None) -> str:
    return f"{alias}:{PACKAGE_LABEL}" if alias else f":{PACKAGE_LABEL}"


"""
Relationship types

The pennsieve-defined relationship types are prefixed with `@` to distinguish
them from user-defined relationship types which only contain letters and
underscores.
"""

INSTANCE_OF = "@INSTANCE_OF"

HAS_PROPERTY = "@HAS_PROPERTY"

IN_ORGANIZATION = "@IN_ORGANIZATION"

IN_DATASET = "@IN_DATASET"

CREATED_BY = "@CREATED_BY"

UPDATED_BY = "@UPDATED_BY"

IN_PACKAGE = "@IN_PACKAGE"

RELATED_TO = "@RELATED_TO"

RESERVED_SCHEMA_RELATIONSHIPS = [
    INSTANCE_OF,
    HAS_PROPERTY,
    IN_ORGANIZATION,
    IN_DATASET,
    CREATED_BY,
    UPDATED_BY,
    IN_PACKAGE,
    RELATED_TO,
]


def instance_of(alias: Optional[str] = None) -> str:
    return f"{alias}:`{INSTANCE_OF}`" if alias else f":`{INSTANCE_OF}`"


def has_property(alias: Optional[str] = None) -> str:
    return f"{alias}:`{HAS_PROPERTY}`" if alias else f":`{HAS_PROPERTY}`"


def in_organization(alias: Optional[str] = None) -> str:
    return f"{alias}:`{IN_ORGANIZATION}`" if alias else f":`{IN_ORGANIZATION}`"


def in_dataset(alias: Optional[str] = None) -> str:
    return f"{alias}:`{IN_DATASET}`" if alias else f":`{IN_DATASET}`"


def created_by(alias: Optional[str] = None) -> str:
    return f"{alias}:`{CREATED_BY}`" if alias else f":`{CREATED_BY}`"


def updated_by(alias: Optional[str] = None) -> str:
    return f"{alias}:`{UPDATED_BY}`" if alias else f":`{UPDATED_BY}`"


def in_package(alias: Optional[str] = None) -> str:
    return f"{alias}:`{IN_PACKAGE}`" if alias else f":`{IN_PACKAGE}`"


def related_to(alias: Optional[str] = None) -> str:
    return f"{alias}:`{RELATED_TO}`" if alias else f":`{RELATED_TO}`"
