import json
import logging
from collections import Sequence, Set
from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import UUID

import neotime
from humps import decamelize

from server.models import (
    JsonDict,
    Model,
    ModelProperty,
    ModelRelationship,
    ModelRelationshipId,
    ModelTopology,
    PackageProxy,
    Record,
    RecordRelationship,
    RelationshipName,
)
from server.models import datatypes as dt
from server.models import is_model_relationship_id
from server.models.legacy import (
    ModelRelationshipStub,
    ProxyLinkConceptInstance,
    ProxyLinkTarget,
)

log = logging.getLogger(__file__)


"""
For identifiers to graph node types that no longer exist (like schema linked
property edges), we reserve a "null" UUID:
"""
NULL_UUID = UUID(int=0)


def filter_model_dict(body: JsonDict) -> JsonDict:
    return {k: v for k, v in body.items() if k in Model.PUBLIC}


def to_concept_dict(m: Model, property_count: int) -> JsonDict:
    json = m.to_dict()
    json["propertyCount"] = property_count

    # Concept locking is unsupported in Neo4j
    json["locked"] = False

    return json


def to_model_property(d: JsonDict) -> ModelProperty:
    d["modelTitle"] = d.pop("conceptTitle")
    d["dataType"] = dt.deserialize(d.pop("dataType"))
    return ModelProperty(
        **{
            decamelize(k): v
            for k, v in d.items()
            if decamelize(k) in ModelProperty.PUBLIC
        }
    )


def to_property_dict(p: ModelProperty) -> JsonDict:
    d = p.to_dict()
    d["conceptTitle"] = d.pop("modelTitle")

    if "defaultValue" not in d:
        d["defaultValue"] = None

    d["dataType"] = to_legacy_data_type(p.data_type)

    return d


def to_legacy_data_type(data_type: Union[JsonDict, dt.DataType]) -> JsonDict:
    """
    Convert to simple datatypes ("String", "Long", etc) instead of JSON objects,
    if possible.

    The frontend expects the "type" field for enums and arrays to be lowercase.
    """
    if not isinstance(data_type, dt.DataType):
        return json.loads(data_type)

    if data_type.is_simple:
        return data_type.into_simple()

    data = data_type.to_dict()

    if data["type"] == "Enum":
        data["type"] = "enum"

    if data["type"] == "Array":
        data["type"] = "array"

    return data


def to_concept_instance(
    r: Record, m: Model, properties: List[ModelProperty]
) -> JsonDict:
    """
    Convert a Record to a legacy concept instance, whose values are represented
    as
        {
            "name": "age",
            "value": 10
        },
        ...

    The legacy model also returns a number of other values for the instance and
    each property which must be pulled from the Model and ModelProperty
    instances of the Record
    """

    def format_value(v):
        if v is None:
            return None
        elif v is True:
            return "true"
        elif v is False:
            return "false"
        elif isinstance(v, str):
            return v
        elif isinstance(v, (Sequence, Set)):
            return [format_value(w) for w in v]
        elif isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, neotime.DateTime):
            return v.to_native().isoformat()
        return v

    property_dict = {p.name: p for p in properties}

    values = sorted(
        [
            {
                "name": k,
                "value": format_value(v),
                "conceptTitle": property_dict[k].model_title,
                "dataType": to_legacy_data_type(property_dict[k].data_type),
                "displayName": property_dict[k].display_name,
                "locked": property_dict[k].locked,
                "required": property_dict[k].required,
                "default": property_dict[k].default,
            }
            # ignore property names (like embedded record entries that
            # aren't defined properties of the record's associated model:
            for k, v in r.values.items()
            if k in property_dict
        ],
        key=lambda p: property_dict[p["name"]].index,
    )

    return {
        "id": r.id,
        "type": m.name,
        "values": values,
        "createdAt": r.created_at.isoformat(),
        "updatedAt": r.updated_at.isoformat(),
        "createdBy": r.created_by,
        "updatedBy": r.updated_by,
    }


def to_record(
    properties: List[ModelProperty], property_values: List[JsonDict]
) -> JsonDict:
    """
    The legacy API expects all property values to be strings, but ``db``
    operates on typed values. This uses the data type of each model property to
    cast the string values to the correct Python type before sending them to the
    database.
    """
    property_map: Dict[str, Dict[str, ModelProperty]] = {p.name: p for p in properties}
    record = {}

    for pair in property_values:
        name = pair["name"]
        value = pair.get("value", None)

        # If a property is not required and lacks a valid value for a type,
        # omit it from `record`, so it will never be sent to the database.
        # This is fine, if `fill_missing=True` is given in `create_record()`
        # or `update_record()` the appropriate properties will be present
        # on the returned record anyway.
        if name not in property_map or (
            not property_map[name].required and value is None
        ):
            continue

        # If a property is required and None is provided, don't attempt to
        # coerce it to a type: just keep it as None:
        if property_map[name].required and value is None:
            record[name] = None
            continue

        record[name] = property_map[name].data_type.into(value)

    return record


def to_relationship_id_or_name(
    id_or_name: str,
) -> Union[ModelRelationshipId, RelationshipName]:
    if is_model_relationship_id(id_or_name):
        return ModelRelationshipId(id_or_name)
    else:
        return RelationshipName(id_or_name)


def to_legacy_relationship(
    relationship: Union[ModelRelationship, ModelRelationshipStub]
) -> JsonDict:
    rel = {
        "name": relationship.name,
        "displayName": relationship.display_name,
        "description": relationship.description,
        "schema": [],
        "createdAt": relationship.created_at.isoformat(),
        "updatedAt": relationship.updated_at.isoformat(),
        "createdBy": relationship.created_by,
        "updatedBy": relationship.updated_by,
        "id": str(relationship.id),
    }
    if isinstance(relationship, ModelRelationshipStub):
        rel["from"] = None
        rel["to"] = None
    else:
        rel["from"] = str(relationship.from_)
        rel["to"] = str(relationship.to)
    return rel


def to_legacy_relationship_instance(relationship: RecordRelationship) -> JsonDict:
    return {
        "createdAt": relationship.created_at.isoformat(),
        "updatedAt": relationship.updated_at.isoformat(),
        "createdBy": relationship.created_by,
        "updatedBy": relationship.updated_by,
        "schemaRelationshipId": str(relationship.model_relationship_id),
        "from": str(relationship.from_),
        "to": str(relationship.to),
        "type": relationship.name,
        "values": [],
        "id": str(relationship.id),
        "name": relationship.name,
        "displayName": relationship.display_name,
    }


def to_schema_linked_property(relationship: ModelRelationship) -> JsonDict:
    return {
        "name": relationship.name,
        "displayName": relationship.display_name,
        "position": 1_000_000 if relationship.index is None else relationship.index,
        "from": str(relationship.from_),
        "to": str(relationship.to),
        "createdAt": relationship.created_at.isoformat(),
        "updatedAt": relationship.updated_at.isoformat(),
        "createdBy": relationship.created_by,
        "updatedBy": relationship.updated_by,
        "type": "schemaLinkedProperty",
        "id": str(relationship.id),
    }


def to_linked_property(relationship: RecordRelationship) -> JsonDict:
    return {
        "schemaLinkedPropertyId": relationship.model_relationship_id,
        "from": str(relationship.from_),
        "to": str(relationship.to),
        "createdAt": relationship.created_at.isoformat(),
        "updatedAt": relationship.updated_at.isoformat(),
        "createdBy": relationship.created_by,
        "updatedBy": relationship.updated_by,
        "id": str(relationship.id),
        "name": relationship.name,
        "displayName": relationship.display_name,
    }


def to_schema_linked_property_target(relationship: ModelRelationship) -> JsonDict:
    return {
        "link": to_schema_linked_property(relationship),
        "concept": str(relationship.from_),
    }


def to_proxy_link_target(body: JsonDict) -> Optional[ProxyLinkTarget]:
    try:
        return ProxyLinkConceptInstance.schema().load(body)
    except:  # noqa: E722
        log.warn(f"Not a proxy link target: {body}")
        return None


def to_proxy_instance(proxyType: str, package_proxy: PackageProxy) -> JsonDict:
    json = package_proxy.to_dict()
    json["externalId"] = package_proxy.package_node_id
    json["proxyType"] = proxyType

    del json["packageId"]

    return json


# TODO: store/use relationship id
def make_proxy_relationship_instance(
    record_id: UUID, package_proxy: PackageProxy, relationship_type: str
) -> JsonDict:
    return {
        "createdAt": package_proxy.created_at.isoformat(),
        "updatedAt": package_proxy.updated_at.isoformat(),
        "createdBy": package_proxy.created_by,
        "updatedBy": package_proxy.updated_by,
        "id": package_proxy.id,
        "schemaRelationshipId": NULL_UUID,  # TODO
        "from": record_id,
        "to": package_proxy.id,
        "type": relationship_type,
        "values": [],
        "name": relationship_type,
        "displayName": relationship_type,
    }


def to_legacy_topology(topology: ModelTopology) -> JsonDict:
    return {
        "id": str(topology.id),
        "name": topology.name,
        "displayName": topology.display_name,
        "description": topology.description,
        "count": topology.count,
        "createdAt": topology.created_at.isoformat(),
        "updatedAt": topology.updated_at.isoformat(),
    }


def to_legacy_package_dto(dto: JsonDict) -> JsonDict:
    content = dto["content"]

    content["id"] = content["nodeId"]
    content["datasetId"] = content["datasetNodeId"]

    return dto
