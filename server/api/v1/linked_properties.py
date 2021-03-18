from typing import Dict, List

from auth_middleware.models import DatasetPermission  # type: ignore
from werkzeug.exceptions import BadRequest, NotFound

from server.auth import permission_required
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.models import (
    CreateRecordRelationship,
    JsonDict,
    ModelId,
    ModelRelationshipId,
    RecordId,
    RecordRelationshipId,
)

from .common import (
    to_linked_property,
    to_schema_linked_property,
    to_schema_linked_property_target,
)


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_all_schema_linked_properties(db: PartitionedDatabase):
    with db.transaction() as tx:
        return [
            to_schema_linked_property_target(relationship)
            for relationship in db.get_outgoing_model_relationships_tx(
                tx=tx, one_to_many=False
            )
        ]


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_schema_linked_properties(db: PartitionedDatabase, id_: ModelId):
    with db.transaction() as tx:
        return [
            to_schema_linked_property_target(relationship)
            for relationship in db.get_outgoing_model_relationships_tx(
                tx=tx, from_model=id_, one_to_many=False
            )
        ]


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def create_schema_linked_property(
    db: PartitionedDatabase, id_: ModelId, body: JsonDict
):
    return (
        to_schema_linked_property(
            db.create_model_relationship(
                from_model=id_,
                name=body["name"],
                to_model=body["to"],
                one_to_many=False,
                display_name=body["display_name"],
                index=body.get("position", None),
            )
        ),
        201,
    )


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def create_schema_linked_properties(
    db: PartitionedDatabase, id_: ModelId, body: JsonDict
):
    with db.transaction() as tx:
        return (
            [
                to_schema_linked_property(
                    db.create_model_relationship_tx(
                        tx,
                        from_model=id_,
                        name=p["name"],
                        to_model=p["to"],
                        one_to_many=False,
                        display_name=p["displayName"],
                        index=p.get("position", None),
                    )
                )
                for p in body
            ],
            201,
        )


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def update_schema_linked_property(
    db: PartitionedDatabase,
    id_: ModelId,
    link_id: ModelRelationshipId,
    body: List[JsonDict],
):

    # Ignore the ID of the model/concept, as it's not needed to actually
    # fetch the linked property:
    return to_schema_linked_property(
        db.update_model_relationship(
            relationship=link_id,
            display_name=body["display_name"],
            index=body.get("position", None),
        )
    )


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def delete_schema_linked_property(
    db: PartitionedDatabase, id_: str, link_id: ModelRelationshipId
):
    # Ignore the ID of the model/concept, as it's not needed to actually
    # fetch the linked property:
    with db.transaction() as tx:
        return db.delete_model_relationship_tx(tx=tx, relationship=link_id)


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def create_linked_property(
    db: PartitionedDatabase, concept_id: str, id_: RecordId, body: JsonDict
):
    (rr, _) = db.create_record_relationship(
        from_record=id_,
        relationship=body["schema_linked_property_id"],
        to_record=body["to"],
    )
    return to_linked_property(rr), 201


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def create_linked_properties(
    db: PartitionedDatabase, concept_id: ModelId, id_: RecordId, body: Dict
):
    with db.transaction() as tx:

        # map of model relationship id -> relationship
        # allows us to make sure model relationships exist
        model_relationship_map = {
            model_relationship.id: model_relationship
            for model_relationship in db.get_outgoing_model_relationships_tx(
                tx=tx, from_model=concept_id, one_to_many=False
            )
        }

        # map of model relationship id -> existing record relationship
        # so we can delete existing relationships such that this behaves as an UPSERT
        existing_record_relationship_map = {
            record_relationship.model_relationship_id: record_relationship
            for record_relationship in db.get_outgoing_record_relationships_tx(
                tx=tx, from_record=id_, one_to_many=False
            )
        }

        payload: List[Dict[str, str]] = list(body["data"])

        # assure each request is or a unique schemaLinkedProperty
        if len(payload) != len(set(p["schemaLinkedPropertyId"] for p in payload)):
            raise BadRequest("duplicate model linked properties")

        to_create: List[CreateRecordRelationship] = []
        for item in payload:
            model_relationship_id = item["schemaLinkedPropertyId"]
            if model_relationship_id not in model_relationship_map.keys():
                raise BadRequest(
                    f"no model linked property exists for {item['schemaLinkedPropertyId']}"
                )

            model_relationship = model_relationship_map[model_relationship_id]

            if model_relationship_id in existing_record_relationship_map.keys():
                db.delete_outgoing_record_relationship_tx(
                    tx=tx,
                    record=id_,
                    relationship=existing_record_relationship_map[
                        model_relationship_id
                    ],
                )

            to_create.append(
                CreateRecordRelationship(
                    from_=id_, to=item["to"], model_relationship=model_relationship
                )
            )

        result = db.create_record_relationship_batch_tx(tx=tx, to_create=to_create)

        data = [to_linked_property(r) for r, _ in result]

        return {"data": data}


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_linked_properties(db: PartitionedDatabase, concept_id: ModelId, id_: RecordId):
    # Ignore the ID of the model/concept, as it's not needed to actually
    # fetch the linked property:
    with db.transaction() as tx:
        return [
            to_linked_property(rr)
            for rr in db.get_outgoing_record_relationships_tx(
                tx=tx, from_record=id_, one_to_many=False
            )
        ]


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def delete_linked_property(
    db: PartitionedDatabase,
    concept_id: ModelId,
    id_: RecordId,
    link_id: RecordRelationshipId,
):
    with db.transaction() as tx:
        return db.delete_outgoing_record_relationship_tx(tx, id_, link_id)
