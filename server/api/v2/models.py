"""
API endpoints for working with models.
"""
from typing import List, Tuple
from uuid import UUID

from auth_middleware.models import DatasetPermission  # type: ignore
from flask import current_app
from werkzeug.exceptions import BadRequest, Conflict, NotFound

from core.clients import (
    AuditLogger,
    PennsieveJobsClient,
    CreateModel,
    CreateModelProperty,
    DeleteModel,
    DeleteModelProperty,
    TraceId,
    UpdateModel,
    UpdateModelProperty,
)
from server.auth import permission_required
from server.config import Config
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.models import JsonDict, ModelId, ModelProperty, ModelRelationshipId


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def create_model(db: PartitionedDatabase, body: JsonDict) -> Tuple[JsonDict, int]:
    model = db.create_model(**body)
    x_bf_trace_id = AuditLogger.trace_id_header()
    # Emit "CreateModel" event:
    PennsieveJobsClient.get().send_changelog_event(
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
        user_id=db.user_id,
        event=CreateModel(id=model.id, name=model.name),
        trace_id=TraceId(x_bf_trace_id),
    )
    return model.to_dict(), 201


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_all_models(
    db: PartitionedDatabase,
) -> List[JsonDict]:
    return [m.to_dict() for m in db.get_models()]


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_model(db: PartitionedDatabase, model_id_or_name: str) -> JsonDict:
    return db.get_model(model_id_or_name).to_dict()


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def update_model(
    db: PartitionedDatabase, model_id_or_name: str, body: JsonDict
) -> JsonDict:
    model = db.update_model(model_id_or_name, **body)
    x_bf_trace_id = AuditLogger.trace_id_header()
    # Emit "UpdateModel" event:
    PennsieveJobsClient.get().send_changelog_event(
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
        user_id=db.user_id,
        event=UpdateModel(id=model.id, name=model.name),
        trace_id=TraceId(x_bf_trace_id),
    )
    return model.to_dict()


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def delete_model(db: PartitionedDatabase, model_id_or_name: str) -> None:
    model = db.delete_model(model_id_or_name)
    x_bf_trace_id = AuditLogger.trace_id_header()
    # Emit "DeleteModel" event:
    PennsieveJobsClient.get().send_changelog_event(
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
        user_id=db.user_id,
        event=DeleteModel(id=model.id, name=model.name),
        trace_id=TraceId(x_bf_trace_id),
    )
    return None


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_all_properties(
    db: PartitionedDatabase, model_id_or_name: str
) -> List[JsonDict]:
    return [p.to_dict() for p in db.get_properties(model_id_or_name)]


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def update_properties(
    db: PartitionedDatabase, model_id_or_name: str, body: List[JsonDict]
):
    x_bf_trace_id = AuditLogger.trace_id_header()
    payload: List[ModelProperty] = ModelProperty.schema().load(body, many=True)
    with db.transaction() as tx:
        model = db.get_model_tx(tx, model_id_or_name)

        properties = db.update_properties_tx(tx, model, *payload)

        PennsieveJobsClient.get().send_changelog_events(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            events=[
                CreateModelProperty(
                    property_name=p.name, model_id=UUID(model.id), model_name=model.name
                )
                if created
                else UpdateModelProperty(
                    property_name=p.name, model_id=UUID(model.id), model_name=model.name
                )
                for p, created in properties
            ],
            trace_id=TraceId(x_bf_trace_id),
        )
        return [p.to_dict() for p, _ in properties]


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def delete_property(
    db: PartitionedDatabase,
    model_id: str,
    property_name: str,
    modify_records: bool = False,
) -> None:

    x_bf_trace_id = AuditLogger.trace_id_header()
    max_record_count = current_app.config[
        "config"
    ].max_record_count_for_property_deletion

    with db.transaction() as tx:
        model = db.get_model_tx(tx, model_id)

        if modify_records:
            record_count = db.model_property_record_count_tx(
                tx, model_id, property_name
            )
            if record_count > 0:
                if record_count > max_record_count:
                    raise BadRequest(
                        f"Cannot delete properties that are used on > {max_record_count} records. This property is used on {record_count}"
                    )
                model_properties = [
                    p
                    for p in db.get_properties_tx(tx, model_id)
                    if p.name == property_name
                ]
                if not model_properties:
                    raise NotFound(f"no such property {property_name} exists")
                updated_records = db.delete_property_from_all_records_tx(
                    tx, model_id, model_properties[0]
                )
                if updated_records != record_count:
                    raise ServerError("the property was not removed from all records")

        deleted = db.delete_property_tx(tx, model_id, property_name)
        if deleted is None:
            raise NotFound(f"Could not delete property [{model_id}.{property_name}]")

        PennsieveJobsClient.get().send_changelog_event(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            event=DeleteModelProperty(
                property_name=deleted.name,
                model_id=UUID(model.id),
                model_name=model.name,
            ),
            trace_id=TraceId(x_bf_trace_id),
        )


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def create_relationship(
    db: PartitionedDatabase, model_id_or_name: str, body: JsonDict
) -> Tuple[JsonDict, int]:
    # connexion converts "type" to "type_":
    return (
        db.create_model_relationship(
            from_model=model_id_or_name,
            name=body["type_"],
            to_model=body["to"],
            one_to_many=body["one_to_many"],
            display_name=body.get("display_name", body["type_"]),
        ).to_dict(),
        201,
    )


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_relationship(
    db: PartitionedDatabase, model_id_or_name: str, relationship_id: ModelRelationshipId
) -> JsonDict:
    relationship = db.get_model_relationship(relationship_id)
    if relationship is None:
        raise NotFound(f"Could not get model relationship [{relationship_id}]")
    return relationship.to_dict()


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_relationships(db: PartitionedDatabase, model_id_or_name: str) -> List[JsonDict]:
    with db.transaction() as tx:
        return [
            r.to_dict()
            for r in db.get_outgoing_model_relationships_tx(
                tx, from_model=model_id_or_name
            )
        ]


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def update_relationship(
    db: PartitionedDatabase,
    model_id_or_name: str,
    relationship_id: ModelRelationshipId,
    body: JsonDict,
) -> JsonDict:
    return db.update_model_relationship(
        relationship=relationship_id, display_name=body["display_name"]
    ).to_dict()


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def delete_relationship(
    db: PartitionedDatabase, model_id_or_name: str, relationship_id: ModelRelationshipId
) -> None:
    with db.transaction() as tx:
        deleted = db.delete_model_relationship_tx(tx=tx, relationship=relationship_id)
        if deleted is None:
            raise NotFound(f"Could not find property [{relationship_id}]")
