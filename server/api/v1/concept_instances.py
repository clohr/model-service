from typing import List, Optional

import connexion
from auth_middleware.models import DatasetPermission  # type: ignore
from werkzeug.exceptions import BadRequest, NotFound

from core.clients import (
    AuditLogger,
    PennsieveJobsClient,
    CreateRecord,
    DeleteRecord,
    TraceId,
    UpdateRecord,
)
from server.auth import permission_required
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.models import JsonDict, OrderByField

from .common import to_concept_instance, to_record


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def create_concept_instance(
    db: PartitionedDatabase, concept_id_or_name: str, body: JsonDict
):
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, concept_id_or_name)
        record = to_record(properties, body["values"])
        records = db.create_records_tx(
            tx, concept_id_or_name, [record], fill_missing=True
        )

        if not records:
            raise BadRequest(
                f"Could not create concept instance [{concept_id_or_name}]"
            )
        record = records[0]

        # Log the created concept instance:
        x_bf_trace_id = AuditLogger.trace_id_header()

        # Emit "CreateRecord" event:
        PennsieveJobsClient.get().send_changelog_event(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            event=CreateRecord(id=record.id, name=record.name, model_id=model.id),
            trace_id=TraceId(x_bf_trace_id),
        )

        AuditLogger.get().message().append("records", str(record.id)).log(x_bf_trace_id)

        return to_concept_instance(record, model, properties), 201


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def create_concept_instance_batch(
    db: PartitionedDatabase, concept_id_or_name: str, body: JsonDict
):
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, concept_id_or_name)
        requests = [to_record(properties, req["values"]) for req in body]
        records = db.create_records_tx(
            tx, concept_id_or_name, requests, fill_missing=True
        )
        instances = [to_concept_instance(r, model, properties) for r in records]
        if not instances:
            raise BadRequest(
                f"Could not create concept instances for [{concept_id_or_name}]"
            )

        # Log the created concept instance:
        x_bf_trace_id = AuditLogger.trace_id_header()

        # Emit "CreateRecord" events:
        PennsieveJobsClient.get().send_changelog_events(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            events=[
                CreateRecord(id=r.id, name=r.name, model_id=model.id) for r in records
            ],
            trace_id=TraceId(x_bf_trace_id),
        )

        AuditLogger.get().message().append(
            "records", *[str(r.id) for r in records]
        ).log(x_bf_trace_id)

        return instances


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_all_concept_instances(
    db: PartitionedDatabase,
    concept_id_or_name: str,
    limit: int,
    offset: int,
    order_by: Optional[str] = None,
    ascending: Optional[bool] = None,
) -> List[JsonDict]:
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, concept_id_or_name)
        results = db.get_all_records_offset_tx(
            tx,
            model=model,
            limit=limit,
            offset=offset,
            fill_missing=True,
            order_by=None
            if order_by is None and ascending is None
            else OrderByField(
                name="created_at" if order_by is None else order_by,
                ascending=True if ascending is None else ascending,
            ),
        )

        x_bf_trace_id = AuditLogger.trace_id_header()
        record_ids = []
        instances = []
        for record in results:
            record_ids.append(str(record.id))
            instances.append(to_concept_instance(record, model, properties))

        AuditLogger.get().message().append("records", *record_ids).log(x_bf_trace_id)

        return instances


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_concept_instance(
    db: PartitionedDatabase, concept_id_or_name: str, concept_instance_id: str
) -> JsonDict:
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, concept_id_or_name)
        record = db.get_record_tx(tx, concept_instance_id, fill_missing=True)
        if record is None:
            raise NotFound(f"Could not get record {concept_instance_id}")
        return to_concept_instance(record, model, properties)


@permission_required(DatasetPermission.EDIT_RECORDS)
@touch_dataset_timestamp
def update_concept_instance(
    db: PartitionedDatabase,
    concept_id_or_name: str,
    concept_instance_id: str,
    body: JsonDict,
) -> JsonDict:
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, concept_id_or_name)
        record = db.get_record_tx(
            tx,
            concept_instance_id,
            embed_linked=False,
            fill_missing=True,
        )
        if record is None:
            raise NotFound(f"Could not get record {concept_instance_id}")

        updated_record = db.update_record_tx(
            tx,
            concept_instance_id,
            to_record(properties, body["values"]),
            fill_missing=True,
        )

        x_bf_trace_id = AuditLogger.trace_id_header()

        # Emit a "UpdateRecord" event:
        PennsieveJobsClient.get().send_changelog_event(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            event=UpdateRecord(
                id=record.id,
                name=record.name,
                model_id=model.id,
                properties=UpdateRecord.compute_diff(
                    properties, record.values, updated_record.values
                ),
            ),
            trace_id=TraceId(x_bf_trace_id),
        )

        return to_concept_instance(updated_record, model, properties)


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def delete_concept_instance(
    db: PartitionedDatabase, concept_id_or_name: str, concept_instance_id: str
) -> None:
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, concept_id_or_name)
        record = db.delete_record_tx(tx, concept_instance_id, properties)

        x_bf_trace_id = AuditLogger.trace_id_header()

        # Emit a "DeleteRecord" event:
        PennsieveJobsClient.get().send_changelog_event(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            event=DeleteRecord(id=record.id, name=record.name, model_id=model.id),
            trace_id=TraceId(x_bf_trace_id),
        )

        return to_concept_instance(record, model, properties)


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def delete_concept_instances(
    db: PartitionedDatabase, concept_id_or_name: str
) -> JsonDict:
    # HACK: request bodies on DELETE requests do not have defined
    # semantics and are not directly support by OpenAPI/Connexion. See
    #  - https://swagger.io/docs/specification/describing-request-body
    #  - https://github.com/zalando/connexion/issues/896
    body = connexion.request.json

    success = []
    errors = []
    events = []

    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, model)

        for instance_id in body:
            try:
                deleted = db.delete_record_tx(tx, instance_id, properties)
                events.append(
                    DeleteRecord(
                        id=deleted.id,
                        name=deleted.name,
                        model_id=model.id,
                    )
                )
            except Exception as e:  # noqa: F841
                errors.append([instance_id, f"Could not delete {instance_id}"])
            else:
                success.append(instance_id)

        x_bf_trace_id = AuditLogger.trace_id_header()

        # Emit a "DeleteRecord" event:
        PennsieveJobsClient.get().send_changelog_events(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            events=events,
            trace_id=TraceId(x_bf_trace_id),
        )

        return {"success": success, "errors": errors}


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_concept_instance_relationship_counts(
    db: PartitionedDatabase,
    concept_id_or_name: str,
    id_: str,
    include_incoming_linked_properties: bool = False,
) -> JsonDict:
    """
    Summary information for concepts related to this instance.

    Package proxies are a special case, with an entry that looks like

        {"name": "package", "displayName": "Files", "count": 4}.
    """
    return [
        summary.to_dict()
        for summary in db.summarize_record(
            id_, include_incoming_linked_properties=include_incoming_linked_properties
        )
    ]
