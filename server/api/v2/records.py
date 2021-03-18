from typing import Dict, List, Optional, Tuple

from auth_middleware.models import DatasetPermission  # type: ignore
from werkzeug.exceptions import NotFound

from core.clients import (
    AuditLogger,
    PennsieveApiClient,
    PennsieveJobsClient,
    CreateRecord,
    DeleteRecord,
    TraceId,
    UpdateRecord,
)
from core.clients.header import with_trace_id_header
from server.auth import auth_header, permission_required
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.errors import RecordRelationshipNotFoundError
from server.models import (
    JsonDict,
    NextPageCursor,
    PackageId,
    PagedResult,
    RecordId,
    RecordRelationshipId,
)


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def create_record(
    db: PartitionedDatabase, model_id_or_name: str, body: JsonDict
) -> Tuple[JsonDict, int]:
    record_values = [body["values"]]
    x_bf_trace_id = AuditLogger.trace_id_header()
    record = db.create_records(model_id_or_name, record_values)[0]

    model = db.get_model(model_id_or_name)
    if model is None:
        raise NotFound(f"Model {model_id_or_name}")

    # Emit "CreateRecord" event:
    PennsieveJobsClient.get().send_changelog_event(
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
        user_id=db.user_id,
        event=CreateRecord(id=record.id, name=record.name, model_id=model.id),
        trace_id=TraceId(x_bf_trace_id),
    )
    return record.to_dict(), 201


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def create_records(
    db: PartitionedDatabase, model_id_or_name: str, body: List[Dict]
) -> Tuple[List[JsonDict], int]:
    x_bf_trace_id = AuditLogger.trace_id_header()
    record_values = [r["values"] for r in body]
    records = db.create_records(model_id_or_name, records=record_values)

    model = db.get_model(model_id_or_name)
    if model is None:
        raise NotFound(f"Model {model_id_or_name}")

    # Emit "CreateRecord" event:
    events = [CreateRecord(id=r.id, name=r.name, model_id=model.id) for r in records]
    PennsieveJobsClient.get().send_changelog_events(
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
        user_id=db.user_id,
        events=events,
        trace_id=TraceId(x_bf_trace_id),
    )
    return [record.to_dict() for record in records], 201


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_record(db: PartitionedDatabase, record_id: RecordId, linked: bool) -> JsonDict:
    record = db.get_record(record_id, embed_linked=linked)
    if record is None:
        raise NotFound(f"Could not get record {record_id}")
    return record.to_dict()


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_all_records(
    db: PartitionedDatabase,
    model_id_or_name: str,
    limit: int,
    linked: bool,
    next_page: Optional[NextPageCursor] = None,
) -> List[JsonDict]:
    x_bf_trace_id = AuditLogger.trace_id_header()
    paged_result = db.get_all_records(
        model_id_or_name, limit=limit, embed_linked=linked, next_page=next_page
    )
    record_ids = []
    for record in paged_result:
        record_ids.append(str(record.id))

    AuditLogger.get().message().append("records", *record_ids).log(
        TraceId(x_bf_trace_id)
    )

    return PagedResult(
        results=paged_result.results, next_page=paged_result.next_page
    ).to_dict()


@permission_required(DatasetPermission.EDIT_RECORDS)
@touch_dataset_timestamp
def update_record(
    db: PartitionedDatabase, record_id: RecordId, body: JsonDict
) -> JsonDict:
    x_bf_trace_id = AuditLogger.trace_id_header()

    record = db.get_record(record_id, embed_linked=False, fill_missing=True)
    if record is None:
        raise NotFound(f"Could not get record {record_id}")

    model = db.get_model_of_record(record)
    if model is None:
        raise NotFound(f"Cound not find model for record {record_id}")

    properties = db.get_properties(model)

    updated_record = db.update_record(record_id, body["values"])

    # Emit "UpdateRecord" event:
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
    return updated_record.to_dict()


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def delete_record(db: PartitionedDatabase, record_id: RecordId) -> None:
    x_bf_trace_id = AuditLogger.trace_id_header()

    model = db.get_model_of_record(record_id)
    if model is None:
        raise NotFound(f"Cound not find model for record {record_id}")

    properties = db.get_properties(model)

    deleted = db.delete_record(record_id, properties)

    # Emit "DeleteRecord" event:
    PennsieveJobsClient.get().send_changelog_event(
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
        user_id=db.user_id,
        event=DeleteRecord(id=deleted.id, name=deleted.name, model_id=model.id),
        trace_id=TraceId(x_bf_trace_id),
    )
    return None


@permission_required(DatasetPermission.VIEW_RECORDS, DatasetPermission.VIEW_FILES)
def get_all_package_proxies(
    db: PartitionedDatabase, record_id: RecordId, limit: int = 100, offset: int = 0
) -> JsonDict:
    total_count, proxies = db.get_package_proxies_for_record(
        record_id, limit=limit, offset=offset
    )

    x_bf_trace_id = AuditLogger.trace_id_header()

    package_proxy_ids = []
    packages = []

    for p in proxies:
        package_proxy_ids.append(str(p.id))
        packages.append(p.to_dict())

    AuditLogger.get().message().append("package-proxies", *package_proxy_ids).log(
        x_bf_trace_id
    )

    return {
        "limit": limit,
        "offset": offset,
        "totalCount": total_count,
        "packages": packages,
    }


@permission_required(DatasetPermission.EDIT_RECORDS, DatasetPermission.VIEW_FILES)
@touch_dataset_timestamp
def create_package_proxy(
    db: PartitionedDatabase, record_id: RecordId, package_id: PackageId, body: JsonDict
) -> Tuple[JsonDict, int]:

    x_bf_trace_id = AuditLogger.trace_id_header()

    package = PennsieveApiClient.get().get_package_ids(
        db.dataset_node_id,
        package_id,
        headers=dict(**auth_header(), **with_trace_id_header(x_bf_trace_id)),
    )
    return (
        db.create_package_proxy(
            record_id, package_id=package.id, package_node_id=package.node_id
        ).to_dict(),
        201,
    )


@permission_required(DatasetPermission.EDIT_RECORDS, DatasetPermission.VIEW_FILES)
@touch_dataset_timestamp
def delete_package_proxy(
    db: PartitionedDatabase, record_id: RecordId, package_id: PackageId
) -> JsonDict:
    return db.delete_package_proxy(record_id, package_id).to_dict()


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def create_relationship(
    db: PartitionedDatabase, record_id: RecordId, body: JsonDict
) -> Tuple[JsonDict, int]:
    (relationship, _) = db.create_record_relationship(
        from_record=record_id, relationship=body["type_"], to_record=body["to"]
    )
    return relationship.to_dict(), 201


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
def get_relationship(
    db: PartitionedDatabase, record_id: RecordId, relationship_id: RecordRelationshipId
) -> JsonDict:
    with db.transaction() as tx:
        return db.get_outgoing_record_relationship_tx(
            tx, record_id, relationship_id
        ).to_dict()


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
def get_relationships(
    db: PartitionedDatabase,
    record_id: RecordId,
    relationship_type: Optional[str] = None,
) -> List[JsonDict]:
    with db.transaction() as tx:
        relationships = db.get_outgoing_record_relationships_tx(
            tx=tx, from_record=record_id, relationship_name=relationship_type
        )
        return [rel.to_dict() for rel in relationships]


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def delete_relationship(
    db: PartitionedDatabase, record_id: RecordId, relationship_id: RecordRelationshipId
) -> RecordRelationshipId:

    with db.transaction() as tx:
        return db.delete_outgoing_record_relationship_tx(tx, record_id, relationship_id)


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def delete_relationships(
    db: PartitionedDatabase, record_id: RecordId, body: JsonDict
) -> Tuple[JsonDict, int]:
    if len(body) == 0:
        return [], 200

    response = []

    # TODO: batch these database calls
    with db.transaction() as tx:
        for record_relationship_id in body:
            try:
                db.delete_outgoing_record_relationship_tx(
                    tx, record_id, record_relationship_id
                )
            except RecordRelationshipNotFoundError:
                record_status = 404
            except Exception as e:
                record_status = 500
            else:
                record_status = 200

            response.append({"id": record_relationship_id, "status": record_status})

    # The 207 response (multi-status) allows parts of a batch request to
    # fail/succeed individually. If all succeed or all fail with the same
    # status, use this status for the overall response. Otherwise, return 207.
    # See https://tools.ietf.org/html/rfc4918#section-13
    unique_status = set(r["status"] for r in response)
    if len(unique_status) == 1:
        response_status = unique_status.pop()
    else:
        response_status = 207

    return response, response_status
