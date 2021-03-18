from typing import Dict, List, Optional, Tuple
from uuid import UUID

from auth_middleware.models import DatasetPermission  # type: ignore
from more_itertools import partition
from werkzeug.exceptions import NotFound

from core.clients import (
    AuditLogger,
    PennsieveApiClient,
    PennsieveJobsClient,
    CreateModel,
    CreateModelProperty,
    DeleteModel,
    DeleteModelProperty,
    TraceId,
    UpdateModel,
    UpdateModelProperty,
)
from core.clients.header import with_trace_id_header
from server.auth import auth_header, permission_required
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.models import JsonDict, Model, ModelId
from server.models.models import OrderBy as ModelOrderBy

from .common import (
    filter_model_dict,
    to_concept_dict,
    to_concept_instance,
    to_legacy_package_dto,
    to_legacy_relationship,
    to_legacy_relationship_instance,
    to_legacy_topology,
    to_linked_property,
    to_model_property,
    to_property_dict,
    to_schema_linked_property,
)


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def create_concept(db: PartitionedDatabase, body: JsonDict) -> Tuple[JsonDict, int]:
    x_bf_trace_id = AuditLogger.trace_id_header()
    model = db.create_model(**filter_model_dict(body))
    # Emit "CreateModel" event:
    PennsieveJobsClient.get().send_changelog_event(
        organization_id=db.organization_id,
        dataset_id=db.dataset_id,
        user_id=db.user_id,
        event=CreateModel(id=UUID(model.id), name=model.name),
        trace_id=TraceId(x_bf_trace_id),
    )
    return to_concept_dict(model, property_count=0), 201


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_all_concepts(db: PartitionedDatabase) -> List[JsonDict]:
    with db.transaction() as tx:
        models = db.get_models_tx(tx)
        property_counts = db.get_property_counts_tx(tx, [model.id for model in models])
        return [to_concept_dict(m, property_counts[m.id]) for m in models]


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_related_concepts(db: PartitionedDatabase, id_or_name: str) -> List[JsonDict]:
    with db.transaction() as tx:
        models = db.get_related_models_tx(tx, id_or_name)
        property_counts = db.get_property_counts_tx(tx, [model.id for model in models])
        return [to_concept_dict(m, property_counts[m.id]) for m in models]


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_concept(db: PartitionedDatabase, concept_id_or_name: str) -> JsonDict:
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        property_count = db.get_property_counts_tx(tx, [model.id])[model.id]
        return to_concept_dict(model, property_count)


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def update_concept(
    db: PartitionedDatabase, concept_id_or_name: str, body: JsonDict
) -> JsonDict:
    x_bf_trace_id = AuditLogger.trace_id_header()
    with db.transaction() as tx:
        model = db.update_model_tx(tx, concept_id_or_name, **filter_model_dict(body))
        property_count = db.get_property_counts_tx(tx, [model.id])[model.id]
        # Emit "UpdateModel" event:
        PennsieveJobsClient.get().send_changelog_event(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            event=UpdateModel(id=UUID(model.id), name=model.name),
            trace_id=TraceId(x_bf_trace_id),
        )
        return to_concept_dict(model, property_count)


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def delete_concept(db: PartitionedDatabase, concept_id_or_name: str) -> JsonDict:
    x_bf_trace_id = AuditLogger.trace_id_header()
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        property_count = db.get_property_counts_tx(tx, [model.id])[model.id]
        # Emit "DeleteModel" event:
        PennsieveJobsClient.get().send_changelog_event(
            organization_id=db.organization_id,
            dataset_id=db.dataset_id,
            user_id=db.user_id,
            event=DeleteModel(id=UUID(model.id), name=model.name),
            trace_id=TraceId(x_bf_trace_id),
        )
        return to_concept_dict(
            db.delete_model_tx(tx, concept_id_or_name), property_count
        )


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_all_properties(
    db: PartitionedDatabase, concept_id_or_name: str
) -> List[JsonDict]:
    return [to_property_dict(p) for p in db.get_properties(concept_id_or_name)]


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def update_properties(
    db: PartitionedDatabase, concept_id_or_name: str, body: List[JsonDict]
) -> List[JsonDict]:
    x_bf_trace_id = AuditLogger.trace_id_header()
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.update_properties_tx(
            tx, model, *[to_model_property(p) for p in body]
        )

        # Emit "UpdateModel" event:
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
        return [to_property_dict(p) for p, _ in properties]


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def delete_property(
    db: PartitionedDatabase, concept_id_or_name: str, property_id: str
) -> None:
    x_bf_trace_id = AuditLogger.trace_id_header()
    with db.transaction() as tx:
        model = db.get_model_tx(tx, concept_id_or_name)

        deleted = db.delete_property_tx(tx, model, property_id)
        if deleted is None:
            raise NotFound(
                f"Could not find property {property_id} of model {concept_id_or_name}"
            )
        # Emit "DeleteModelProperty" event:
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


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_related(
    db: PartitionedDatabase,
    concept_id: str,
    id_: str,
    target_concept_id_or_name: str,
    relationship_order_by: Optional[str] = None,
    record_order_by: Optional[str] = None,
    ascending: Optional[bool] = True,
    limit: int = 100,
    offset: int = 0,
    include_incoming_linked_properties: bool = False,
) -> List[JsonDict]:
    with db.transaction() as tx:

        model = db.get_model_tx(tx, target_concept_id_or_name)
        properties = db.get_properties_tx(tx, target_concept_id_or_name)
        order_by: Optional[ModelOrderBy] = None
        asc = ascending if ascending is not None else True

        if record_order_by is not None:
            order_by = ModelOrderBy.field(name=record_order_by, ascending=asc)
        elif relationship_order_by is not None:
            order_by = ModelOrderBy.relationship(
                type=relationship_order_by, ascending=asc
            )
        else:
            order_by = ModelOrderBy.field(
                name="created_at", ascending=True
            )  # default order for backwards compatibility

        related = db.get_related_records_tx(
            tx,
            start_from=id_,
            model_name=target_concept_id_or_name,
            order_by=order_by,
            limit=limit,
            offset=offset,
            include_incoming_linked_properties=include_incoming_linked_properties,
        )
        return [
            (
                to_legacy_relationship_instance(rr)
                if rr.one_to_many
                else to_linked_property(rr),
                to_concept_instance(r, model, properties),
            )
            for (rr, r) in related
        ]


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_graph_summary(db: PartitionedDatabase) -> JsonDict:
    return db.summarize().to_dict()


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_topology(db: PartitionedDatabase, id_: str) -> JsonDict:
    return [to_legacy_topology(t) for t in db.topology(id_)]


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_schema_graph(db: PartitionedDatabase) -> JsonDict:
    with db.transaction() as tx:
        structure = db.graph_schema_structure_tx(tx)
        models: List[Model] = structure.models
        model_ids: List[ModelId] = [m.id for m in models]
        property_counts: Dict[ModelId, int] = db.get_property_counts_tx(tx, model_ids)
        one_to_one, one_to_many = partition(
            lambda r: r.one_to_many, structure.relationships
        )

        legacy_models = [
            dict(type="concept", **to_concept_dict(m, property_counts[m.id]))
            for m in models
        ]
        legacy_model_relationships = [
            dict(type="schemaRelationship", **to_legacy_relationship(r))
            for r in one_to_many
        ]
        legacy_schema_linked_properties = [
            to_schema_linked_property(r) for r in one_to_one
        ]

        return (
            legacy_models + legacy_model_relationships + legacy_schema_linked_properties
        )


@permission_required(DatasetPermission.VIEW_RECORDS, DatasetPermission.VIEW_FILES)
def get_files(
    db: PartitionedDatabase,
    concept_id: str,
    id_: str,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "createdAt",
    ascending: bool = True,
) -> JsonDict:

    x_bf_trace_id = AuditLogger.trace_id_header()

    _, package_proxies = db.get_package_proxies_for_record(
        id_, limit=limit, offset=offset
    )

    package_proxies = list(package_proxies)

    # If any packages cannot be found they will be ignored in this response
    # TODO: https://app.clickup.com/t/3gaec4
    packages = PennsieveApiClient.get().get_packages(
        db.dataset_node_id,
        package_ids=[proxy.package_id for proxy in package_proxies],
        headers=dict(**auth_header(), **with_trace_id_header(x_bf_trace_id)),
    )

    package_proxy_ids = [str(p.id) for p in package_proxies]
    package_ids = packages.keys()

    AuditLogger.get().message().append("package-proxies", *package_proxy_ids).append(
        "packages", *package_ids
    ).log(TraceId(x_bf_trace_id))

    # Yes, this response is crazy: an array of two-tuples (arrays), containing a
    # single object with the proxy id, and the package DTO.
    return [
        [{"id": proxy.id}, to_legacy_package_dto(packages[proxy.package_id])]
        for proxy in package_proxies
        if proxy.package_id in packages
    ]


@permission_required(DatasetPermission.VIEW_RECORDS, DatasetPermission.VIEW_FILES)
def get_files_paged(
    db: PartitionedDatabase,
    concept_id: str,
    id_: str,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "createdAt",
    ascending: bool = True,
) -> JsonDict:

    x_bf_trace_id = AuditLogger.trace_id_header()

    total_count, package_proxies = db.get_package_proxies_for_record(
        id_, limit=limit, offset=offset
    )

    package_proxies = list(package_proxies)

    # If any packages cannot be found they will be ignored in this response
    # TODO: https://app.clickup.com/t/3gaec4
    packages = PennsieveApiClient.get().get_packages(
        db.dataset_node_id,
        package_ids=[proxy.package_id for proxy in package_proxies],
        headers=dict(**auth_header(), **with_trace_id_header(x_bf_trace_id)),
    )

    package_proxy_ids = [str(p.id) for p in package_proxies]
    package_ids = packages.keys()

    AuditLogger.get().message().append("package-proxies", *package_proxy_ids).append(
        "packages", *package_ids
    ).log(TraceId(x_bf_trace_id))

    return {
        "limit": limit,
        "offset": offset,
        "totalCount": total_count,
        "results": [
            [{"id": proxy.id}, to_legacy_package_dto(packages[proxy.package_id])]
            for proxy in package_proxies
            if proxy.package_id in packages
        ],
    }
