from typing import List, Optional
from uuid import UUID

import connexion
from auth_middleware.models import DatasetPermission  # type: ignore

from core.clients import AuditLogger, PennsieveApiClient, TraceId
from core.clients.header import with_trace_id_header
from server.auth import auth_header, permission_required
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.models import JsonDict, PackageNodeId, get_record_id

from ...errors import InvalidPackageProxyLinkTargetError
from .common import (
    NULL_UUID,
    make_proxy_relationship_instance,
    to_concept_instance,
    to_proxy_instance,
    to_proxy_link_target,
)

# Default (and only) proxy type
PROXY_TYPE = "package"


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_proxy_concept(db: PartitionedDatabase, proxy_type: str) -> JsonDict:
    """
    The legacy concepts-service created a special-case model that all proxy
    nodes were records of.

    Return a stub here.
    """
    return {"id": NULL_UUID, "proxyType": PROXY_TYPE}


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_all_proxy_instances(db: PartitionedDatabase, proxy_type: str) -> List[JsonDict]:
    with db.transaction() as tx:

        proxy_instances = []
        package_proxy_ids = []
        record_ids = []

        x_bf_trace_id = AuditLogger.trace_id_header()

        for package_proxy, record in db.get_all_package_proxies_tx(tx):
            proxy_instances.append(to_proxy_instance(PROXY_TYPE, package_proxy))
            package_proxy_ids.append(str(package_proxy.id))
            record_ids.append(str(record.id))

        AuditLogger.get().message().append(
            "package-proxies", *package_proxy_ids
        ).append("records", *record_ids).log(x_bf_trace_id)

        return proxy_instances


@permission_required(DatasetPermission.CREATE_DELETE_RECORD)
@touch_dataset_timestamp
def create_proxy_instance(
    db: PartitionedDatabase, proxy_type: str, body: JsonDict
) -> List[JsonDict]:
    response = []

    with db.transaction() as tx:

        x_bf_trace_id = AuditLogger.trace_id_header()
        link_targets = []
        package_ids = []

        for target in body["targets"]:

            link_target = target["linkTarget"]
            relationship_type = target["relationshipType"]

            link_targets.append(link_target)

            concept_link_target = to_proxy_link_target(link_target)
            if concept_link_target is None:
                raise InvalidPackageProxyLinkTargetError(link_target=str(body))

            package = PennsieveApiClient.get().get_package_ids(
                db.dataset_node_id,
                body["external_id"],
                headers=dict(**auth_header(), **with_trace_id_header(x_bf_trace_id)),
            )

            package_ids.append(str(package.id))

            package_proxy = db.create_package_proxy_tx(
                tx=tx,
                record=concept_link_target.id,
                package_id=package.id,
                package_node_id=package.node_id,
                legacy_relationship_type=relationship_type,
            )

            linkResult = {
                "proxyInstance": to_proxy_instance(PROXY_TYPE, package_proxy),
                "relationshipInstance": make_proxy_relationship_instance(
                    concept_link_target.id, package_proxy, relationship_type
                ),
            }

            response.append(linkResult)

    AuditLogger.get().message().append("link-targets", *link_targets).append(
        "packages", *package_ids
    ).log(x_bf_trace_id)

    return response, 201


@permission_required(DatasetPermission.VIEW_RECORDS)
def get_proxy_instance(db: PartitionedDatabase, proxy_type: str, id_: UUID) -> JsonDict:

    return to_proxy_instance(PROXY_TYPE, db.get_package_proxy(id_))


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def delete_proxy_instance(
    db: PartitionedDatabase, proxy_type: str, id_: UUID
) -> JsonDict:

    db.delete_package_proxy_by_id(id_)
    return None


@permission_required(DatasetPermission.MANAGE_RECORD_RELATIONSHIPS)
@touch_dataset_timestamp
def delete_proxy_instances(db: PartitionedDatabase, proxy_type: str) -> List[JsonDict]:

    # HACK: request bodies on DELETE requests do not have defined
    # semantics and are not directly support by OpenAPI/Connexion. See
    #  - https://swagger.io/docs/specification/describing-request-body
    #  - https://github.com/zalando/connexion/issues/896
    body = connexion.request.json

    # HACK:
    # since we're pulling directly from the raw body, names will not be camel cased:
    source_record_id = get_record_id(body.get("sourceRecordId"))
    proxy_instance_ids: List[str] = body.get("proxyInstanceIds", [])

    with db.transaction() as tx:
        return db.delete_package_proxies_tx(tx, source_record_id, *proxy_instance_ids)


# Note: `proxy_type` is ignored as we only support `proxy_type`="package".
@permission_required(DatasetPermission.VIEW_RECORDS)
def get_proxy_relationship_counts(
    db: PartitionedDatabase, proxy_type: str, node_id: str
) -> List[JsonDict]:
    return [
        count.to_dict()
        for count in db.get_proxy_relationship_counts(PackageNodeId(node_id))
    ]


# Note: `proxy_type` is ignored as we only support `proxy_type`="package".
@permission_required(DatasetPermission.VIEW_RECORDS)
def get_records_related_to_package(
    db: PartitionedDatabase,
    proxy_type: str,
    package_id: str,
    concept_id_or_name: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    relationship_order_by: Optional[str] = None,
    record_order_by: Optional[str] = None,
    ascending: bool = False,
) -> List[JsonDict]:
    with db.transaction() as tx:

        x_bf_trace_id = AuditLogger.trace_id_header()
        model = db.get_model_tx(tx, concept_id_or_name)
        properties = db.get_properties_tx(tx, concept_id_or_name)

        results = []
        package_proxy_ids = []
        record_ids = []

        for pp, r in db.get_records_related_to_package_tx(
            tx=tx,
            package_id=PackageNodeId(package_id),
            related_model_id_or_name=concept_id_or_name,
            limit=limit,
            offset=offset,
            relationship_order_by=relationship_order_by,
            record_order_by=record_order_by,
            ascending=ascending,
        ):
            package_proxy_ids.append(str(pp.id))
            record_ids.append(str(r.id))
            t = (
                # All package-to-record relationships are defined with the
                # internal `@IN_PACKAGE` relationship type:
                #   (Package)<-[`@IN_PACKAGE`]-(Record)
                # For legacy consistency, we just use the generic "belongs_to"
                # here:
                make_proxy_relationship_instance(r.id, pp, "belongs_to"),
                to_concept_instance(r, model, properties),
            )
            results.append(t)

        AuditLogger.get().message().append(
            "package-proxies", *package_proxy_ids
        ).append("records", *record_ids).log(x_bf_trace_id)

        return results
