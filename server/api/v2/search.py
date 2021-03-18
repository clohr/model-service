import csv
import json
from collections import defaultdict
from itertools import chain, islice
from typing import Dict, Iterator, List, Optional, Set, Tuple

import neotime
from audit_middleware import Auditor, TraceId
from auth_middleware.claim import Claim  # type: ignore
from auth_middleware.role import OrganizationId as RoleOrganizationId
from connexion.exceptions import OAuthProblem  # type: ignore
from flask import Response, current_app
from more_itertools import unique_everseen
from werkzeug.exceptions import Forbidden

from core.clients import AuditLogger, PennsieveApiClient
from core.clients.header import with_trace_id_header
from core.dtos.api import Dataset
from server.auth import auth_header
from server.models import DatasetId, JsonDict, ModelProperty, OrderDirection, UserId
from server.models import datatypes as dt
from server.models.query import Operator
from server.models.search import (
    DatasetFilter,
    ModelFilter,
    PropertyFilter,
    SearchDownloadRequest,
    SuggestedValues,
)

from ...db.search import SearchDatabase
from ..v1.properties import Unit


def prop_key(p: ModelProperty) -> Tuple[str, str]:
    return (p.name, p.data_type.to_json())


def authorize_search(organization_id: int, trace_id: TraceId, token_info: Claim):

    if not token_info.is_user_claim:
        raise OAuthProblem("Requires a user claim")

    if not token_info.has_organization_access(RoleOrganizationId(organization_id)):
        raise Forbidden

    user_id = UserId(token_info.content.node_id)

    datasets = PennsieveApiClient.get().get_datasets(
        headers=dict(**auth_header(), **with_trace_id_header(trace_id))
    )

    return SearchDatabase(
        db=current_app.config["db"],
        organization_id=organization_id,
        user_id=user_id,
        datasets=datasets,
    )


def property_to_suggestion(p: ModelProperty, operators: List[Operator]) -> JsonDict:
    return {
        "name": p.name,
        "displayName": p.display_name,
        "dataType": p.data_type.to_dict(),
        "operators": [op.value for op in operators],
    }


# TODO: return linked properties
def records(
    organization_id: int,
    token_info: Claim,
    limit: int,
    offset: int,
    order_direction: str,
    body: JsonDict,
    order_by: Optional[str] = None,
) -> JsonDict:

    x_bf_trace_id = AuditLogger.trace_id_header()

    db = authorize_search(organization_id, x_bf_trace_id, token_info)

    property_filters: List[PropertyFilter] = PropertyFilter.schema().load(
        body["filters"], many=True
    )
    dataset_filters = [DatasetFilter(d) for d in body.get("datasets", [])]

    results, total_count = db.search_records(
        model_filter=ModelFilter(body["model"]),
        property_filters=property_filters,
        dataset_filters=dataset_filters,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_direction=OrderDirection.parse(order_direction),
    )
    results = list(results)

    # Deduplicate the set of models and properites represented in the results.
    # TODO: do this in Neo4j so we don't have to send duplicate data over the wire
    model_properties = {result.model_id: result.properties for result in results}

    datasets = {result.model_id: result.dataset for result in results}

    # Write to the audit log:
    AuditLogger.get().message().append("organization", organization_id).append(
        "datasets", *[str(ds.id) for ds in datasets.values()]
    ).append("records", *[str(result.record.id) for result in results]).log(
        x_bf_trace_id
    )

    return {
        "models": [
            {
                "id": model_id,
                "properties": [p.to_dict() for p in properties],
                "dataset": datasets[model_id],
            }
            for model_id, properties in model_properties.items()
        ],
        "records": [
            {"modelId": result.model_id, **result.record.to_dict()}
            for result in results
        ],
        "totalCount": total_count,
        "limit": limit,
        "offset": offset,
    }


DATASET_COLUMN_NAME = "Dataset Name"
RECORD_DOWNLOAD_EXTENSION = ".csv"


def csv_error(e: Exception, name: str = "") -> str:
    return f"There was an error generating {name}{RECORD_DOWNLOAD_EXTENSION}: {e}"


class Line:
    def __init__(self):
        self._line = None

    def write(self, line):
        self._line = line

    def read(self):
        return self._line


def iter_csv(
    db: SearchDatabase,
    model_filter: ModelFilter,
    property_filters: List[PropertyFilter],
    dataset_filters: List[DatasetFilter],
    columns: Dict[str, str],
    date_columns: Set[str],
    datasets: Dict[int, Dataset],
    logger: Auditor,
    x_bf_trace_id: TraceId,
) -> Iterator[str]:
    """
    Streaming record iterator.

    In order for the response to stream end to end, the database transaction
    must be started inside this iterator so that the transaction context is
    bound to this scope even after the endpoint returns.
    """
    line = Line()
    writer = csv.DictWriter(line, [DATASET_COLUMN_NAME, *columns.values()])

    def format_value(key, value):
        if key in columns.keys():
            if key in date_columns:
                if isinstance(value, list):
                    return ", ".join(format_value(key, v) for v in value)
                elif isinstance(value, neotime.Date):
                    return str(value)
                else:
                    return value.date()
            elif isinstance(value, list):
                return ", ".join(format_value(key, v) for v in value)
            else:
                return str(value)

    with db.transaction() as tx:
        results = db.search_records_csv(
            tx,
            model_filter=model_filter,
            property_filters=property_filters,
            dataset_filters=dataset_filters,
        )
        writer.writeheader()
        yield line.read()

        # Batch records to limit size of audit log requests
        for batch in grouper(1000, results):
            record_ids = []

            for record, dataset in batch:
                record_ids.append(str(record.id))
                csv_record = {}

                for key, value in record.values.items():
                    csv_record[columns[key]] = format_value(key, value)

                csv_record[DATASET_COLUMN_NAME] = datasets[dataset.id].name

                writer.writerow(csv_record)
                yield line.read()

            # TODO: it would be great if we could correlate the dataset ID of
            # each record with the record ID. Can we group records by dataset?
            if record_ids:
                logger.message().append("records", *record_ids).log(x_bf_trace_id)


def grouper(n, iterable):
    """
    >>> list(grouper(3, 'ABCDEFG'))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterable = iter(iterable)
    return iter(lambda: list(islice(iterable, n)), [])


def records_csv(organization_id: int, token_info: Claim, body: JsonDict) -> Response:

    x_bf_trace_id = AuditLogger.trace_id_header()

    try:

        search_params = SearchDownloadRequest.schema().load(json.loads(body["data"]))

        db = authorize_search(organization_id, x_bf_trace_id, token_info)

        datasets = {ds.int_id: ds for ds in db.datasets}

        # Write to the audit log:
        logger = AuditLogger.get()
        logger.message().append("organization", organization_id).append(
            "datasets", *[ds.int_id for ds in db.datasets]
        ).log(x_bf_trace_id)

        model_properties = db.suggest_properties(search_params.model)
        columns = {}
        date_columns = set()
        for _, p, _ in model_properties:

            if dt.DataType.is_date(p.data_type):
                date_columns = date_columns.union({p.name})

            unit = dt.DataType.get_unit(p.data_type)
            unit_display_name = None if unit is None else Unit.find_display_name(unit)

            display_name = (
                p.display_name
                if unit is None
                else f"{p.display_name} ({unit if unit_display_name is None else unit_display_name})"
            )

            columns[p.name] = display_name

        response = Response(
            iter_csv(
                db,
                search_params.model,
                search_params.filters,
                search_params.datasets,
                columns,
                date_columns,
                datasets,
                logger,
                x_bf_trace_id,
            ),
            mimetype="text/csv",
        )
        response.headers[
            "Content-Disposition"
        ] = f"attachment; filename={search_params.model.name}{RECORD_DOWNLOAD_EXTENSION}"
        return response

    except Exception as e:
        response = Response(csv_error(e))
        response.headers["Content-Disposition"] = "attachment; filename=!ERROR.txt"
        return response


def packages(
    organization_id: int, token_info: Claim, limit: int, offset: int, body: JsonDict
) -> JsonDict:

    x_bf_trace_id = AuditLogger.trace_id_header()

    db = authorize_search(organization_id, x_bf_trace_id, token_info)

    api_client = PennsieveApiClient.get()

    property_filters: List[PropertyFilter] = PropertyFilter.schema().load(
        body["filters"], many=True
    )

    dataset_filters = [DatasetFilter(d) for d in body.get("datasets", [])]

    # 1) Run the query, and get all package
    results, total_count = db.search_packages(
        model_filter=ModelFilter(body["model"]),
        property_filters=property_filters,
        dataset_filters=dataset_filters,
        limit=limit,
        offset=offset,
    )
    results = list(results)

    # 2) Group packages by dataset - the API endpoint to get datasets requires
    # a dataset ID in the URL.
    packages_by_dataset = defaultdict(list)
    for result in results:
        packages_by_dataset[result.dataset].append(result.package)

    package_dtos = []
    package_ids = []

    # 3) Get all package DTOs
    for dataset, packages in packages_by_dataset.items():

        dtos = api_client.get_packages(
            dataset.node_id,
            [package.id for package in packages],
            headers=dict(**auth_header(), **with_trace_id_header(x_bf_trace_id)),
        )

        # 4) Reorder the DTOs in the response to match the order that packages
        # came out of the database. If a package has been deleted from Pennsieve
        # API, but not from Neo4j, it will be missing from the response. Ignore
        # this for now.
        # TODO: https://app.clickup.com/t/2c3ec9
        for package in packages:
            package_ids.append(package.id)
            if package.id in dtos:
                package_dtos.append(dtos[package.id])

    # Write to the audit log:
    AuditLogger.get().message().append("organization", organization_id).append(
        "datasets", *[str(r.dataset.id) for r in results]
    ).append("packages", *package_ids).log(x_bf_trace_id)

    return {
        "packages": package_dtos,
        "totalCount": total_count,
        "limit": limit,
        "offset": offset,
    }


def autocomplete_models(
    organization_id: int,
    token_info: Claim,
    dataset_id: Optional[int] = None,
    related_to: Optional[str] = None,
) -> JsonDict:

    x_bf_trace_id = AuditLogger.trace_id_header()

    db = authorize_search(organization_id, x_bf_trace_id, token_info)

    ds_id = None if dataset_id is None else DatasetId(dataset_id)

    datasets_and_models = list(
        db.suggest_models(dataset_id=ds_id, related_to=related_to)
    )
    datasets = {d for (d, _) in datasets_and_models}
    models = unique_everseen(
        (m for (_, m) in datasets_and_models), key=lambda m: m.name
    )

    # Write to the audit log:
    AuditLogger.get().message().append("organization", organization_id).append(
        "datasets", *[str(ds.id) for ds in datasets]
    ).log(x_bf_trace_id)

    return {"models": [model.to_dict() for model in models]}


def autocomplete_model_properties(
    organization_id: int,
    model_name: str,
    token_info: Claim,
    dataset_id: Optional[int] = None,
) -> JsonDict:

    x_bf_trace_id = AuditLogger.trace_id_header()

    db = authorize_search(organization_id, x_bf_trace_id, token_info)

    ds_id = None if dataset_id is None else DatasetId(dataset_id)

    datasets_properties_operators = list(
        db.suggest_properties(
            model_filter=ModelFilter(name=model_name), dataset_id=ds_id
        )
    )

    datasets = {d for (d, _, _) in datasets_properties_operators}
    properties_and_operators = unique_everseen(
        [(p, op) for (_, p, op) in datasets_properties_operators],
        key=lambda t: prop_key(t[0]),
    )

    # Write to the audit log:
    AuditLogger.get().message().append("organization", organization_id).append(
        "datasets", *[str(ds.id) for ds in datasets]
    ).log(x_bf_trace_id)

    # If a name is a duplicate, include its type in the output display name
    # to disambinguate:
    return [property_to_suggestion(p, ops) for (p, ops) in properties_and_operators]


def autocomplete_model_property_values(
    organization_id: int,
    model_name: str,
    property_name: str,
    token_info: Claim,
    dataset_id: Optional[int] = None,
    prefix: Optional[str] = None,
    unit: Optional[str] = None,
    limit: Optional[int] = 10,
) -> List[SuggestedValues]:

    x_bf_trace_id = AuditLogger.trace_id_header()

    db = authorize_search(organization_id, x_bf_trace_id, token_info)

    ds_id = None if dataset_id is None else DatasetId(dataset_id)

    suggested_values: List[Tuple[Dataset, SuggestedValues]] = db.suggest_values(
        model_name=model_name,
        model_property_name=property_name,
        dataset_id=ds_id,
        matching_prefix=prefix,
        unit=unit,
        limit=limit,
    )

    datasets: List[Dataset] = [d for d, _ in suggested_values]

    # Write to the audit log:
    AuditLogger.get().message().append("organization", organization_id).append(
        "datasets", *[str(ds.id) for ds in datasets]
    ).log(x_bf_trace_id)

    # Group properties by data type
    grouped_suggestions = defaultdict(list)

    for _, suggestion in suggested_values:
        grouped_suggestions[suggestion.property_.data_type.to_json()].append(suggestion)

    return [
        {
            "property": property_to_suggestion(
                suggestions[0].property_, suggestions[0].operators
            ),
            "values": list(chain.from_iterable(sv.values for sv in suggestions)),
        }
        for suggestions in grouped_suggestions.values()
    ]
