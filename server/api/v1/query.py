from typing import Dict, List, Optional, cast

from auth_middleware.models import DatasetPermission  # type: ignore

from core.clients import AuditLogger, TraceId
from server.auth import permission_required
from server.db import PartitionedDatabase, QueryRunner
from server.models import JsonDict, Model, ModelProperty
from server.models import OrderByField as ModelOrderBy
from server.models import Record
from server.models.legacy import GraphQuery
from server.models.query import Aggregate, Operator, OrderBy, UserQuery

from .common import to_concept_instance


def to_predicate_operand(op: str) -> (Operator, bool):
    op_table = {
        "eq": (Operator.EQUALS, False),
        "neq": (Operator.EQUALS, True),
        "lt": (Operator.LESS_THAN, False),
        "lte": (Operator.LESS_THAN_EQUALS, False),
        "gt": (Operator.GREATER_THAN, False),
        "gte": (Operator.GREATER_THAN_EQUALS, False),
    }
    if op not in op_table:
        raise NotImplementedError(f"query: operation [{op}] not supported")
    return op_table[op]


def to_user_query(legacy_query: GraphQuery) -> (UserQuery, str):
    """
    Convert a legacy query to a "modern" query.
    """
    q = UserQuery()

    if legacy_query.type.is_concept:
        source_model = legacy_query.type.type
    else:
        raise NotImplementedError("query: proxy packages not support")

    # For filters: the model to use should be the same as that of .[type]:
    for f in legacy_query.filters:
        (op, negate) = to_predicate_operand(f.predicate.operation)
        q.with_filter(
            model=source_model,
            field=f.key,
            op=op,
            argument=f.predicate.value,
            negate=negate,
        )

    # Join aliases are defined by the `key` section of a join and later
    # referenced by a selection like `{ "Concepts": { "joinKeys": ["site", "name"] }`

    # For joins -- connect two records by any relationship that exists:
    for join in legacy_query.joins:

        if join.target_type.is_concept:
            target_model: str = join.target_type.type
        else:
            raise NotImplementedError(f"join: proxy packages not support")

        # Note:
        #
        # A "join key" is an alias assigned to a clause in the [join] section
        # of a query that is later referenced by an optional [select] clause
        # in the query, such as
        #
        #   "select": { "Concepts": { "joinKeys": ["site", "name"] } },
        #
        # where [joins] is
        #
        #   "joins": [
        #     {
        #       "targetType": { "concept": { "type": "site" } },
        #       "filters": [],
        #       "key": "site"
        #     },
        #     {
        #       "targetType": { "concept": { "type": "site" } },
        #       "filters": [
        #         { "key": "name", "predicate": { "operation": "eq", "value": "Philly" } }
        #       ],
        #       "key": "name"
        #     }
        #   ]
        #
        join_key: Optional[str] = join.key

        if join.filters:
            for f in join.filters:
                (op, negate) = to_predicate_operand(f.predicate.operation)
                q.with_filter(
                    model=target_model,
                    field=f.key,
                    op=op,
                    argument=f.predicate.value,
                    negate=negate,
                )
        else:
            q.connect_to(model=target_model)

        if join_key is not None:
            q.alias(join_key, target_model)

    # Ordering:
    if legacy_query.order_by is not None:
        if legacy_query.order_by.field in ModelOrderBy.CREATED_AT_FIELDS:
            q.order_by(OrderBy.created_at(ascending=legacy_query.order_by.ascending))
        elif legacy_query.order_by.field in ModelOrderBy.UPDATED_AT_FIELDS:
            q.order_by(OrderBy.updated_at(ascending=legacy_query.order_by.ascending))
        else:
            q.order_by(
                OrderBy(
                    field=legacy_query.order_by.field,
                    ascending=legacy_query.order_by.ascending,
                )
            )

    # Selection:
    if legacy_query.select is not None:
        # if we have an aggregating query like
        # "select": { "GroupCount": { "field": "name", "key": "site" } },
        if legacy_query.select.is_group_count:
            q.aggregate(
                Aggregate.group_count(
                    field=legacy_query.select.field, model=legacy_query.select.key
                )
            )
        else:
            for join_key in legacy_query.select.join_keys:
                q.select_model(join_key)

    return (q, source_model)


@permission_required(DatasetPermission.VIEW_RECORDS)
def run(db: PartitionedDatabase, body: JsonDict) -> List[JsonDict]:

    x_bf_trace_id = AuditLogger.trace_id_header()

    # connexion renames properties called "type" to "type_":
    body["type"] = body.pop("type_")

    # connexion converted CamelCased keys to snake_case:
    legacy_query = GraphQuery.schema().load(body)
    limit = int(legacy_query.limit) if legacy_query.limit is not None else 25
    offset = int(legacy_query.offset) if legacy_query.offset is not None else 0

    (user_query, src_model_id_or_name) = to_user_query(legacy_query)
    qr = QueryRunner(db, user_query)

    results = []
    record_ids: List[str] = []

    with db.transaction() as tx:

        src_model: Model = qr.get_model_tx(tx, cast(str, src_model_id_or_name))
        models: Dict[str, Model] = qr.get_models_tx(tx, src_model_id_or_name)
        model_properties: Dict[str, List[ModelProperty]] = qr.get_model_properties_tx(
            tx, src_model_id_or_name
        )

        for r in qr.run_tx(
            tx=tx, source_model=src_model_id_or_name, limit=limit, offset=offset
        ):

            # if a single model was selected: expect `List[Record]`:
            if isinstance(r, Record):
                record_ids.append(str(r.id))
                results.append(
                    {
                        "targetValue": to_concept_instance(
                            r, src_model, model_properties[src_model.name]
                        )
                    }
                )
            else:
                if qr.query and qr.query.is_aggregating:
                    results.append(r)
                else:
                    # otherwise, expect `List[Dict[str, Record]]`
                    result = {}

                    for model_name_or_alias, record_data in r.items():

                        # If `model_name` is an alias, resolve it:
                        model_name: str = (
                            qr.resolve_model_alias(model_name_or_alias)
                            or model_name_or_alias
                        )

                        if model_name not in models:
                            models[model_name] = qr.get_model_tx(
                                tx, cast(str, model_name)
                            )
                            model_properties.update(
                                qr.get_model_properties_tx(tx, cast(str, model_name))
                            )

                        if src_model.name == model_name:
                            result.update(
                                {
                                    "targetValue": to_concept_instance(
                                        record_data,
                                        models[model_name],
                                        model_properties[model_name],
                                    )
                                }
                            )
                        else:
                            result.update(
                                {
                                    model_name_or_alias: to_concept_instance(
                                        record_data,
                                        models[model_name],
                                        model_properties[model_name],
                                    )
                                }
                            )
                        record_ids.append(str(record_data.id))

                    results.append(result)

        AuditLogger.get().message().append("records", *record_ids).log(x_bf_trace_id)

        return results
