import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from neo4j import GraphDatabase, Session, Transaction  # type: ignore

from core.dtos.api import Dataset as DatasetDTO
from core.util import normalize_datetime

from ..errors import (
    InvalidDatasetError,
    ModelNotFoundError,
    ModelPropertyNotFoundError,
    OperationError,
)
from ..models import (
    Dataset,
    DatasetId,
    GraphValue,
    Model,
    ModelProperty,
    ModelPropertyId,
    NativeScalar,
    OrderDirection,
    OrganizationId,
    Package,
    PackageProxy,
    Record,
    UserId,
)
from ..models import datatypes as dt
from ..models import get_dataset_id, get_model_property_id
from ..models.query import Operator as op
from ..models.search import (
    DatasetFilter,
    ModelFilter,
    ModelSuggestion,
    PackageSearchResult,
    PropertyFilter,
    SearchResult,
    SuggestedValues,
)
from ..models.validate import validate_property_name
from . import Database, Transactional, labels
from .util import match_clause

log = logging.getLogger(__file__)


class SearchDatabase(Transactional):
    def __init__(
        self,
        db: Database,
        organization_id: OrganizationId,
        user_id: UserId,
        datasets: List[DatasetDTO],
    ):
        self.db = db
        self.organization_id = organization_id
        self.user_id = user_id
        self.datasets = datasets
        self.dataset_ids = [DatasetId(dataset.int_id) for dataset in self.datasets]

    def get_driver(self) -> GraphDatabase:
        return self.db.get_driver()

    @classmethod
    def get_operators(cls, datatype: dt.DataType) -> List[op]:
        if isinstance(datatype, dt.Boolean):
            return [op.EQUALS, op.NOT_EQUALS]
        elif isinstance(datatype, (dt.Double, dt.Long, dt.Date)):
            return [
                op.EQUALS,
                op.NOT_EQUALS,
                op.LESS_THAN,
                op.LESS_THAN_EQUALS,
                op.GREATER_THAN,
                op.GREATER_THAN_EQUALS,
            ]
        elif isinstance(datatype, dt.String):
            return [op.EQUALS, op.NOT_EQUALS, op.STARTS_WITH]
        elif isinstance(datatype, (dt.Enumeration, dt.Array)):
            return [op.CONTAINS]
        else:
            raise ValueError(f"No operator for type: {str(datatype)}")

    def suggest_models(
        self, dataset_id: Optional[DatasetId] = None, related_to: Optional[str] = None
    ) -> Iterator[Tuple[Dataset, ModelSuggestion]]:
        """
        Suggest a list of `Model` instances for the current list of datasets.
        """
        if not self.dataset_ids:
            return iter(())

        if dataset_id is not None and dataset_id not in self.dataset_ids:
            raise InvalidDatasetError(str(dataset_id))

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_ids=self.dataset_ids
        )

        match_dataset_clause = ""
        if dataset_id is not None:
            match_dataset_clause = """AND d.id = $dataset_id"""
            kwargs["dataset_id"] = dataset_id

        match_related_model_clause = ""
        if related_to is not None:
            match_related_model_clause = f"""
            MATCH (m)-[{labels.related_to()} *0..1]-({labels.model("n")} {{ name: $related_to }})
            """
            kwargs["related_to"] = related_to

        cql = f"""
        MATCH  ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        WHERE d.id IN $dataset_ids
        // Only return models with at least one record
        AND size((m)<-[{labels.instance_of()}]-()) > 0

        {match_dataset_clause}
        {match_related_model_clause}

        RETURN DISTINCT d.id           AS dataset_id,
                        d.node_id      AS dataset_node_id,
                        m.name         AS name,
                        m.display_name AS display_name
        ORDER BY display_name ASC
        """
        nodes = self.execute_single(
            cql,
            reserved_relationship_types=labels.RESERVED_SCHEMA_RELATIONSHIPS,
            **kwargs,
        ).records()

        return (
            (
                Dataset(
                    id=DatasetId(node["dataset_id"]), node_id=node["dataset_node_id"]
                ),
                ModelSuggestion(name=node["name"], display_name=node["display_name"]),
            )
            for node in nodes
        )

    def _assert_model_exists_in_datasets(
        self, tx: Union[Session, Transaction], model_name: str
    ):
        """
        Validate that at least model exists in the datasets specified when
        this `Search` instance was constructed.
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_ids=self.dataset_ids,
            model_name=model_name,
        )
        cql = f"""
        MATCH ({labels.organization()} {{ id: $organization_id }})
              <-[{labels.in_organization()}]-({labels.dataset("d")})
              <-[{labels.in_dataset()}]-({labels.model("m")})
              -[{labels.has_property()}]->({labels.model_property("p")})
        WHERE d.id IN $dataset_ids AND m.name = $model_name
        RETURN COUNT(m.name) AS count
        """
        count = tx.run(cql, kwargs).single().get("count")
        if count == 0:
            raise ModelNotFoundError(model_name)

    def _get_properties(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Optional[Union[ModelProperty, ModelPropertyId, str]] = None,
        model_property_name: Optional[str] = None,
        dataset_id: Optional[DatasetId] = None,
        unit: Optional[str] = None,
    ) -> Iterator[Tuple[Dataset, ModelProperty]]:
        """
        Get an iterator over matching model properties based on a model ID
        or name and an optional dataset ID.
        """
        if dataset_id is not None and dataset_id not in self.dataset_ids:
            raise InvalidDatasetError(str(dataset_id))

        if model_id_or_name is None and model_property_name is None:
            return iter([])

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_ids=self.dataset_ids
        )

        match_model_clause = ""
        if model_id_or_name is not None:
            match_model_clause = "AND " + match_clause(
                "model_id_or_name",
                model_id_or_name,
                kwargs,
                property_operator="=",
                qualifier="m",
            )

        match_property_clause = ""
        if model_property_name is not None:
            match_property_clause = """AND toLower(p.name) = $property_name"""
            kwargs["property_name"] = cast(
                GraphValue, model_property_name.strip().lower()
            )

        match_dataset_clause = ""
        if dataset_id is not None:
            match_dataset_clause = """AND d.id = $dataset_id"""
            kwargs["dataset_id"] = dataset_id

        match_unit_clause = ""
        if unit is not None:
            match_unit_clause = """
            AND (
              apoc.json.path(p.data_type).unit = $unit OR apoc.json.path(p.data_type).items.unit = $unit
            )
            """
            kwargs["unit"] = unit

        cql = f"""
        MATCH ({labels.organization()} {{ id: $organization_id }})
              <-[{labels.in_organization()}]-({labels.dataset("d")})
              <-[{labels.in_dataset()}]-({labels.model("m")})
              -[{labels.has_property()}]->({labels.model_property("p")})

        // Only return properties for models with at least one record
        WHERE size((m)<-[{labels.instance_of()}]-()) > 0

        MATCH (p)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (p)-[{labels.updated_by("updated")}]->({labels.user("u")})

        WHERE d.id IN $dataset_ids
              {match_model_clause}
              {match_property_clause}
              {match_dataset_clause}
              {match_unit_clause}

        RETURN d.id       AS dataset_id,
               d.node_id  AS dataset_node_id,
               p,
               c.node_id  AS created_by,
               u.node_id  AS updated_by,
               created.at AS created_at,
               updated.at AS updated_at

        ORDER BY p.index
        """

        results = tx.run(cql, **kwargs)

        return (
            (
                Dataset(id=node["dataset_id"], node_id=node["dataset_node_id"]),
                cast(
                    ModelProperty,
                    ModelProperty.from_node(
                        **node["p"],
                        created_by=node["created_by"],
                        updated_by=node["updated_by"],
                        created_at=node["created_at"],
                        updated_at=node["updated_at"],
                    ),
                ),
            )
            for node in results
        )

    def suggest_properties(
        self, model_filter: ModelFilter, dataset_id: Optional[DatasetId] = None
    ) -> Iterator[Tuple[Dataset, ModelProperty, List[op]]]:
        """
        Return a list of model properties for a specified model in a given
        dataset.
        """
        if dataset_id is not None and dataset_id not in self.dataset_ids:
            raise InvalidDatasetError(str(dataset_id))

        with self.transaction() as tx:

            self._assert_model_exists_in_datasets(tx, model_filter.name)
            datasets_and_properties = self._get_properties(
                tx, model_id_or_name=model_filter.name, dataset_id=dataset_id
            )

            return (
                (d, p, self.get_operators(p.data_type))
                for (d, p) in datasets_and_properties
            )

    def _get_property_range_values(
        self, tx: Union[Session, Transaction], property_id: ModelPropertyId
    ) -> Optional[List[Union[int, float, datetime]]]:
        """
        Gets the [minimum, maximum] values records associated model property.

        This is meant to apply to numeric values (like longs and doubles), as
        well as dates.

        If no minimum or maximum value is found, None is returned.
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_ids=self.dataset_ids,
            property_id=str(property_id),
        )
        cql = f"""
        MATCH ({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.has_property()}]->({labels.model_property("p")} {{ id: $property_id }})
        MATCH (m)
              -[{labels.in_dataset()}]->({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        WHERE  d.id IN $dataset_ids
        RETURN MIN(r[p.name]) AS minimum_value,
               MAX(r[p.name]) AS maximum_value
        """
        results = tx.run(cql, kwargs).single()
        min_value = results.get("minimum_value")
        max_value = results.get("maximum_value")
        return (
            None if min_value is None or max_value is None else [min_value, max_value]
        )

    def _get_property_array_values(
        self,
        tx: Union[Session, Transaction],
        property_id: ModelPropertyId,
        matching_prefix: Optional[str] = None,
        limit: int = 10,
    ) -> Iterator[str]:
        """
        Get all values for an array property.

        - If `matching_prefix` is provided, only values starting with `matching_prefix`
          will be matched.

        - If `matching_prefix` is omitted, the top `limit` most frequent values
          will be returned.
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_ids=self.dataset_ids,
            property_id=str(property_id),
            limit=limit,
        )

        cql = f"""
        MATCH ({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.has_property()}]->({labels.model_property("p")} {{ id: $property_id }})
        MATCH (m)
              -[{labels.in_dataset()}]->({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        WHERE  d.id IN $dataset_ids
        UNWIND r[p.name] AS value
        """
        if matching_prefix is not None:
            kwargs["prefix"] = matching_prefix.strip().lower()
            cql += f"""
            WITH value
            WHERE toLower(toString(value)) STARTS WITH $prefix
            """
        cql += """
        WITH COUNT(toLower(toString(value))) AS counts, value
        ORDER BY counts DESC
        RETURN value
        LIMIT toInteger($limit)
        """

        results = tx.run(cql, **kwargs)

        return (node["value"] for node in results)

    def _get_property_string_values(
        self,
        tx: Union[Session, Transaction],
        property_id: ModelPropertyId,
        matching_prefix: Optional[str] = None,
        limit: int = 10,
    ) -> Iterator[str]:
        """
        Get all string values for a property.

        - If `matching_prefix` is provided, only values starting with `matching_prefix`
          will be matched.

        - If `matching_prefix` is omitted, the top `limit` most frequent values
          will be returned.
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_ids=self.dataset_ids,
            property_id=str(property_id),
            limit=limit,
        )

        cql = f"""
        MATCH ({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.has_property()}]->({labels.model_property("p")} {{ id: $property_id }})
        MATCH (m)
              -[{labels.in_dataset()}]->({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        WHERE d.id IN $dataset_ids
        """
        if matching_prefix is not None:
            kwargs["prefix"] = matching_prefix.strip().lower()
            cql += f"""
            AND toLower(r[p.name]) STARTS WITH $prefix
            WITH r[p.name] AS value
            ORDER BY value ASC
            """
        else:
            cql += f"""
            WITH COUNT(toLower(r[p.name])) AS counts,
                 r[p.name]                 AS value
            ORDER BY counts DESC
            """
        cql += """
        RETURN DISTINCT value
        LIMIT  toInteger($limit)
        """

        return (node["value"] for node in tx.run(cql, **kwargs))

    def suggest_values(
        self,
        model_name: str,
        model_property_name: str,
        matching_prefix: Optional[str] = None,
        dataset_id: Optional[DatasetId] = None,
        unit: Optional[str] = None,
        limit: int = 10,
    ) -> List[Tuple[Dataset, SuggestedValues]]:
        """
        Suggest values for a property. Suggestions work based on the datatype
        of the specified property:

        For booleans:
            - Return [`True`, `False`]

        For long/double:
            - Return [minimum value, maximum value]. If a `unit` argumenet is
              provided, only return values of this unit.

        For dates:
            - Return [earliest date, latest date]

        For enumerations:
            - Return allowed values

        For arrays:
            - Return the enumeration list (if present)
            - Otherwise, the top K most frequent array values and return those

        For strings:
            - Return first K strings matching the given prefix, or if no
              prefix is provided, return the top-k most frequent strings
        """
        with self.transaction() as tx:

            datasets_and_properties = self._get_properties(
                tx,
                model_id_or_name=model_name,
                model_property_name=model_property_name,
                dataset_id=dataset_id,
                unit=unit,
            )

            property_to_dataset = {}
            possible_properties: List[ModelProperty] = []

            for (d, p) in datasets_and_properties:
                possible_properties.append(p)
                property_to_dataset[p.id] = d

            if not possible_properties:
                raise OperationError(
                    f"No matching values found",
                    cause=ModelPropertyNotFoundError(
                        model="*", property_name=model_property_name
                    ),
                )

            suggested_values: List[Tuple[Dataset, SuggestedValues]] = []

            for p in possible_properties:

                prop_values: List[NativeScalar] = []

                if isinstance(p.data_type, dt.Boolean):
                    prop_values = [True, False]
                elif isinstance(p.data_type, (dt.Long, dt.Double, dt.Date)):
                    prop_values = cast(
                        List[NativeScalar],
                        self._get_property_range_values(tx=tx, property_id=p.id) or [],
                    )
                    if isinstance(p.data_type, dt.Date):
                        prop_values = [normalize_datetime(dt) for dt in prop_values]

                elif isinstance(p.data_type, dt.String):
                    prop_values = list(
                        self._get_property_string_values(
                            tx=tx,
                            property_id=p.id,
                            matching_prefix=matching_prefix,
                            limit=limit,
                        )
                    )
                elif isinstance(p.data_type, dt.Enumeration):
                    prop_values = p.data_type.enum or []
                elif isinstance(p.data_type, dt.Array):
                    if p.data_type.enum is not None:
                        prop_values = p.data_type.enum or []
                    else:
                        prop_values = cast(
                            List[NativeScalar],
                            list(
                                self._get_property_array_values(
                                    tx=tx,
                                    property_id=p.id,
                                    matching_prefix=matching_prefix,
                                    limit=limit,
                                )
                            ),
                        )
                else:
                    raise ValueError(f"Unsupported datatype: {str(p.data_type)}")

                suggested = SuggestedValues(
                    property_=p,
                    operators=self.get_operators(p.data_type),
                    values=cast(List[GraphValue], prop_values),
                )

                suggested_values.append((property_to_dataset[p.id], suggested))

        return suggested_values

    def search_records_csv(
        self,
        tx,
        model_filter: ModelFilter,
        property_filters: Optional[List[PropertyFilter]] = None,
        dataset_filters: Optional[List[DatasetFilter]] = None,
    ) -> Generator[Tuple[Record, Dataset], None, None]:
        """
        TODO: Include linked property stubs
        """
        if property_filters is None:
            property_filters = []

        dataset_ids = self.filter_dataset_ids(dataset_filters)

        root_model, adjacent_models = TargetModel.group_property_filters(
            model_filter, property_filters
        )

        cql = f"""
        MATCH ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        WHERE d.id IN $dataset_ids
        AND m.name = $model_name
        {root_model.match_model_properties("m") if root_model else ""}

        {"".join(adjacent_model.match_models("m") for adjacent_model in adjacent_models)}

        MATCH ({labels.record("r")})-[{labels.instance_of()}]->(m)
        {"WHERE " + root_model.match_record_properties("r") if root_model else ""}

        {"".join(adjacent_model.match_records("r") for adjacent_model in adjacent_models)}

        MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})
        MATCH (r)-[{labels.instance_of()}]->({labels.model("m")})

        RETURN d AS dataset, {{
              r: r,
              created_by: created_by.node_id,
              updated_by: updated_by.node_id,
              created_at: created.at,
              updated_at: updated.at
        }} AS record
        """

        kwargs = dict(
            organization_id=self.organization_id,
            dataset_ids=dataset_ids,
            model_name=model_filter.name,
            **(root_model.parameters() if root_model else {}),
        )
        for adjacent_model in adjacent_models:
            kwargs.update(adjacent_model.parameters())

        log.debug(cql)
        log.debug(kwargs)

        result = tx.run(cql, **kwargs)

        for node in result.records():
            yield (
                (
                    Record.from_node(
                        node["record"]["r"],
                        created_by=node["record"]["created_by"],
                        updated_by=node["record"]["updated_by"],
                        created_at=node["record"]["created_at"],
                        updated_at=node["record"]["updated_at"],
                    ),
                    Dataset.from_node(node["dataset"]),
                )
            )

            # HACK: Neo4j stores the entire result set in-memory in a `Graph` object.
            # This causes a memory leak when streaming CSVs with millions of records.
            #
            # These calls delete the `Graph` store after deserializing each result
            #
            # Internally, the `PackStreamHydrator` [1] deserializes Neo4j
            # entities from raw bytes. `hydrate` calls out to
            # `hydration_functions` [2] which dispatches by entity type. For
            # example, `put_node` [3] hydrates a `Node`, whose constructor then
            # inserts the `Node` into the `Graph` [4]
            #
            # [1] https://github.com/neo4j/neo4j-python-driver/blob/b3c6aba9595e6899bc104f6160eb9003a3a657ab/neo4j/types/__init__.py#L56-L75
            # [2] https://github.com/neo4j/neo4j-python-driver/blob/b3c6aba9595e6899bc104f6160eb9003a3a657ab/neo4j/types/graph.py#L367-L374
            # [3] https://github.com/neo4j/neo4j-python-driver/blob/b3c6aba9595e6899bc104f6160eb9003a3a657ab/neo4j/types/graph.py#L72-L76
            # [4] https://github.com/neo4j/neo4j-python-driver/blob/b3c6aba9595e6899bc104f6160eb9003a3a657ab/neo4j/types/graph.py#L193
            #
            # TODO: verify these calls are correct when upgrading to 4.x.x driver

            result._hydrant.graph._nodes.clear()
            result._hydrant.graph._relationships.clear()

    def search_records(
        self,
        model_filter: ModelFilter,
        property_filters: Optional[List[PropertyFilter]] = None,
        dataset_filters: Optional[List[DatasetFilter]] = None,
        limit: int = 25,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_direction: OrderDirection = OrderDirection.ASC,
    ) -> Tuple[Iterator[SearchResult], int]:
        """
        TODO: Include linked property stubs
        """
        if property_filters is None:
            property_filters = []

        dataset_ids = self.filter_dataset_ids(dataset_filters)

        root_model, adjacent_models = TargetModel.group_property_filters(
            model_filter, property_filters
        )

        if order_by is None:
            order_by_cql = ""
        # TODO: raise CannotSortRecords error if the result set is larger than a max limit
        else:
            order_by_cql = (
                f"ORDER BY coalesce(r[$order_by], r.`@id`) {order_direction.value}"
            )

        cql = f"""
        MATCH ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        WHERE d.id IN $dataset_ids
        AND m.name = $model_name
        {root_model.match_model_properties("m") if root_model else ""}

        {"".join(adjacent_model.match_models("m") for adjacent_model in adjacent_models)}

        MATCH ({labels.record("r")})-[{labels.instance_of()}]->(m)
        {"WHERE " + root_model.match_record_properties("r") if root_model else ""}

        {"".join(adjacent_model.match_records("r") for adjacent_model in adjacent_models)}

        WITH COLLECT(DISTINCT r) AS records
        WITH records, SIZE(records) AS total_count
        UNWIND records AS r

        WITH r, total_count
        {order_by_cql}
        SKIP toInteger($offset)
        LIMIT toInteger($limit)

        MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})
        MATCH (r)-[{labels.instance_of()}]->({labels.model("m")})

        WITH total_count, m, {{
              r: r,
              created_by: created_by.node_id,
              updated_by: updated_by.node_id,
              created_at: created.at,
              updated_at: updated.at
        }} AS record

        MATCH (m)-[{labels.in_dataset()}]->({labels.dataset("d")})
        MATCH (m)-[{labels.has_property()}]->({labels.model_property("p")})
        MATCH (p)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (p)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

        RETURN m AS model, d AS dataset, record, total_count, COLLECT(p {{
             .*,
             created_by: created_by.node_id,
             updated_by: updated_by.node_id,
             created_at: created.at,
             updated_at: updated.at
        }}) AS properties
        """

        kwargs = dict(
            organization_id=self.organization_id,
            dataset_ids=dataset_ids,
            order_by=order_by,
            limit=limit,
            offset=offset,
            model_name=model_filter.name,
            **(root_model.parameters() if root_model else {}),
        )
        for adjacent_model in adjacent_models:
            kwargs.update(adjacent_model.parameters())

        log.debug(cql)
        log.debug(kwargs)

        result = self.execute_single(cql, **kwargs)

        # TODO raise not found?
        head = result.peek()
        if head is None:
            return iter(()), 0

        return (
            (
                SearchResult(
                    model_id=node["model"]["id"],
                    properties=sorted(
                        [
                            cast(ModelProperty, ModelProperty.from_node(**p))
                            for p in node["properties"]
                        ],
                        key=lambda p: p.index,
                    ),
                    record=Record.from_node(
                        node["record"]["r"],
                        created_by=node["record"]["created_by"],
                        updated_by=node["record"]["updated_by"],
                        created_at=node["record"]["created_at"],
                        updated_at=node["record"]["updated_at"],
                    ),
                    dataset=Dataset.from_node(node["dataset"]),
                )
                for node in result.records()
            ),
            head["total_count"],
        )

    def search_packages(
        self,
        model_filter: ModelFilter,
        property_filters: Optional[List[PropertyFilter]] = None,
        dataset_filters: Optional[List[DatasetFilter]] = None,
        limit: int = 25,
        offset: int = 0,
    ) -> Tuple[Iterator[PackageSearchResult], int]:

        if property_filters is None:
            property_filters = []

        dataset_ids = self.filter_dataset_ids(dataset_filters)

        root_model, adjacent_models = TargetModel.group_property_filters(
            model_filter, property_filters
        )

        cql = f"""
        MATCH ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        WHERE d.id IN $dataset_ids
        AND m.name = $model_name
        {root_model.match_model_properties("m") if root_model else ""}

        {"".join(adjacent_model.match_models("m") for adjacent_model in adjacent_models)}

        MATCH ({labels.record("r")})-[{labels.instance_of()}]->(m)
        {"WHERE " + root_model.match_record_properties("r") if root_model else ""}

        {"".join(adjacent_model.match_records("r") for adjacent_model in adjacent_models)}

        MATCH (r)-[{labels.in_package()}]->({labels.package("p")})

        WITH COLLECT(DISTINCT p) AS packages
        WITH packages, SIZE(packages) AS total_count
        UNWIND packages AS p

        MATCH (p)-[{labels.in_dataset()}]->({labels.dataset("d")})

        WITH p, d, total_count
        ORDER BY d.id, p.package_id
        SKIP toInteger($offset)
        LIMIT toInteger($limit)

        RETURN p AS package, d AS dataset, total_count
        """

        kwargs = dict(
            organization_id=self.organization_id,
            dataset_ids=dataset_ids,
            limit=limit,
            offset=offset,
            model_name=model_filter.name,
            **(root_model.parameters() if root_model else {}),
        )
        for adjacent_model in adjacent_models:
            kwargs.update(adjacent_model.parameters())

        log.debug(cql)
        log.debug(kwargs)

        result = self.execute_single(cql, **kwargs)

        # TODO raise not found?
        head = result.peek()
        if head is None:
            return iter(()), 0

        return (
            (
                PackageSearchResult(
                    package=Package(
                        id=node["package"]["package_id"],
                        node_id=node["package"]["package_node_id"],
                    ),
                    dataset=Dataset.from_node(node["dataset"]),
                )
                for node in result.records()
            ),
            head["total_count"],
        )

    def filter_dataset_ids(
        self, dataset_filters: Optional[List[DatasetFilter]]
    ) -> List[DatasetId]:
        """
        Combine the provided dataset filters with the allowed datasets to get
        the final list of dataset ids to target in the search.

        Raises:
          - InvalidDatasetError
        """
        if not dataset_filters:
            return self.dataset_ids
        # HACK: match the current frontend expectation that multiple dataset ids
        # means that we should AND them together. Since a record can only be in
        # one dataset, that means no results.
        # TODO: update this handling once dataset filters are more fleshed out.
        elif len(dataset_filters) > 1:
            return []
        elif dataset_filters[0].id in self.dataset_ids:
            return [dataset_filters[0].id]
        else:
            raise InvalidDatasetError(str(dataset_filters[0].id))


@dataclass
class TargetModel:
    """
    An `TargetModel` represents one or more filters on a record adjacent to
    the target record.

    TODO: expand this to more than one hop.
    """

    i: int
    model: str
    property_filters: List[PropertyFilter]

    def model_alias(self) -> str:
        return f"model_{self.i}"

    def model_name(self) -> str:
        return f"model_{self.i}_name"

    def property_alias(self, j) -> str:
        return f"model_{self.i}_property_{j}"

    def property_name(self, j) -> str:
        return f"model_{self.i}_property_{j}_name"

    def property_value(self, j) -> str:
        return f"model_{self.i}_property_{j}_value"

    def property_unit(self, j) -> str:
        return f"model_{self.i}_property_{j}_unit"

    def record_alias(self) -> str:
        return f"record_{self.i}"

    def record_relationship_path(self) -> str:
        return f"record_relationship_{self.i}"

    def model_relationship_path(self) -> str:
        return f"model_relationship_{self.i}"

    def match_models(self, root_model_alias: str) -> str:
        return f"""
        MATCH {self.model_relationship_path()} = ({root_model_alias})-[{labels.related_to()} *0..1]-({labels.model(self.model_alias())})
        WHERE {self.model_alias()}.name = ${self.model_name()}
        {self.match_model_properties()}
        """

    def match_records(self, root_record_alias: str) -> str:
        return f"""
        MATCH {self.record_relationship_path()} = ({root_record_alias})-[*0..1]-({labels.record(self.record_alias())})
        WHERE NONE(relationship in rels({self.record_relationship_path()}) WHERE type(relationship) IN $reserved_relationship_types)
        AND {self.match_record_properties()}
        MATCH ({self.record_alias()})-[{labels.instance_of()}]->({self.model_alias()})
        """

    def match_model_properties(self, model_alias: Optional[str] = None) -> str:
        """
        Ensure that the properties of this neighbor model match all the
        conditions of these property filters.

        If a scientific unit is provided with search, exclude all other unit
        dimensions. The `unit` field can be defined either at the top-level of
        the `data_type`, or nested in the `items` object,
        """
        if model_alias is None:
            model_alias = self.model_alias()

        return "        ".join(
            f"""
            MATCH ({model_alias})-[{labels.has_property()}]->({labels.model_property(self.property_alias(j))} {{ name: ${self.property_name(j)} }})
            """.strip()
            + (
                f"""
            WHERE apoc.json.path({self.property_alias(j)}.data_type).unit = ${self.property_unit(j)}
            OR apoc.json.path({self.property_alias(j)}.data_type).items.unit = ${self.property_unit(j)}
            """.strip()
                if property_filter.unit
                else ""
            )
            + "\n"
            for j, property_filter in enumerate(self.property_filters)
        )

    def match_record_properties(self, record_alias: Optional[str] = None) -> str:
        """
        Ensure that the properties of this neighbor record match all the
        conditions of these property filters.

        Match is case insensitive for string type property values. If the
        given property value is a string, convert to lowercase and check
        against the record's value (also converted to lowercase).
        """
        if record_alias is None:
            record_alias = self.record_alias()

        def build_clause(index: int, prop_filter: PropertyFilter):
            if prop_filter.operator == op.EQUALS:
                return f"""
                CASE
                    WHEN apoc.meta.cypher.isType({record_alias}[${self.property_name(index)}], 'STRING')
                    THEN {record_alias}[${self.property_name(index)}] =~ ('(?i)' + ${self.property_value(index)})
                    ELSE {record_alias}[${self.property_name(index)}] {prop_filter.operator.value} ${self.property_value(index)}
                END
                """
            elif prop_filter.operator == op.NOT_EQUALS:
                return f"""
                CASE
                    WHEN apoc.meta.cypher.isType({record_alias}[${self.property_name(index)}], 'STRING')
                    THEN NOT({record_alias}[${self.property_name(index)}] =~ ('(?i)' + ${self.property_value(index)}))
                    ELSE {record_alias}[${self.property_name(index)}] {prop_filter.operator.value} ${self.property_value(index)}
                END
                """
            elif prop_filter.operator == op.CONTAINS:
                # The "contains" operator has dual function: for strings, it operates as one would expect for
                # searching for a substring in a target; for arrays, it searches for the given item in the property
                cql = f"""
                CASE
                    WHEN apoc.meta.cypher.isType({record_alias}[${self.property_name(index)}], 'STRING')
                    THEN toLower({record_alias}[${self.property_name(index)}]) {prop_filter.operator.value} toLower(${self.property_value(index)})
                """
                # For arrays, we need to explicitly use the "IN" operator:
                for t in ["STRING", "INTEGER", "FLOAT", "BOOLEAN", "ANY"]:
                    cql += f"""
                        WHEN apoc.meta.cypher.isType({record_alias}[${self.property_name(index)}], 'LIST OF {t}')
                        THEN ${self.property_value(index)} IN {record_alias}[${self.property_name(index)}]
                    """
                cql += f"""
                ELSE {record_alias}[${self.property_name(index)}] {prop_filter.operator.value} ${self.property_value(index)}
                END
                """

                return cql

            return f"""
            CASE
                WHEN apoc.meta.cypher.isType({record_alias}[${self.property_name(index)}], 'STRING')
                THEN toLower({record_alias}[${self.property_name(index)}]) {prop_filter.operator.value} toLower(${self.property_value(index)})
                ELSE {record_alias}[${self.property_name(index)}] {prop_filter.operator.value} ${self.property_value(index)}
            END
            """

        return "        AND ".join(
            [
                build_clause(j, property_filter)
                for j, property_filter in enumerate(self.property_filters)
            ]
        )

    def parameters(self) -> Dict[str, GraphValue]:
        parameters: Dict[str, GraphValue] = {
            self.model_name(): self.model,
            "reserved_relationship_types": labels.RESERVED_SCHEMA_RELATIONSHIPS,
        }
        for j, property_filter in enumerate(self.property_filters):
            parameters[self.property_name(j)] = property_filter.property_
            parameters[self.property_value(j)] = property_filter.value
            parameters[self.property_unit(j)] = property_filter.unit

        return parameters

    @classmethod
    def group_property_filters(
        self, model_filter: ModelFilter, property_filters: List[PropertyFilter]
    ) -> Tuple[Optional["TargetModel"], List["TargetModel"]]:
        """
        Combine a collection of property filters so that all filters that apply
        to the same model are part of the same `TargetModel`.

        Return a tuple of (root model, adjacent model) filters. If no filters
        exist for the root model, return `None` as the first item in the tuple.
        """
        grouped: Dict[str, List[PropertyFilter]] = defaultdict(list)
        for property_filter in property_filters:
            validate_property_name(property_filter.property_)
            grouped[property_filter.model].append(property_filter)

        root_filters = grouped.pop(model_filter.name, None)
        return (
            TargetModel(0, model_filter.name, root_filters) if root_filters else None,
            [
                TargetModel(i + 1, model, filters)
                for (i, (model, filters)) in enumerate(grouped.items())
                if model != model_filter.name
            ],
        )
