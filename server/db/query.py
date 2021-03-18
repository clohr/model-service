import logging
from enum import Enum
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple, Union, cast
from uuid import UUID

from neo4j import Session, Transaction  # type: ignore

from ..errors import InfeasibleQueryError, ModelNotFoundError
from ..models import GraphValue, Model, ModelProperty, Record, RecordStub
from ..models.query import GroupCount, Predicate, RelationshipTarget, UserQuery
from . import ModelId, labels
from .core import PartitionedDatabase
from .util import map_traversal_nodes_to_variables, match_relationship_clause

log = logging.getLogger(__file__)


RecordsByModel = Dict[str, Record]
AggregateCounts = Dict[str, int]


DEFAULT_RECORD_LIMIT: int = 25
DEFAULT_MAX_HOPS: int = 4


class EmbedLinked(str, Enum):
    """
    Record linking mode

    STUB = A "stub" of a record will be embedded consisting of a record ID
           and a title.

    FULL = The entire `Record` instance will be embedded.
    """

    STUB = "STUB"
    FULL = "FULL"


class QueryRunner:
    """
    A runner of user-defined queries.
    """

    def __init__(self, db: PartitionedDatabase, query: Optional[UserQuery] = None):
        self._db = db
        self._query = query

    def get_model_tx(
        self, tx: Union[Session, Transaction], model: Union[Model, ModelId, str]
    ) -> Optional[Model]:
        """
        [[TRANSACTIONAL]]

        Get a model.
        """
        return self._db.get_model_tx(tx, model)

    def resolve_model_alias(self, alias: str) -> Optional[Union[Model, ModelId, str]]:
        if not self._query:
            return None
        return self._query.aliases.get(alias)

    def get_models_tx(
        self,
        tx: Union[Session, Transaction],
        *additional_models: Union[Model, ModelId, str],
    ) -> Dict[str, Model]:
        """
        [[TRANSACTIONAL]]

        Get all models occurring in `query`, along with any additional models
        specified.
        """
        lookup = set(
            self.resolve_model_alias(m) or m
            for m in ((self._query and self._query.selection) or set())
        ) | set(cast(Tuple[str], additional_models))

        return {m.name: m for m in [self._db.get_model_tx(tx, m) for m in lookup]}

    def get_model_properties_tx(
        self,
        tx: Union[Session, Transaction],
        *additional_models: Union[Model, ModelId, str],
    ) -> Dict[str, List[ModelProperty]]:
        """
        [[TRANSACTIONAL]]

        Get all model properties occurring in `query`, along with any additional models
        specified.
        """
        return {
            m.name: self._db.get_properties_tx(tx, m)
            for m in self.get_models_tx(tx, *additional_models).values()
        }

    @property
    def query(self) -> Optional[UserQuery]:
        return self._query

    def run_tx(
        self,
        tx: Union[Session, Transaction],
        source_model: Union[Model, ModelId, str],
        max_hops: Optional[int] = 4,
        embed_linked: Optional[EmbedLinked] = None,
        limit: int = 25,
        offset: int = 0,
    ) -> Union[List[Record], List[Dict[str, Record]], List[Dict[str, int]]]:

        """
        [[TRANSACTIONAL]]

        Run a user-defined query.

        See `run()`
        """
        __next_rvar = 0

        def new_rvar() -> str:
            """
            Allocate a new record variable
            """
            nonlocal __next_rvar
            v = f"r{__next_rvar}"
            __next_rvar += 1
            return v

        def resolve_model(tx, m: Union[Model, ModelId, str]) -> Model:
            if isinstance(m, Model):
                return m
            else:
                model_: Optional[Model] = self._db.get_model_tx(tx, m)
                if model_ is not None:
                    return model_
            raise ModelNotFoundError(m)

        # Predicate conditions to include in the generated record query.
        # Note: # `Predicate` (and its subsets `RelationshipTarget`) will
        # result in a # relationship join being generated.
        predicates: List[Predicate] = []
        connections: List[RelationshipTarget] = []

        # The originating model for the query. This is starting node for
        # any generated query paths:
        source: Model = resolve_model(tx, source_model)

        # --- GROUPING --------------------------------------------------------

        # Fetch the group model if an aggregate operation like group counting
        # is specified:
        group_model: Optional[Model] = None
        group_field: Optional[str] = None
        counts: Dict[str, int] = {}

        if self._query and self._query.aggregation:
            assert isinstance(
                self._query.aggregation, GroupCount
            ), "Unknown aggregation type"
            group_count: GroupCount = self._query.aggregation

            if group_count.model:
                group_model = self._db.get_model_tx(tx, group_count.model)

            if group_count.field:
                group_field = group_count.field

            # If not model is specified, use the source model:
            group_model = group_model or source

        # This is True if counting mode is enabled:
        count_groups_mode: bool = group_model is not None or group_field is not None

        # ---------------------------------------------------------------------

        # Map Model ID -> Model:
        model_by_id: Dict[ModelId, Model] = {source.id: source}

        # The source is always considered the first record variable:
        rvars: Dict[ModelId, str] = {source.id: f"r{new_rvar()}"}

        # Models that will be selected ("projected") later:
        selection: List[Model] = [source]

        property_map: Dict[Model, Dict[str, ModelProperty]] = {
            source: {p.name: p for p in self._db.get_properties_tx(tx, source)}
        }

        if self._query is not None:

            # Gather predicates/connections and resolve their models by
            # name if required:
            for target in self._query.connections + cast(
                List[RelationshipTarget], self._query.filters
            ):

                m = resolve_model(tx, target.model)
                model_by_id[m.id] = m

                # Rewrite the predicate/connection to reference the model
                # object itself:
                t = target.with_model(m)

                if isinstance(t, Predicate):
                    predicates.append(t)
                elif isinstance(t, RelationshipTarget):
                    connections.append(t)

                # Assign a query variable to every *UNIQUE* model ID:
                if source.id != m.id:
                    m_id = str(m.id)
                    if m_id not in rvars:
                        rvars[m.id] = new_rvar()

                property_map[m] = {p.name: p for p in self._db.get_properties_tx(tx, m)}

            # For selection models, resolve each model name to an actual
            # `Model` and store its properties:

            if self._query.selection:
                for model_name in self._query.selection:
                    m = resolve_model(
                        tx, self._query.aliases.get(model_name, model_name)
                    )
                    m_id = str(m.id)
                    if m_id not in rvars:
                        rvars[m.id] = new_rvar()

                    # add the model to the selection list:
                    selection.append(m)

                    # store the selection model's properties:
                    property_map[m] = {
                        p.name: p for p in self._db.get_properties_tx(tx, m)
                    }

        # For each predicate/connection model that isn't `source`,
        # generate find the shortest undirected path:
        target_models: List[Model] = [
            cast(Model, t.model)
            for t in (cast(List[RelationshipTarget], predicates) + connections)
            if cast(Model, t.model).id != source.id
        ]

        if target_models:
            shortest_paths_cql = f"""
            MATCH ({labels.model("m")} {{ id: $source_id }})
                  -[{labels.in_dataset()}]->({labels.dataset("d")})
                  -[{labels.in_organization()}]->({labels.organization("o")})

            MATCH ({labels.model("n")})-[{labels.in_dataset()}]->(d)
            WHERE n.id IN $target_ids

            MATCH p = shortestPath((m)-[{labels.related_to()} *..{max_hops}]-(n))
            RETURN p
            """
            target_ids = [m.id for m in target_models]
            path_kwargs = dict(
                source_id=source.id,
                target_ids=target_ids,
                reserved_relationship_types=labels.RESERVED_SCHEMA_RELATIONSHIPS,
                organization_id=self._db.organization_id,
                dataset_id=self._db.dataset_id,
            )
            paths = [
                r["p"] for r in tx.run(shortest_paths_cql, **path_kwargs).records()
            ]

            relationships = list(chain.from_iterable(p.relationships for p in paths))

            if not relationships:
                raise InfeasibleQueryError(
                    source_model=str(source.id),
                    target_models=[m.name for m in target_models],
                )

            # A map of model IDs to query variable names
            # (record instances, e.g. "r0", "r1", etc.)
            rvars.update(
                map_traversal_nodes_to_variables(*relationships, start=len(rvars) + 1)
            )
        else:
            paths = []

        # If any intermediate models were found in a relationship, add them.
        # Also, generate model variables:
        mvars: Dict[ModelId, str] = {}

        for i, model_id in enumerate(rvars.keys()):
            mvars[model_id] = f"m{i}"
            if model_id not in model_by_id:
                model_by_id[model_id] = resolve_model(tx, model_id)

        conditions: List[str] = []

        # Generate MATCH "instance" check clauses for each model here:
        for model_id in rvars.keys():

            rvar: str = labels.record(rvars[model_id])
            mvar: str = labels.model(mvars[model_id])

            # For the source, check the dataset and organization as well:
            if source.id == model_id:
                cql = f"""
                MATCH ({rvar})
                      -[{labels.instance_of()}]->({mvar} {{ id: "{model_id}" }})
                      -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                      -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }}) // {model_by_id[model_id].name}
                """
            else:
                cql = f"""MATCH ({rvar})-[{labels.instance_of()}]->({mvar} {{ id: "{model_id}" }}) // {model_by_id[model_id].name}"""
            conditions.append(cql)

        # Add relationship clauses:
        conditions.extend(
            match_relationship_clause(
                {str(model_id): rvar for model_id, rvar in rvars.items()}, *paths
            )
        )

        # Add filtering WHERE conditions and parameters:
        clauses = [
            p.cql(i, model=rvars[p.model.id])  # type: ignore
            for i, p in enumerate(predicates)  # type: ignore
        ]

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self._db.organization_id, dataset_id=self._db.dataset_id
        )

        for i, predicate in enumerate(predicates):
            kwargs.update(predicate.parameters(i))

        # use these for the actual selection of models below:
        select_models: Set[Model] = set(
            [cast(Model, c.model) for c in connections]
        ) | set(selection)

        if embed_linked:
            cql = (
                "\n".join(c.strip() for c in conditions)
                + (f"\nWHERE {' AND '.join(clauses)}" if clauses else "")
                + f"\nMATCH ({rvars[source.id]})-[{labels.created_by('created')}]->({labels.user('created_by')})"
                + f"\nMATCH ({rvars[source.id]})-[{labels.updated_by('updated')}]->({labels.user('updated_by')})"
                + f"\nOPTIONAL MATCH ({mvars[source.id]})-[{labels.related_to('mr')} {{ one_to_many: false }}]->({labels.model('m_other')})"
                + f"\nOPTIONAL MATCH ({rvars[source.id]})-[rr]->({labels.record('r_related')})"
                + f"\nWHERE mr.type = TYPE(rr)"
            )
            cql += f"\nRETURN DISTINCT "
            for m in select_models:
                cql += f"\n       {rvars[m.id]}, "
            cql += (
                f"\n       created_by.node_id                 AS created_by,"
                + f"\n       updated_by.node_id               AS updated_by,"
                + f"\n       created.at                       AS created_at,"
                + f"\n       updated.at                       AS updated_at,"
                + f"\n       COLLECT(DISTINCT TYPE(rr))       AS relations,"
                + f"\n       COLLECT(DISTINCT r_related)      AS neighbors"
            )
        else:
            cql = (
                "\n".join(c.strip() for c in conditions)
                + (f"\nWHERE {' AND '.join(clauses)}" if clauses else "")
                + f"\nMATCH ({rvars[source.id]})-[{labels.created_by('created')}]->({labels.user('created_by')})"
                + f"\nMATCH ({rvars[source.id]})-[{labels.updated_by('updated')}]->({labels.user('updated_by')})"
            )
            cql += f"\nRETURN DISTINCT "
            for m in select_models:
                cql += f"\n       {rvars[m.id]}, "
            cql += (
                f"\n       created_by.node_id                 AS created_by,"
                + f"\n       updated_by.node_id               AS updated_by,"
                + f"\n       created.at                       AS created_at,"
                + f"\n       updated.at                       AS updated_at,"
                + f"\n       []                               AS relations,"
                + f"\n       []                               AS neighbors"
            )

        if self._query and self._query.ordering is not None:
            cql += f"\nORDER BY "
            if self._query.ordering.is_created_at:
                cql += f"created_at"
            elif self._query.ordering.is_updated_at:
                cql += f"updated_at"
            else:
                cql += f"{rvars[source.id]}[$order_by_field]"
                kwargs["order_by_field"] = self._query.ordering.field
            cql += " " + (
                "ASCENDING" if self._query.ordering.ascending else "DESCENDING"
            )

        # Limit/offset should not be applied when grouping

        if offset is not None and not count_groups_mode:
            cql += f"\nSKIP {offset}"

        if limit is not None and not count_groups_mode:
            cql += f"\nLIMIT {limit}"

        log.debug(cql)

        # Handle results
        # ---------------------------------------------------------------------
        nodes = tx.run(cql, **kwargs).records()

        # TODO: consider turning this function into a generator/use yield
        # instead of building up a `List[Record]` / `Dict[str, List[Record]]`.
        single_records: List[Record] = []
        multiple_records: List[Dict[str, Record]] = []

        # If `source` is the only model selected, we just want to return
        # a list of records, as opposed to a list of dicts or records,
        # indexed by model name.
        single_model: bool = len(select_models) == 1

        for node in nodes:

            # If multiple models are selected, construct a dict, where
            # the key of the dict is the name of the model, and the value is
            # the associated recor value:
            records_per_model: Dict[str, Record] = {}

            # For every selected model:
            for s in select_models:

                s = cast(Model, s)

                # Convert the result to a record:
                record = Record.from_node(
                    node[rvars[s.id]],
                    property_map=property_map[s],
                    created_by=node["created_by"],
                    updated_by=node["updated_by"],
                    created_at=node["created_at"],
                    updated_at=node["updated_at"],
                    fill_missing=True,
                )

                # Embed the record stubs to the record:
                if embed_linked:
                    for (relation, neighbor) in zip(
                        node["relations"], node["neighbors"]
                    ):
                        record.embed(relation, RecordStub.from_node(neighbor))

                if count_groups_mode:
                    r, key = record.to_dict()["values"], None
                    if group_model:
                        if group_model == s and group_field is not None:
                            key = r.get(group_field)
                    else:
                        key = r.get(group_field)
                    # Ignore null keys:
                    if key is not None:
                        counts[key] = counts.setdefault(key, 0) + 1
                else:
                    if single_model:
                        single_records.append(record)
                    else:
                        output_as_selected = True

                        # Additionally, link the record to any aliases of the
                        # selected model:
                        for alias_name, aliased_model in (
                            self._query and self._query.aliases.items()
                        ) or []:
                            if aliased_model == s.name:
                                output_as_selected = False
                                records_per_model[alias_name] = record

                        if output_as_selected:
                            # If an alias was not provided for the selected model
                            # link the record to the selected model:
                            records_per_model[s.name] = record

            if not single_model:
                multiple_records.append(records_per_model)

        if count_groups_mode:
            return [counts]

        return single_records if single_model else multiple_records

    def run(
        self,
        source_model: Union[Model, ModelId, str],
        max_hops: Optional[int] = DEFAULT_MAX_HOPS,
        embed_linked: Optional[EmbedLinked] = None,
        limit: int = DEFAULT_RECORD_LIMIT,
        offset: int = 0,
    ) -> Union[List[Record], List[RecordsByModel], List[AggregateCounts]]:

        """
        Run a user-defined query.

        Process
        -------

        Performing the query is a multistep process:

        1) Find the shortest path in the model-schema between the source
           and filter models.

        2) Declare record variables that are linked to models in the
           user's dataset using MATCH patterns.

        3) Add MATCH patterns for relationships from records appearing in
           step (2) that reference models specified for filtering conditions
           to apply to the query.

        3) Apply filtering conditions to the query.
        """
        with self._db.transaction() as tx:
            return self.run_tx(
                tx=tx,
                source_model=source_model,
                max_hops=max_hops,
                embed_linked=embed_linked,
                limit=limit,
                offset=offset,
            )
