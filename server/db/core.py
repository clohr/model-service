import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, cast
from uuid import UUID, uuid4

from neo4j import GraphDatabase, Session, Transaction, basic_auth  # type: ignore
from typing_extensions import Protocol

from ..config import Config
from ..errors import (
    CannotSortRecordsError,
    DuplicateModelRelationshipError,
    ExceededTimeLimitError,
    ImmutablePropertyError,
    LegacyModelRelationshipNotFoundError,
    ModelNotFoundError,
    ModelPropertyNotFoundError,
    ModelRelationshipNotFoundError,
    ModelServiceError,
    MultiplePropertyTitleError,
    OperationError,
    PackageProxyNotFoundError,
    RecordNotFoundError,
    RecordRelationshipNotFoundError,
    ReservedModelNameError,
)
from ..models import (
    CreateRecordRelationship,
    DatasetDeletionCounts,
    DatasetDeletionSummary,
    DatasetId,
    DatasetNodeId,
    GraphSchemaStructure,
    GraphValue,
    Model,
    ModelId,
    ModelProperty,
    ModelPropertyId,
    ModelRelationship,
    ModelRelationshipId,
    ModelTopology,
    NextPageCursor,
    OrganizationId,
    OrganizationNodeId,
    PackageId,
    PackageNodeId,
    PackageProxy,
    PackageProxyId,
    PagedResult,
    ProxyRelationshipCount,
    Record,
    RecordId,
    RecordRelationship,
    RecordRelationshipId,
    RecordStub,
    RecordSummary,
    RelationshipName,
    TopologySummary,
    UserNodeId,
    get_model_id,
    get_model_relationship_id,
    get_package_proxy_id,
    get_record_id,
    get_record_relationship_id,
    get_relationship_type,
    is_model_relationship_id,
    is_reserved_model_name,
)
from ..models.datatypes import deserialize
from ..models.legacy import CreateModelRelationship, ModelRelationshipStub
from ..models.models import OrderBy as ModelOrderBy
from ..models.models import OrderByField as ModelOrderByField
from ..models.models import OrderByRelationship as ModelOrderByRelationship
from ..models.validate import (
    validate_default_value,
    validate_model_name,
    validate_property_name,
    validate_records,
    validate_relationship_name,
)
from . import labels, metrics
from .assert_ import AssertionHelper
from .util import match_clause

log = logging.getLogger(__file__)


# Maximum batch duration processing time, in milliseconds:
MAX_RUN_DURATION: int = 5000

# Maximum batch size for record processing:
MAX_BATCH_SIZE: int = 10000


"""
==============================================================================
Required Indices

The following index definitions should exists in the database:

- CREATE INDEX ON :Record(`@sort_key`)

  Needed for `get_all_records` to return records in sorted order

==============================================================================
"""


class Transactional(Protocol):
    """
    Classes that support database transactions, etc. must implement this
    protocol.
    """

    def get_driver(self) -> GraphDatabase:
        raise NotImplementedError

    @contextmanager
    def transaction(self):
        """
        Start a new transaction, yielding the transaction object to the scope
        of the `with` block.

        - If no rollbacks or commits occur in the scope of the `with` block
          on `tx`, the transaction is still considered active and will be
          committed upon exiting the block scope.

        - If an unhandled exception is raised in the scope of the `with` block
          and the transaction is still open, a rollback will be issued and
          the originating exception will be re-raised.

        - If an explicit rollback (or commit) occurs in the scope of the
          `with` block on `tx`, the transaction is considered closed from the
          perspective of Neo4j. Do further action will be taken.
        """
        with self.session() as s:
            tx = s.begin_transaction()
            try:
                yield tx
            except Exception as e:
                if not tx.closed():
                    tx.rollback()
                raise e
            else:
                if not tx.closed():
                    tx.commit()

    @contextmanager
    def session(self):
        with self.get_driver().session() as s:
            yield s

    def execute_single(self, cmd: str, **kwargs):
        """
        Execute a single statement with autocommit.
        """
        with self.session() as s:
            return s.run(cmd, **kwargs)


class Database(Transactional):
    def __init__(
        self, uri: str, user: str, password: str, max_connection_lifetime: int
    ):
        """
        Instantiate a new database connection.
        """
        self.driver = GraphDatabase.driver(
            uri=uri,
            auth=basic_auth(user, password),
            max_connection_lifetime=max_connection_lifetime,
        )

    def get_driver(self) -> GraphDatabase:
        return self.driver

    @classmethod
    def from_config(cls, config: Config) -> "Database":
        """
        Given a configuration, create a new database connection.
        """
        return cls(
            config.neo4j_url,
            config.neo4j_user,
            config.neo4j_password,
            config.neo4j_max_connection_lifetime,
        )

    @classmethod
    def from_server(cls) -> "Database":
        """
        Get the database connection from the Flask application scope.
        """
        from flask import current_app

        return current_app.config["db"]

    def execute(self, cmd, **kwargs):
        with self.session() as s:
            return s.run(cmd, **kwargs)

    def initialize_organization(
        self,
        tx: Union[Session, Transaction],
        organization_id: OrganizationId,
        organization_node_id: Optional[OrganizationNodeId] = None,
    ) -> Tuple[OrganizationId, Optional[OrganizationNodeId]]:
        """
        Create the nodes for the given organization.

        Note: `tx` can be any object that supplies a `run()` method.
        """

        cql = f"""
        MERGE ({labels.organization("o")} {{ id: $organization_id }})
            ON CREATE SET o.id      = toInteger($organization_id),
                          o.node_id = $organization_node_id
            ON MATCH  SET o.node_id = COALESCE($organization_node_id, o.node_id)
        RETURN o.id       AS organization_id,
               o.node_id  AS organization_node_id
        """
        ids = tx.run(
            cql,
            organization_id=organization_id,
            organization_node_id=organization_node_id,
        ).single()

        return (
            OrganizationId(ids["organization_id"]),
            OrganizationNodeId(ids["organization_node_id"])
            if ids["organization_node_id"]
            else None,
        )

    def initialize_organization_and_dataset(
        self,
        tx: Union[Session, Transaction],
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        organization_node_id: Optional[OrganizationNodeId] = None,
        dataset_node_id: Optional[DatasetNodeId] = None,
    ) -> Tuple[
        OrganizationId, DatasetId, Optional[OrganizationNodeId], Optional[DatasetNodeId]
    ]:
        """
        Create the nodes for the given organization and dataset.

        Note: `tx` can be any object that supplies a `run()` method.
        """

        cql = f"""
        MERGE ({labels.organization("o")} {{ id: $organization_id }})
            ON CREATE SET o.id      = toInteger($organization_id),
                          o.node_id = $organization_node_id
            ON MATCH  SET o.node_id = COALESCE($organization_node_id, o.node_id)
        MERGE (o)<-[{labels.in_organization()}]-({labels.dataset("d")} {{ id: $dataset_id }})
            ON CREATE SET d.id                = toInteger($dataset_id),
                          d.node_id           = $dataset_node_id
            ON MATCH  SET d.node_id = COALESCE($dataset_node_id, d.node_id)
        RETURN o.id       AS organization_id,
               d.id       AS dataset_id,
               o.node_id  AS organization_node_id,
               d.node_id  AS dataset_node_id
        """
        ids = tx.run(
            cql,
            organization_id=organization_id,
            dataset_id=dataset_id,
            organization_node_id=organization_node_id,
            dataset_node_id=dataset_node_id,
        ).single()

        return (
            OrganizationId(ids["organization_id"]),
            DatasetId(ids["dataset_id"]),
            OrganizationNodeId(ids["organization_node_id"])
            if ids["organization_node_id"]
            else None,
            DatasetNodeId(ids["dataset_node_id"]) if ids["dataset_node_id"] else None,
        )

    def get_dataset_id(self, dataset_node_id: DatasetNodeId) -> Optional[DatasetId]:
        """
        Resolve a dataset by node ID to an integer ID.
        """

        cql = f"""
        MATCH  ({labels.dataset("d")} {{ node_id: $dataset_node_id }})
        RETURN d.id AS id
        LIMIT 1
        """
        result = self.execute(cql, dataset_node_id=dataset_node_id).single()

        if not result:
            return None

        return DatasetId(result["id"])

    def get_dataset_node_id(self, dataset_id: DatasetId) -> Optional[DatasetNodeId]:
        """
        Resolve a dataset ID to a node ID.
        """

        cql = f"""
        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
        RETURN d.node_id AS node_id
        LIMIT 1
        """
        result = self.execute(cql, dataset_id=dataset_id).single()

        if not result:
            return None

        return DatasetNodeId(result["node_id"])

    def get_dataset_ids(self, organization_id: OrganizationId) -> Set[DatasetId]:
        """
        Return a listing of all datasets.
        """
        cql = f"""
        MATCH ({labels.dataset("d")})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        RETURN DISTINCT d.id AS id
        """
        result = self.execute(cql, organization_id=organization_id).records()

        return {DatasetId(node["id"]) for node in result}

    def count_child_nodes(
        self, organization_id: OrganizationId, dataset_id: DatasetId
    ) -> int:
        """
        Count the number of dependent nodes in a dataset.
        """
        cql = f"""
        MATCH (c)
              -[*1..3]->({labels.dataset("d")}  {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        RETURN COUNT(c) AS child_nodes
        """
        return self.execute(
            cql, organization_id=organization_id, dataset_id=dataset_id
        ).single()["child_nodes"]

    def get_one(self) -> int:
        """
        Returns 1.
        """

        cql = f"""
        RETURN 1 as health
        """

        result = self.execute(cql).single()

        return int(result["health"])


class PartitionedDatabase(Transactional):
    @classmethod
    def get_from_env(
        cls,
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        user_id: UserNodeId,
        organization_node_id: Optional[OrganizationNodeId] = None,
        dataset_node_id: Optional[DatasetNodeId] = None,
    ) -> "PartitionedDatabase":
        """
        Get the database connection using the environment for looking up
        configuration details.
        """
        config = Config()
        return cls(
            db=Database.from_config(config),
            organization_id=organization_id,
            dataset_id=dataset_id,
            user_id=user_id,
            organization_node_id=organization_node_id,
            dataset_node_id=dataset_node_id,
        )

    @classmethod
    def get_from_server(
        cls,
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        user_id: UserNodeId,
        organization_node_id: Optional[OrganizationNodeId] = None,
        dataset_node_id: Optional[DatasetNodeId] = None,
    ) -> "PartitionedDatabase":
        """
        Get the database connection from the Flask application scope for
        configuration details.
        """
        from flask import current_app

        return cls(
            db=current_app.config["db"],
            organization_id=organization_id,
            dataset_id=dataset_id,
            user_id=user_id,
            organization_node_id=organization_node_id,
            dataset_node_id=dataset_node_id,
        )

    def __init__(
        self,
        db: Database,
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        user_id: UserNodeId,
        organization_node_id: Optional[OrganizationNodeId] = None,
        dataset_node_id: Optional[DatasetNodeId] = None,
    ):
        self.db = db
        self.organization_id = organization_id
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.organization_node_id = organization_node_id
        self.dataset_node_id = dataset_node_id
        self.assert_ = AssertionHelper(self)

    def get_driver(self) -> GraphDatabase:
        return self.db.get_driver()

    # -------------------------------------------------------------------------
    # INTERNAL API
    # -------------------------------------------------------------------------

    @property
    def _unrestricted(self) -> Database:
        """
        Access the underlying unrestricted view of the database.
        """
        return self.db

    def _get_model_from_record_tx(
        self, tx: Union[Session, Transaction], record: Union[Record, RecordId]
    ) -> Optional[Model]:
        """
        Get a model from a record in the context of a transaction by following
        the record's "INSTANCE_OF" relationship to the corresponding `Model` node.

        Note: `tx` can be any object that supplies a `run()` method.
        """

        record_id = get_record_id(record)
        cql = f"""
        MATCH  ({labels.record("r")} {{ `@id`: $id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        MATCH (m)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (m)-[{labels.updated_by("updated")}]->({labels.user("u")})

        RETURN m,
               size(()-[{labels.instance_of()}]->(m)) AS count,
               c.node_id                              AS created_by,
               u.node_id                              AS updated_by,
               created.at                             AS created_at,
               updated.at                             AS updated_at
        """
        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            id=str(record_id),
        ).single()

        if not node:
            return None

        return cast(
            Model,
            Model.from_node(
                **node["m"],
                count=node["count"],
                created_by=node["created_by"],
                updated_by=node["updated_by"],
                created_at=node["created_at"],
                updated_at=node["updated_at"],
            ),
        )

    def _count_outgoing_records_tx(
        self,
        tx: Union[Session, Transaction],
        from_record: Union[Record, RecordId],
        model_relationship_id: ModelRelationshipId,
    ) -> int:
        """
        Counts the number of outgoing relationships from a record with a
        `model_id` matching the ID of a model relationship.

        Note: `tx` can be any object that supplies a `run()` method.
        """

        cql = f"""
        MATCH ({labels.record("r1")} {{ `@id`: $from_record }})
              -[r {{ model_relationship_id: $model_relationship_id }}]->({labels.record("r2")})
        RETURN COUNT(r) AS count
        """
        from_record_id = get_record_id(from_record)

        return (
            tx.run(
                cql,
                from_record=str(from_record_id),
                model_relationship_id=str(model_relationship_id),
            )
            .single()
            .value()
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def create_model_tx(
        self,
        tx: Union[Session, Transaction],
        name: str,
        display_name: str,
        description: str = "",
        template_id: Optional[UUID] = None,
    ) -> Model:
        """
        [[TRANSACTIONAL]]

        Create a new model.

        Raises
        ------
        OperationError
        """
        if is_reserved_model_name(name):
            raise OperationError(
                "couldn't create model", cause=ReservedModelNameError(name)
            )

        # Each model contains a non-decreasing "count" of the number of
        # records attached to it. Every time a record of the model's type is
        # created, the counter is incremented, and the new count is assigned
        # to the record's `@sort_key` property. The purpose of the sort key is
        # to enable faster pagination  when retrieving a large number of
        # records
        cql = f"""
        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        MERGE ({labels.user("u")} {{ node_id: $user_id }})
        CREATE ({labels.model("m")} {{
          `@max_sort_key`: 0,
          id: randomUUID(),
          name: $name,
          display_name: $display_name,
          description: $description,
          template_id: $template_id
        }})
        CREATE (m)-[{labels.in_dataset()}]->(d)
        CREATE (m)-[{labels.created_by("created")} {{ at: datetime() }}]->(u)
        CREATE (m)-[{labels.updated_by("updated")} {{ at: datetime() }}]->(u)
        RETURN m,
               created.at AS created_at,
               updated.at AS updated_at
        """

        self.db.initialize_organization_and_dataset(
            tx,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            organization_node_id=self.organization_node_id,
            dataset_node_id=self.dataset_node_id,
        )

        try:
            validate_model_name(name)
            self.assert_.model_name_does_not_exist(tx, name)
        except ModelServiceError as e:
            raise OperationError("couldn't create model", cause=e)

        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            name=name,
            display_name=display_name,
            description=description,
            template_id=template_id,
        ).single()

        return cast(
            Model,
            Model.from_node(
                **node["m"],
                count=0,
                created_at=node["created_at"],
                updated_at=node["updated_at"],
                created_by=self.user_id,
                updated_by=self.user_id,
            ),
        )

    def create_model(
        self,
        name: str,
        display_name: str,
        description: str = "",
        template_id: Optional[UUID] = None,
    ) -> Model:
        """
        Create a new model.

        Raises
        ------
        OperationError
        """
        with self.transaction() as tx:
            # Transaction required:
            return self.create_model_tx(
                tx=tx,
                name=name,
                display_name=display_name,
                description=description,
                template_id=template_id,
            )

    def update_model_tx(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, ModelId, str],
        name: str,
        display_name: str,
        description: str,
        template_id: Optional[UUID] = None,
    ) -> Model:
        """
        [[TRANSACTIONAL]]

        Update a given model.

        Raises
        ------
        - ModelNotFoundError
        - OperationError
        """

        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        WITH m, (m.name <> $name) AS name_changed
        SET m += {{
           name: $name,
           display_name: $display_name,
           description: $description,
           template_id: $template_id
        }}

        WITH m, name_changed
        MATCH (m)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (m)-[{labels.updated_by("last_updated")}]->({labels.user()})

        DELETE last_updated
        MERGE ({labels.user("u")} {{ node_id: $user_id }})
        CREATE (m)-[{labels.updated_by("updated")} {{ at: datetime() }}]->(u)

        RETURN m,
               name_changed,
               size(()-[{labels.instance_of()}]->(m)) AS count,
               c.node_id                              AS created_by,
               u.node_id                              AS updated_by,
               created.at                             AS created_at,
               updated.at                             AS updated_at
        """

        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            model_id_or_name=model_id_or_name,
            name=name,
            display_name=display_name,
            description=description,
            template_id=template_id,
        ).single()

        if node is None:
            raise OperationError(
                "couldn't update model", cause=ModelNotFoundError(model_id_or_name)
            )

        if node["name_changed"]:
            try:
                validate_model_name(name)
                self.assert_.model_name_is_unique(tx, name)
            except ModelServiceError as e:  # noqa F841
                raise OperationError("couldn't update model", cause=e)

        return cast(
            Model,
            Model.from_node(
                **node["m"],
                count=node["count"],
                created_by=node["created_by"],
                updated_by=node["updated_by"],
                created_at=node["created_at"],
                updated_at=node["updated_at"],
            ),
        )

    def update_model(
        self,
        model_id_or_name: Union[Model, ModelId, str],
        name: str,
        display_name: str,
        description: str,
        template_id: Optional[UUID] = None,
    ) -> Model:
        """
        Update a given model.

        Raises
        ------
        - ModelNotFoundError
        - OperationError
        """
        with self.transaction() as tx:
            # Multiple operations in `update_model_tx()` - tx required
            return self.update_model_tx(
                tx=tx,
                model_id_or_name=model_id_or_name,
                name=name,
                display_name=display_name,
                description=description,
                template_id=template_id,
            )

    def get_model_tx(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, ModelId, str],
    ) -> Model:
        """
        [[TRANSACTIONAL]]

        Get a model by its ID, given a session or transaction object supporting
        the `run()` method.

        Using `size()` instead of `count()` in Cypher results in a constant-time
        lookup when computing the size of the set of outgoing relationships of a
        given relationship type. This is possible because the `IS A`
        relationship is only allowed for Pennsieve-defined relationships, so
        `MATCH (n:Model)-[:`@IS_A`]->(m) RETURN COUNT(m)` gives the same results as
        `RETURN size(()-[:`@IS_A`])->(m)) AS count`.

        See https://neo4j.com/developer/kb/how-do-i-improve-the-performance-of-counting-number-of-relationships-on-a-node/

        Raises
        ------
        ModelNotFoundError
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_id=self.dataset_id
        )
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        MATCH (m)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (m)-[{labels.updated_by("updated")}]->({labels.user("u")})

        RETURN m,
               size(()-[{labels.instance_of()}]->(m)) AS count,
               c.node_id                              AS created_by,
               u.node_id                              AS updated_by,
               created.at                             AS created_at,
               updated.at                             AS updated_at
        """
        node = tx.run(cql, **kwargs).single()
        if node is None:
            raise ModelNotFoundError(str(model_id_or_name))

        return cast(
            Model,
            Model.from_node(
                **node["m"],
                count=node["count"],
                created_by=node["created_by"],
                updated_by=node["updated_by"],
                created_at=node["created_at"],
                updated_at=node["updated_at"],
            ),
        )

    def get_model(
        self, model_id_or_name: Union[Model, ModelId, str]
    ) -> Optional[Model]:
        """
        Get a model by its ID.

        Raises
        ------
        ModelNotFoundError
        """
        with self.session() as s:
            # Single lookup - session usage OK with get_model_tx()
            return self.get_model_tx(tx=s, model_id_or_name=model_id_or_name)

    def get_model_of_record(self, record: Union[Record, RecordId]) -> Optional[Model]:
        """
        Get the corresponding model of a record.
        """
        with self.transaction() as tx:
            # Multiple operations in `_get_model_from_record_tx()` - tx needed:
            return self._get_model_from_record_tx(tx, record)

    def get_models_tx(self, tx: Union[Session, Transaction]) -> List[Model]:
        """
        [[TRANSACTIONAL]]

        Get all models in the database for the current organization and
        dataset.
        """

        cql = f"""
        MATCH  ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        MATCH (m)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (m)-[{labels.updated_by("updated")}]->({labels.user("u")})

        RETURN m,
               size(()-[{labels.instance_of()}]->(m)) AS count,
               c.node_id                              AS created_by,
               u.node_id                              AS updated_by,
               created.at                             AS created_at,
               updated.at                             AS updated_at
        """
        nodes = tx.run(
            cql, organization_id=self.organization_id, dataset_id=self.dataset_id
        ).records()

        return [
            cast(
                Model,
                Model.from_node(
                    **node["m"],
                    count=node["count"],
                    created_by=node["created_by"],
                    updated_by=node["updated_by"],
                    created_at=node["created_at"],
                    updated_at=node["updated_at"],
                ),
            )
            for node in nodes
        ]

    def get_models(self) -> List[Model]:
        """
        Get all models in the database for the current organization and
        dataset.
        """
        with self.session() as s:
            # Session OK - `get_models_tx()` only performs 1 operation:
            return self.get_models_tx(tx=s)

    def delete_model_tx(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, ModelId, str],
    ) -> Model:
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_id=self.dataset_id
        )
        """
        [[TRANSACTIONAL]]

        Delete a model and all the model's properties from the graph.

        Model deletion is only allowed if no records exist for the model.

        Raises
        ------
        OperationError
        """
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        OPTIONAL MATCH (m)-[{labels.has_property()}]->({labels.model_property("p")})

        DETACH DELETE p
        DETACH DELETE m
        """
        model = self.get_model_tx(tx, model_id_or_name)
        if model is None:
            raise OperationError(
                "couldn't delete model", cause=ModelNotFoundError(model_id_or_name)
            )

        self.assert_.model_has_no_records(tx, model_id_or_name)

        tx.run(cql, **kwargs).single()

        return model

    def delete_model(self, model_id_or_name: Union[Model, ModelId, str]) -> Model:
        """
        Delete a model and all the model's properties from the graph.

        Model deletion is only allowed if no records exist for the model.

        Raises
        ------
        OperationError
        """
        with self.transaction() as tx:
            # Transaction required for `delete_model_tx()`
            return self.delete_model_tx(tx=tx, model_id_or_name=model_id_or_name)

    def update_properties_tx(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, ModelId, str],
        *properties: ModelProperty,
    ) -> List[Tuple[ModelProperty, bool]]:
        """
        Add or update properties of a model.

        Raises
        ------
        - ImmutablePropertyError
        - MultiplePropertyTitleError
        """
        model: Model = self.get_model_tx(tx, model_id_or_name)

        for prop in properties:
            validate_property_name(prop.name)
            validate_default_value(prop)

        # INVARIANT: the number of True `model_title` properties must not be
        # greater than 1:
        model_titles = [p.name for p in properties if p.model_title]

        if len(model_titles) > 1:
            raise OperationError(
                "couldn't update model properties",
                cause=MultiplePropertyTitleError(str(model), model_titles),
            )

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            # Update the index of each property to match the order the property
            # list was passed in. This allows for consistent ordering on the
            # frontend.
            # The data type must by stringified - Neo4j does not allow nested maps
            properties=[
                replace(p, index=index).to_dict_with_string_datatype(camel_case=False)
                for index, p in enumerate(properties)
            ],
        )

        # https://neo4j.com/developer/kb/updating-a-node-but-returning-its-state-from-before-the-update/
        #
        # Note: `id: COALESCE(node.id, 0) is used to make sure an ID is defined
        # for matching when using MERGE. If node.id is null, the query will fail
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_ident", model, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        UNWIND $properties AS node
        MERGE ({labels.model_property("p")} {{ id: COALESCE(node.id, 0) }})
            ON CREATE SET p.id            = COALESCE(node.id, randomUUID()),
                          p.name          = trim(node.name),
                          p.display_name  = node.display_name,
                          p.data_type     = node.data_type,
                          p.description   = node.description,
                          p.index         = node.index,
                          p.locked        = node.locked,
                          p.default       = node.default,
                          p.default_value = node.default_value,
                          p.model_title   = node.model_title,
                          p.required      = node.required,
                          p.`@name`       = trim(node.name),
                          p.`@data_type`  = node.data_type,
                          p.`@created`    = true
            ON MATCH SET  p.display_name  = node.display_name,
                          p.description   = node.description,
                          p.index         = node.index,
                          p.locked        = node.locked,
                          p.default       = node.default,
                          p.default_value = node.default_value,
                          p.model_title   = node.model_title,
                          p.required      = node.required,
                          // note - ORDER MATTERS - replacing with node.data_type first results in values being the same
                          p.`@data_type`  = p.data_type,
                          p.data_type     = node.data_type,
                          p.`@name`       = trim(node.name),
                          p.`@created`    = false
        MERGE (m)-[{labels.has_property()}]->(p)

        // -------- add created/updated provenance --------------------

        MERGE ({labels.user("u")} {{ node_id: $user_id }})

        WITH p, u,
             p.`@created` AS node_created,
             datetime()   AS now

        OPTIONAL MATCH (p)-[{labels.updated_by("last_updated")}]->({labels.user()})
        DELETE last_updated
        CREATE (p)-[{labels.updated_by()} {{ at: now }}]->(u)

        FOREACH (p IN CASE WHEN node_created THEN [p] ELSE [] END |
            CREATE (p)-[{labels.created_by()} {{ at: now }}]->(u))

        WITH p

        MATCH (p)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (p)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

        // ------------------------------------------------------------

        WITH   p, updated_by, created_by, created, updated,
               p.`@created`   AS node_created,
               p.`@name`      AS new_name,
               p.`@data_type` AS old_data_type

        REMOVE p.`@created`,
               p.`@name`,
               p.`@data_type`

        RETURN p,
               created_by.node_id AS created_by,
               updated_by.node_id AS updated_by,
               created.at         AS created_at,
               updated.at         AS updated_at,
               node_created,
               old_data_type,
               new_name <> p.name           AS name_changed,     // immutable
               old_data_type <> p.data_type AS data_type_changed // immutable

        ORDER BY p.index
        """

        collected: List[Tuple[ModelProperty, bool]] = []

        nodes = tx.run(cql, **kwargs).records()

        if nodes is None:
            raise OperationError(
                "couldn't update model properties", cause=ModelNotFoundError(str(model))
            )

        # Check that no immutable properties have been changed:
        for node in nodes:
            for immutable_property in ModelProperty.IMMUTABLE:
                if node[f"{immutable_property}_changed"]:
                    if immutable_property == "data_type":
                        old_data_type = deserialize(node["old_data_type"])
                        new_data_type = deserialize(node["p"]["data_type"])
                        assert old_data_type is not None and new_data_type is not None

                        if not old_data_type.only_unit_changed(new_data_type):
                            raise OperationError(
                                "couldn't update model properties",
                                cause=ImmutablePropertyError(
                                    str(model), immutable_property
                                ),
                            )
                    else:
                        raise OperationError(
                            "couldn't update model properties",
                            cause=ImmutablePropertyError(
                                str(model), immutable_property
                            ),
                        )

            collected.append(
                (
                    ModelProperty(
                        created_by=node["created_by"],
                        updated_by=node["updated_by"],
                        created_at=node["created_at"],
                        updated_at=node["updated_at"],
                        **node["p"],
                    ),
                    node["node_created"],
                )
            )

        # Check that all property names and display names are unique, and that
        # at most 1 property can have `model_title=True`.
        try:
            self.assert_.single_model_title(tx, model)
            self.assert_.unique_property_names(tx, model)
            self.assert_.unique_property_display_names(tx, model)
        except Exception as e:
            raise OperationError("couldn't update model properties", cause=e)

        return collected

    def update_properties(
        self, model: Union[Model, ModelId, str], *properties: ModelProperty
    ) -> List[ModelProperty]:
        """
        Add or update properties of a model.

        Raises
        ------
        - ImmutablePropertyError
        - MultiplePropertyTitleError
        """
        with self.transaction() as tx:
            return [p for p, _ in self.update_properties_tx(tx, model, *properties)]

    def get_properties_tx(
        self, tx: Union[Session, Transaction], model: Union[Model, ModelId, str]
    ) -> List[ModelProperty]:
        """
        [[TRANSACTIONAL]]

        Get all properties of a model in the context of a transaction.

        Note: `tx` can be any object that supplies a `run()` method.
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_id=self.dataset_id
        )
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_ident", model, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        MATCH (m)-[{labels.has_property()}]->({labels.model_property("p")})

        MATCH (p)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (p)-[{labels.updated_by("updated")}]->({labels.user("u")})

        RETURN p,
               c.node_id  AS created_by,
               u.node_id  AS updated_by,
               created.at AS created_at,
               updated.at AS updated_at

        ORDER BY p.index
        """

        nodes = tx.run(cql, **kwargs).records()

        return [
            cast(
                ModelProperty,
                ModelProperty.from_node(
                    **node["p"],
                    created_by=node["created_by"],
                    updated_by=node["updated_by"],
                    created_at=node["created_at"],
                    updated_at=node["updated_at"],
                ),
            )
            for node in nodes
        ]

    def get_properties(self, model: Union[Model, ModelId, str]) -> List[ModelProperty]:
        """
        Get all properties of a model.
        """
        with self.session() as s:
            return self.get_properties_tx(tx=s, model=model)

    def get_property_counts_tx(
        self, tx: Union[Session, Transaction], model_ids: List[ModelId]
    ) -> Dict[ModelId, int]:
        """
        [[TRANSACTIONAL]]

        Count the number of properties for a model in the context of a transaction.

        Note: `tx` can be any object that supplies a `run()` method.
        """
        cql = f"""
        UNWIND $model_ids AS model_id

        MATCH ({labels.model("m")} {{ id: model_id }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        RETURN model_id, size((m)-[{labels.has_property()}]->()) AS property_count
        """
        nodes = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            model_ids=model_ids,
        ).records()

        return {node["model_id"]: node["property_count"] for node in nodes}

    def get_property_counts(self, model_ids: List[ModelId]) -> Dict[ModelId, int]:
        """
        Count the number of properties for several models
        """
        with self.session() as session:
            return self.get_property_counts_tx(tx=session, model_ids=model_ids)

    def delete_property(
        self,
        model: Union[Model, ModelId, str],
        property_: Union[ModelProperty, ModelPropertyId, str],
    ) -> Optional[ModelProperty]:
        with self.transaction() as tx:
            return self.delete_property_tx(tx, model, property_)

    def delete_property_tx(
        self,
        tx: Union[Session, Transaction],
        model: Union[Model, ModelId, str],
        property_: Union[ModelProperty, ModelPropertyId, str],
    ) -> Optional[ModelProperty]:
        """
        Delete a property from a model.

        Note: Property deletion will be stopped if a connected `Record` node
        contains a property with the same name as that of the specified
        property.
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_id=self.dataset_id
        )
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("model_ident", model, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        MATCH (m)-[{labels.has_property()}]->({labels.model_property("p")} {{ {match_clause("property_ident", property_, kwargs)} }})

        MATCH (p)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (p)-[{labels.updated_by("updated")}]->({labels.user("u")})

        WITH p,
             properties(p) AS fields,
             c.node_id  AS created_by,
             u.node_id  AS updated_by,
             created.at AS created_at,
             updated.at AS updated_at

        DETACH DELETE p
        RETURN fields, created_by, updated_by, created_at, updated_at
        """

        self.assert_.model_property_not_used(tx, model, property_)

        node = tx.run(cql, **kwargs).single()

        if node is None:
            return None

        return cast(
            ModelProperty,
            ModelProperty.from_node(
                **node["fields"],
                created_by=node["created_by"],
                updated_by=node["updated_by"],
                created_at=node["created_at"],
                updated_at=node["updated_at"],
            ),
        )

    def get_model_relationship_tx(
        self, tx: Union[Session, Transaction], id: ModelRelationshipId
    ) -> Optional[ModelRelationship]:
        """
        [[TRANSACTIONAL]]

        Gets a single model relationship.
        """

        cql = f"""
        MATCH  ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.model("n")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)
        MATCH (m)-[{labels.related_to("r")} {{ id: $relationship_id }}]->(n)
        RETURN r,
               r.type AS type,
               m.id   AS from,
               n.id   AS to
        """
        relationship_id = get_model_relationship_id(id)
        relationship = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            relationship_id=str(relationship_id),
        ).single()

        if not relationship:
            return None

        return ModelRelationship(
            id=relationship["r"]["id"],
            name=relationship["r"]["name"],
            type=relationship["type"],
            from_=relationship["from"],
            to=relationship["to"],
            one_to_many=relationship["r"]["one_to_many"],
            display_name=relationship["r"]["display_name"],
            description=relationship["r"]["description"],
            created_at=relationship["r"]["created_at"],
            updated_at=relationship["r"]["updated_at"],
            created_by=relationship["r"]["created_by"],
            updated_by=relationship["r"]["updated_by"],
            index=relationship["r"]["index"],
        )

    def get_model_relationship(
        self, id: ModelRelationshipId
    ) -> Optional[ModelRelationship]:
        """
        Gets a single model relationship.
        """
        with self.session() as session:
            return self.get_model_relationship_tx(session, id)

    def get_outgoing_model_relationships_tx(
        self,
        tx: Union[Session, Transaction],
        from_model: Optional[Union[Model, ModelId, str]] = None,
        one_to_many: Optional[bool] = None,
    ) -> Iterator[ModelRelationship]:
        """
        [[TRANSACTIONAL]]

        Gets all outgoing relationships from a model.
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            one_to_many=one_to_many,
        )

        from_model_clause = (
            f""" {{ {match_clause("from_model_id_or_name", from_model, kwargs)} }}"""
            if from_model is not None
            else ""
        )

        one_to_many_clause = (
            f" {{ one_to_many: $one_to_many }}" if one_to_many is not None else ""
        )

        cql = f"""
        MATCH ({labels.model("m")}{from_model_clause})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        MATCH (m)
              -[{labels.related_to("r")} {one_to_many_clause}]->({labels.model("n")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        RETURN r,
               m.id    AS from,
               n.id    AS to
        ORDER BY r.index
        """

        results = tx.run(cql, **kwargs)

        return (
            ModelRelationship(
                id=rel["r"]["id"],
                name=rel["r"]["name"],
                type=rel["r"]["type"],
                from_=rel["from"],
                to=rel["to"],
                one_to_many=rel["r"]["one_to_many"],
                display_name=rel["r"]["display_name"],
                description=rel["r"]["description"],
                created_by=rel["r"]["created_by"],
                updated_by=rel["r"]["updated_by"],
                created_at=rel["r"]["created_at"],
                updated_at=rel["r"]["updated_at"],
                index=rel["r"]["index"],
            )
            for rel in results
        )

    def create_model_relationship_tx(
        self,
        tx: Union[Session, Transaction],
        from_model: Union[Model, ModelId],
        name: Union[RelationshipName, str],
        to_model: Union[Model, ModelId],
        one_to_many: bool = True,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        index: Optional[int] = None,
    ) -> ModelRelationship:
        """
        =======================================================================
        Internal method: add a new relationship between two Models.
        =======================================================================

        Note: an `id` parameter is provided to allow for a model relationship
        to be created from a model relationship stub, if needed using the same
        ID.

        The `name` of the relationship is a backwards-compatible identifier that
        allows values from the Python client and the web app that have a UUID
        suffix, eg `belongs_to_478e215d-04ec-4cdf-ac8b-d5289601c9f7`. The `type`
        of the relationship is an uppercase Neo4j relationship type, with UUID
        suffix removed. Differentiating `name` and `type` will hopefully help
        keep us below the Neo4j limit of 65,000 relationship types in the graph.

        Raises
        ------
        OperationError
        """
        name = validate_relationship_name(name)
        relationship_type = get_relationship_type(name)
        display_name = display_name or name
        description = description or ""

        self.assert_.model_relationship_name_is_unique(
            tx, relationship_name=name, from_model=from_model, to_model=to_model
        )

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            one_to_many=one_to_many,
            name=name,
            relationship_type=relationship_type,
            display_name=display_name,
            description=description,
            index=index,
        )

        cql = f"""
        WITH datetime() AS now
        MATCH  ({labels.model("m")} {{ {match_clause("from_model", from_model, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.model("n")} {{ {match_clause("to_model", to_model, kwargs)} }})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        CREATE (m)-[{labels.related_to("r")} {{
          id                    : randomUUID(),
          type                  : $relationship_type,
          name                  : $name,
          one_to_many           : $one_to_many,
          display_name          : $display_name,
          description           : $description,
          created_at            : now,
          updated_at            : now,
          created_by            : $user_id,
          updated_by            : $user_id,
          index                 : $index
        }} ]->(n)

        RETURN r,
               r.type AS type,
               m.id   AS from,
               n.id   AS to
        """

        relationship = tx.run(cql, **kwargs).single()

        if not relationship:
            raise OperationError(
                "couldn't create model relationship",
                cause=ModelRelationshipNotFoundError(
                    str(from_model), str(to_model), relationship_type
                ),
            )

        return ModelRelationship(
            id=relationship["r"]["id"],
            name=relationship["r"]["name"],
            type=relationship["type"],
            from_=relationship["from"],
            to=relationship["to"],
            one_to_many=relationship["r"]["one_to_many"],
            display_name=relationship["r"]["display_name"],
            description=relationship["r"]["description"],
            created_at=relationship["r"]["created_at"],
            updated_at=relationship["r"]["updated_at"],
            created_by=relationship["r"]["created_by"],
            updated_by=relationship["r"]["updated_by"],
            index=relationship["r"]["index"],
        )

    def create_model_relationship(
        self,
        from_model: Union[Model, ModelId],
        name: Union[RelationshipName, str],
        to_model: Union[Model, ModelId],
        one_to_many: bool = True,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        index: Optional[int] = None,
    ) -> ModelRelationship:
        """
        Add a new relationship between two Models.

        Raises
        ------
        OperationError
        """
        with self.transaction() as tx:
            return self.create_model_relationship_tx(
                tx=tx,
                from_model=from_model,
                name=name,
                to_model=to_model,
                one_to_many=one_to_many,
                display_name=display_name,
                description=description,
                index=index,
            )

    def update_model_relationship(
        self,
        relationship: Union[ModelRelationship, ModelRelationshipId, RelationshipName],
        display_name: str,
        index: Optional[int] = None,
    ) -> Optional[ModelRelationship]:
        """
        Updates an exist model relationship.
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            display_name=display_name,
            index=index,
        )

        cql = f"""
        MATCH  ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.model("n")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)
        MATCH (m)-[{labels.related_to("r")} {{ {match_clause("relationship_id_or_name", relationship, kwargs)} }}]->(n)
        SET r.display_name = $display_name,
            r.index        = COALESCE($index, r.index),
            r.updated_at   = datetime(),
            r.updated_by   = $user_id
        RETURN r,
               r.type AS type,
               m.id   AS from,
               n.id   AS to
        """
        rel = self.execute_single(cql, **kwargs).single()

        if not rel:
            return None

        return ModelRelationship(
            id=rel["r"]["id"],
            type=rel["type"],
            name=rel["r"]["name"],
            from_=rel["from"],
            to=rel["to"],
            one_to_many=rel["r"]["one_to_many"],
            display_name=rel["r"]["display_name"],
            description=rel["r"]["description"],
            created_at=rel["r"]["created_at"],
            updated_at=rel["r"]["updated_at"],
            created_by=rel["r"]["created_by"],
            updated_by=rel["r"]["updated_by"],
            index=rel["r"]["index"],
        )

    def delete_model_relationship_tx(
        self,
        tx: Union[Session, Transaction],
        relationship: Union[ModelRelationship, ModelRelationshipId, str],
    ) -> Optional[ModelRelationshipId]:
        """
        Deletes a Model relationship.
        """

        # See if any records exist with the same type as that of the
        # model-to-model relationship. Record relationships contain a property
        # `model_relationship_id` that specifies the ID of the parent model
        # relationship that the record relationship is an instance of.
        check_record_relations_cql = f"""
        MATCH  ({labels.model("m1")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.model("m2")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)
        MATCH (m1)-[{labels.related_to("mrel")} {{ id: $relationship_id }}]->(m2)
        MATCH (m1)<-[{labels.instance_of()}]-({labels.record("r1")})
                   -[rrel {{ model_relationship_id: mrel.id }}]->({labels.record("r2")})
                   -[{labels.instance_of()}]->(m2)
        RETURN TYPE(rrel) AS type,
               m1.id      AS model_from,
               m2.id      AS model_to,
               r1.id      AS record_from,
               r2.id      AS record_to
        """

        delete_relationship_cql = f"""
        MATCH  ({labels.model("m1")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.model("m2")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)
        MATCH (m1)-[{labels.related_to("r_v2")} {{ id: $relationship_id }}]->(m2)
        OPTIONAL MATCH (m1)-[r {{ id: $relationship_id }}]->(m2)
        WHERE NOT TYPE(r) IN $reserved_relationship_types
        DELETE r, r_v2
        RETURN r
        """

        relationship_id = get_model_relationship_id(relationship)

        r = tx.run(
            check_record_relations_cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            relationship_id=str(relationship_id),
        ).single()

        if r:
            raise OperationError(
                "couldn't delete model relationship",
                cause=ModelRelationshipNotFoundError(
                    type=r["type"], model_from=r["model_from"], model_to=r["model_to"]
                ),
            )

        rel = tx.run(
            delete_relationship_cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            relationship_id=str(relationship_id),
            reserved_relationship_types=labels.RESERVED_SCHEMA_RELATIONSHIPS,
        ).single()

        if not rel:
            # Explicitly rollback:
            tx.rollback()
            return None

        return relationship_id

    def create_records_tx(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, ModelId, str],
        records: List[Dict[str, GraphValue]],
        fill_missing: bool = False,
    ) -> List[Record]:
        """
        [[TRANSACTIONAL]]

        Create multiple new records.

        The record ID is stored as `@id` in order not to conflict with any
        user-defined properties named `id`.

        Options
        -------
        - If `fill_missing=True`, any defined properties of a record that do
          not have a value or a default, and are not required, will be set
          to `None`, rather than being omitted from the returned record.
        """

        # Build keywords to unwind the record list
        properties: List[ModelProperty] = self.get_properties_tx(tx, model_id_or_name)
        property_map: Dict[str, ModelProperty] = {p.name: p for p in properties}

        # Check record values against data types:
        validate_records(properties, *records).check()

        # Coerce record values to the proper datatype
        # TODO: do this in one step with `validate_records`
        for record in records:
            for name, value in record.items():
                if value is not None:
                    record[name] = property_map[name].data_type.into(value)

        kwargs: Dict[str, GraphValue] = cast(
            Dict[str, GraphValue],
            dict(
                organization_id=self.organization_id,
                dataset_id=self.dataset_id,
                user_id=str(self.user_id),
                records=records,
            ),
        )

        # See the note on `create_models()` regarding a description of how
        # the record `@sort_key` property works.
        cql = f"""
        MATCH  ({labels.model("m")} {{ {match_clause("match_id_or_name", model_id_or_name, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
        UNWIND $records as record
        CREATE ({labels.record("r")} {{
          `@sort_key`: 0,
          `@id`: randomUUID()
        }})
        SET r += record
        SET m.`@max_sort_key` = m.`@max_sort_key` + 1
        SET r.`@sort_key` = m.`@max_sort_key`
        CREATE (r)-[{labels.instance_of()}]->(m)

        MERGE ({labels.user("u")} {{ node_id: $user_id }})
        CREATE (r)-[{labels.created_by("created")} {{ at: datetime() }}]->(u)
        CREATE (r)-[{labels.updated_by("updated")} {{ at: datetime() }}]->(u)

        RETURN r,
               created.at AS created_at,
               updated.at AS updated_at
        """
        nodes = tx.run(cql, **kwargs).records()

        return [
            Record.from_node(
                node["r"],
                property_map=property_map,
                fill_missing=fill_missing,
                created_by=self.user_id,
                updated_by=self.user_id,
                created_at=node["created_at"],
                updated_at=node["updated_at"],
            )
            for node in nodes
        ]

    def create_records(
        self,
        model_id_or_name: Union[Model, ModelId, str],
        records: List[Dict[str, GraphValue]],
        fill_missing: bool = False,
    ) -> List[Record]:
        """
        Create multiple new records.

        Options
        -------
        - If `fill_missing=True`, any defined properties of a record that do
          not have a value or a default, will be set to `None`, rather than
          being omitted from the returned record.
        """
        with self.transaction() as tx:
            return self.create_records_tx(
                tx=tx,
                model_id_or_name=model_id_or_name,
                records=records,
                fill_missing=fill_missing,
            )

    def create_record(
        self,
        model_id_or_name: Union[Model, ModelId, str],
        record: Dict[str, GraphValue],
        fill_missing: bool = False,
    ) -> Record:
        """
        Create a single new record.

        Options
        -------
        - If `fill_missing=True`, any defined properties of a record that do
          not have a value or a default, will be set to `None`, rather than
          being omitted from the returned record.
        """
        with self.transaction() as tx:
            return self.create_records_tx(
                tx=tx,
                model_id_or_name=model_id_or_name,
                records=[record],
                fill_missing=fill_missing,
            )[0]

    def model_property_record_count(self, model_id: str, property_name: str) -> int:
        with self.transaction() as tx:
            return self.model_property_record_count_tx(tx, model_id, property_name)

    def model_property_record_count_tx(
        self, tx: Union[Session, Transaction], model_id: str, property_name: str
    ) -> int:
        cql = f"""
        MATCH  ({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")} {{ id: $model_id }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        WHERE NOT r[$prop_name] IS NULL

        RETURN COUNT(r) as count
        """

        result = tx.run(
            cql,
            model_id=model_id,
            prop_name=property_name,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
        ).single()

        return result["count"]

    def delete_property_from_all_records(
        self, model_id: str, model_property: ModelProperty
    ) -> int:
        with self.transaction() as tx:
            return self.delete_property_from_all_records_tx(
                tx, model_id, model_property
            )

    def delete_property_from_all_records_tx(
        self,
        tx: Union[Session, Transaction],
        model_id: str,
        model_property: ModelProperty,
    ) -> int:

        cql = f"""
        MATCH  ({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")} {{ id: $model_id }})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        WHERE NOT r[$property_name] IS NULL

        WITH r
        LIMIT $limit

        SET r += $null_property_map

        RETURN COUNT(r)
        """

        property_name = model_property.name
        null_property_map = {property_name: None}

        def call_periodically(cql) -> int:
            periodic_cql = f"""
            CALL apoc.periodic.commit("{cql}", {{
              model_id: $model_id,
              property_name: $property_name,
              null_property_map: $null_property_map,
              dataset_id: $dataset_id,
              organization_id: $organization_id,
              limit: $limit
            }})
            """
            result = tx.run(
                periodic_cql,
                model_id=model_id,
                property_name=property_name,
                null_property_map=null_property_map,
                organization_id=self.organization_id,
                dataset_id=self.dataset_id,
                limit=10000,
            ).single()

            if result["failedBatches"] > 0:
                raise Exception(result["batchErrors"])

            if result["failedCommits"] > 0:
                raise Exception(result["commitErrors"])

            return result["updates"]

        return call_periodically(cql)

    def update_record_tx(
        self,
        tx: Union[Session, Transaction],
        record: Union[Record, RecordId],
        values: Dict[str, GraphValue],
        fill_missing: bool = False,
    ) -> Record:
        """
        [[TRANSACTIONAL]]

        Update an existing record.

        Options
        -------
        - If `fill_missing=True`, any defined properties of a record that do
          not have a value or a default, will be set to `None`, rather than
          being omitted from the returned record.

        Raises
        ------
        RecordNotFoundError
        """

        cql = f"""
        MATCH  ({labels.record("r")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        SET r += $values
        WITH r

        MATCH (r)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (r)-[{labels.updated_by("last_updated")}]->({labels.user()})

        DELETE last_updated
        MERGE ({labels.user("u")} {{ node_id: $user_id }})
        CREATE (r)-[{labels.updated_by("updated")} {{ at: datetime() }}]->(u)

        RETURN r,
               c.node_id  AS created_by,
               u.node_id  AS updated_by,
               created.at AS created_at,
               updated.at AS updated_at
        """
        record_id = get_record_id(record)

        model = self._get_model_from_record_tx(tx, record_id)
        if not model:
            raise RecordNotFoundError(record_id)

        properties: List[ModelProperty] = self.get_properties_tx(tx, ModelId(model.id))
        property_map: Dict[str, ModelProperty] = {p.name: p for p in properties}

        # If any properties are not in the request, they need to be removed from
        # the record.  Overwrite any existing values with `null`.
        for p in properties:
            if p.name not in values:
                values[p.name] = None

        validate_records(properties, values).check()

        # TODO: do this in validate_records
        for name, value in values.items():
            if value is not None:
                values[name] = property_map[name].data_type.into(value)

        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            record_id=str(record_id),
            values=values,
        ).single()

        return Record.from_node(
            node["r"],
            property_map=property_map,
            fill_missing=fill_missing,
            created_by=node["created_by"],
            updated_by=node["updated_by"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
        )

    def update_record(
        self,
        record: Union[Record, RecordId],
        values: Dict[str, GraphValue],
        fill_missing: bool = False,
    ) -> Record:
        """
        Update an existing record.

        Options
        -------
        - If `fill_missing=True`, any defined properties of a record that do
          not have a value or a default, will be set to `None`, rather than
          being omitted from the returned record.

        Raises
        ------
        RecordNotFoundError
        """
        with self.transaction() as tx:
            return self.update_record_tx(
                tx=tx, record=record, values=values, fill_missing=fill_missing
            )

    def get_record_tx(
        self,
        tx: Union[Session, Transaction],
        record: Union[Record, RecordId],
        embed_linked: bool = True,
        fill_missing: bool = False,
    ) -> Optional[Record]:
        """
        See `get_record()`
        """
        if isinstance(record, Record):
            return record

        if embed_linked:
            cql = f"""
            MATCH  ({labels.record("r_source")} {{ `@id`: $record_id }})
                  -[{labels.instance_of()}]->({labels.model("m")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

            MATCH (r_source)-[{labels.created_by("created")}]->({labels.user("created_by")})
            MATCH (r_source)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

            OPTIONAL MATCH (m)-[{labels.related_to("mr")} {{ one_to_many: false }}]->({labels.model()})-[{labels.in_dataset()}]->(d)-[{labels.in_organization()}]->(o)
            OPTIONAL MATCH (r_source)-[rr {{ model_relationship_id: mr.id }}]->({labels.record("r_related")})

            RETURN m                  AS model,
                   r_source           AS src,
                   created_by.node_id AS created_by,
                   updated_by.node_id AS updated_by,
                   created.at         AS created_at,
                   updated.at         AS updated_at,
                   COLLECT(TYPE(rr))  AS relations,
                   COLLECT(r_related) AS neighbors
            LIMIT 1
            """
        else:
            cql = f"""
            MATCH  ({labels.record("r")} {{ `@id`: $record_id }})
                  -[{labels.instance_of()}]->({labels.model("m")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

            MATCH (r_source)-[{labels.created_by("created")}]->({labels.user("created_by")})
            MATCH (r_source)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

            RETURN m                  AS model,
                   r                  AS src,
                   created_by.node_id AS created_by,
                   updated_by.node_id AS updated_by,
                   created.at         AS created_at,
                   updated.at         AS updated_at,
                   []                 AS relations,
                   []                 AS neighbors
            LIMIT 1
            """

        record_id = get_record_id(record)

        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            record_id=str(record_id),
        ).single()

        if node is None:
            # Read-only; no rollback
            return None

        properties: List[ModelProperty] = self.get_properties_tx(
            tx, node["model"]["id"]
        )
        property_map: Dict[str, ModelProperty] = {p.name: p for p in properties}

        record = Record.from_node(
            node["src"],
            property_map=property_map,
            fill_missing=fill_missing,
            created_by=node["created_by"],
            updated_by=node["updated_by"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
        )

        if embed_linked:
            # Embed the record stubs to the record:
            for (relation, neighbor) in zip(node["relations"], node["neighbors"]):
                record.embed(relation, RecordStub.from_node(neighbor.items()))

        return record

    def get_record(
        self,
        record: Union[Record, RecordId],
        embed_linked: bool = True,
        fill_missing: bool = False,
    ) -> Optional[Record]:
        """
        Get a single record by its ID.

        Options
        -------

        - If `embed_linked=True`, any records that are joined by a relationship
          will be included as stubs in the record returned from this method.

          For a neighboring node to be embeddable, the model node `(m:Model)`
          of the fetched record must have a neighboring `(n:Model)` node
          connected by way of a relation `(m)-[s]->(n)`, where
          `s.one_of_many = false`, and associated record nodes `r[m]` and
          `r[n]` must be joined by a relationship `t` where
          `TYPE(t) = TYPE(s)`.

        - If `fill_missing=True`, any defined properties of a record that do
          not have a value or a default will be set to `None`, rather than
          being omitted from the returned record.
        """
        with self.transaction() as tx:
            return self.get_record_tx(
                tx=tx,
                record=record,
                embed_linked=embed_linked,
                fill_missing=fill_missing,
            )

    def get_all_records_cursor_tx(
        self,
        tx: Union[Session, Transaction],
        model_id_or_name: Union[Model, str],
        limit: int = 1000,
        next_page: Optional[NextPageCursor] = None,
        embed_linked: bool = True,
        fill_missing: bool = False,
    ) -> Optional[PagedResult]:
        """
        [[TRANSACTIONAL]]

        See `get_all_records()`

        TODO: the performance of this is broken with many datasets in the graph.
        Fix it.

        Raises
        ------
        ModelNotFoundError
        """
        if not next_page:
            next_page = NextPageCursor(0)

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            limit=limit,
            next_page_start=next_page,
        )

        if embed_linked:
            cql = f"""
            MATCH  ({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
            MATCH ({labels.record("r_source")})-[{labels.instance_of()}]->(m)
            USING INDEX {labels.record("r_source")}(`@sort_key`)
            WHERE r_source.`@sort_key` > $next_page_start

            WITH r_source, m
            LIMIT toInteger($limit)

            MATCH (r_source)-[{labels.created_by("created")}]->({labels.user("created_by")})
            MATCH (r_source)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

            OPTIONAL MATCH (m)-[{labels.related_to("mr")} {{ one_to_many: false }}]->({labels.model()})
            OPTIONAL MATCH (r_source)-[rr]->({labels.record("r_related")})
            WHERE TYPE(rr) = mr.type

            RETURN m                  AS model,
                   r_source           AS src,
                   created_by.node_id AS created_by,
                   updated_by.node_id AS updated_by,
                   created.at         AS created_at,
                   updated.at         AS updated_at,
                   COLLECT(TYPE(rr))  AS relations,
                   COLLECT(r_related) AS neighbors
            """
        else:
            cql = f"""
            MATCH  ({labels.record("r")})
                  -[{labels.instance_of()}]->({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
                  -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})
            USING INDEX {labels.record("r")}(`@sort_key`)
            WHERE r.`@sort_key` > $next_page_start

            WITH r, m
            LIMIT toInteger($limit)

            MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
            MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

            RETURN m                  AS model,
                   r                  AS src,
                   created_by.node_id AS created_by,
                   updated_by.node_id AS updated_by,
                   created.at         AS created_at,
                   updated.at         AS updated_at,
                   []                 AS relations,
                   []                 AS neighbors
            """

        nodes = tx.run(cql, **kwargs).records()

        if nodes is None:
            raise ModelNotFoundError(model_id_or_name)

        properties: List[ModelProperty] = self.get_properties_tx(tx, model_id_or_name)
        property_map: Dict[str, ModelProperty] = {p.name: p for p in properties}

        records = []
        next_page_cursor: Optional[NextPageCursor] = None

        for node in nodes:
            sort_key = node["src"]["@sort_key"]
            r = Record.from_node(
                node["src"],
                property_map=property_map,
                fill_missing=fill_missing,
                created_by=node["created_by"],
                updated_by=node["updated_by"],
                created_at=node["created_at"],
                updated_at=node["updated_at"],
            )
            next_page_cursor = NextPageCursor(sort_key)

            if embed_linked:
                # Embed the record stubs to the record:
                for (relation, neighbor) in zip(node["relations"], node["neighbors"]):
                    r.embed(relation, RecordStub.from_node(neighbor.items()))

            records.append(r)

        return PagedResult(results=records, next_page=next_page_cursor)

    def get_all_records_offset_tx(
        self,
        tx: Union[Session, Transaction],
        model: Model,
        limit: Optional[int] = None,
        offset: int = 0,
        embed_linked: bool = True,
        fill_missing: bool = False,
        order_by: Optional[ModelOrderByField] = None,
        max_records: int = 100000,
    ) -> Iterator[Record]:
        """
        [[TRANSACTIONAL]]

        Get all records of a given model. This query can be paginated by passing
        a non-null LIMIT.

        Raises
        ------
        - CannotSortRecordsError
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            limit=limit,
            offset=offset,
        )

        above_max = model.count > max_records

        if order_by is None and above_max:
            order_by_cql = ""
        elif order_by is None and not above_max:
            order_by_cql = "ORDER BY r.`@sort_key`"
        elif order_by is not None and above_max:
            raise CannotSortRecordsError(model.count, max_records)
        elif order_by is not None and not above_max:
            if order_by.is_created_at:
                order_by_cql = f"ORDER BY created.at {order_by.direction.value}\n"
            elif order_by.is_updated_at:
                order_by_cql = f"ORDER BY updated.at {order_by.direction.value}\n"
            else:
                kwargs["order_by"] = order_by.name
                order_by_cql = f"ORDER BY r[$order_by] {order_by.direction.value}"
            order_by_cql += ", r.`@sort_key`"
        else:
            raise NotImplementedError

        if embed_linked:
            cql = f"""
            MATCH  ({labels.record("r")})
                  -[{labels.instance_of()}]->({labels.model("m")} {{ {match_clause("model_id_or_name", model.id, kwargs)} }})
                  -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

            MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
            MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

            WITH r, m, created, updated, created_by, updated_by
            {order_by_cql}
            SKIP toInteger($offset)
            { "LIMIT toInteger($limit)" if limit is not None else "" }

            OPTIONAL MATCH (m)-[{labels.related_to("mr")} {{ one_to_many: false }}]->({labels.model()})
            OPTIONAL MATCH (r)-[rr]->({labels.record("r_related")})
            WHERE rr.model_relationship_id = mr.id

            RETURN m                  AS model,
                   r                  AS src,
                   created_by.node_id AS created_by,
                   updated_by.node_id AS updated_by,
                   created.at         AS created_at,
                   updated.at         AS updated_at,
                   COLLECT(TYPE(rr))  AS relations,
                   COLLECT(r_related) AS neighbors
            """
        else:
            cql = f"""
            MATCH  ({labels.record("r")})
                  -[{labels.instance_of()}]->({labels.model("m")} {{ {match_clause("model_id_or_name", model.id, kwargs)} }})
                  -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

            MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
            MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

            WITH r, m, created, updated, created_by, updated_by
            {order_by_cql}
            SKIP toInteger($offset)
            { "LIMIT toInteger($limit)" if limit is not None else "" }

            RETURN m                  AS model,
                   r                  AS src,
                   created_by.node_id AS created_by,
                   updated_by.node_id AS updated_by,
                   created.at         AS created_at,
                   updated.at         AS updated_at,
                   []                 AS relations,
                   []                 AS neighbors
            """

        nodes = tx.run(cql, **kwargs).records()

        properties: List[ModelProperty] = self.get_properties_tx(tx, model.id)
        property_map: Dict[str, ModelProperty] = {p.name: p for p in properties}

        for node in nodes:
            r = Record.from_node(
                node["src"],
                property_map=property_map,
                fill_missing=fill_missing,
                created_by=node["created_by"],
                updated_by=node["updated_by"],
                created_at=node["created_at"],
                updated_at=node["updated_at"],
            )

            if embed_linked:
                # Embed the record stubs to the record:
                for (relation, neighbor) in zip(node["relations"], node["neighbors"]):
                    r.embed(relation, RecordStub.from_node(neighbor.items()))

            yield r

    def get_all_records(
        self,
        model_id_or_name: Union[Model, str],
        limit: int = 1000,
        next_page: Optional[NextPageCursor] = None,
        embed_linked: bool = True,
        fill_missing: bool = False,
    ) -> Optional[PagedResult]:
        """
        Fetches all records in this dataset by page.

        Options
        -------

        The ``offset`` parameter is only provided for backwards-compatibility
        with the legacy API. In general, the ``next_page`` cursor should be
        preferred.

        If `embed_linked=True`, any records that are joined by a relationship
        will be included as stubs in the record returned from this method.

        For a neighboring node to be embeddable, the model node `(m:Model)` of
        the fetched record must have a neighboring `(n:Model)` node connected
        by way of a relation `(m)-[s]->(n)`, where `s.one_of_many = false`,
        and associated record nodes `r[m]` and `r[n]` must be joined by a
        relationship `t` where `TYPE(t) = TYPE(s)`.

        If `fill_missing=True`, any defined properties of a record that do
        not have a value or a default will be set to `None`, rather than
        being omitted from the returned record.
        """
        with self.transaction() as tx:
            return self.get_all_records_cursor_tx(
                tx=tx,
                model_id_or_name=model_id_or_name,
                limit=limit,
                next_page=next_page,
                embed_linked=embed_linked,
                fill_missing=fill_missing,
            )

    def delete_record_tx(
        self,
        tx: Union[Session, Transaction],
        record: Union[Record, RecordId],
        properties: Optional[List[ModelProperty]],
    ) -> Record:
        """
        [[TRANSACTIONAL]]

        Delete a record.

        Raises
        ------
        RecordNotFoundError
        """

        cql = f"""
        MATCH  ({labels.record("r")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset()} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization()} {{ id: $organization_id }})

        MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

        WITH r,
             properties(r) AS deleted,
             created_by.node_id AS created_by,
             updated_by.node_id AS updated_by,
             created.at         AS created_at,
             updated.at         AS updated_at

        DETACH DELETE r

        RETURN deleted, created_by, updated_by, created_at, updated_at
        """
        record_id = get_record_id(record)

        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            record_id=str(record_id),
        ).single()

        if not node:
            raise RecordNotFoundError(record_id)

        return Record.from_node(
            node["deleted"],
            property_map={p.name: p for p in properties} if properties else None,
            fill_missing=True,
            created_by=node["created_by"],
            updated_by=node["updated_by"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
        )

    def delete_record(
        self,
        record: Union[Record, RecordId],
        properties: Optional[List[ModelProperty]],
    ) -> Record:
        """
        Delete a record.

        Raises
        ------
        RecordNotFoundError
        """
        with self.transaction() as tx:
            # Safe to use session as `delete_record()` is a single operation
            return self.delete_record_tx(tx=tx, record=record, properties=properties)

    def create_record_relationship_batch_tx(
        self, tx: Union[Session, Transaction], to_create: List[CreateRecordRelationship]
    ) -> Iterator[Tuple[RecordRelationship, ModelRelationship]]:
        """
        Add a relationship between two or more records.

        Constraint
        ----------
        In order for a relationship to be created between two records,
        a corresponding relationship must exist between the models the records
        are instances of.

        Raises
        ------
        - ModelRelationshipNotFoundError
        - MultipleRelationshipsViolationError
        """
        to_create_by_model_relationship: Dict[
            ModelRelationship, List[Dict[str, str]]
        ] = defaultdict(list)

        # Group record relationships by model relationship so that each query
        # can by string-formatted with the relationship type string.
        for relation in to_create:
            assert relation.model_relationship is not None

            model_relationship = self.assert_.model_relationship_exists(
                tx, relation.from_, relation.model_relationship, relation.to
            )

            to_create_by_model_relationship[model_relationship].append(
                {
                    "from": str(relation.from_),
                    "to": str(relation.to),
                    "model_relationship_id": str(model_relationship.id),
                    "name": model_relationship.name,
                    "display_name": model_relationship.display_name,
                }
            )

        for model_relationship, records in to_create_by_model_relationship.items():
            cql = f"""
            UNWIND $records AS record
            MATCH ({labels.record("r_from")} {{ `@id`: record.from }})
                  -[{labels.instance_of()}]->({labels.model("m_from")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

            MATCH ({labels.record("r_to")} {{ `@id`: record.to }})
                  -[{labels.instance_of()}]->({labels.model("m_to")})
                  -[{labels.in_dataset()}]->(d)
                  -[{labels.in_organization()}]->(o)

            MERGE (r_from)-[r:`{model_relationship.type}` {{ model_relationship_id: record.model_relationship_id }}]->(r_to)
            ON CREATE SET
                r.id         = randomUUID(),
                r.created_at = datetime(),
                r.created_by = $user_id,
                r.updated_at = datetime(),
                r.updated_by = $user_id
            ON MATCH SET
                r.updated_at = datetime(),
                r.updated_by = $user_id

            RETURN r,
                   r_from.`@id`        AS from,
                   r_to.`@id`          AS to,
                   TYPE(r)             AS type,
                   record.name         AS name,
                   record.display_name AS display_name
            """
            results = tx.run(
                cql,
                organization_id=self.organization_id,
                dataset_id=self.dataset_id,
                user_id=str(self.user_id),
                records=records,
            )

            for result in results:
                assert (
                    result["type"] == model_relationship.type
                ), f"{result['type']} not equal to {model_relationship.type}"

                record_relationship = RecordRelationship(
                    id=result["r"]["id"],
                    from_=result["from"],
                    to=result["to"],
                    type=result["type"],
                    model_relationship_id=result["r"]["model_relationship_id"],
                    name=result["name"],
                    display_name=result["display_name"],
                    one_to_many=model_relationship.one_to_many,
                    created_at=result["r"]["created_at"],
                    updated_at=result["r"]["updated_at"],
                    created_by=result["r"]["created_by"],
                    updated_by=result["r"]["updated_by"],
                )

                # If `model_relationship` `one_to_many` is False, validate that no more
                # than one relationship exists; if not, raise an exception and roll back
                # the transaction.
                self.assert_.one_or_many_condition_holds(
                    tx, record_relationship.from_, model_relationship
                )

                yield (record_relationship, model_relationship)

    def create_record_relationship_tx(
        self,
        tx: Union[Session, Transaction],
        from_record: Union[Record, RecordId],
        relationship_id_or_name: Union[
            ModelRelationship, ModelRelationshipId, RelationshipName
        ],
        to_record: Union[Record, RecordId],
    ) -> Tuple[RecordRelationship, ModelRelationship]:
        """
        Add a relationship between two or more records.

        Constraint
        ----------
        In order for a relationship to be created between two records,
        a corresponding relationship must exist between the models the records
        are instances of.

        Raises
        ------
        - ModelRelationshipNotFoundError
        - MultipleRelationshipsViolationError
        """
        from_record = get_record_id(from_record)
        to_record = get_record_id(to_record)

        # Check that a corresponding model relationship exists:
        model_relationship: ModelRelationship = self.assert_.model_relationship_exists(
            tx, from_record, relationship_id_or_name, to_record
        )

        relationship_type = get_relationship_type(model_relationship.name)

        cql = f"""
        MATCH ({labels.record("r_from")} {{ `@id`: $from_record }})
              -[{labels.instance_of()}]->({labels.model("m_from")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        MATCH ({labels.record("r_to")} {{ `@id`: $to_record }})
              -[{labels.instance_of()}]->({labels.model("m_to")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        MERGE (r_from)-[r:`{relationship_type}` {{ model_relationship_id: $model_relationship_id }}]->(r_to)
        ON CREATE SET
            r.id         = randomUUID(),
            r.created_at = datetime(),
            r.created_by = $user_id,
            r.updated_at = datetime(),
            r.updated_by = $user_id
        ON MATCH SET
            r.updated_at = datetime(),
            r.updated_by = $user_id

        RETURN r,
               r_from.`@id` AS from,
               r_to.`@id`   AS to,
               TYPE(r)      AS type
        """

        result = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            from_record=str(from_record),
            to_record=str(to_record),
            model_relationship_id=str(model_relationship.id),
        ).single()

        record_relationship = RecordRelationship(
            id=result["r"]["id"],
            from_=result["from"],
            to=result["to"],
            type=result["type"],
            model_relationship_id=result["r"]["model_relationship_id"],
            name=model_relationship.name,
            display_name=model_relationship.display_name,
            one_to_many=model_relationship.one_to_many,
            created_at=result["r"]["created_at"],
            updated_at=result["r"]["updated_at"],
            created_by=result["r"]["created_by"],
            updated_by=result["r"]["updated_by"],
        )

        # If `model_relationship` `one_to_many` is False, validate that no more
        # than one relationship exists; if not, raise an exception and roll back
        # the transaction.
        self.assert_.one_or_many_condition_holds(tx, from_record, model_relationship)

        return (record_relationship, model_relationship)

    def create_record_relationship(
        self,
        from_record: Union[Record, RecordId],
        relationship: Union[ModelRelationship, ModelRelationshipId, RelationshipName],
        to_record: Union[Record, RecordId],
    ) -> Tuple[RecordRelationship, ModelRelationship]:
        """
        Add a relationship between two or more records.

        Constraint
        ----------
        In order for a relationship to be created between two records,
        a corresponding relationship must exist between the models the records
        are instances of.

        Raises
        ------
        - ModelRelationshipNotFoundError
        - MultipleRelationshipsViolationError
        """
        with self.transaction() as tx:
            try:
                (rr, mr) = self.create_record_relationship_tx(
                    tx, from_record, relationship, to_record
                )
            except ModelServiceError as e:
                raise OperationError("couldn't create record relationship", cause=e)

            return (rr, mr)

    def get_record_relationship_tx(
        self,
        tx: Union[Session, Transaction],
        model_relationship: Union[ModelRelationshipId, RelationshipName],
        record_relationship: Union[RecordRelationship, RecordRelationshipId, str],
    ) -> Optional[RecordRelationship]:
        """
        Get a record relationship by ID.

        Raises
        ------
        - RecordRelationshipNotFoundError
        """
        if isinstance(record_relationship, RecordRelationship):
            return record_relationship

        record_relationship_id = get_record_relationship_id(record_relationship)

        cql = f"""
        MATCH  ({labels.record("r_from")})
              -[{labels.instance_of()}]->({labels.model("m_from")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH (m_from)-[{labels.related_to("mr")}]-({labels.model("m_to")})
        WHERE mr.id = $model_relationship OR mr.name = $model_relationship
        MATCH (r_from)-[rr {{ id: $record_relationship_id, model_relationship_id: mr.id }}]->({labels.record("r_to")})
        RETURN r_from.id        AS from,
               r_to.id          AS to,
               TYPE(rr)         AS type,
               rr.id            AS id,
               mr.id            AS model_relationship_id,
               mr.name          AS name,
               mr.display_name  AS display_name,
               mr.one_to_many   AS one_to_many,
               rr.created_at    AS created_at,
               rr.updated_at    AS updated_at,
               rr.created_by    AS created_by,
               rr.updated_by    AS updated_by
        """
        rel = self.execute_single(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            record_relationship_id=str(record_relationship_id),
            model_relationship=str(model_relationship),
        ).single()

        if rel is None:
            raise RecordRelationshipNotFoundError(
                record_id=None, relationship_id=record_relationship_id
            )

        return cast(RecordRelationship, RecordRelationship.from_node(**rel))

    def get_outgoing_record_relationship_tx(
        self,
        tx: Union[Session, Transaction],
        record: Union[Record, RecordId],
        relationship: Union[RecordRelationship, RecordRelationshipId],
    ) -> Optional[RecordRelationship]:
        """
        Get a record relationship by ID.

        Raises
        ------
        - RecordRelationshipNotFoundError
        """
        if isinstance(relationship, RecordRelationship):
            return relationship

        record_id = get_record_id(record)
        relationship_id = get_record_relationship_id(relationship)

        cql = f"""
        MATCH  ({labels.record("r_from")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model("m_from")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.record("r_to")})
              -[{labels.instance_of()}]->({labels.model("m_to")})
              -[{labels.in_dataset()}]->(d)
        MATCH (m_from)-[{labels.related_to("mr")}]-(m_to)
        MATCH (r_from)-[rr {{ id: $relationship_id, model_relationship_id: mr.id }}]->(r_to)
        RETURN r_from.id        AS from,
               r_to.id          AS to,
               TYPE(rr)         AS type,
               rr.id            AS id,
               mr.id            AS model_relationship_id,
               mr.name          AS name,
               mr.display_name  AS display_name,
               mr.one_to_many   AS one_to_many,
               rr.created_at    AS created_at,
               rr.updated_at    AS updated_at,
               rr.created_by    AS created_by,
               rr.updated_by    AS updated_by
        """
        rel = self.execute_single(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            record_id=str(record_id),
            relationship_id=str(relationship_id),
        ).single()

        if rel is None:
            raise RecordRelationshipNotFoundError(
                record_id=record_id, relationship_id=relationship_id
            )

        return cast(RecordRelationship, RecordRelationship.from_node(**rel))

    def get_outgoing_record_relationships_tx(
        self,
        tx: Union[Session, Transaction],
        from_record: Union[Record, RecordId],
        one_to_many: Optional[bool] = None,
    ) -> Iterator[RecordRelationship]:
        """
        Get all record relationships with the given relationship type.

        - If no type is provided, all record-level relationships will be
          returned.

        - If `one_to_many` is not specified, all relationships, regardless of
          cardinality will be returned.
        """
        from_record = get_record_id(from_record)
        cql = f"""
        MATCH  ({labels.record("r_from")} {{ `@id`: $from_record }})
              -[{labels.instance_of()}]->({labels.model("m_from")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.record("r_to")})
              -[{labels.instance_of()}]->({labels.model("m_to")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        MATCH (m_from)-[{labels.related_to("mr")}]->(m_to) { "WHERE mr.one_to_many = $one_to_many" if one_to_many is not None else "" }
        MATCH (r_from)-[rr {{ model_relationship_id: mr.id }}]->(r_to)
        RETURN r_from.`@id`     AS from,
               r_to.`@id`       AS to,
               TYPE(mr)         AS type,
               rr.id            AS id,
               mr.id            AS model_relationship_id,
               mr.name          AS name,
               mr.display_name  AS display_name,
               mr.one_to_many   AS one_to_many,
               rr.created_at    AS created_at,
               rr.updated_at    AS updated_at,
               rr.created_by    AS created_by,
               rr.updated_by    AS updated_by
        ORDER BY mr.index
        """
        results = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            from_record=str(from_record),
            one_to_many=one_to_many,
        )

        return (
            cast(RecordRelationship, RecordRelationship.from_node(**rel))
            for rel in results
        )

    def get_record_relationships_by_model_with_relationship_stub_tx(
        self, tx: Union[Session, Transaction], relationship: ModelRelationshipStub
    ) -> Iterator[RecordRelationship]:
        """
        [[TRANSACTIONAL]]

        Given a model relationship, return an iterator over all associated
        record relationships.
        """
        cql = f"""
        MATCH  ({labels.record("r_from")})
                -[{labels.instance_of()}]->({labels.model("m_from")})
                -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.record("r_to")})
                  -[{labels.instance_of()}]->({labels.model("m_to")})
                  -[{labels.in_dataset()}]->(d)
                  -[{labels.in_organization()}]->(o)
            MATCH (m_from)-[{labels.related_to("mr")} {{ name: $model_relationship_name }}]->(m_to)
            MATCH (r_from)-[rr {{ model_relationship_id: mr.id  }}]->(r_to)
            RETURN r_from.`@id`     AS from,
                   r_to.`@id`       AS to,
                   TYPE(rr)         AS type,
                   rr.id            AS id,
                   mr.id            AS model_relationship_id,
                   mr.name          AS name,
                   mr.display_name  AS display_name,
                   mr.one_to_many   AS one_to_many,
                   rr.created_by    AS created_by,
                   rr.updated_by    AS updated_by,
                   rr.created_at    AS created_at,
                   rr.updated_at    AS updated_at
            """

        results = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            model_relationship_name=relationship.name,
        )

        return (
            cast(RecordRelationship, RecordRelationship.from_node(**relationship))
            for relationship in results
        )

    def get_record_relationships_by_model_tx(
        self,
        tx: Union[Session, Transaction],
        model: Union[ModelRelationship, ModelRelationshipId],
    ) -> Iterator[RecordRelationship]:
        """
        [[TRANSACTIONAL]]

        Given a model relationship, return an iterator over all associated
        record relationships.
        """
        model_relationship_id = get_model_relationship_id(model)

        cql = f"""
        MATCH  ({labels.record("r_from")})
              -[{labels.instance_of()}]->({labels.model("m_from")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.record("r_to")})
              -[{labels.instance_of()}]->({labels.model("m_to")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)
        MATCH (m_from)-[{labels.related_to("mr")} {{ id: $model_relationship_id }}]->(m_to)
        MATCH (r_from)-[rr {{ model_relationship_id: $model_relationship_id }}]->(r_to)
        RETURN r_from.`@id`     AS from,
               r_to.`@id`       AS to,
               TYPE(rr)         AS type,
               rr.id            AS id,
               mr.id            AS model_relationship_id,
               mr.name          AS name,
               mr.display_name  AS display_name,
               mr.one_to_many   AS one_to_many,
               rr.created_by    AS created_by,
               rr.updated_by    AS updated_by,
               rr.created_at    AS created_at,
               rr.updated_at    AS updated_at
        """

        results = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            model_relationship_id=str(model_relationship_id),
        )

        return (
            cast(RecordRelationship, RecordRelationship.from_node(**relationship))
            for relationship in results
        )

    def delete_record_relationships_tx(
        self,
        tx: Union[Session, Transaction],
        *relationships: Union[RecordRelationship, RecordRelationshipId, str],
    ) -> List[RecordRelationshipId]:
        """
        Delete a record relationship.

        WARNING: this is a bad query. Since relationship properties can not be
        indexed, this requires a scan over all record-level relationships in the
        graph. `r2` is explictly *not* matched against a model in `d` - doing so
        makes the query performance even worse.

        This *must* only be used for the legacy `relationships/instancs/bulk`
        deletion endpoint. Use `delete_outgoing_record_relationship_tx` instead.
        """
        cql = f"""
        MATCH  ({labels.record("r1")})
              -[{labels.instance_of()}]->({labels.model("m1")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        UNWIND $relationship_ids AS relationship_id
        MATCH (r1)-[r {{ id: relationship_id }}]->(r2:Record)
        WITH r, r.id AS id
        DELETE r
        RETURN id
        """

        relationship_ids: List[str] = [
            str(get_record_relationship_id(rid)) for rid in relationships
        ]

        deleted = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            relationship_ids=relationship_ids,
        ).records()

        return [rel["id"] for rel in deleted]

    def delete_outgoing_record_relationship_tx(
        self,
        tx: Union[Session, Transaction],
        record: Union[Record, RecordId],
        relationship: Union[RecordRelationship, RecordRelationshipId, str],
    ) -> RecordRelationshipId:
        """
        Delete a record relationship.

        Raises
        ------
        - RecordRelationshipNotFoundError
        """
        cql = f"""
        MATCH  ({labels.record("r1")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model()})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.record("r2")})
              -[{labels.instance_of()}]->({labels.model()})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)
        MATCH (r1)-[r {{ id: $relationship_id }}]->(r2)
        WITH r, r.id AS id
        DELETE r
        RETURN id
        """
        record_id = get_record_id(record)
        relationship_id = get_record_relationship_id(relationship)

        deleted = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            record_id=str(record_id),
            relationship_id=str(relationship_id),
        ).single()

        if deleted is None:
            raise RecordRelationshipNotFoundError(
                record_id=record_id, relationship_id=relationship_id
            )

        return deleted["id"]

    def create_package_proxy_tx(
        self,
        tx: Union[Session, Transaction],
        record: Union[Record, RecordId],
        package_id: int,
        package_node_id: str,
        legacy_relationship_type: str = labels.PROXY_RELATIONSHIP_TYPE,
    ) -> PackageProxy:
        """
        [[TRANSACTIONAL]]

        See `create_package_proxy()`
        """
        record_id = get_record_id(record)

        cql = f"""
        MATCH  ({labels.record("r")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        MERGE ({labels.package("pp")} {{ package_id: $package_id }})-[{labels.in_dataset()}]->(d)
        ON CREATE SET pp.package_node_id = $package_node_id

        MERGE (r)-[{labels.in_package("proxy_relationship")} {{ relationship_type: $relationship_type }}]->(pp)
        ON CREATE SET
            proxy_relationship.id                = randomUUID(),
            proxy_relationship.proxy_instance_id = randomUUID(),
            proxy_relationship.created_at        = datetime(),
            proxy_relationship.created_by        = $user_id
        SET proxy_relationship.updated_at        = datetime(),
            proxy_relationship.updated_by        = $user_id

        RETURN pp,
               proxy_relationship.id                AS id,
               proxy_relationship.proxy_instance_id AS proxy_instance_id,
               proxy_relationship.relationship_type AS relationship_type,
               proxy_relationship.created_at        AS created_at,
               proxy_relationship.updated_at        AS updated_at
        """
        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            record_id=str(record_id),
            package_id=package_id,
            package_node_id=package_node_id,
            relationship_type=legacy_relationship_type,
        ).single()

        if node is None:
            raise RecordNotFoundError(record_id)

        return PackageProxy(
            id=node["id"],
            proxy_instance_id=node["proxy_instance_id"],
            relationship_type=node["relationship_type"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
            created_by=self.user_id,
            updated_by=self.user_id,
            **node["pp"],
        )

    def create_package_proxy(
        self,
        record: Union[Record, RecordId],
        package_id: int,
        package_node_id: str,
        legacy_relationship_type: str = labels.PROXY_RELATIONSHIP_TYPE,
    ) -> PackageProxy:
        """
        Link a record to a package proxy.

        An important thing to note is that package proxies are *relationships*
        between `Record` and `Package` nodes. The id for a package proxy
        lives on the relationship between the two.

        Package proxies can be given an optional `legacy_relationship_type`
        argument to be backwards-compatible with the v1 API. This value is not
        used as the Neo4j relationship type, but is stored as a property on the
        relationship in addition to the ID. By default this is `belongs_to` -
        the relationship type hardcoded by the Python client and the frontend.

        Raises
        ------
        RecordNotFoundError
        """
        with self.session() as s:
            # Safe to use session with `create_package_proxy_tx()`:
            return self.create_package_proxy_tx(
                tx=s,
                record=record,
                package_id=package_id,
                package_node_id=package_node_id,
                legacy_relationship_type=legacy_relationship_type,
            )

    def get_package_proxy(
        self, package_proxy: Union[PackageProxy, PackageProxyId]
    ) -> PackageProxy:
        """
        Raises
        ------
        - RecordNotFoundError
        - PackageProxyNotFoundError
        """
        package_proxy_id = get_package_proxy_id(package_proxy)

        cql = f"""
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")} {{ id: $package_proxy_id }}]-({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        RETURN pp,
               proxy_relationship.id                AS id,
               proxy_relationship.proxy_instance_id AS proxy_instance_id,
               proxy_relationship.relationship_type AS relationship_type,
               proxy_relationship.created_by        AS created_by,
               proxy_relationship.updated_by        AS updated_by,
               proxy_relationship.created_at        AS created_at,
               proxy_relationship.updated_at        AS updated_at
        """

        node = self.execute_single(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            package_proxy_id=str(package_proxy_id),
        ).single()

        if node is None:
            raise PackageProxyNotFoundError(package_proxy_id)

        return PackageProxy(
            id=node["id"],
            proxy_instance_id=node["proxy_instance_id"],
            relationship_type=node["relationship_type"],
            created_by=node["created_by"],
            updated_by=node["updated_by"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
            **node["pp"],
        )

    def get_package_proxies_for_record(
        self, record: Union[Record, RecordId], limit: int, offset: int
    ) -> Tuple[int, List[PackageProxy]]:
        """
        Raises
        ------
        RecordNotFoundError
        """
        record_id = get_record_id(record)

        cql = f"""
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")}]-({labels.record("r")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        RETURN pp,
               proxy_relationship.id                AS id,
               proxy_relationship.proxy_instance_id AS proxy_instance_id,
               proxy_relationship.relationship_type AS relationship_type,
               proxy_relationship.created_by        AS created_by,
               proxy_relationship.updated_by        AS updated_by,
               proxy_relationship.created_at        AS created_at,
               proxy_relationship.updated_at        AS updated_at

        ORDER BY pp.package_id
        SKIP toInteger($offset)
        LIMIT toInteger($limit)
        """

        with self.transaction() as tx:
            self.assert_.record_exists(tx, record)
            total_count = self.count_package_proxies_for_record_tx(tx, record)
            nodes = tx.run(
                cql,
                organization_id=self.organization_id,
                dataset_id=self.dataset_id,
                record_id=str(record_id),
                limit=limit,
                offset=offset,
            ).records()

        return (
            total_count,
            [
                PackageProxy(
                    id=node["id"],
                    proxy_instance_id=node["proxy_instance_id"],
                    relationship_type=node["relationship_type"],
                    created_by=node["created_by"],
                    updated_by=node["updated_by"],
                    created_at=node["created_at"],
                    updated_at=node["updated_at"],
                    **node["pp"],
                )
                for node in nodes
            ],
        )

    def count_package_proxies_for_record_tx(
        self, tx: Union[Session, Transaction], record: Union[Record, RecordId]
    ) -> int:
        """
        Count the total number of proxy packages associated with a record.

        Raises
        ------
        RecordNotFoundError
        """
        record_id = get_record_id(record)

        cql = f"""
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")}]-({labels.record("r")} {{ `@id`: $record_id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        RETURN COUNT(pp) AS total_count
        """
        node = tx.run(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            record_id=str(record_id),
        ).single()

        if node is None:
            raise RecordNotFoundError(record_id)

        return node["total_count"]

    def count_packages_tx(self, tx: Union[Session, Transaction]) -> int:
        """
        See `count_packages()`.
        """

        cql = f"""
        MATCH ({labels.package("p")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        RETURN COUNT(p) AS package_count
        """
        return (
            tx.run(
                cql, organization_id=self.organization_id, dataset_id=self.dataset_id
            )
            .single()
            .get("package_count")
        )

    def count_packages(self) -> int:
        """
        Count all packages in the current (organization, dataset) partition.
        """
        with self.session() as s:
            return self.count_packages_tx(tx=s)

    def get_package_tx(
        self, tx: Union[Session, Transaction], package_id: PackageId
    ) -> Optional[PackageNodeId]:
        """
        See `get_package()`.
        """

        cql = f"""
        MATCH ({labels.package("p")} {{ package_id: $package_id }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        RETURN p.package_node_id AS package_node_id
        LIMIT 1
        """
        package_node_id = (
            tx.run(
                cql,
                organization_id=self.organization_id,
                dataset_id=self.dataset_id,
                package_id=package_id,
            )
            .single()
            .get("package_node_id")
        )

        if package_node_id is None:
            return None
        return PackageNodeId(package_node_id)

    def get_package(self, package_id: PackageId) -> Optional[PackageNodeId]:
        """
        Get a package by its integer ID, return its node ID (if it exists).
        """
        with self.session() as s:
            return self.get_package_tx(tx=s, package_id=package_id)

    def get_all_package_proxies_tx(
        self, tx: Union[Session, Transaction]
    ) -> Iterator[Tuple[PackageProxy, Record]]:
        """
        Get all package proxy nodes.

        Note: this is used by the legacy v1 API and the model exporter.
        """
        cql = f"""
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")}]-({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

        RETURN {{
          r: r,
          created_by: created_by.node_id,
          created_at: created.at,
          updated_by: updated_by.node_id,
          updated_at: updated.at
        }} AS record, pp {{
          .*,
          id:                proxy_relationship.id,
          proxy_instance_id: proxy_relationship.proxy_instance_id,
          relationship_type: proxy_relationship.relationship_type,
          created_by:        proxy_relationship.created_by,
          updated_by:        proxy_relationship.updated_by,
          created_at:        proxy_relationship.created_at,
          updated_at:        proxy_relationship.updated_at
        }} AS package_proxy
        """

        results = tx.run(
            cql, organization_id=self.organization_id, dataset_id=self.dataset_id
        )

        return (
            (
                PackageProxy(**node["package_proxy"]),
                Record.from_node(
                    node["record"]["r"],
                    created_by=node["record"]["created_by"],
                    updated_by=node["record"]["updated_by"],
                    created_at=node["record"]["created_at"],
                    updated_at=node["record"]["updated_at"],
                ),
            )
            for node in results
        )

    def delete_package_proxy(
        self,
        src_record: Union[Record, RecordId],
        package_id: Union[PackageId, PackageNodeId],
    ) -> PackageProxy:
        """
        Raises
        ------
        RecordNotFoundError
        """
        src_record_id = get_record_id(src_record)

        cql = f"""
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")}]-({labels.record("r")} {{ `@id`: $src_record_id }})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        WHERE pp.package_node_id = $package_id OR pp.package_id = $package_id

        WITH proxy_relationship,
             proxy_relationship.id                AS id,
             proxy_relationship.proxy_instance_id AS proxy_instance_id,
             proxy_relationship.relationship_type AS relationship_type,
             proxy_relationship.created_by        AS created_by,
             proxy_relationship.updated_by        AS updated_by,
             proxy_relationship.created_at        AS created_at,
             proxy_relationship.updated_at        AS updated_at,
             pp.package_id                        AS package_id,
             pp.package_node_id                   AS package_node_id

        DELETE proxy_relationship

        RETURN id,
               proxy_instance_id,
               relationship_type,
               package_id,
               package_node_id,
               created_by,
               updated_by,
               created_at,
               updated_at
        """
        node = self.execute_single(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            src_record_id=str(src_record_id),
            package_id=package_id,
        ).single()

        if node is None:
            raise RecordNotFoundError(src_record_id)

        return PackageProxy(
            id=node["id"],
            proxy_instance_id=node["proxy_instance_id"],
            package_id=node["package_id"],
            package_node_id=node["package_node_id"],
            relationship_type=node["relationship_type"],
            created_by=node["created_by"],
            updated_by=node["updated_by"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
        )

    def delete_package_proxy_by_id(
        self,
        package_proxy: Union[PackageProxy, PackageProxyId, str],
    ) -> None:
        """
        Raises
        ------
        PackageProxyNotFoundError
        """
        with self.transaction() as tx:
            # Transaction required:
            self.delete_package_proxies_tx(tx, None, package_proxy)

    def delete_package_proxies_tx(
        self,
        tx: Union[Session, Transaction],
        src_record: Optional[Union[Record, RecordId]],
        *package_proxies: Union[PackageProxy, PackageProxyId, str],
    ) -> List[str]:
        """
        Delete multiple package proxies.

        Raises
        ------
        PackageProxyNotFoundError
        """
        package_proxy_ids = [
            str(get_package_proxy_id(package_proxy))
            for package_proxy in package_proxies
        ]

        kwargs = {
            "organization_id": self.organization_id,
            "dataset_id": self.dataset_id,
            "package_proxy_ids": package_proxy_ids,
        }

        match_src_record_clause = ""
        if src_record is not None:
            match_src_record_clause = """{ `@id`: $src_record_id }"""
            kwargs["src_record_id"] = str(get_record_id(src_record))

        cql = f"""
        UNWIND $package_proxy_ids AS package_proxy_id
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")} {{ id: package_proxy_id }}]-({labels.record("r")} {match_src_record_clause})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        WITH proxy_relationship, proxy_relationship.id AS id
        DELETE proxy_relationship
        RETURN id
        """
        nodes = tx.run(cql, **kwargs).records()
        result_node_ids = set([str(node["id"]) for node in nodes])

        for pp_id in package_proxy_ids:
            if pp_id not in result_node_ids:
                raise PackageProxyNotFoundError(PackageProxyId(UUID(pp_id)))

        return list(result_node_ids)

    # TODO: create index on packageId?
    def get_proxy_relationship_counts(
        self, package_id: Union[PackageId, PackageNodeId]
    ) -> List[ProxyRelationshipCount]:
        cql = f"""
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")}]-({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WHERE pp.package_node_id = $package_id OR pp.package_id = $package_id
        RETURN m.name         AS name,
               m.display_name AS display_name,
               COUNT(m.name)  AS count
        """
        nodes = self.execute_single(
            cql,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            package_id=package_id,
        ).records()

        return [
            ProxyRelationshipCount(
                name=node["name"],
                display_name=node["display_name"],
                count=node["count"],
            )
            for node in nodes
        ]

    def get_records_related_to_package_tx(
        self,
        tx: Union[Session, Transaction],
        package_id: Union[PackageId, PackageNodeId],
        related_model_id_or_name: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        relationship_order_by: Optional[str] = None,
        record_order_by: Optional[str] = None,
        ascending: bool = False,
    ) -> Iterator[Tuple[PackageProxy, Record]]:
        """
        Get all records related to a given package matching a specified
        model.
        """
        model = self.get_model_tx(tx, related_model_id_or_name)
        properties: List[ModelProperty] = self.get_properties_tx(tx, model)
        property_map: Dict[str, ModelProperty] = {p.name: p for p in properties}

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            package_id=package_id,
            model_id=str(model.id),
        )

        cql = f"""
        MATCH  ({labels.package("pp")})
             <-[{labels.in_package("proxy_relationship")}]-({labels.record("r")})
              -[{labels.instance_of()}]->({labels.model("m")} {{ id: $model_id }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WHERE pp.package_node_id = $package_id OR pp.package_id = $package_id

        MATCH (r)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (r)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

        RETURN // record
               r,
               created.at AS r_created_at,
               updated.at AS r_updated_at,
               // package proxy
               pp,
               proxy_relationship.id                AS pp_id,
               proxy_relationship.proxy_instance_id AS proxy_instance_id,
               proxy_relationship.relationship_type AS relationship_type,
               proxy_relationship.created_by        AS pp_created_by,
               proxy_relationship.updated_by        AS pp_updated_by,
               proxy_relationship.created_at        AS pp_created_at,
               proxy_relationship.updated_at        AS pp_updated_at

        """

        if record_order_by is not None:
            cql += f"""
            ORDER BY r[$record_order_by] {"ASC" if ascending else "DESC"}
            """
            kwargs["record_order_by"] = record_order_by

        if offset is not None:
            cql += """
            SKIP toInteger($offset)
            """
            kwargs["offset"] = offset

        if limit is not None:
            cql += """
            LIMIT toInteger($limit)
            """
            kwargs["limit"] = limit

        # Relationship ordering isn't supported, as all relationships between
        # packages and records are connected by a singular relationship type
        # (:`@IN_PACKAGE`). Due to this, it only makes sense to sort by
        # record.

        results = tx.run(cql, **kwargs)

        return (
            (
                PackageProxy(
                    id=node["pp_id"],
                    proxy_instance_id=node["proxy_instance_id"],
                    relationship_type=node["relationship_type"],
                    created_by=node["pp_created_by"],
                    updated_by=node["pp_updated_by"],
                    created_at=node["pp_created_at"],
                    updated_at=node["pp_updated_at"],
                    **node["pp"],
                ),
                Record.from_node(
                    node["r"],
                    property_map=property_map,
                    fill_missing=True,
                    created_by=self.user_id,
                    updated_by=self.user_id,
                    created_at=node["r_created_at"],
                    updated_at=node["r_updated_at"],
                ),
            )
            for node in results
        )

    # -------------------------------------------------------------------------
    # LEGACY API
    # -------------------------------------------------------------------------

    def get_model_relationships_tx(
        self,
        tx: Union[Session, Transaction],
        from_: Optional[Union[Model, ModelId, str]] = None,
        relation: Optional[Union[ModelRelationshipId, str]] = None,
        to: Optional[Union[Model, ModelId, str]] = None,
        one_to_many: Optional[bool] = None,
    ) -> List[ModelRelationship]:
        """
        Get all model relationships in the partitioned view of the database.

        By default this returns linked properties *and* regular relationships.
        If `one_to_many` is `False` this only returns linked properties, if
        `True` it only returns regular relationships.

        [[TRANSACTIONAL]]
        """

        from_id: Optional[ModelId] = get_model_id(from_) if from_ is not None else None
        to_id: Optional[ModelId] = get_model_id(to) if to is not None else None

        kwargs: Dict[str, GraphValue] = dict(
            dataset_id=self.dataset_id, organization_id=self.organization_id
        )
        cql = f"""
        MATCH  ({labels.model("m")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.model("n")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        MATCH (m)-[{labels.related_to("r")}]->(n)
        """
        conditions = []

        if relation is not None:
            conditions.append(
                "r.id = $relationship_id_or_name OR r.name = $relationship_id_or_name\n"
            )
            kwargs["relationship_id_or_name"] = str(relation)

        if from_id is not None:
            conditions.append("m.id = $from_model_id\n")
            kwargs["from_model_id"] = str(from_id)

        if to_id is not None:
            conditions.append("n.id = $to_model_id\n")
            kwargs["to_model_id"] = str(to_id)

        if one_to_many is not None:
            conditions.append("r.one_to_many = $one_to_many\n")
            kwargs["one_to_many"] = one_to_many

        if len(conditions) > 0:
            cql += "WHERE " + (" AND ".join(conditions))

        cql += """
        RETURN r,
               r.type  AS type,
               m.id    AS from,
               n.id    AS to
        """

        relationships = tx.run(cql, **kwargs).records()

        return [
            ModelRelationship(
                id=rel["r"]["id"],
                name=rel["r"]["name"],
                type=rel["type"],
                from_=rel["from"],
                to=rel["to"],
                one_to_many=rel["r"]["one_to_many"],
                display_name=rel["r"]["display_name"],
                description=rel["r"]["description"],
                created_by=rel["r"]["created_by"],
                updated_by=rel["r"]["updated_by"],
                created_at=rel["r"]["created_at"],
                updated_at=rel["r"]["updated_at"],
                index=rel["r"]["index"],
            )
            for rel in relationships
        ]

    def get_related_models_tx(
        self, tx: Union[Session, Transaction], start_from: Union[Model, ModelId, str]
    ) -> List[Model]:
        """
        Get all models related to the specified model by way of records
        connected to it.
        """
        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_id=self.dataset_id
        )

        cql = f"""
        MATCH  ({labels.record("r_from")})
              -[{labels.instance_of()}]->({labels.model("m_from")} {{ {match_clause("start_model", start_from, kwargs)} }})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.record("r_to")})
              -[{labels.instance_of()}]->({labels.model("m_to")})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        MATCH (r_from)-[r]-(r_to)  // undirected

        MATCH (m_to)-[{labels.created_by("created")}]->({labels.user("c")})
        MATCH (m_to)-[{labels.updated_by("updated")}]->({labels.user("u")})

        RETURN DISTINCT
               m_to,
               size(()-[{labels.instance_of()}]->(m_to)) AS count,
               created.at                         AS created_at,
               c.node_id                          AS created_by,
               updated.at                         AS updated_at,
               u.node_id                          AS updated_by
        """
        start_from_model: Model = self.get_model_tx(tx, start_from)

        nodes = tx.run(cql, **kwargs).records()

        related: List[Model] = [
            cast(
                Model,
                Model.from_node(
                    **node["m_to"],
                    count=node["count"],
                    created_by=node["created_by"],
                    updated_by=node["updated_by"],
                    created_at=node["created_at"],
                    updated_at=node["updated_at"],
                ),
            )
            for node in nodes
        ]

        if not related:
            return []

        return [start_from_model] + related

    def get_related_models(self, start_from: Union[Model, ModelId, str]) -> List[Model]:
        """
        Get all models related to the specified model by way of records
        connected to it.
        """
        with self.session() as s:
            return self.get_related_models_tx(tx=s, start_from=start_from)

    def get_related_records_tx(
        self,
        tx: Union[Session, Transaction],
        start_from: Union[Record, RecordId],
        model_name: str,
        order_by: Optional[ModelOrderBy] = None,
        limit: int = 100,
        offset: int = 0,
        include_incoming_linked_properties: bool = False,
    ) -> Iterator[Tuple[RecordRelationship, Record]]:
        """
        Get related records from a starting record of a specific model type.

        Note: directionality is ignored when relating the record specified by
        `start_from` to any records whose corresponding model name matches
        `model_name`.

        This includes incoming linked properties, but *not* outgoing linked
        properties.  Outgoing linked properties are displayed as real
        properties, but there can be many incoming linked property relationships.

        The following table illustrates which relationships are returned from
        this method:

        +------+--------+--------+
        |      |INCOMING|OUTGOING|
        +------+--------+--------+
        |NORMAL|  yes   |  yes   |
        +------+--------+--------+
        |LINKED|  yes   |  no    |
        +------+--------+--------+

        """
        from_record = get_record_id(start_from)

        kwargs: Dict[str, GraphValue] = dict(
            dataset_id=self.dataset_id,
            organization_id=self.organization_id,
            from_record=str(from_record),
            offset=offset,
            limit=limit,
            include_incoming_linked_properties=include_incoming_linked_properties,
        )

        related: List[Tuple[RecordRelationship, Record]] = []

        target_model = self.get_model_tx(tx, model_name)

        property_map: Dict[str, ModelProperty] = {
            p.name: p for p in self.get_properties_tx(tx, target_model)
        }

        cql = f"""
        MATCH  ({labels.record("r")} {{ `@id`: $from_record }})
              -[{labels.instance_of()}]->({labels.model("mr")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        MATCH  ({labels.record("s")})
              -[{labels.instance_of()}]->({labels.model("ms")} {{ {match_clause("model_name", model_name, kwargs)} }})
              -[{labels.in_dataset()}]->(d)
              -[{labels.in_organization()}]->(o)

        MATCH (r)-[record_relationship]-(s)
        WITH *, r = endNode(record_relationship) AS is_incoming

        MATCH (mr)-[{labels.related_to("model_relationship")}]-(ms)
        WHERE model_relationship.id = record_relationship.model_relationship_id
        AND (model_relationship.one_to_many = TRUE OR ($include_incoming_linked_properties AND is_incoming))

        MATCH (s)-[{labels.created_by("created")}]->({labels.user("created_by")})
        MATCH (s)-[{labels.updated_by("updated")}]->({labels.user("updated_by")})

        RETURN model_relationship.type         AS type,
               model_relationship.name         AS name,
               model_relationship.display_name AS display_name,
               model_relationship.one_to_many  AS one_to_many,
               record_relationship             AS rel,

               CASE WHEN is_incoming THEN s.`@id` ELSE r.`@id` END AS from,
               CASE WHEN is_incoming THEN r.`@id` ELSE s.`@id` END AS to,

               s                  AS s,
               created.at         AS s_created_at,
               created_by.node_id AS s_created_by,
               updated.at         AS s_updated_at,
               updated_by.node_id AS s_updated_by
        """

        if order_by is not None:

            cql += "ORDER BY "

            if order_by.is_field:

                order_by_field = cast(ModelOrderByField, order_by)
                asc = "ASCENDING" if order_by_field.ascending else "DESCENDING"

                if order_by_field.is_created_at:
                    cql += f"s_created_at {asc}\n"
                elif order_by_field.is_updated_at:
                    cql += f"s_updated_at {asc}\n"
                else:
                    if order_by_field.name not in property_map:
                        raise OperationError(
                            f"couldn't get get related records for model [{target_model.name}]",
                            cause=ModelPropertyNotFoundError(
                                model=target_model.name,
                                property_name=order_by_field.name,
                            ),
                        )
                    cql += f"s[$order_by_field_name] {asc}\n"
                    kwargs["order_by_field_name"] = order_by_field.name

            elif order_by.is_relationship:

                order_by_relationship = cast(ModelOrderByRelationship, order_by)
                asc = "ASCENDING" if order_by_relationship.ascending else "DESCENDING"

                if not order_by_relationship.is_supported_type:
                    raise NotImplementedError(
                        f"get_related_records: sorting on [relationship].[{order_by_relationship.type}] not supported"
                    )

                # ignore the actual value (label, whatever):
                cql += f"type {asc}\n"

            # In all cases, break ties with the sort by key. Records created in
            # the same transaction will have the same "created_at" date, but the
            # sort key is monotonically increasing.
            cql += ", s.`@sort_key` ASCENDING"
        else:
            cql += "ORDER BY s.`@sort_key` ASCENDING"

        cql += f"""
        SKIP toInteger($offset)
        LIMIT toInteger($limit)
        """

        nodes = tx.run(cql, **kwargs).records()

        for node in nodes:
            rr = RecordRelationship(
                id=node["rel"]["id"],
                from_=node["from"],
                to=node["to"],
                type=node["type"],
                model_relationship_id=node["rel"]["model_relationship_id"],
                name=node["name"],
                display_name=node["display_name"],
                one_to_many=node["one_to_many"],
                created_by=node["rel"]["created_by"],
                updated_by=node["rel"]["updated_by"],
                created_at=node["rel"]["created_at"],
                updated_at=node["rel"]["updated_at"],
            )
            r = Record.from_node(
                node["s"],
                property_map=property_map,
                fill_missing=True,
                created_at=node["s_created_at"],
                created_by=node["s_created_by"],
                updated_at=node["s_updated_at"],
                updated_by=node["s_updated_by"],
            )

            yield (rr, r)

    def get_related_records(
        self,
        start_from: Union[Record, RecordId],
        model_name: str,
        order_by: Optional[ModelOrderBy] = None,
        limit: int = 100,
        offset: int = 0,
        include_incoming_linked_properties: bool = False,
    ) -> List[Tuple[RecordRelationship, Record]]:
        """
        Get all models related to the specified model by way of records
        connected to it.
        """
        with self.transaction() as tx:
            return list(
                self.get_related_records_tx(
                    tx=tx,
                    start_from=start_from,
                    model_name=model_name,
                    order_by=order_by,
                    limit=limit,
                    offset=offset,
                    include_incoming_linked_properties=include_incoming_linked_properties,
                )
            )

    def create_model_relationship_stub_tx(
        self,
        tx: Union[Session, Transaction],
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ModelRelationshipStub:
        """
        Add a new relationship stub between two Models. A relationship stub
        is a relationship where only the type is specified (and reserved), but
        no "from" and "to" `Model` nodes are specified. This is needed to
        accomodate the Python client worflow

        ```
        nc_one = new_model.create_record(values)
        nc_two = new_model.create_record({'an_integer': 1, 'a_bool': False, 'a_string': '', 'a_datetime': datetime.datetime.now()})
        nc_three = new_model.create_record({'an_integer': 10000, 'a_bool': False, 'a_string': '43132312', 'a_datetime': datetime.datetime.now()})
        nc_four = new_model.create_record({'a_datetime': datetime.datetime.now()})

        new_relationship = dataset.create_relationship_type(f"New_Relationship_{current_ts()}", "a new relationship")

        nr_one = new_relationship.relate(nc_one, nc_two)
        nr_four = new_relationship.relate(nc_four, nc_one)
        nr_five = new_relationship.relate(nc_four, nc_two)
        ```
        """
        relationship_name = validate_relationship_name(name)
        relationship_type = get_relationship_type(name)
        display_name = display_name or name
        description = description or ""

        # Creating models and creating relationship stubs are the two ways to
        # initialize a dataset in Neo4j.
        self.db.initialize_organization_and_dataset(
            tx,
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            organization_node_id=self.organization_node_id,
            dataset_node_id=self.dataset_node_id,
        )

        # Don't allow duplicate stub names
        if len(list(self.get_model_relationship_stubs_tx(tx, relationship_name))) > 0:
            raise OperationError(
                f"model relationship already exists",
                cause=DuplicateModelRelationshipError(
                    name, from_model="*", to_model="*"
                ),
            )

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            user_id=str(self.user_id),
            relationship_type=relationship_type,
            name=name,
            display_name=display_name,
            description=description,
        )
        cql = f"""
        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        CREATE ({labels.model_relationship_stub("r")} {{
                 id: randomUUID(),
                 name: $name,
                 display_name: $display_name,
                 description: $description,
                 type: $relationship_type,
                 created_at: datetime(),
                 updated_at: datetime(),
                 created_by: $user_id,
                 updated_by: $user_id
               }})
        CREATE (r)-[{labels.in_dataset()}]->(d)
        RETURN r
        """
        relationship = tx.run(cql, **kwargs).single()

        if not relationship:
            raise OperationError(
                f"couldn't create model relationship [{relationship_type}]",
                cause=LegacyModelRelationshipNotFoundError(id=None, name=name),
            )

        return ModelRelationshipStub(
            id=relationship["r"]["id"],
            type=relationship["r"]["type"],
            name=relationship["r"]["name"],
            display_name=relationship["r"]["display_name"],
            description=relationship["r"]["description"],
            created_at=relationship["r"]["created_at"],
            updated_at=relationship["r"]["updated_at"],
            created_by=relationship["r"]["created_by"],
            updated_by=relationship["r"]["updated_by"],
        )

    def create_model_relationship_stub(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ModelRelationshipStub:
        with self.transaction() as tx:
            return self.create_model_relationship_stub_tx(
                tx, name=name, display_name=display_name, description=description
            )

    def get_model_relationship_stub_tx(
        self,
        tx: Union[Session, Transaction],
        relation: Union[ModelRelationship, ModelRelationshipId, str],
    ) -> ModelRelationshipStub:
        id_or_name: str = (
            str(relation)
            if is_model_relationship_id(relation)
            else str(cast(RelationshipName, relation))
        )

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            id_or_name=id_or_name,
        )

        cql = f"""
            MATCH  ({labels.model_relationship_stub("r")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
            WHERE r.id = $id_or_name OR r.name = $id_or_name
            RETURN r
            """

        relationship = tx.run(cql, **kwargs).single()

        if not relationship:
            raise OperationError(
                f"couldn't get model relationship [{id_or_name}]",
                cause=LegacyModelRelationshipNotFoundError(
                    id=id_or_name, name=id_or_name
                ),
            )

        return ModelRelationshipStub(
            id=relationship["r"]["id"],
            type=relationship["r"]["type"],
            name=relationship["r"]["name"],
            display_name=relationship["r"]["display_name"],
            description=relationship["r"]["description"],
            created_at=relationship["r"]["created_at"],
            updated_at=relationship["r"]["updated_at"],
            created_by=relationship["r"]["created_by"],
            updated_by=relationship["r"]["updated_by"],
        )

    def get_model_relationship_stubs_tx(
        self,
        tx: Union[Session, Transaction],
        relationship_name: Optional[RelationshipName] = None,
    ) -> Iterator[ModelRelationshipStub]:
        """
        [[TRANSACTIONAL]]

        Fetch all model relationship stubs matching the given relationship
        type.

        Background
        ----------
        The legacy API supports creating a relationship with only a type given
        and no `from` or `to` nodes. Model relationship "stubs" are needed to
        provide this functionality.
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id, dataset_id=self.dataset_id
        )
        cql = f"""
        MATCH  ({labels.model_relationship_stub("r")})
              -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        """
        if relationship_name is not None:
            cql += f"""
            WHERE r.name = $relationship_name
            """
            kwargs["relationship_name"] = relationship_name
        cql += """
        RETURN r
        """

        relationships = tx.run(cql, **kwargs).records()

        return (
            cast(ModelRelationshipStub, ModelRelationshipStub.from_node(**nodes["r"]))
            for nodes in relationships
        )

    def get_model_relationship_stub(
        self, relation: Union[ModelRelationshipId, RelationshipName]
    ) -> ModelRelationshipStub:
        """
        Fetch a relationship stub matching the provided relationship type.

        Background
        ----------
        The legacy API supports creating a relationship with only a type given
        and no `from` or `to` nodes. Model relationship "stubs" are needed to
        provide this functionality.
        """

        with self.transaction() as tx:
            return self.get_model_relationship_stub_tx(tx=tx, relation=relation)

    def delete_model_relationship_stub_tx(
        self,
        tx: Union[Session, Transaction],
        relation: Union[RelationshipName, ModelRelationshipId],
    ) -> Optional[ModelRelationshipId]:
        """
        Delete a relationship stub.
        """

        kwargs: Dict[str, GraphValue] = dict(
            organization_id=self.organization_id,
            dataset_id=self.dataset_id,
            id_or_name=str(relation),
        )

        cql = f"""
            MATCH  ({labels.model_relationship_stub("r")})
                -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
            WHERE r.id = $id_or_name OR r.name = $id_or_name
            WITH * , r.id as id
            DETACH DELETE r
            RETURN id
            """

        relationship = tx.run(cql, **kwargs).single()

        if not relationship:
            return None

        return relationship["id"]

    def create_legacy_record_relationship_batch_tx(
        self,
        tx: Union[Session, Transaction],
        to_create: List[CreateRecordRelationship],
        relation: Union[ModelRelationshipId, RelationshipName],
    ) -> List[Tuple[RecordRelationship, ModelRelationship]]:
        """
        Legacy: create a record relationship

        There are three ways to create a new record relationship:

        [1] From an existing model relationship
        [2] From a stub - need to create the model relationship
        [3] With an explicit request to create a new model relationship

        """
        try:
            model_relationships: List[ModelRelationship] = []
            memo_relationship: Dict[
                Tuple[Model, Model], ModelRelationship
            ] = {}  # Dict used to cache the calls to get_model_relationships_tx
            memo_stub: Dict[
                Union[ModelRelationshipId, RelationshipName], ModelRelationshipStub
            ] = {}

            for relationship in to_create:
                if relationship.model_relationship_to_create is None:
                    from_model: Optional[Model] = self._get_model_from_record_tx(
                        tx, relationship.from_
                    )
                    if from_model is None:
                        raise OperationError(
                            "couldn't find 'from' model",
                            cause=ModelNotFoundError(str(from_model)),
                        )

                    to_model: Optional[Model] = self._get_model_from_record_tx(
                        tx, relationship.to
                    )
                    if to_model is None:
                        raise OperationError(
                            "couldn't find 'to' model",
                            cause=ModelNotFoundError(str(to_model)),
                        )

                    # [1] If the model relationship already exists that's great - use it.
                    try:
                        potential_model_relationships = [
                            memo_relationship[from_model, to_model]
                        ]
                    except KeyError:
                        potential_model_relationships = self.get_model_relationships_tx(
                            tx,
                            from_=from_model,
                            to=to_model,
                            relation=relation,
                            one_to_many=True,
                        )

                    if len(potential_model_relationships) > 1:
                        raise OperationError(
                            "found duplicate model relationships",
                            cause=DuplicateModelRelationshipError(
                                relationship_name=str(relation),
                                from_model=from_model,
                                to_model=to_model,
                            ),
                        )
                    elif len(potential_model_relationships) == 1:
                        memo_relationship[
                            from_model, to_model
                        ] = potential_model_relationships[0]
                        model_relationship = potential_model_relationships[0]
                    else:
                        # [2] Otherwise, try and create the model relationship from a relationship stub
                        try:
                            stub = memo_stub[relation]
                        except KeyError:
                            stub = self.get_model_relationship_stub_tx(
                                tx, relation=relation
                            )
                            memo_stub[relation] = stub

                        model_relationship = self.create_model_relationship_tx(
                            tx,
                            from_model=from_model,
                            name=stub.name,
                            to_model=to_model,
                            one_to_many=True,
                            display_name=stub.display_name,
                        )
                        memo_relationship[from_model, to_model] = model_relationship
                else:
                    # [3] Create the model relationship from the request
                    model_relationship = self.create_model_relationship_tx(
                        tx,
                        from_model=get_model_id(
                            relationship.model_relationship_to_create.from_
                        ),
                        name=relationship.model_relationship_to_create.name,
                        to_model=get_model_id(
                            relationship.model_relationship_to_create.to
                        ),
                        one_to_many=True,
                        display_name=relationship.model_relationship_to_create.display_name,
                    )

                relationship.model_relationship = model_relationship

            record_relationships = list(
                self.create_record_relationship_batch_tx(tx, to_create)
            )

        except Exception as e:
            log.error("couldn't create record relationships", exc_info=True)
            raise OperationError("couldn't create record relationships", cause=e)

        return record_relationships

    def summarize(self) -> TopologySummary:
        """
        Generate a summary of the graph.
        """
        return metrics.DatabaseMetrics(self).summarize()

    def summarize_record(
        self, record: Union[Record, RecordId], include_incoming_linked_properties: bool
    ) -> List[RecordSummary]:
        """
        Produce a summary of a specific record.
        """
        return metrics.DatabaseMetrics(self).summarize_record(
            record, include_incoming_linked_properties
        )

    def graph_schema_structure_tx(self, tx: Union[Session, Transaction]):
        """
        Get the schema-structure objects (models and model relationships)
        of the graph.
        """
        models = self.get_models_tx(tx)
        relationships = self.get_model_relationships_tx(tx)

        return GraphSchemaStructure(models=models, relationships=relationships)

    def topology(
        self, model_id_or_name: Union[Model, ModelId, str]
    ) -> List[ModelTopology]:
        """
        Generate a topology entry for a given concept.

        A topology is the information about a concept, along with a count of
        all records connected associated records of that type

        Raises
        ------
        RecordNotFoundError
        """
        return metrics.DatabaseMetrics(self).topology(model_id_or_name)

    def delete_dataset(
        self, batch_size: int = 1000, duration: int = 5000
    ) -> DatasetDeletionSummary:
        """
        Delete all data for this dataset, optionally providing a maximum
        time limit in milliseconds. As many nodes will be deleted in the
        given time limit.

        This method is reentrant and can be called multiple times if an
        invocation is interrupted.

        Datasets marked with the property `deleting` = `true` are in the process
        of having all child nodes removed and are not safe to use.
        """
        if batch_size > MAX_BATCH_SIZE:
            raise OperationError(
                f"Batch size is too large: must be less than {MAX_BATCH_SIZE}"
            )

        if duration > MAX_RUN_DURATION:
            raise OperationError(
                f"Maximum run time is too long: must be less than {MAX_RUN_DURATION}"
            )

        def shortcircuit_return(alias):
            return f"""
            WITH {alias}
            CALL apoc.util.validate((apoc.date.currentTimestamp() - $start_time) > $duration, 'timeout', null)
            RETURN COUNT({alias}) AS processed
            """

        # Establish a start time for further operations:
        with self.session() as s:
            start_time = s.run(
                "WITH apoc.date.currentTimestamp() AS start_time RETURN start_time"
            ).single()["start_time"]

        def call_periodically(
            cql: str, check_initial_tx: bool = False
        ) -> Tuple[int, bool]:
            count, i = 0, 0
            kwargs = {
                "organization_id": self.organization_id,
                "dataset_id": self.dataset_id,
                "limit": batch_size,
                "duration": duration,
                "start_time": start_time,
            }

            while True:
                with self.transaction() as tx:
                    try:
                        result = tx.run(cql, **kwargs).single()
                        tx.commit()
                        processed = 0 if result is None else result["processed"]
                        count += processed
                        if processed == 0:
                            return count, True
                    except Exception as e:
                        tx.rollback()
                        message = str(e)
                        if "apoc.util.validate" in message and "timeout" in message:
                            # If the batch size was too large for the initial call,
                            # signal that the batch size is too large, or the
                            # duration is too short to accomplish the unit of work:
                            if i == 0 and check_initial_tx:
                                raise ExceededTimeLimitError
                            else:
                                # A subsequent batch failed to be updated in time.
                                # Rollback the current back, but allow the another
                                # call to be made again in the future.
                                break
                        else:
                            raise OperationError("Couldn't delete dataset", cause=e)
                i += 1

            return count, False

        def summary(
            models: int = 0,
            properties: int = 0,
            records: int = 0,
            packages: int = 0,
            relationship_stubs: int = 0,
            done: bool = False,
        ) -> DatasetDeletionSummary:
            counts = DatasetDeletionCounts(
                models=models,
                properties=properties,
                records=records,
                packages=packages,
                relationship_stubs=relationship_stubs,
            )
            return DatasetDeletionSummary(done=done, counts=counts)

        # Indicate the dataset is marked for deletion:
        cql = f"""
        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        SET d.deleting = true
        RETURN d
        """
        self.execute_single(
            cql, organization_id=self.organization_id, dataset_id=self.dataset_id
        ).single()

        cql = f"""
        MATCH ({labels.package('pp')})
             -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
             -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH pp
        LIMIT $limit
        DETACH DELETE pp
        {shortcircuit_return("pp")}
        """
        package_count, package_done = call_periodically(cql)
        logging.info(f"Deleted {package_count} packages")
        if not package_done:
            return summary(packages=package_count)

        cql = f"""
        MATCH ({labels.record("r")})
             -[{labels.instance_of()}]->({labels.model("m")})
             -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
             -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH r
        LIMIT $limit
        DETACH DELETE r
        {shortcircuit_return("r")}
        """
        record_count, record_done = call_periodically(cql, check_initial_tx=True)
        logging.info(f"Deleted {record_count} records")
        if not record_done:
            return summary(packages=package_count, records=record_count)

        cql = f"""
        MATCH ({labels.model_property("p")})
            <-[{labels.has_property()}]->({labels.model("m")})
             -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
             -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH p
        LIMIT $limit
        DETACH DELETE p
        {shortcircuit_return("p")}
        """
        property_count, property_done = call_periodically(cql)
        logging.info(f"Deleted {property_count} properties")
        if not property_done:
            return summary(
                packages=package_count, records=record_count, properties=property_count
            )

        cql = f"""
        MATCH ({labels.model("m")})
             -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
             -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH m
        LIMIT $limit
        DETACH DELETE m
        {shortcircuit_return("m")}
        """
        model_count, model_done = call_periodically(cql)
        logging.info(f"Deleted {model_count} models")
        if not model_done:
            return summary(
                packages=package_count,
                records=record_count,
                properties=property_count,
                models=model_count,
            )

        cql = f"""
        MATCH ({labels.model_relationship_stub("mr")})
             -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
             -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH mr
        LIMIT $limit
        DETACH DELETE mr
        {shortcircuit_return("mr")}
        """
        relationship_stub_count, relationship_stub_done = call_periodically(cql)
        logging.info(f"Deleted {relationship_stub_count} model relationship stubs")
        if not relationship_stub_done:
            return summary(
                packages=package_count,
                records=record_count,
                properties=property_count,
                models=model_count,
                relationship_stubs=relationship_stub_count,
            )

        # Only when models, properties, records, packages, and relationship stubs
        # are removed, can the parent dataset be removed:
        cql = f"""
        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
             -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        WITH d
        DETACH DELETE d
        RETURN COUNT(*) AS count
        """
        dataset_count = self.execute_single(
            cql, organization_id=self.organization_id, dataset_id=self.dataset_id
        ).single()["count"]
        logging.info(f"Deleted {dataset_count} dataset")

        return summary(
            done=True,
            packages=package_count,
            records=record_count,
            properties=property_count,
            models=model_count,
            relationship_stubs=relationship_stub_count,
        )
