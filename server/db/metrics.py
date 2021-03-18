from collections import Counter
from typing import Any, Dict, List, Union, cast

from neo4j import Session, Transaction  # type: ignore

from ..errors import OperationError
from ..models import (
    GraphValue,
    Model,
    ModelId,
    ModelSummary,
    ModelTopology,
    Record,
    RecordId,
    RecordSummary,
    RelationshipSummary,
    RelationshipTypeSummary,
    TopologySummary,
    get_record_id,
)
from . import core, labels
from .util import match_clause


class DatabaseMetrics:
    """
    A wrapper class than provides summary, topology, and metrics information
    on a database.
    """

    def __init__(self, db: "core.PartitionedDatabase"):
        self._db = db

    def _get_model_summary_tx(
        self, tx: Union[Session, Transaction]
    ) -> List[ModelSummary]:

        cql = f"""
        MATCH ({labels.model("m")})
               -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
               -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})
        RETURN m.name                                 AS name,
               size(()-[{labels.instance_of()}]->(m)) AS count
        """

        nodes = tx.run(
            cql,
            organization_id=self._db.organization_id,
            dataset_id=self._db.dataset_id,
        ).records()

        return [ModelSummary(name=node["name"], count=node["count"]) for node in nodes]

    def _get_relationship_summary_tx(
        self, tx: Union[Session, Transaction]
    ) -> List[RelationshipSummary]:
        """
        TODO: optimize this. This query aggregates over every relationship
        in the dataset which is not going to be scale for large datasets.

        We will probably need to cache counts on each model relationship to make
        this fast.
        """
        cql = f"""
        MATCH ({labels.dataset("d")} {{ id: $dataset_id }})
              -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

        MATCH (d)<-[{labels.in_dataset()}]-({labels.model("m")})
                  -[{labels.related_to("model_relationship")}]->({labels.model("n")})
                  -[{labels.in_dataset()}]->(d)

        MATCH (m)<-[{labels.instance_of()}]-({labels.record()})
                  -[record_relationship]->({labels.record()})
                  -[{labels.instance_of()}]->(n)
        WHERE record_relationship.model_relationship_id = model_relationship.id

        RETURN model_relationship.name    AS name,
               m.id                       AS from,
               n.id                       AS to,
               COUNT(record_relationship) AS count
        """
        nodes = tx.run(
            cql,
            organization_id=self._db.organization_id,
            dataset_id=self._db.dataset_id,
        ).records()

        return [
            RelationshipSummary(
                name=node["name"],
                from_=node["from"],
                to=node["to"],
                count=node["count"],
            )
            for node in nodes
        ]

    def _get_relationship_type_summary_tx(
        self, relationships: List[RelationshipSummary]
    ) -> List[RelationshipTypeSummary]:

        relationship_types: Dict[str, int] = Counter()
        for r in relationships:
            relationship_types[r.name] += r.count

        return [
            RelationshipTypeSummary(name=r, count=count)
            for r, count in relationship_types.items()
        ]

    def summarize(self):
        """
        Generate a summary of the graph.
        """
        with self._db.transaction() as tx:

            model_summaries = self._get_model_summary_tx(tx)
            relationship_summaries = self._get_relationship_summary_tx(tx)
            relationship_type_summaries = self._get_relationship_type_summary_tx(
                relationship_summaries
            )
            model_count = len(model_summaries)
            model_record_count = sum(m.count for m in model_summaries)
            relationship_count = len(relationship_summaries)
            relationship_record_count = sum(r.count for r in relationship_summaries)
            relationship_type_count = len(relationship_type_summaries)

            return TopologySummary(
                model_summary=model_summaries,
                relationship_summary=relationship_summaries,
                relationship_type_summary=relationship_type_summaries,
                model_count=model_count,
                model_record_count=model_record_count,
                relationship_count=relationship_count,
                relationship_record_count=relationship_record_count,
                relationship_type_count=relationship_type_count,
            )

    def topology(
        self, model_id_or_name: Union[Model, ModelId, str]
    ) -> List[ModelTopology]:
        """
        Gets the topology for a given model, which consists of a list of
        connected models and associated record counts per model.
        """
        with self._db.transaction() as tx:

            try:
                # Check that the model exists:
                self._db.get_model_tx(tx, model_id_or_name)

                kwargs: Dict[str, GraphValue] = dict(
                    organization_id=self._db.organization_id,
                    dataset_id=self._db.dataset_id,
                )

                cql = f"""
                MATCH ({labels.model("m")} {{ {match_clause("model_id_or_name", model_id_or_name, kwargs)} }})
                      -[{labels.in_dataset()}]->({labels.dataset("d")} {{ id: $dataset_id }})
                      -[{labels.in_organization()}]->({labels.organization("o")} {{ id: $organization_id }})

                MATCH (m)-[{labels.related_to()}]-({labels.model("n")})

                WITH *, size(()-[{labels.instance_of()}]->(n)) AS count
                WHERE count > 0

                MATCH (n)-[{labels.created_by("created")}]->({labels.user("c")})
                MATCH (n)-[{labels.updated_by("updated")}]->({labels.user("u")})

                RETURN n.id            AS id,
                       n.name          AS name,
                       n.description   AS description,
                       n.display_name  AS display_name,
                       count           AS count,
                       created.at      AS created_at,
                       updated.at      AS updated_at
                """
                nodes = tx.run(cql, **kwargs).records()

                return [ModelTopology(**node) for node in nodes]

            except Exception as e:
                raise OperationError("couldn't get model topology", cause=e)

    def summarize_record(
        self,
        record: Union[Record, RecordId],
        include_incoming_linked_properties: bool = False,
    ) -> List[RecordSummary]:
        """
        Aggregate the number of related records by model.

        This includes regular record relationships and incoming linked
        properties, but *not* outgoing linked properties.  Outgoing linked
        properties are displayed as real properties, but there can be many
        incoming linked property relationships.
        """
        record_id = get_record_id(record)

        with self._db.transaction() as tx:

            self._db.assert_.record_exists(tx, record)

            cql = f"""
            MATCH ({labels.record("from_record")} {{ `@id`: $record_id }})
                  -[{labels.instance_of()}]->({labels.model("from_model")})
                  -[{labels.in_dataset()}]->({labels.dataset("d")}{{ id: $dataset_id }})
                  -[{labels.in_organization()}]->({labels.organization()}{{ id: $organization_id }})

            MATCH (from_record)
                  -[record_relationship]-({labels.record("to_record")})
                  -[{labels.instance_of()}]->({labels.model("to_model")})
                  -[{labels.in_dataset()}]->(d)
            WHERE NOT(TYPE(record_relationship) IN $reserved_relationship_types)
            WITH *, from_record = endNode(record_relationship) AS is_incoming

            MATCH (from_model)-[{labels.related_to("model_relationship")}]-(to_model)
            WHERE model_relationship.id = record_relationship.model_relationship_id
            AND (model_relationship.one_to_many = TRUE OR ($include_incoming_linked_properties AND is_incoming))

            RETURN to_model.name           AS name,
                   to_model.display_name   AS display_name,
                   COUNT(*)                AS count
            ORDER BY count DESCENDING
            """
            nodes = tx.run(
                cql,
                organization_id=self._db.organization_id,
                dataset_id=self._db.dataset_id,
                record_id=str(record_id),
                reserved_relationship_types=labels.RESERVED_SCHEMA_RELATIONSHIPS,
                include_incoming_linked_properties=include_incoming_linked_properties,
            ).records()

            package_count = self._db.count_package_proxies_for_record_tx(tx, record)

            return [RecordSummary(**node) for node in nodes] + (
                [
                    RecordSummary(
                        name="package", display_name="Files", count=package_count
                    )
                ]
                if package_count
                else []
            )
