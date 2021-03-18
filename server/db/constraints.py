import logging

from .core import Database

log = logging.getLogger(__file__)


class IntegrityError(Exception):
    pass


def check_integrity(db: Database) -> None:

    log.info("Checking datasets...")

    cql = """
    MATCH (d:Dataset)-[:`@IN_ORGANIZATION`]->(o:Organization)
    WITH d, COUNT(o) AS num_organizations
    WHERE num_organizations <> 1
    RETURN COUNT(*) > 0 AS error
    """
    if db.execute(cql).single()["error"]:
        raise IntegrityError("Dataset linked to more than one organization")

    log.info("Checking models...")

    cql = """
    MATCH (m:Model)-[:`@IN_DATASET`]->(d:Dataset)
    WITH m, COUNT(d) AS num_datasets
    WHERE num_datasets <> 1
    RETURN COUNT(*) > 0 AS error
    """
    if db.execute(cql).single()["error"]:
        raise IntegrityError("Model is in zero or many datasets")

    log.info("Checking records...")

    cql = """
    MATCH (r:Record)-[:`@INSTANCE_OF`]->(m:Model)
    WITH r, COUNT(m) AS num_models
    WHERE num_models <> 1
    RETURN COUNT(*) > 0 AS error
    """
    if db.execute(cql).single()["error"]:
        raise IntegrityError("Record is instance of no or many models")

    log.info("Checking model relationships...")

    cql = """
    MATCH (m:Model)-[model_relationship]->(n:Model)
    WHERE TYPE(model_relationship) <> "@RELATED_TO"
    AND NOT (m)-[:`@RELATED_TO` {id: model_relationship.id}]->(n)
    RETURN COUNT(*) > 0 AS error
    """
    if db.execute(cql).single()["error"]:
        raise IntegrityError(
            "Found model relationships without `@RELATED_TO` relationship"
        )

    log.info("Checking record relationships...")

    cql = """
    MATCH (r1:Record)-[record_relationship]->(r2:Record)
    MATCH (r1)-[:`@INSTANCE_OF`]->(m1:Model)
    MATCH (r2)-[:`@INSTANCE_OF`]->(m2:Model)

    OPTIONAL MATCH (m1)-[mr1:`@RELATED_TO` {id: record_relationship.model_relationship_id}]->(m2)

    WITH * WHERE mr1 IS NULL

    RETURN COUNT(*) > 0 AS error
    """
    if db.execute(cql).single()["error"]:
        raise IntegrityError("Record relationship does not have model relationships")
