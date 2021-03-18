"""
This defines the indices and uniqueness constraints that are needed for
the service to be performant and maintain a correct data model.
"""
from typing import Optional

import structlog  # type: ignore

from ..config import Config
from ..db import Database

logger = structlog.get_logger(__name__)

INDEXES = [
    "CREATE INDEX ON :Record(`@sort_key`)",
    "CREATE INDEX ON :Dataset(id)",  # Not unique across organizations
    "CREATE INDEX ON :Model(name)",
    "CREATE INDEX ON :ModelProperty(name)",
    "CREATE INDEX ON :ModelRelationshipStub(name)",
    "CREATE INDEX ON :Package(package_id)",  # Not unique across organizations
]

CONSTRAINTS = [
    "CREATE CONSTRAINT ON (o:Organization) ASSERT o.id IS UNIQUE",
    "CREATE CONSTRAINT ON (o:Organization) ASSERT o.node_id IS UNIQUE",
    "CREATE CONSTRAINT ON (d:Dataset) ASSERT d.node_id IS UNIQUE",
    "CREATE CONSTRAINT ON (m:Model) ASSERT m.id IS UNIQUE",
    "CREATE CONSTRAINT ON (u:User) ASSERT u.node_id IS UNIQUE",
    "CREATE CONSTRAINT ON (p:ModelProperty) ASSERT p.id IS UNIQUE",
    "CREATE CONSTRAINT ON (r:Record) ASSERT r.`@id` IS UNIQUE",
    "CREATE CONSTRAINT ON (p:Package) ASSERT p.package_node_id IS UNIQUE",
    "CREATE CONSTRAINT ON (m:ModelRelationshipStub) ASSERT m.id IS UNIQUE",
]


def setup(db: Optional[Database] = None):
    """
    Setups up the indices and constraints required by the service to function
    properly
    """
    if db is None:
        db = Database.from_config(Config())

    def apply(session, cql):
        try:
            session.run(cql).single()
            logger.info(f"OK     - {cql}")
        except Exception as e:
            logger.error(f"FAILED - {cql}\n\t{e}")
            raise e

    assert db is not None  # For mypy

    with db.session() as session:
        for cql in INDEXES:
            apply(session, cql)
        for cql in CONSTRAINTS:
            apply(session, cql)
        apply(session, "CALL db.awaitIndexes")


if __name__ == "__main__":
    setup()
