import re
from itertools import chain
from typing import Dict, List, Optional, Union
from uuid import UUID

import neo4j  # type: ignore

from ..models import (
    GraphValue,
    Model,
    ModelId,
    ModelProperty,
    ModelPropertyId,
    ModelRelationship,
    ModelRelationshipId,
    Record,
    RecordId,
    RelationshipName,
)
from ..models import datatypes as dt


def match_clause(
    name: str,
    value: Union[
        Model,
        ModelId,
        Record,
        RecordId,
        ModelProperty,
        ModelPropertyId,
        ModelRelationship,
        ModelRelationshipId,
        RelationshipName,
        UUID,
        str,
    ],
    kwargs: Dict[str, GraphValue] = dict(),
    qualifier: Optional[str] = None,
    property_operator: str = ":",
) -> str:
    """
    Generates a match clause string for matching against any database object
    that possesses `id` and `name` properties.

    If `value` is parseable as a UUID, `name` is assumed to equal "id",
    otherwise, it is set as "name".

    If `qualifier` is provided, the rendered value of "id" or "name" will be
    that of `{qualifier}`.id or `{qualifier}`.name.
    """
    if isinstance(value, Model):
        value = ModelId(value.id)
    elif isinstance(value, ModelProperty):
        value = ModelPropertyId(value.id)
    elif isinstance(value, Record):
        value = RecordId(value.id)
    elif isinstance(value, ModelRelationship):
        value = ModelRelationshipId(value.id)

    use_qualifier: str = ""
    if qualifier is not None:
        use_qualifier = f"`{qualifier}`."

    match_id_or_name = (
        f"{use_qualifier}id {property_operator} ${name}"
        if dt.is_uuid(value)
        else f"{use_qualifier}name {property_operator} ${name}"
    )
    kwargs[name] = value
    return match_id_or_name


def displayable(s: str) -> str:
    """
    Converts a string into a "displayable" string, like one suitable for
    use as a title, etc.

    Non-alphanumeric characters  (including "_") are converted into whitespace.
    """
    return re.sub(r"[\W_]+", " ", s).capitalize()


def map_traversal_nodes_to_variables(
    *relationships: neo4j.Relationship, start: int = 0
) -> Dict[ModelId, str]:
    """
    Given a traversal path composed of a series of relationships, generate a
    Cypher variable for each node in the traversal, returning a map from model
    id to variable.

    Variables are `r{start}`, `r{start+1}`, ...
    """
    assert len(relationships) > 0

    return {
        ModelId(id): f"r{k}"
        for (k, id) in enumerate(
            set(
                chain.from_iterable(
                    (r.start_node["id"], r.end_node["id"]) for r in relationships
                )
            ),
            start=start or 0,
        )
    }


def match_relationship_clause(
    variables: Dict[str, str], *paths: neo4j.Path, use_types=False
) -> List[str]:
    """
    Generates a CQL MATCH clause for query based on the given path:

    If the given path `p`, contains a relationships

      - (a:Foo)-[:UP]->(b:Bar)
      - (b:Bar)-[:DOWN]->(c:Baz)

    then the clause

      MATCH (a:Foo)-[:UP]->(b:Bar)-[:DOWN]->(c:Baz)

    will be generated.
    """
    clauses = []

    for path in paths:
        clause = []

        # Determine the directionality/ordering of the relationships in the
        # path:
        #
        # Either (1) [(a)-[r1]-(b), (b)-[r2]-(c)], or
        #        (2) [(b)-[r2]-(c), (a)-[r1]-(b)] (reversed)
        #
        # If (2), reverse the relationship list prior to processing:
        rs = list(path.relationships)

        # Check if the ordering need to be reversed:
        if len(rs) > 1 and rs[0].start_node["id"] == rs[1].end_node["id"]:
            rs.reverse()

        for i, r in enumerate(rs):
            rstart, rtype, rend = (
                str(r.start_node["id"]),
                r["type"],
                str(r.end_node["id"]),
            )
            if i == 0:
                clause.append(f"({variables[rstart]})")
            if use_types:
                clause.append(f"-[:{rtype}]")
            else:
                clause.append("-")
            clause.append(f"-({variables[rend]})")
        clauses.append("MATCH " + ("".join(clause)))
    return clauses
