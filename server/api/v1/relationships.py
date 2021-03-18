from typing import List, Optional, Tuple

from auth_middleware.models import DatasetPermission  # type: ignore
from werkzeug.exceptions import NotFound

from server.api.v1.common import to_legacy_relationship, to_relationship_id_or_name
from server.auth import permission_required
from server.db import PartitionedDatabase
from server.decorators import touch_dataset_timestamp
from server.errors import OperationError
from server.models import (
    JsonDict,
    ModelRelationshipId,
    get_model_relationship_id,
    is_model_relationship_id,
)


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_concept_relationship(
    db: PartitionedDatabase, relationship_id_or_name: str
) -> JsonDict:
    relationship = None
    with db.transaction() as tx:
        try:
            if is_model_relationship_id(relationship_id_or_name):
                # Relationship ID:
                id_: ModelRelationshipId = get_model_relationship_id(
                    relationship_id_or_name
                )
                relationship = db.get_model_relationship_tx(
                    tx, id_
                ) or db.get_model_relationship_stub_tx(tx, id_)

            else:
                # DANGER: Do *not* change the order of this. Relationship stubs must
                # be returned *first* to not break compatibility with the Python client.
                # See https://app.clickup.com/t/426zh9 for motivation.

                # Relationship name:
                relation = to_relationship_id_or_name(relationship_id_or_name)

                try:
                    relationship = db.get_model_relationship_stub_tx(
                        tx=tx, relation=relation
                    )
                except OperationError:
                    relationships = db.get_model_relationships_tx(
                        tx=tx, relation=relation, one_to_many=True
                    )
                    if relationships:
                        relationship = relationships[0]
                    else:
                        relationship = None

        except OperationError:
            # Treat as not found
            pass

    if relationship is None:
        raise NotFound(f"Could not get model relationship [{relationship_id_or_name}]")

    return to_legacy_relationship(relationship)


@permission_required(DatasetPermission.VIEW_GRAPH_SCHEMA)
def get_concept_relationships(db: PartitionedDatabase, **kwargs) -> List[JsonDict]:
    from_: Optional[ModelRelationshipId] = kwargs.get("from", None)
    to: Optional[ModelRelationshipId] = kwargs.get("to", None)

    with db.transaction() as tx:
        real_relationships = [
            to_legacy_relationship(r)
            for r in db.get_model_relationships_tx(
                tx=tx, from_=from_, relation=None, to=to, one_to_many=True
            )
        ]

        if from_ or to:
            return real_relationships

        relationship_stubs = [
            to_legacy_relationship(r) for r in db.get_model_relationship_stubs_tx(tx)
        ]

        # DANGER: Do *not* change the order of this. Relationship stubs must
        # be returned *last* to not break the Python client.
        # See https://app.clickup.com/t/426zh9 for motivation.
        return real_relationships + relationship_stubs


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def create_concept_relationship(
    db: PartitionedDatabase, body: JsonDict
) -> Tuple[JsonDict, int]:
    from_model = body.get("from", None)
    to_model = body.get("to", None)
    name = body["name"]
    display_name = body["display_name"]
    description = body["description"]

    with db.transaction() as tx:
        if from_model is None and to_model is None:
            relationship = to_legacy_relationship(
                db.create_model_relationship_stub_tx(
                    tx, name=name, display_name=display_name, description=description
                )
            )
        else:
            relationship = to_legacy_relationship(
                db.create_model_relationship_tx(
                    tx,
                    from_model=from_model,
                    name=name,
                    display_name=display_name,
                    to_model=to_model,
                    one_to_many=True,
                    description=description,
                )
            )

    return relationship, 201


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def update_concept_relationship(
    db: PartitionedDatabase,
    relationship_id_or_name: ModelRelationshipId,
    body: JsonDict,
) -> JsonDict:

    relationship = db.update_model_relationship(
        relationship=to_relationship_id_or_name(relationship_id_or_name),
        display_name=body["display_name"],
    )
    if relationship is None:
        raise NotFound(f"Could not get model relationship [{relationship_id_or_name}]")
    return [to_legacy_relationship(relationship)]


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA)
@touch_dataset_timestamp
def delete_concept_relationship(
    db: PartitionedDatabase, relationship_id_or_name: ModelRelationshipId
) -> JsonDict:

    with db.transaction() as tx:
        id_or_name = to_relationship_id_or_name(relationship_id_or_name)
        deleted_id = db.delete_model_relationship_stub_tx(tx=tx, relation=id_or_name)
        if deleted_id is None:
            deleted_id = db.delete_model_relationship_tx(tx=tx, relationship=id_or_name)

    if deleted_id is None:
        raise NotFound(f"Could not delete model relationship [{deleted_id}]")
    return [str(deleted_id)]
