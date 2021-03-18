from typing import List, Optional

from auth_middleware.models import DatasetPermission  # type: ignore

from server.auth import permission_required
from server.db import EmbedLinked, PartitionedDatabase, QueryRunner
from server.models import JsonDict
from server.models.query import UserQuery


@permission_required(DatasetPermission.VIEW_RECORDS)
def run(
    db: PartitionedDatabase,
    source_model_id_or_name: str,
    linked: bool,
    body: Optional[JsonDict] = None,
    limit: int = 25,
    offset: int = 0,
) -> List[JsonDict]:
    user_query = UserQuery.schema().load(body)
    qr = QueryRunner(db, user_query)
    return [
        r.to_dict()
        for r in qr.run(
            source_model_id_or_name,
            embed_linked=EmbedLinked.STUB if linked else None,
            limit=limit,
            offset=offset,
        )
    ]
