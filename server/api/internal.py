from auth_middleware.models import DatasetPermission  # type: ignore

from server.auth import permission_required
from server.db import PartitionedDatabase
from server.models import JsonDict


@permission_required(DatasetPermission.MANAGE_GRAPH_SCHEMA, service_only=True)
def delete_dataset(
    db: PartitionedDatabase, batch_size: int = 1000, duration: int = 5000
) -> JsonDict:
    return db.delete_dataset(batch_size=batch_size, duration=duration).to_dict()
