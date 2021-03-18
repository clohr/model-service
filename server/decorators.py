import structlog  # type: ignore
from auth_middleware import JwtConfig  # type: ignore
from auth_middleware.claim import Claim, ServiceClaim, UserClaim  # type: ignore
from auth_middleware.models import RoleType  # type: ignore
from auth_middleware.role import (  # type: ignore
    DatasetId,
    DatasetRole,
    OrganizationId,
    OrganizationRole,
)

from core.clients import PennsieveApiClient, VictorOpsClient
from server.config import Config
from server.db import PartitionedDatabase

logger = structlog.get_logger(__name__)


def to_utf8(encoded):
    try:
        return str(encoded, "utf-8")
    except:
        return encoded


def touch_dataset_timestamp(func):
    """
    Decorator that wraps a route and updates the `updatedAt` timestamp of the
    current dataset in API after every successful request.

    This decorator must be used *after* the `permission_required` decorator
    because this decorator expects to receive a `PartitionedDatabase` as a
    parameter.
    """

    def wrapper(db: PartitionedDatabase, **kwargs):
        result = func(db, **kwargs)

        config = Config.from_app()
        jwt = service_claim(db.organization_id, db.dataset_id, config.jwt_config)

        try:
            PennsieveApiClient.get().touch_dataset(
                db.organization_id,
                db.dataset_id,
                headers={"Authorization": f"Bearer {jwt}"},
            )
        except Exception as e:
            VictorOpsClient.get().warning(
                f"organization/{db.organization_id}/dataset/{db.dataset_id}",
                f"Couldn't touch dataset {db.dataset_id} for organization={db.organization_id}",
            )
            logger.warn(
                f"couldn't touch timestamp for organization/{db.organization_id}/dataset/{db.dataset_id}: {e}"
            )

        return result

    # Connexion does not play well with `functools.wraps`. See note in
    # `auth.permission_required` and
    # https://github.com/zalando/connexion/issues/142#issuecomment-184279033
    # Unfortunately, the `decorator` library did not work correctly here.
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    wrapper.__dict__.update(func.__dict__)

    return wrapper


def service_claim(organization_id, dataset_id, jwt_config: JwtConfig) -> str:
    data = ServiceClaim(
        roles=[
            OrganizationRole(id=OrganizationId(organization_id), role=RoleType.OWNER),
            DatasetRole(id=DatasetId(dataset_id), role=RoleType.OWNER),
        ]
    )
    claim = Claim.from_claim_type(data, seconds=30)
    return to_utf8(claim.encode(jwt_config))
