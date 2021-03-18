from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import connexion  # type: ignore
from auth_middleware.claim import Claim  # type: ignore
from auth_middleware.models import DatasetPermission  # type: ignore
from auth_middleware.models import Role as JwtRole  # type: ignore
from auth_middleware.models import RoleType
from auth_middleware.role import DatasetId as RoleDatasetId  # type: ignore
from auth_middleware.role import DatasetRole
from auth_middleware.role import OrganizationId as RoleOrganizationId
from connexion.exceptions import OAuthProblem  # type: ignore
from flask import has_request_context
from jwt.exceptions import DecodeError  # type: ignore
from werkzeug.exceptions import Forbidden, Locked

from core.clients import PennsieveApiClient
from core.clients.header import trace_id_header_dict
from core.errors import ExternalRequestError

from .config import Config
from .db import Database, PartitionedDatabase
from .errors import InvalidDatasetError, InvalidOrganizationError, LockedDatasetError
from .models import (
    DatasetId,
    DatasetNodeId,
    OrganizationId,
    OrganizationNodeId,
    UserNodeId,
    get_dataset_id,
    get_organization_id,
)


def decode_token(token: str) -> Claim:
    """
    Decode a JWT into a claim.

    Raises
    ------
    - connexion.exceptions.OAuthProblem
    - jwt.exceptions.ExpiredSignatureError
    """
    try:
        claim = Claim.from_token(token, Config.from_app().jwt_config)
    except DecodeError:
        raise connexion.exceptions.OAuthProblem(description="Invalid token")

    # This is a hack to work around Connexion internals.  The security wrapper
    # expects the claim to be a dictionary, but we have already built the
    # `Claim` object. The wrapper is trying to extract a user, which we don't
    # need so stub out `get` to always return `None`
    #
    # Ref: https://github.com/zalando/connexion/blob/08e4536e5e6c284aaabcfb6fa159c738dbae7758/connexion/decorators/security.py#L297

    claim.get = lambda x, y=None: None

    return claim


def fetch_dataset_id(claim: Claim, node_id: DatasetNodeId) -> DatasetId:
    """
    Given a dataset's node ID, attempt to look up its integer ID, first from
    the JWT claim, then the database, then from the Pennsieve API.

    Raises
    ------
    ExternalRequestError
    """
    for role in claim.content.roles:
        if role.type == JwtRole.DATASET_ROLE and role.node_id == node_id:
            return DatasetId(role.id.id)

    dataset_id = Database.from_server().get_dataset_id(node_id)
    if dataset_id is not None:
        return dataset_id

    return DatasetId(
        PennsieveApiClient.get()
        .get_dataset(node_id, headers=dict(**auth_header(), **trace_id_header_dict()))
        .int_id
    )


def into_dataset_id(claim: Claim, raw_dataset_id: str) -> DatasetId:
    """
    Attempt resolve the given dataset identifier to a dataset integer ID.

    - If `raw_dataset_id` is an integer, it will be returned as-is, and will
      not be resolved to an existing dataset until it is actually used.

    - If `raw_dataset_id` is a node ID ("N:dataset:1234"), the ID will be
      resolved to an existing dataset. If the dataset exists, its integer ID
      will be returned. If it does not exist, a `InvalidDatasetError` will
      be raised.

    Raises
    ------
    InvalidDatasetError
    """
    try:
        return get_dataset_id(raw_dataset_id)
    except InvalidDatasetError:
        return fetch_dataset_id(claim, node_id=DatasetNodeId(raw_dataset_id))


def permission_required(*permissions: DatasetPermission, service_only: bool = False):
    """
    Decorator that wraps a route, peels off the JWT `token_info` and `user`
    arguments, checks that the dataset id matches that of the route, and calls
    the inner function with an added `organization_id` argument.

    The default signature of a route in `connexion` is

        ```
        def route(dataset_id: str,
                  body, *params,
                  token_info: Claim,
                  user: Optional[str],
                  organization_id: Optional[str] = None):
            ...
        ```

    When wrapped by this decorator the signature becomes

        ```
        @permission_required(DatasetPermission.EDIT_RECORDS)
        def route(db: PartitionedDatabase, *params):
            ...
        ```
    """

    def decorator(func):

        # It would be nice to use `functools.wraps` to preserve the docstring,
        # (if any) but `connexion` introspects the function signature of the
        # route to figure out what arguments to pass, and will error since the
        # wrapped function does not have a `token_info` argument.

        # `dataset_id` is the dataset ID (integer or node ID) passed in the
        # route.
        def inner(
            dataset_id: str,  # route dataset ID
            token_info: Claim,
            organization_id: Optional[str] = None,  # route organization ID (v1=None)
            user: Optional[str] = None,
            **kwargs
        ):
            if service_only and not token_info.is_service_claim:
                raise OAuthProblem("Route only accessible with a service claim")

            try:
                # Attempt to parse the organization and dataset IDs as integers:
                organization_int_id: Optional[OrganizationId] = (
                    None
                    if organization_id is None
                    else get_organization_id(organization_id)
                )

                # Depending on the version of the API that is accessed, this
                # might be a an integer ID or a node ID. In the latter case,
                # we need to resolve the node ID to an integer ID and store
                # it in the database for later use.

                dataset_int_id = into_dataset_id(token_info, dataset_id)

                auth_context = validate_claim(
                    token_info=token_info,
                    dataset_id=dataset_int_id,
                    permissions=cast(List[DatasetPermission], permissions),
                    organization_id=organization_int_id,
                )

                def is_viewer_request():
                    for permission in permissions:
                        if not RoleType.VIEWER.has_permission(permission):
                            return False
                    return True

                if auth_context.locked and not is_viewer_request():
                    raise LockedDatasetError(str(dataset_int_id))

            except OAuthProblem:
                raise
            except (InvalidOrganizationError, InvalidDatasetError) as e:
                raise OAuthProblem(str(e))
            except ExternalRequestError as e:
                if e.is_client_error:
                    raise OAuthProblem(str(e))
                else:
                    raise e

            return func(
                db=PartitionedDatabase.get_from_server(
                    dataset_id=DatasetId(auth_context.dataset_id),
                    organization_id=OrganizationId(auth_context.organization_id),
                    user_id=UserNodeId(auth_context.user_node_id),
                    organization_node_id=OrganizationNodeId(
                        auth_context.organization_node_id
                    )
                    if auth_context.organization_node_id
                    else None,
                    dataset_node_id=DatasetNodeId(auth_context.dataset_node_id)
                    if auth_context.dataset_node_id
                    else None,
                ),
                **kwargs
            )

        # Default transformations provided by functools.wraps
        inner.__name__ = func.__name__
        inner.__qualname__ = func.__qualname__
        inner.__doc__ = func.__doc__
        inner.__annotations__ = func.__annotations__
        inner.__dict__.update(func.__dict__)

        return inner

    return decorator


# *****************************************************************************
# Note:
#
# auth-service does not rewrite routes containing integer ids to node ids.
# If the route receives an integer id, replace it with the node id from the
# JWT
#
# *** If an organization ID is not supplied, it is assumed the v1 API is being
# used. ***
#
# *****************************************************************************


SERVICE_USER_NODE_ID: UserNodeId = UserNodeId("N:user:service")


@dataclass
class AuthContext:
    organization_id: int
    dataset_id: int
    user_node_id: str
    organization_node_id: Optional[str] = None
    dataset_node_id: Optional[str] = None
    locked: Optional[bool] = None


def auth_header() -> Dict[str, str]:
    """
    Return the JWT authorization header from the current request.
    """
    if has_request_context():
        return {"Authorization": connexion.request.headers["Authorization"]}
    return {}


def validate_claim(
    token_info: Claim,
    dataset_id: DatasetId,
    permissions: List[DatasetPermission],
    organization_id: Optional[OrganizationId] = None,
) -> AuthContext:

    # API v1 does not specify an organization. Additionally, for v1 API
    # compatability, we assume the organization of context is the head
    # organization in the given JWT claim:
    if organization_id is None:
        if token_info.head_organization_node_id is None:
            raise OAuthProblem("Missing organization node id")
        organization_int_id: OrganizationId = OrganizationId(
            token_info.head_organization_id.id
        )
    else:
        organization_int_id = OrganizationId(organization_id)

    dataset_int_id = dataset_id

    if token_info.is_user_claim:
        if token_info.content.node_id is None:
            raise OAuthProblem("Missing user node ID")
        user_node_id = UserNodeId(token_info.content.node_id)
    else:
        user_node_id = SERVICE_USER_NODE_ID

    auth_organization_id = RoleOrganizationId(organization_int_id)
    auth_dataset_id = RoleDatasetId(dataset_int_id)

    if not token_info.has_organization_access(auth_organization_id):
        raise Forbidden

    for permission in permissions:
        if not token_info.has_dataset_access(auth_dataset_id, permission):
            raise Forbidden

    # (invariant):
    # These roles should never be None and are assumed to be valid given the
    # checks above.
    organization_role = token_info.get_role(auth_organization_id)
    dataset_role = token_info.get_role(auth_dataset_id)

    def get_locked():
        for role in token_info.content.roles:
            # we do not pass through the locked field for a wildcard role,
            # since by definition the wildcard means we don't know individual datasets are locked or not
            if role.id == auth_dataset_id and isinstance(role, DatasetRole):
                return role.locked
        return None

    return AuthContext(
        organization_id=organization_int_id,
        dataset_id=dataset_id,
        user_node_id=user_node_id,
        organization_node_id=organization_role.node_id,
        dataset_node_id=dataset_role.node_id,
        locked=get_locked(),
    )
