import pytest
from auth_middleware.claim import Claim, UserClaim
from auth_middleware.models import DatasetPermission, RoleType
from auth_middleware.role import DatasetId, DatasetRole, OrganizationRole
from connexion.exceptions import OAuthProblem
from werkzeug.exceptions import Forbidden

from core.dtos import api
from server.auth import permission_required
from server.errors import ExternalRequestError, LockedDatasetError

DEFAULT_USER_ID: int = 12345

DEFAULT_USER_NODE_ID: str = "N:user:U-T"

TOKEN_EXPIRATION_S: int = 10


@permission_required(DatasetPermission.EDIT_RECORDS)
def sample_update_route(db, body):
    def test(expected_organization_id, expected_dataset_id):
        assert db.organization_id == expected_organization_id
        assert db.dataset_id == expected_dataset_id
        assert body == {"k": 1}

    return test


@permission_required(DatasetPermission.VIEW_RECORDS)
def sample_view_route(db, body):
    def test(expected_organization_id, expected_dataset_id):
        assert db.organization_id == expected_organization_id
        assert db.dataset_id == expected_dataset_id
        assert body == {"k": 1}

    return test


def test_permission_required_decorator(app_context, valid_organization, valid_dataset):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id,
                    node_id=dataset_node_id,
                    role=RoleType.OWNER,
                    locked=False,
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    sample_update_route(
        dataset_id=str(dataset_id.id),
        token_info=claim,
        organization_id=str(organization_id.id),
        body={"k": 1},
    )(organization_id.id, dataset_id.id)


def test_authorization_requires_integer_organization_id(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id, node_id=dataset_node_id, role=RoleType.OWNER
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(OAuthProblem):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_node_id),
            body={"k": 1},
        )(organization_id.id, dataset_id.id)


def test_authorization_allows_dataset_integer_id(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id,
                    node_id=dataset_node_id,
                    role=RoleType.OWNER,
                    locked=False,
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    sample_update_route(
        dataset_id=dataset_id.id,
        token_info=claim,
        organization_id=str(organization_id.id),
        body={"k": 1},
    )(organization_id.id, dataset_id.id)


def test_authorization_rejects_nonexistent_dataset_integer_id(
    valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id, node_id=dataset_node_id, role=RoleType.OWNER
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(Forbidden):
        sample_update_route(
            dataset_id=9999,
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )(organization_id.id, dataset_id.id)


def test_authorization_allows_dataset_node_id(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id,
                    node_id=dataset_node_id,
                    role=RoleType.OWNER,
                    locked=False,
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    sample_update_route(
        dataset_id=dataset_node_id,
        token_info=claim,
        organization_id=str(organization_id.id),
        body={"k": 1},
    )(organization_id.id, dataset_id.id)


def test_authorization_resolves_dataset_id_from_api_with_wildcard_claim(
    app_context, api_client, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_dataset_response = api.Dataset(
        id=dataset_node_id, int_id=dataset_id.id, name="foo"
    )

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(id=DatasetId("*"), role=RoleType.EDITOR),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    sample_view_route(
        dataset_id=dataset_node_id,
        token_info=claim,
        organization_id=str(organization_id.id),
        body={"k": 1},
    )(organization_id.id, dataset_id.id)


def test_authorization_rejects_nonexistent_dataset_node_ids_with_wildcard_claim(
    app_context, api_client, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.raise_exception(
        ExternalRequestError(
            status_code=404,
            method="GET",
            url="/datasets/N:dataset:does-not-exist",
            content="Dataset does not exist",
        )
    )

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(id="*", role=RoleType.EDITOR),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(OAuthProblem, match="Dataset does not exist"):
        sample_update_route(
            dataset_id="N:dataset:does-not-exist",
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )(organization_id.id, dataset_id.id)


def test_authorization_requires_user_node_id(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id, node_id=dataset_node_id, role=RoleType.OWNER
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(OAuthProblem, match="Missing user node ID"):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )


def test_permission_requires_an_organization_role_specifier(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                DatasetRole(id=dataset_id, node_id=dataset_node_id, role=RoleType.OWNER)
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(Forbidden):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )


def test_permission_requires_a_dataset_role_specifier(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                )
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(Forbidden):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )


def test_permission_required_to_access_organization_datasets(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                DatasetRole(id=dataset_id, node_id=dataset_node_id, role=RoleType.OWNER)
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(Forbidden):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )


def test_permission_required_to_access_a_specific_organization(
    app_context, valid_organization, other_valid_organization, valid_dataset
):
    organization_id_1, organization_node_id_1 = valid_organization
    organization_id_2, organization_node_id_2 = other_valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id_2,
                    node_id=organization_node_id_1,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id, node_id=organization_node_id_2, role=RoleType.OWNER
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(Forbidden):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_id_1.id),
            body={"k": 1},
        )


def test_permission_required_to_a_access_specific_dataset(
    app_context, valid_organization, valid_dataset, other_valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id_1, _ = valid_dataset
    dataset_id_2, dataset_node_id_2 = other_valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id_2, node_id=dataset_node_id_2, role=RoleType.OWNER
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(Forbidden):
        sample_update_route(
            dataset_id=str(dataset_id_1.id),
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )


def test_permission_required_raises_forbidden_when_dataset_role_is_too_low(
    app_context, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=dataset_id, node_id=dataset_node_id, role=RoleType.VIEWER
                ),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    # sample_route requires EDITOR permissions, which are higher than VIEWER:
    with pytest.raises(Forbidden):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )


def test_authorization_allows_updates_with_wildcard_claim(
    app_context, api_client, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_dataset_response = api.Dataset(
        id=dataset_node_id, int_id=dataset_id.id, name="foo"
    )

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                # token validation should strip any locked field from wildcard claims
                DatasetRole(id=DatasetId("*"), role=RoleType.EDITOR, locked=True),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    sample_update_route(
        dataset_id=str(dataset_id.id),
        token_info=claim,
        organization_id=str(organization_id.id),
        body={"k": 1},
    )


def test_authorization_rejects_updates_with_locked_true_claim(
    app_context, api_client, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_dataset_response = api.Dataset(
        id=dataset_node_id, int_id=dataset_id.id, name="foo"
    )

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(id=dataset_id, role=RoleType.EDITOR, locked=True),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    with pytest.raises(LockedDatasetError):
        sample_update_route(
            dataset_id=str(dataset_id.id),
            token_info=claim,
            organization_id=str(organization_id.id),
            body={"k": 1},
        )


def test_authorization_allows_updates_with_locked_false_claim(
    app_context, api_client, valid_organization, valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset

    api_client.get_dataset_response = api.Dataset(
        id=dataset_node_id, int_id=dataset_id.id, name="foo"
    )

    claim = Claim.from_claim_type(
        UserClaim(
            id=DEFAULT_USER_ID,
            node_id=DEFAULT_USER_NODE_ID,
            roles=[
                OrganizationRole(
                    id=organization_id,
                    node_id=organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(id=dataset_id, role=RoleType.EDITOR, locked=False),
            ],
        ),
        TOKEN_EXPIRATION_S,
    )

    sample_update_route(
        dataset_id=str(dataset_id.id),
        token_info=claim,
        organization_id=str(organization_id.id),
        body={"k": 1},
    )
