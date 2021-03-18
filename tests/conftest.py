import os
import uuid
from typing import List, Tuple

import boto3
import pytest
from audit_middleware import Auditor, GatewayHost
from auth_middleware.claim import Claim, ServiceClaim, UserClaim
from auth_middleware.models import RoleType
from auth_middleware.role import (
    DatasetId,
    DatasetRole,
    OrganizationId,
    OrganizationRole,
)
from flask import Flask, current_app
from moto import mock_sqs

from server.app import create_app
from server.config import Config
from server.db import Database, PartitionedDatabase, SearchDatabase, constraints, index
from server.logging import configure_logging

from . import fixtures
from .mocks import (
    MockPennsieveApiClient,
    create_audit_logger,
    create_pennsieve_jobs_client,
    create_victor_ops_client,
)

# 5 minutes
JWT_EXPIRATION_SECS = 60 * 60


def to_utf8(encoded):
    try:
        return str(encoded, "utf-8")
    except:
        return encoded


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    aws_env = {
        "AWS_ACCESS_KEY_ID": None,
        "AWS_SECRET_ACCESS_KEY": None,
        "AWS_SECURITY_TOKEN": None,
        "AWS_SESSION_TOKEN": None,
    }

    # Save original values; add mock keys
    for key in aws_env:
        aws_env[key] = os.environ.get(key)
        os.environ[key] = "testing"

    yield

    # Revert original values
    for key, value in aws_env.items():
        os.environ[key] = value or ""


@pytest.fixture(scope="session", autouse=True)
def enable_logging(request):
    configure_logging()


@pytest.fixture(scope="session")
def config():
    return Config()


@pytest.fixture(scope="session")
def setup_indexes(config):
    db = Database.from_config(config)
    index.setup()
    db.driver.close()


@pytest.fixture(scope="session")
def valid_organization() -> Tuple[OrganizationId, str]:
    return (OrganizationId(1), "N:organization:1")


@pytest.fixture(scope="session")
def other_valid_organization() -> Tuple[OrganizationId, str]:
    return (OrganizationId(2), "N:organization:2")


@pytest.fixture(scope="session")
def invalid_organization() -> Tuple[OrganizationId, str]:
    return (OrganizationId(2), "N:organization:2")


@pytest.fixture(scope="session")
def valid_dataset() -> Tuple[DatasetId, str]:
    return (DatasetId(1), "N:dataset:A-B")


@pytest.fixture(scope="session")
def other_valid_dataset() -> Tuple[DatasetId, str]:
    return (DatasetId(2), "N:dataset:C-D")


@pytest.fixture(scope="session")
def valid_user():
    return (12345, "N:user:U-S-E-R")


@pytest.fixture(scope="session")
def other_user():
    return (3, "N:user:3")


@pytest.fixture(scope="function")
def random_uuid() -> str:
    return uuid.uuid4()


@pytest.fixture(scope="function")
def trace_id() -> str:
    return "1234-5678"


@pytest.fixture
def neo4j(config, setup_indexes):
    """
    Configure the Neo4j instance.
    """
    db = Database.from_config(config)
    # Make sure the database is empty
    with db.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield db

    constraints.check_integrity(db)
    db.driver.close()


@pytest.fixture(scope="function")
def partitioned_db(valid_organization, valid_dataset, valid_user, neo4j):
    from server import models as m

    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset
    _, user_node_id = valid_user

    return PartitionedDatabase(
        db=neo4j,
        organization_id=m.OrganizationId(organization_id.id),
        dataset_id=m.DatasetId(dataset_id.id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )


@pytest.fixture(scope="function")
def other_partitioned_db(valid_organization, valid_dataset, other_user, neo4j):
    from server import models as m

    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset
    _, user_node_id = other_user

    return PartitionedDatabase(
        db=neo4j,
        organization_id=m.OrganizationId(organization_id.id),
        dataset_id=m.DatasetId(dataset_id.id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )


@pytest.fixture(scope="function")
def another_partitioned_db(neo4j, valid_organization, valid_user, other_valid_dataset):
    """
    Same organization, different dataset.

    TODO:rename this
    """
    from server import models as m

    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = other_valid_dataset
    _, user_node_id = valid_user

    return PartitionedDatabase(
        db=neo4j,
        organization_id=m.OrganizationId(organization_id.id),
        dataset_id=m.DatasetId(dataset_id.id),
        user_id=user_node_id,
        organization_node_id=organization_node_id,
        dataset_node_id=dataset_node_id,
    )


@pytest.fixture(scope="function")
def configure_search_database():
    def build(pdb: PartitionedDatabase, dataset_ids: List[DatasetId]):
        return SearchDatabase(pdb.db, pdb.organization_id, pdb.user_id, datasets=[])

    return build


@pytest.fixture(scope="function")
def api_client():
    return MockPennsieveApiClient("localhost")


@pytest.fixture(scope="function")
def sqs(aws_credentials):
    with mock_sqs():
        yield boto3.client("sqs", region_name="us-east-1")


@pytest.fixture(scope="function")
def jobs_client(sqs):
    queue = sqs.create_queue(QueueName="test-job-sqs-queue")
    return create_pennsieve_jobs_client(sqs, queue["QueueUrl"])


@pytest.fixture(scope="function")
def victor_ops_client():
    return create_victor_ops_client("localhost", "dev-data-management")


@pytest.fixture(scope="function")
def audit_logger():
    return create_audit_logger(GatewayHost("localhost"))


@pytest.fixture
def app_context(config, api_client, jobs_client, victor_ops_client, neo4j):
    """
    Stub out a Flask app context without creating a full app/client. Useful for
    testing decorators and other middleware.
    """
    app = Flask(__name__)
    with app.app_context():
        current_app.config["config"] = config
        current_app.config["db"] = neo4j
        current_app.config["api_client"] = api_client
        current_app.config["jobs_client"] = jobs_client
        current_app.config["victor_ops_client"] = victor_ops_client
        yield


@pytest.fixture(scope="function")
def client(neo4j, api_client, jobs_client, victor_ops_client, audit_logger):
    app = create_app(
        db=neo4j,
        api_client=api_client,
        jobs_client=jobs_client,
        victor_ops_client=victor_ops_client,
        audit_logger=audit_logger,
    ).app
    app.testing = True
    return app.test_client()


@pytest.fixture(scope="function")
def configure_client(neo4j, api_client, jobs_client, victor_ops_client):
    def new_client(audit_logger):
        app = create_app(
            db=neo4j,
            api_client=api_client,
            jobs_client=jobs_client,
            victor_ops_client=victor_ops_client,
            audit_logger=audit_logger,
        ).app
        app.testing = True
        return app.test_client()

    return new_client


def config_request(method, client, headers, valid_organization, valid_dataset):
    dataset_id, _ = valid_dataset
    client_keywords = ["json", "query_string"]
    client_kwargs = {"headers": headers}

    def req(url, **kwargs):
        for kw in client_keywords:
            client_kwargs[kw] = kwargs.pop(kw, None)
        m = getattr(client, method)
        if "dataset" not in kwargs:
            kwargs["dataset"] = dataset_id.id
        url = url.format(**kwargs)
        return m(url, **client_kwargs)

    return req


@pytest.fixture(scope="function")
def get(client, auth_headers, trace_id_headers, valid_organization, valid_dataset):
    headers = dict(**auth_headers, **trace_id_headers)
    return config_request("get", client, headers, valid_organization, valid_dataset)


@pytest.fixture(scope="function")
def configure_get(
    configure_client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    def configure_inner(audit_logger):
        headers = dict(**auth_headers, **trace_id_headers)
        return config_request(
            "get",
            configure_client(audit_logger),
            headers,
            valid_organization,
            valid_dataset,
        )

    return configure_inner


@pytest.fixture(scope="function")
def post(client, auth_headers, trace_id_headers, valid_organization, valid_dataset):
    headers = dict(**auth_headers, **trace_id_headers)
    return config_request("post", client, headers, valid_organization, valid_dataset)


@pytest.fixture(scope="function")
def configure_post(
    configure_client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    def configure_inner(audit_logger):
        headers = dict(**auth_headers, **trace_id_headers)
        return config_request(
            "post",
            configure_client(audit_logger),
            headers,
            valid_organization,
            valid_dataset,
        )

    return configure_inner


@pytest.fixture(scope="function")
def put(client, auth_headers, trace_id_headers, valid_organization, valid_dataset):
    headers = dict(**auth_headers, **trace_id_headers)
    return config_request("put", client, headers, valid_organization, valid_dataset)


@pytest.fixture(scope="function")
def configure_put(
    configure_client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    def configure_inner(audit_logger):
        headers = dict(**auth_headers, **trace_id_headers)
        return config_request(
            "put",
            configure_client(audit_logger),
            headers,
            valid_organization,
            valid_dataset,
        )

    return configure_inner


@pytest.fixture(scope="function")
def delete(client, auth_headers, trace_id_headers, valid_organization, valid_dataset):
    headers = dict(**auth_headers, **trace_id_headers)
    return config_request("delete", client, headers, valid_organization, valid_dataset)


@pytest.fixture(scope="function")
def configure_delete(
    configure_client, auth_headers, trace_id_headers, valid_organization, valid_dataset
):
    def configure_inner(audit_logger):
        headers = dict(**auth_headers, **trace_id_headers)
        return config_request(
            "delete",
            configure_client(audit_logger),
            headers,
            valid_organization,
            valid_dataset,
        )

    return configure_inner


@pytest.fixture(scope="function")
def authorized_user_token(
    config, valid_organization, valid_dataset, other_valid_dataset, valid_user
):
    organization_id, organization_node_id = valid_organization
    dataset_id_1, dataset_node_id_1 = valid_dataset
    dataset_id_2, dataset_node_id_2 = other_valid_dataset

    user_id, user_node_id = valid_user
    data = UserClaim(
        id=user_id,
        node_id=user_node_id,
        roles=[
            OrganizationRole(
                id=organization_id, node_id=organization_node_id, role=RoleType.OWNER
            ),
            DatasetRole(
                id=dataset_id_1,
                node_id=dataset_node_id_1,
                role=RoleType.OWNER,
                locked=False,
            ),
            DatasetRole(
                id=dataset_id_2,
                node_id=dataset_node_id_2,
                role=RoleType.OWNER,
                locked=False,
            ),
        ],
    )
    claim = Claim.from_claim_type(data, seconds=JWT_EXPIRATION_SECS)
    return to_utf8(claim.encode(config.jwt_config))


@pytest.fixture(scope="function")
def authorized_service_token(
    config, valid_organization, valid_dataset, other_valid_dataset
):
    organization_id, organization_node_id = valid_organization
    dataset_id_1, dataset_node_id_1 = valid_dataset
    dataset_id_2, dataset_node_id_2 = other_valid_dataset

    data = ServiceClaim(
        roles=[
            OrganizationRole(
                id=organization_id, node_id=organization_node_id, role=RoleType.OWNER
            ),
            DatasetRole(
                id=dataset_id_1,
                node_id=dataset_node_id_1,
                role=RoleType.OWNER,
                locked=False,
            ),
            DatasetRole(
                id=dataset_id_2,
                node_id=dataset_node_id_2,
                role=RoleType.OWNER,
                locked=False,
            ),
        ]
    )

    claim = Claim.from_claim_type(data, seconds=JWT_EXPIRATION_SECS)
    return to_utf8(claim.encode(config.jwt_config))


@pytest.fixture(scope="function")
def auth_headers(authorized_user_token):
    return {"Authorization": "Bearer {}".format(authorized_user_token)}


@pytest.fixture(scope="function")
def auth_headers_for_service(authorized_service_token):
    return {"Authorization": "Bearer {}".format(authorized_service_token)}


@pytest.fixture(scope="function")
def trace_id_headers(trace_id):
    return {Auditor.TRACE_ID_HEADER: trace_id}


@pytest.fixture(scope="function")
def unauthorized_user_token(jwt_config, invalid_organization, valid_dataset):
    organization_id, organization_node_id = invalid_organization
    dataset_id, dataset_node_id = valid_dataset
    data = UserClaim(
        id=12345,
        roles=[
            OrganizationRole(
                id=organization_id, node_id=organization_node_id, role=RoleType.OWNER
            ),
            DatasetRole(id=dataset_id, node_id=dataset_node_id, role=RoleType.OWNER),
        ],
    )
    claim = Claim.from_claim_type(data, seconds=JWT_EXPIRATION_SECS)
    return to_utf8(claim.encode(jwt_config))


@pytest.fixture(scope="function")
def expired_user_token(jwt_config, valid_organization, valid_dataset):
    organization_id, organization_node_id = valid_organization
    dataset_id, dataset_node_id = valid_dataset
    data = UserClaim(
        id=12345,
        roles=[
            OrganizationRole(
                id=organization_id, node_id=organization_node_id, role=RoleType.OWNER
            ),
            DatasetRole(id=dataset_id, node_id=dataset_node_id, role=RoleType.OWNER),
        ],
    )
    claim = Claim.from_claim_type(data, -1)
    return to_utf8(claim.encode(jwt_config))


@pytest.fixture(scope="function")
def organization_token(config, valid_organization, other_valid_dataset, valid_user):
    organization_id, organization_node_id = valid_organization
    user_id, user_node_id = valid_user

    data = UserClaim(
        id=user_id,
        node_id=user_node_id,
        roles=[
            OrganizationRole(
                id=organization_id, node_id=organization_node_id, role=RoleType.OWNER
            )
        ],
    )
    claim = Claim.from_claim_type(data, seconds=JWT_EXPIRATION_SECS)
    return to_utf8(claim.encode(config.jwt_config))


@pytest.fixture(scope="function")
def sample_patient_db(partitioned_db):
    return fixtures.sample_patient_db(partitioned_db)


@pytest.fixture(scope="function")
def movie_db(partitioned_db):
    return fixtures.movie_db(partitioned_db)


@pytest.fixture(scope="function")
def large_partitioned_db(partitioned_db):
    return fixtures.large_db(partitioned_db)
