from dataclasses import dataclass

import pytest
from blackfynn import Blackfynn

from core.clients import PennsieveApiClient


@dataclass
class Package:
    id: int
    node_id: str


@pytest.fixture
def bf():
    return Blackfynn()


@pytest.fixture
def dataset(bf):
    for dataset in bf.datasets():
        if len(dataset.items) > 0:
            return dataset

    raise Exception("No dataset in this organization has packages - aborting")


@pytest.fixture
def package(bf, dataset):
    """
    Get the package id of any package in the dataset.

    This uses raw requests because the regular methods exposed by the Python
    client do not return integer ids.
    """
    json = bf._api.datasets._get(f"/{dataset.id}/packages?pageSize=1")["packages"][0][
        "content"
    ]
    return Package(id=json["id"], node_id=json["nodeId"])


@pytest.fixture
def auth_header(bf):
    """
    This will be a JWT when used by the service, but the tests need a session
    token to get through the gateway.
    """
    return {"Authorization": f"Bearer {bf._api.token}"}


@pytest.mark.integration
def test_get_packages(bf, dataset, package, auth_header, trace_id_headers):
    client = PennsieveApiClient(bf.settings.api_host)
    packages = client.get_packages(
        dataset.id, [package.id], headers=dict(**auth_header, **trace_id_headers)
    )

    assert packages[package.id]["content"]["nodeId"] == package.node_id
    assert packages[package.id]["content"]["id"] == package.id


@pytest.mark.integration
def test_get_packages_ignores_deleted_packages(
    bf, dataset, package, auth_header, trace_id_headers
):
    client = PennsieveApiClient(bf.settings.api_host)
    packages = client.get_packages(
        dataset.id,
        ["N:package:does-not-exist"],
        headers=dict(**auth_header, **trace_id_headers),
    )
    assert len(packages) == 0


@pytest.mark.integration
def test_get_package_by_node_id(bf, dataset, package, auth_header, trace_id_headers):
    client = PennsieveApiClient(bf.settings.api_host)
    response = client.get_package_ids(
        dataset.id, package.node_id, headers=dict(**auth_header, **trace_id_headers)
    )

    assert response.id == package.id
    assert response.node_id == package.node_id


@pytest.mark.integration
def test_get_datasets(bf, dataset, package, auth_header, trace_id_headers):
    client = PennsieveApiClient(bf.settings.api_host)
    datasets = client.get_datasets(headers=dict(**auth_header, **trace_id_headers))
    assert len(datasets) > 0

    dataset_ids = client.get_dataset_ids(
        headers=dict(**auth_header, **trace_id_headers)
    )
    assert sorted(d.int_id for d in datasets) == sorted(dataset_ids)
