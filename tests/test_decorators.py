import pytest
from requests.exceptions import ConnectionError

from server.decorators import touch_dataset_timestamp
from server.errors import ExternalRequestError


@touch_dataset_timestamp
def sample_route(db):
    return "OK"


@touch_dataset_timestamp
def sample_error_route(db):
    raise Exception("could not complete request")


def test_touch_dataset_timestamp(partitioned_db, api_client, app_context):
    api_client.clear_requests()

    assert sample_route(partitioned_db) == "OK"
    assert api_client.touch_dataset_requests == [
        {
            "organization_id": partitioned_db.organization_id,
            "dataset_id": partitioned_db.dataset_id,
        }
    ]


def test_do_not_touch_dataset_timestamp_when_request_fails(
    partitioned_db, api_client, app_context
):
    api_client.clear_requests()

    with pytest.raises(Exception):
        sample_error_route(partitioned_db)

    assert api_client.touch_dataset_requests == []


def test_call_victor_ops_when_touch_dataset_timestamp_fails(
    partitioned_db, victor_ops_client, api_client, app_context
):
    api_client.clear_requests()
    api_client.raise_exception(ConnectionError())
    assert sample_route(partitioned_db)
    assert victor_ops_client._alert.call_count == 1
    assert victor_ops_client._alert.call_args[0][0] == "WARNING"
    assert (
        victor_ops_client._alert.call_args[0][1]
        == f"organization/{partitioned_db.organization_id}/dataset/{partitioned_db.dataset_id}"
    )
