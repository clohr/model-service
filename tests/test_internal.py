def test_dataset_deletion_denied_for_user_claims(
    large_partitioned_db, client, auth_headers, valid_organization, valid_dataset
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=1000&duration=5000",
        headers=auth_headers,
    )
    assert r.status_code == 401


def test_dataset_deletion_allowed_for_service_claims(
    large_partitioned_db,
    client,
    auth_headers_for_service,
    valid_organization,
    valid_dataset,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=1000&duration=5000",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 200


def test_dataset_deletion_enforce_query_parameter_limits(
    large_partitioned_db,
    client,
    auth_headers_for_service,
    valid_organization,
    valid_dataset,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    # batchSize is too small
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=1",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 400

    # batchSize is too large
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=100000",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 400

    # duration is too small
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?duration=1",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 400

    # duration is too large
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?duration=10001",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 400


def test_dataset_deletion_rolled_back_batch_size(
    large_partitioned_db,
    client,
    auth_headers_for_service,
    valid_organization,
    valid_dataset,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    # Trying to delete with too big a batch size to complete with too short a
    # duration will fail:
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=10000&duration=10",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 408
    # Make sure no records were actually deleted:
    assert large_partitioned_db.summarize().model_record_count == 15000

    # Trying again with a longer duration batch_size will succeed:
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=10000&duration=5000",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 200
    status = r.json
    assert status["done"] is True
    assert status["counts"]["records"] >= 1000

    after_summary = large_partitioned_db.summarize()
    assert after_summary.model_count == 0
    assert after_summary.model_record_count == 0


def test_dataset_deletion_rolled_back_duration(
    large_partitioned_db,
    client,
    auth_headers_for_service,
    valid_organization,
    valid_dataset,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    # Trying to delete with too big a batch size to complete with too short a
    # duration will fail:
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=10000&duration=10",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 408
    # Make sure no records were actually deleted:
    assert large_partitioned_db.summarize().model_record_count == 15000

    # Trying again with a smaller batch size and longer duration will succeed:
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=1000&duration=500",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 200
    status = r.json
    assert status["done"] is False

    # Trying again with a big batch size and longer duration will succeed:
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?batchSize=1000&duration=5000",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 200
    status = r.json
    assert status["done"] is True

    after_summary = large_partitioned_db.summarize()
    assert after_summary.model_count == 0
    assert after_summary.model_record_count == 0


def test_dataset_deletion_with_time_limit(
    large_partitioned_db,
    client,
    auth_headers_for_service,
    valid_organization,
    valid_dataset,
):
    organization_id, _ = valid_organization
    dataset_id, _ = valid_dataset

    model_count, record_count = 0, 0

    # Models should remain as long as records still exist:
    before_summary = large_partitioned_db.summarize()

    # First delete call shouldn't finish it:
    r = client.delete(
        f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?duration=100",
        headers=auth_headers_for_service,
    )
    assert r.status_code == 200
    status = r.json
    assert status["done"] is False
    model_count += status["counts"]["models"]
    record_count += status["counts"]["records"]

    assert before_summary.model_count > 0
    assert before_summary.model_record_count > 0

    for _ in range(100):
        # First delete call shouldn't finish it:
        r = client.delete(
            f"/internal/organizations/{organization_id.id}/datasets/{dataset_id.id}?duration=1000",
            headers=auth_headers_for_service,
        )
        assert r.status_code == 200
        status = r.json
        model_count += status["counts"]["models"]
        record_count += status["counts"]["records"]
        if status["done"] is True:
            break
    else:
        assert status["done"] is True

    after_summary = large_partitioned_db.summarize()
    assert after_summary.model_count == 0
    assert after_summary.model_record_count == 0

    assert before_summary.model_count == model_count
    assert before_summary.model_record_count == record_count
