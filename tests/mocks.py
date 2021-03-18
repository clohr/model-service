import copy
from typing import Dict, List, Union
from unittest.mock import Mock, patch
from uuid import uuid4

from audit_middleware import Auditor, GatewayHost, TraceId

from core.clients import PennsieveApiClient, PennsieveJobsClient, VictorOpsClient
from core.dtos.api import Dataset
from server.errors import MissingTraceId
from server.models import DatasetId, OrganizationId


# TODO: replace with unittest.mock
class MockPennsieveApiClient(PennsieveApiClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_response()
        self.clear_requests()
        self.exc = None

    def _render(self, response):
        if self.exc:
            raise self.exc
        assert response is not None
        return copy.deepcopy(response)

    def get_packages(
        self,
        dataset_id: str,
        package_ids: List[Union[int, str]],
        headers: Dict[str, str],
    ):
        return self._render(self.get_packages_response)

    def get_dataset(self, dataset_id: Union[int, str], headers: Dict[str, str]):
        return self._render(self.get_dataset_response)

    def get_datasets(self, headers: Dict[str, str]) -> List[Dataset]:
        return self._render(self.get_datasets_response)

    def get_dataset_ids(self, headers: Dict[str, str]) -> List[DatasetId]:
        return self._render(self.get_dataset_ids_response)

    def touch_dataset(
        self,
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        headers: Dict[str, str],
    ) -> None:
        if self.exc:
            raise self.exc
        self.touch_dataset_requests.append(
            {"organization_id": organization_id, "dataset_id": dataset_id}
        )
        return None

    def clear_response(self):
        self.get_dataset_response = None
        self.get_dataset_ids_response = None
        self.get_packages_response = None
        self.get_datasets_response = None

    def clear_requests(self):
        self.touch_dataset_requests = []

    def raise_exception(self, exc: Exception):
        self.exc = exc

    def clear_exception(self):
        self.exc = None


def create_victor_ops_client(*args, **kwargs):
    class MockVictorOpsClient(VictorOpsClient):
        def __init__(self):
            super().__init__(*args, **kwargs)

    client = MockVictorOpsClient()
    client._alert = Mock()
    return client


def create_pennsieve_jobs_client(*args, **kwargs):
    class MockPennsieveJobsClient(PennsieveJobsClient):
        def __init__(self):
            super().__init__(*args, **kwargs)

    client = MockPennsieveJobsClient()

    def call_send_changelog_event(*args, **kwargs):
        return MockPennsieveJobsClient.send_changelog_event(client, *args, **kwargs)

    def call_send_changelog_events(*args, **kwargs):
        return MockPennsieveJobsClient.send_changelog_events(client, *args, **kwargs)

    client.send_changelog_event = Mock()
    client.send_changelog_event.side_effect = call_send_changelog_event

    client.send_changelog_events = Mock()
    client.send_changelog_events.side_effect = call_send_changelog_events

    return client


def create_audit_logger(host) -> Mock:
    class MockAuditor(Auditor):
        def __init__(self):
            self.host = host

    auditor = MockAuditor()
    auditor.enhance = Mock()
    return auditor
