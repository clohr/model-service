from dataclasses import dataclass
from typing import Dict, List, Union
from urllib.parse import urlparse

import requests
import structlog  # type: ignore
from audit_middleware import Auditor  # type: ignore
from flask import current_app, has_request_context
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from ..dtos import api as pennsieve_api
from ..errors import ExternalRequestError, PackagesNotFoundError
from ..types import DatasetId, DatasetNodeId, OrganizationId
from .header import with_trace_id_header

logger = structlog.get_logger(__name__)


def normalize_host(host: str) -> str:
    netloc = urlparse(host).netloc
    if netloc:
        return f"https://{netloc}"
    return f"https://{host}"


@dataclass
class PackageIds:
    id: int
    node_id: str

    @classmethod
    def from_package_dto(cls, dto):
        return cls(id=dto["content"]["id"], node_id=dto["content"]["nodeId"])


class PennsieveApiClient:
    def __init__(self, host):
        self.host = normalize_host(host)

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @classmethod
    def get(cls) -> "PennsieveApiClient":
        """
        Get a configured API client from the Flask app config
        """
        return current_app.config["api_client"]

    def get_dataset(
        self, dataset_id: Union[DatasetId, DatasetNodeId], headers: Dict[str, str]
    ) -> pennsieve_api.Dataset:
        url = f"{self.host}/datasets/{str(dataset_id)}"

        log = logger.bind(pennsieve=dict(dataset_id=dataset_id, url=url))

        resp = self.session.get(url, headers=headers)  # type: ignore
        if resp.status_code != 200:
            raise ExternalRequestError.from_response(resp)

        json = resp.json()
        log.info("Got response", response=json)

        return pennsieve_api.Dataset.schema().load(json["content"])

    def get_datasets(self, headers: Dict[str, str]) -> List[pennsieve_api.Dataset]:
        """
        This is liable to be slow since this endpoint returns a lot of
        extraneous information in the DTO. We may need to cache this response,
        or add a faster endpoint to API.

        See https://app.clickup.com/t/36bpem
        """
        url = f"{self.host}/datasets"

        resp = self.session.get(url, headers=headers)  # type: ignore
        if resp.status_code != 200:
            raise ExternalRequestError.from_response(resp)

        return pennsieve_api.Dataset.schema().load(
            (dto["content"] for dto in resp.json()), many=True
        )

    def get_dataset_ids(self, headers: Dict[str, str]) -> List[DatasetId]:
        """
        Return all dataset IDs that the user has access to.
        """
        return [
            DatasetId(dataset.int_id) for dataset in self.get_datasets(headers=headers)
        ]

    def get_packages(
        self,
        dataset_node_id: DatasetNodeId,
        package_ids: List[Union[int, str]],
        headers: Dict[str, str],
    ):
        assert dataset_node_id is not None

        params = [("packageId", id_) for id_ in package_ids]

        url = f"{self.host}/datasets/{dataset_node_id}/packages/batch"
        log = logger.bind(
            pennsieve=dict(dataset_id=dataset_node_id, package_ids=package_ids, url=url)
        )

        resp = self.session.get(url, params=params, headers=headers)  # type: ignore
        if resp.status_code != 200:
            raise ExternalRequestError.from_response(resp)

        json = resp.json()
        log.info("Got response", response=json)

        # If any packages cannot be found they will be ignored in this response
        # TODO: actually delete package proxies, and raise an error here.
        # See https://app.clickup.com/t/3gaec4
        if json.get("failures"):
            log.error("Proxy packages not found", missing=json["failures"])

        return {package["content"]["id"]: package for package in json["packages"]}

    def get_package_ids(
        self,
        dataset_node_id: DatasetNodeId,
        package_id: Union[int, str],
        headers: Dict[str, str],
    ) -> PackageIds:

        return PackageIds.from_package_dto(
            list(self.get_packages(dataset_node_id, [package_id], headers).values())[0]
        )

    def touch_dataset(
        self,
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        headers: Dict[str, str],
    ) -> None:
        url = f"{self.host}/internal/datasets/{dataset_id}/touch"

        log = logger.bind(
            pennsieve=dict(
                organization_id=organization_id, dataset_id=dataset_id, url=url
            )
        )

        headers["X-ORGANIZATION-INT-ID"] = str(organization_id)

        resp = self.session.post(url, headers=headers, timeout=1.0)  # type: ignore
        if resp.status_code != 200:
            log.error("Error touching dataset updatedAt timestamp", exc_info=True)
            raise ExternalRequestError.from_response(resp)

        log.info("Touched dataset timestamp")
