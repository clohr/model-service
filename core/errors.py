from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

from dataclasses_json import LetterCase, dataclass_json  # type: ignore

from .types import GraphValue, ModelId, ModelRelationshipId, PackageProxyId, RecordId


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelServiceError(Exception):
    message: str = field(init=False)

    def __str__(self):
        return self.message


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ExternalRequestError(ModelServiceError):
    status_code: int
    method: str
    url: str
    content: str

    @classmethod
    def from_response(cls, response) -> "ExternalRequestError":
        return cls(
            status_code=response.status_code,
            method=response.request.method,
            url=response.request.url,
            content=response.content,
        )

    @property
    def is_informational(self):
        return self.status_code >= 100 and self.status_code < 200

    @property
    def is_successful(self):
        return self.status_code >= 200 and self.status_code < 300

    @property
    def is_redirection(self):
        return self.status_code >= 300 and self.status_code < 400

    @property
    def is_client_error(self):
        return self.status_code >= 400 and self.status_code < 500

    @property
    def is_server_error(self):
        return self.status_code >= 500 and self.status_code < 600

    def __post_init__(self):
        self.message = (
            f"{self.status_code} error: {self.method} {self.url}: {self.content}"
        )


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class PackagesNotFoundError(ModelServiceError):
    package_ids: List[int]

    def __post_init__(self):
        self.message = f"packages not found: {self.package_ids}"
