import datetime
import json
import uuid
from collections import Sequence, Set
from dataclasses import InitVar, dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple, Union
from uuid import UUID

from audit_middleware import TraceId  # type: ignore
from dataclasses_json import LetterCase, dataclass_json  # type: ignore
from flask import current_app, has_request_context

from core import DatasetId, OrganizationId, UserNodeId
from core.types import GraphValue, JsonDict
from core.util import is_datetime, normalize_datetime
from server.models import ModelProperty
from server.models.datatypes import DataType


def timestamp() -> str:
    """
    Return an ISO8601 timestamp without microseconds.
    """
    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Event:
    event_type: ClassVar[str]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CreateModel(Event):
    event_type: ClassVar[str] = "CREATE_MODEL"
    id: UUID
    name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class UpdateModel(Event):
    event_type: ClassVar[str] = "UPDATE_MODEL"
    id: UUID
    name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DeleteModel(Event):
    event_type: ClassVar[str] = "DELETE_MODEL"
    id: UUID
    name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CreateModelProperty(Event):
    event_type: ClassVar[str] = "CREATE_MODEL_PROPERTY"
    property_name: str
    model_id: UUID
    model_name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class UpdateModelProperty(Event):
    event_type: ClassVar[str] = "UPDATE_MODEL_PROPERTY"
    property_name: str
    model_id: UUID
    model_name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DeleteModelProperty(Event):
    event_type: ClassVar[str] = "DELETE_MODEL_PROPERTY"
    property_name: str
    model_id: UUID
    model_name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CreateRecord(Event):
    event_type: ClassVar[str] = "CREATE_RECORD"
    id: UUID
    name: Optional[str]
    model_id: UUID


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class UpdateRecord(Event):
    event_type: ClassVar[str] = "UPDATE_RECORD"
    id: UUID
    name: Optional[str]
    model_id: UUID
    properties: List["PropertyDiff"]

    @dataclass_json(letter_case=LetterCase.CAMEL)
    @dataclass
    class PropertyDiff:
        name: str
        data_type: DataType
        old_value: GraphValue
        new_value: GraphValue

    @classmethod
    def compute_diff(
        cls,
        properties: List[ModelProperty],
        old_values: Dict[str, GraphValue],
        new_values: Dict[str, GraphValue],
    ) -> List[PropertyDiff]:
        """
        Diff old and new property values.
        """
        diff = []

        for p in properties:
            old_value = cls.format_value(old_values.get(p.name))
            new_value = cls.format_value(new_values.get(p.name))

            if old_value != new_value:
                diff.append(
                    UpdateRecord.PropertyDiff(
                        name=p.name,
                        data_type=p.data_type,
                        old_value=old_value,
                        new_value=new_value,
                    )
                )

        return diff

    @classmethod
    def format_value(cls, v):
        """
        Hack: convert NeoTime instances to dates to work around issues with
        dataclasses-json encoding.

        Cannot use a custom dataclass encoder for this because of a bug in
        nested dataclasses.
        """
        if is_datetime(v):
            return normalize_datetime(v).isoformat()
        elif isinstance(v, (Sequence, Set)) and not isinstance(v, str):
            return [cls.format_value(w) for w in v]
        else:
            return v


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DeleteRecord(Event):
    event_type: ClassVar[str] = "DELETE_RECORD"
    id: UUID
    name: Optional[str]
    model_id: UUID


class MessageJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, UUID):
            return str(o)
        return json.JSONEncoder.default(self, o)


class PennsieveJobsClient:
    def __init__(self, sqs_client, sqs_queue_id: str):
        self.sqs_client = sqs_client
        self.sqs_queue_id = sqs_queue_id

    @classmethod
    def get(cls) -> "PennsieveJobsClient":
        """
        Get a configured Pennsieve jobs client from the Flask app config
        """
        return current_app.config["jobs_client"]

    def send_changelog_event(
        self,
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        user_id: UserNodeId,
        event: Event,
        trace_id: TraceId,
    ):
        return self.send_changelog_events(
            organization_id=organization_id,
            dataset_id=dataset_id,
            user_id=user_id,
            events=[event],
            trace_id=trace_id,
        )

    def send_changelog_events(
        self,
        organization_id: OrganizationId,
        dataset_id: DatasetId,
        user_id: UserNodeId,
        events: List[Event],
        trace_id: TraceId,
    ):
        body = {
            "DatasetChangelogEventJob": {
                "organizationId": organization_id,
                "datasetId": dataset_id,
                "userId": user_id,
                "events": [
                    {
                        "eventType": e.event_type,
                        "eventDetail": e.to_dict(),  # type: ignore
                        "timestamp": timestamp(),
                    }
                    for e in events
                ],
                "traceId": trace_id,
                "id": str(uuid.uuid4()),
            }
        }

        return self.sqs_client.send_message(
            QueueUrl=self.sqs_queue_id,
            MessageBody=json.dumps(body, cls=MessageJSONEncoder),
        )
