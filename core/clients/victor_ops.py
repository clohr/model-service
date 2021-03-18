from typing import Optional

import requests
import structlog  # type: ignore
from flask import current_app, has_request_context

logger = structlog.get_logger(__name__)


class VictorOpsClient:
    def __init__(self, url: str, routing_key: str):
        self.alert_url = f"{url}/{routing_key}"

    @classmethod
    def get(cls) -> "VictorOpsClient":
        """
        Get a configured VictorOps client from the Flask app config
        """
        return current_app.config["victor_ops_client"]

    def _alert(
        self,
        message_type: str,
        entity_id: str,
        entity_display_name: Optional[str] = None,
        state_message: Optional[str] = None,
    ):
        return requests.post(
            self.alert_url,
            data=dict(
                message_type=message_type,
                entity_id=entity_id,
                entity_display_name=entity_display_name,
                state_message=state_message,
            ),
        )

    def critical(
        self,
        entity_id: str,
        entity_display_name: Optional[str] = None,
        state_message: Optional[str] = None,
    ):
        return self._alert("CRITICAL", entity_id, entity_display_name, state_message)

    def warning(
        self,
        entity_id: str,
        entity_display_name: Optional[str] = None,
        state_message: Optional[str] = None,
    ):
        return self._alert("WARNING", entity_id, entity_display_name, state_message)

    def ack(
        self,
        entity_id: str,
        entity_display_name: Optional[str] = None,
        state_message: Optional[str] = None,
    ):
        return self._alert(
            "ACKNOWLEDGEMENT", entity_id, entity_display_name, state_message
        )

    def info(
        self,
        entity_id: str,
        entity_display_name: Optional[str] = None,
        state_message: Optional[str] = None,
    ):
        return self._alert("INFO", entity_id, entity_display_name, state_message)
