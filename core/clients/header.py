from typing import Dict

import connexion  # type: ignore
from audit_middleware import Auditor, TraceId  # type: ignore
from flask import has_request_context  # type: ignore

from server.errors import MissingTraceId


def trace_id_header() -> TraceId:
    x_bf_trace_id = connexion.request.headers.get(Auditor.TRACE_ID_HEADER)
    if x_bf_trace_id is None:
        raise MissingTraceId
    return TraceId(x_bf_trace_id)


def trace_id_header_dict() -> Dict[str, str]:
    if has_request_context():
        trace_id = trace_id_header()
        return {Auditor.TRACE_ID_HEADER: str(trace_id)}
    return {}


def with_trace_id_header(trace_id: TraceId) -> Dict[str, str]:
    return {Auditor.TRACE_ID_HEADER: str(trace_id)}
