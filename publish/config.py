import os
from dataclasses import dataclass

from core import Config


@dataclass(frozen=True)
class PublishConfig(Config):
    s3_bucket: str
    s3_publish_key: str
