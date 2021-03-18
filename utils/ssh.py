import os
import subprocess
import sys
import time


class SSHTunnel:
    """
    Open a SSH tunnel through a Pennsieve jumpbox.

    You must have an SSH alias configured.
    """

    def __init__(self, remote_host, remote_port, local_port, jumpbox="non-prod"):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.jumpbox = jumpbox

        # If False, fake the tunnel and connect directly to the remote service
        self.enabled = jumpbox is not None

    @property
    def port(self):
        if self.enabled:
            return self.local_port
        else:
            return self.remote_port

    @property
    def host(self):
        if self.enabled:
            return "localhost"
        else:
            return self.remote_host

    def __enter__(self):
        command = [
            "ssh",
            "-o ExitOnForwardFailure=yes",
            f"-L {self.local_port}:{self.remote_host}:{self.remote_port}",
            self.jumpbox,
            "sleep 30",
        ]

        if self.enabled:
            self.tunnel = subprocess.Popen(command, stderr=subprocess.STDOUT)
            time.sleep(2)

        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.tunnel.kill()
            self.tunnel.wait()
