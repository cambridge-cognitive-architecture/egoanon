"""Keep the model-backed levels off the network."""
from __future__ import annotations

import os
import socket
from contextlib import contextmanager

SETTINGS = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_XET": "1", "YOLO_OFFLINE": "1", "YOLO_AUTOINSTALL": "False",
    "ULTRALYTICS_ANALYTICS": "False",
}


class NetworkBlocked(OSError):
    """Something tried to use the network during an offline-only step."""


def go_offline() -> None:
    """Call before any model library is imported: they read these on import."""
    os.environ.update(SETTINGS)


@contextmanager
def offline():
    """Refuse Python-level network connections inside the block.

    Native code can bypass this, so for sensitive video also disconnect the machine.
    """
    go_offline()
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex
    lookup = socket.getaddrinfo

    def guard(original):
        def call(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                raise NetworkBlocked("the network is disabled for this step")
            return original(sock, address)
        return call

    def no_lookup(*args, **kwargs):
        raise NetworkBlocked("the network is disabled for this step")

    socket.socket.connect = guard(connect)
    socket.socket.connect_ex = guard(connect_ex)
    socket.getaddrinfo = no_lookup
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
        socket.getaddrinfo = lookup
