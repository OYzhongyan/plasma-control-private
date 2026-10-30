#
# Copyright @2025 ENN Energy(enn.cn)
#
# Docker Socket Predictor (no fusion_control dependency).
# TCP socket protocol: INIT / RESET / STEP / EXIT.
#

from __future__ import annotations

import json
import socket
import time
from abc import abstractmethod
from io import BlockingIOError
from typing import Any, TextIO

import numpy as np


class DockerSocketPredictor:
    """Base class for docker container socket-based predictors."""

    def __init__(
        self, name: str = "DockerSocketPredictor", config: dict[str, Any] | None = None
    ):
        config = config or {}
        self.name = name
        self.config = config

        self.host: str = config.get("host", "127.0.0.1")
        self.port: int = config.get("port", 5558)
        self.timeout: float = config.get("timeout", 300.0)
        self.auto_connect: bool = config.get("auto_connect", False)

        self._socket: socket.socket | None = None
        self._reader: TextIO | None = None
        self._is_connected: bool = False
        self._is_initialized: bool = False
        self._selected_port: int | None = None

        self.init_config: dict[str, Any] = self._get_init_config(config)

        if self.auto_connect:
            self.connect()

    def _get_init_config(self, config: dict[str, Any]) -> dict[str, Any]:
        init_config = config.get("init_config", {})
        if hasattr(init_config, "to_dict"):
            init_config = init_config.to_dict()
        return dict(init_config)

    def _select_port(self) -> int:
        return self.port

    def connect(self) -> None:
        if self._is_connected:
            return
        port = self._select_port()
        self._do_connect(self.host, port)
        self._protocol_init(self.init_config)

    def _do_connect(self, host: str, port: int) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((host, port))
            self._reader = self._socket.makefile("r", encoding="utf-8", newline="\n")
            self._is_connected = True
            self._is_initialized = True
            self._selected_port = port
        except Exception as exc:
            self._do_disconnect()
            raise ConnectionError(f"Failed to connect to {host}:{port}: {exc}") from exc

    def disconnect(self) -> None:
        if self._is_connected:
            try:
                self._protocol_exit()
            except Exception:
                pass
        self._do_disconnect()

    def _do_disconnect(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._is_connected = False
        self._is_initialized = False

    def ensure_connected(self) -> None:
        if not self._is_connected:
            self.connect()

    def is_connection_alive(self) -> bool:
        if self._socket is None or self._reader is None:
            return False
        try:
            self._socket.setblocking(False)
            try:
                data = self._socket.recv(1, socket.MSG_PEEK)
                if data == b"":
                    return False
            except BlockingIOError:
                pass
            except (ConnectionError, OSError):
                return False
            finally:
                self._socket.setblocking(True)
                self._socket.settimeout(self.timeout)
            return True
        except Exception:
            return False

    def reconnect(self) -> None:
        self._do_disconnect()
        self.connect()

    def _send_line(self, line: str) -> None:
        if self._socket is None:
            raise ConnectionError("Not connected")
        self._socket.sendall(f"{line}\n".encode())

    def _send_lines(self, *lines: str) -> None:
        """Send multiple lines as a single sendall() so they arrive atomically.

        The MATLAB server peeks for an optional params/action line shortly after
        reading the command line. If the two sendall() calls from the client
        land in separate TCP segments with a gap >~10ms (easily induced by OS
        jitter, GC, or heavy load), the server misses the follow-up line and
        later interprets it as the next command, which poisons the stream and
        causes the next `step()` to receive `ERROR: Unknown command\n`,
        surfacing as `json.JSONDecodeError: Expecting value: line 1 column 1`.
        """
        if self._socket is None:
            raise ConnectionError("Not connected")
        payload = "".join(f"{line}\n" for line in lines).encode()
        self._socket.sendall(payload)

    def _send_json(self, data: dict[str, Any] | Any) -> None:
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        self._send_line(json.dumps(data))

    def _read_line(self) -> str:
        if self._reader is None:
            raise ConnectionError("Not connected")
        line = self._reader.readline()
        if line == "":
            raise ConnectionError("Server closed the connection")
        return line.rstrip("\n")

    def _protocol_init(self, init_config: dict[str, Any]) -> None:
        if hasattr(init_config, "to_dict"):
            init_config = init_config.to_dict()
        self._send_lines("INIT", json.dumps(init_config))
        response = self._read_line()
        if response != "INIT_OK":
            raise ConnectionError(f"Initialization failed: {response}")
        self._is_initialized = True

    def _protocol_reset(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params:
            if hasattr(params, "to_dict"):
                params = params.to_dict()
            self._send_lines("RESET", json.dumps(params))
        else:
            self._send_line("RESET")
        return json.loads(self._read_line())

    def _protocol_step(self, action: np.ndarray) -> dict[str, Any]:
        self._send_lines("STEP", " ".join(map(str, action)))
        return json.loads(self._read_line())

    def _protocol_exit(self) -> None:
        self._send_line("EXIT")
        try:
            self._read_line()
        except Exception:
            pass

    @abstractmethod
    def step(self, action: np.ndarray) -> dict[str, Any]:
        pass

    def reset(self, **kwargs) -> dict[str, Any]:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not self._is_initialized or not self.is_connection_alive():
                    self.reconnect()
                return self._protocol_reset(kwargs if kwargs else None)
            except (ConnectionError, BrokenPipeError, OSError, json.JSONDecodeError):
                if attempt < max_retries - 1:
                    self._do_disconnect()
                    time.sleep(1)
                else:
                    raise
        raise ConnectionError("Reset failed after max retries")

    def close(self) -> None:
        self.disconnect()

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host,
            "port": self._selected_port or self.port,
            "connected": self._is_connected,
            "initialized": self._is_initialized,
        }

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
