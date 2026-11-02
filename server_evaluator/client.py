"""Only API requests carry the platform bearer token; downloads never do."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import urlsplit
import requests


class PlatformError(RuntimeError):
    pass


class DownloadError(RuntimeError):
    pass


class FileTooLarge(ValueError):
    pass


def check_url(value):
    if not isinstance(value, str):
        raise ValueError("Expected an HTTP(S) URL string")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Expected HTTP(S) URL without embedded credentials")
    return value.rstrip("/")


def rows_from(data):
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for key in ("items", "data", "results", "submissions"):
            if isinstance(data.get(key), list): return data[key]
    raise PlatformError("Platform list response must be an array (or items/data/results/submissions array)")


class PlatformClient:
    def __init__(self, base_url, problem_id, *, username=None, password=None, token=None, timeout=30):
        self.base_url = check_url(base_url)
        self.problem_id = problem_id
        self.username, self.password, self._token = username, password, token
        self.timeout = (10, timeout)
        self.session = requests.Session()

    def login(self):
        if not self.username or not self.password:
            raise PlatformError("Set PLATFORM_USERNAME and PLATFORM_PASSWORD, or a valid PLATFORM_TOKEN; expired token needs fresh credentials")
        try:
            response = self.session.post(self.base_url + "/api/auth/login/teacher", json={"username": self.username, "password": self.password}, timeout=self.timeout, allow_redirects=False)
        except requests.RequestException as exc:
            raise PlatformError("Platform login network error") from exc
        if response.status_code != 200:
            raise PlatformError(f"Platform login returned HTTP {response.status_code}")
        try: self._token = response.json()["token"]
        except (ValueError, KeyError, TypeError) as exc: raise PlatformError("Login response is missing token") from exc
        if not isinstance(self._token, str) or not self._token:
            raise PlatformError("Login returned an invalid token")

    def request(self, method, path, **kwargs):
        if not self._token: self.login()
        for attempt in range(2):
            try:
                response = self.session.request(method, self.base_url + path, headers={"Authorization": f"Bearer {self._token}"}, timeout=self.timeout, allow_redirects=False, **kwargs)
            except requests.RequestException as exc:
                raise PlatformError(f"Platform {method} {path} network error; result may require reconciliation") from exc
            if response.status_code == 401 and attempt == 0:
                self.login(); continue
            if response.status_code != 200:
                raise PlatformError(f"Platform {method} {path} returned HTTP {response.status_code}")
            try: return response.json()
            except ValueError as exc: raise PlatformError(f"Platform {path} returned invalid JSON") from exc
        raise PlatformError("Authentication failed")

    def pending(self, sources, limit):
        rows = []
        for source in sources:
            path = "/api/submissions/offline-pending" if source == "offline" else "/api/submissions"
            params = {"problem_id": self.problem_id, "limit": limit}
            if source == "manual": params.update(source="manual", status="pending")
            rows.extend(rows_from(self.request("GET", path, params=params)))
        return rows

    def records(self):
        return rows_from(self.request("GET", "/api/submissions", params={"problem_id": self.problem_id}))

    def report(self, items):
        return self.request("POST", "/api/submissions/offline-grading-batch", json={"items": items})

    def download(self, url, destination, max_bytes):
        try: check_url(url)
        except ValueError as exc: raise DownloadError("Invalid submission file URL") from exc
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".part")
        try:
            # Deliberately not self.session: no auth headers, cookies or API token.
            with requests.get(url, stream=True, timeout=self.timeout) as response:
                if response.status_code != 200:
                    raise DownloadError(f"File download returned HTTP {response.status_code}")
                total = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(65536):
                        total += len(chunk)
                        if total > max_bytes: raise FileTooLarge("Submission file exceeds configured size limit")
                        handle.write(chunk)
            temporary.replace(destination)
        except requests.RequestException as exc:
            raise DownloadError("File download network error") from exc
        finally:
            if temporary.exists(): temporary.unlink()

    def close(self): self.session.close()
