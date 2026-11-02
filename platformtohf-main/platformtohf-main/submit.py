"""Run Static Space F2b scoring in local headless Chromium/Pyodide."""
from __future__ import annotations
import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
MAX_BYTES = 128 * 1024 * 1024


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextmanager
def local_page():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(HERE)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def read_json(path):
    if not path.is_file():
        raise ValueError(f"JSON file not found: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"JSON file exceeds 128 MiB: {path}")
    text = path.read_text(encoding="utf-8-sig")
    def reject(value):
        raise ValueError(f"Invalid JSON constant {value} in {path}")
    try:
        json.loads(text, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    return text


def run_browser(url, infer_text, target_text, tag, timeout):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt, then run: python -m playwright install chromium") from exc
    with sync_playwright() as playwright:
        with playwright.chromium.launch(headless=True) as browser:
            page = browser.new_page()
            page.set_default_timeout(timeout * 1000)
            page.set_default_navigation_timeout(timeout * 1000)
            response = page.goto(url, wait_until="domcontentloaded")
            if response is None or not response.ok:
                raise RuntimeError(f"Static page failed to load: {url} (HTTP {response.status if response else 'no response'})")
            page.wait_for_function("typeof window.scoreTrajectoryFiles === 'function'")
            return page.evaluate("""async ({infer, target, tag, timeout}) => {
                let timer;
                try {
                    return await Promise.race([
                        window.scoreTrajectoryFiles(infer, target, tag, timeout),
                        new Promise((_, reject) => {
                            timer = setTimeout(() => reject(new Error('Pyodide scoring timed out; check CDN access')), timeout);
                        })
                    ]);
                } finally { clearTimeout(timer); }
            }""", {"infer": infer_text, "target": target_text, "tag": tag, "timeout": timeout * 1000})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infer_json", type=Path, help="Full F2b rollout JSON document (not JSONL)")
    parser.add_argument("target_json", type=Path, help="Official F2b target JSON")
    parser.add_argument("--space-url", default=os.environ.get("HF_SPACE_URL", ""), help="Optional public Static Space app root URL; default serves local files")
    parser.add_argument("--tag", default=os.environ.get("TAG", ""))
    parser.add_argument("--output", type=Path, help="Also save score JSON to this file")
    parser.add_argument("--timeout", type=float, default=300, help="Page/compute timeout in seconds")
    args = parser.parse_args(argv)
    try:
        infer_text, target_text = read_json(args.infer_json), read_json(args.target_json)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if args.output and args.output.resolve() in {args.infer_json.resolve(), args.target_json.resolve()}:
            raise ValueError("--output must not overwrite either input JSON file")
        if args.space_url:
            url = urlsplit(args.space_url)
            if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password:
                raise ValueError("--space-url must be an HTTP(S) app URL without embedded credentials")
            if url.path.rstrip("/").endswith("/evaluate"):
                raise ValueError("Use a Static Space app root URL, not the old /evaluate endpoint")
            result = run_browser(args.space_url, infer_text, target_text, args.tag, args.timeout)
        else:
            with local_page() as url:
                result = run_browser(url, infer_text, target_text, args.tag, args.timeout)
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError("Browser returned an invalid scoring result")
        output = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0
    except Exception as exc:
        print(json.dumps({"success": False, "score": None, "scoreJson": {}, "errorMsg": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
