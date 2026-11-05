"""Poll -> download -> grade -> durably cache -> report/reconcile."""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from .client import DownloadError, FileTooLarge, PlatformError
from .grader import GradingUnavailable, InvalidSubmission

LOG = logging.getLogger(__name__)


class Worker:
    def __init__(self, client, store, grader, downloads, *, sources=("offline", "manual"), limit=100, max_bytes=128*1024*1024, download_retries=3, stop_event=None):
        self.client, self.store, self.grader = client, store, grader
        self.downloads = Path(downloads)
        self.sources, self.limit, self.max_bytes = sources, limit, max_bytes
        self.download_retries = download_retries
        self.stop_event = stop_event

    def eligible(self, task, pending=True):
        return (isinstance(task, dict) and isinstance(task.get("id"), str) and bool(task["id"])
                and task.get("problem_id") == self.client.problem_id
                and task.get("source") in self.sources
                and (not pending or task.get("status") == "pending"))

    def cycle(self):
        pending = self.client.pending(self.sources, self.limit)
        seen = set()
        for task in pending:
            if self.stop_event is not None and self.stop_event.is_set():
                break
            if not self.eligible(task) or task["id"] in seen:
                continue
            sid = task["id"]; seen.add(sid)
            self.store.register(task)
            cached = self.store.get(sid)
            if cached["result"] is not None or cached["reported"]: continue
            destination = self.downloads / (hashlib.sha256(sid.encode()).hexdigest() + getattr(self.grader, "download_suffix", ".json"))
            try:
                self.client.download(task.get("file_url", ""), destination, self.max_bytes)
                if hasattr(self.grader, "grade_file"):
                    result = self.grader.grade_file(destination)
                else:
                    result = self.grader.grade_bytes(destination.read_bytes())
            except GradingUnavailable as exc:
                self.store.note_error(sid, str(exc)[:1500])
                LOG.warning("submission=%s evaluation deferred: %s", sid, exc)
                break  # Avoid failing or repeatedly probing a broken simulator for every team.
            except DownloadError as exc:
                self.store.attempt(sid, str(exc))
                attempts = self.store.get(sid)["attempts"]
                if attempts < self.download_retries:
                    LOG.warning("submission=%s download failed; retry %s/%s", sid, attempts, self.download_retries)
                    continue
                result = {"metrics_scores": {}, "status": "failed", "evaluator_message": f"Download failed after {attempts} attempts: {exc}"}
            except (InvalidSubmission, FileTooLarge) as exc:
                result = {"metrics_scores": {}, "status": "failed", "evaluator_message": str(exc)[:1500]}
            self.store.save_result(sid, result)
            LOG.info("submission=%s scored status=%s", sid, result["status"])
            if hasattr(self.grader, "grade_file"):
                self.flush_results({task["id"]: task})
        self.flush_results({task["id"]: task for task in pending if self.eligible(task)})
        return self.store.outstanding_count()

    def reconcile(self):
        records = {task["id"]: task for task in self.client.records() if self.eligible(task, pending=False)}
        for cached in self.store.unreported():
            remote = records.get(cached["id"])
            if remote and remote.get("status") in {"completed", "failed"}:
                # This may be an earlier successful write whose HTTP response was lost.
                self.store.mark_reported(cached["id"])
                LOG.info("submission=%s reconciled platform status=%s", cached["id"], remote["status"])
        return records

    def flush_results(self, fresh_pending=None):
        if not self.store.unreported(): return
        try:
            # The all-records endpoint may be paginated. Fresh pending results
            # remain valid candidates even when absent from that separate page.
            records = {**(fresh_pending or {}), **self.reconcile()}
            items = [row["result"] for row in self.store.unreported()
                     if row["id"] in records and self.eligible(records[row["id"]])]
            for offset in range(0, len(items), 200):
                batch = items[offset:offset+200]
                response = self.client.report(batch)
                if not isinstance(response, dict):
                    LOG.warning("Unrecognized batch response; will reconcile cached results")
                    continue
                errors = response.get("errors", [])
                updated, failed = response.get("updated"), response.get("failed")
                if isinstance(errors, list):
                    allowed_ids = {item["submission_id"] for item in batch}
                    for error in errors:
                        if not isinstance(error, dict): continue
                        sid = error.get("submission_id", error.get("id"))
                        if sid not in allowed_ids: continue
                        message = str(error.get("message", error.get("error", "Batch item rejected"))).replace("\n", " ")[:500]
                        self.store.note_error(sid, message)
                        LOG.warning("submission=%s report rejected: %s", sid, message)
                if type(updated) is int and updated == len(batch) and failed == 0 and errors == []:
                    for item in batch: self.store.mark_reported(item["submission_id"])
                else:
                    allowed = {item["submission_id"] for item in batch}
                    error_ids = {e.get("submission_id", e.get("id")) for e in errors if isinstance(e, dict)} if isinstance(errors, list) else set()
                    results = response.get("results", [])
                    if isinstance(results, list):
                        for result in results:
                            if not isinstance(result, dict): continue
                            sid = result.get("submission_id", result.get("id"))
                            if sid in allowed and sid not in error_ids and result.get("success") is True:
                                self.store.mark_reported(sid)
                    LOG.warning("Batch partial/uncertain response: updated=%s failed=%s errors=%s; retained unconfirmed results", updated, failed, len(errors) if isinstance(errors, list) else "unknown")
            if self.store.unreported(): self.reconcile()
        except PlatformError as exc:
            LOG.warning("Result reporting deferred: %s", exc)
