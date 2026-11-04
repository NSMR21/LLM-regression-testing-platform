"""HTTP client for a RegressionLens server (used by ``rlens run`` in server mode and by CI)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from .. import __version__
from ..spec import ExecutionSpec, ModelSpec, PromptSpec, SuiteSpec

TERMINAL = {"completed", "errored", "cancelled"}


class ClientError(Exception):
    pass


class Client:
    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        project: str = "default",
        timeout_s: float = 90.0,
        http: httpx.Client | None = None,
    ):
        headers = {"User-Agent": f"rlens/{__version__}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            headers["X-RLens-Project"] = project
        if http is not None:
            http.headers.update(headers)
            self.http = http
        else:
            if not url:
                raise ClientError("no server URL (set --url or RLENS_URL)")
            self.http = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=timeout_s)

    def request(self, method: str, path: str, retries: int = 3, **kwargs) -> Any:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                last = exc
                if method != "GET" or attempt == retries:
                    break
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail")
                except ValueError:
                    detail = response.text[:500]
                raise ClientError(f"HTTP {response.status_code} on {method} {path}: {detail}")
            return response.json()
        raise ClientError(f"cannot reach server: {last}")

    def start_run(
        self,
        suite: SuiteSpec,
        prompt: PromptSpec,
        model: ModelSpec,
        execution: ExecutionSpec,
        baseline: str | None = None,
        fresh_baseline: bool = False,
        ci_meta: dict[str, Any] | None = None,
        run_seed: int | None = None,
    ) -> dict[str, Any]:
        body = {
            "suite": suite.model_dump(mode="json"),
            "prompt": prompt.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "execution": execution.model_dump(mode="json"),
            "baseline": baseline,
            "fresh_baseline": fresh_baseline,
            "ci_meta": ci_meta or {},
            "run_seed": run_seed,
        }
        return self.request("POST", "/api/v1/runs", json=body)

    def wait(self, run_id: str, timeout_s: float = 3600, on_status: Callable[[str], None] | None = None, poll_s: float = 30) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last = None
        while True:
            remaining = deadline - time.monotonic()
            result = self.request("GET", f"/api/v1/runs/{run_id}/verdict", params={"wait": max(0, min(poll_s, remaining))})
            progress = result.get("progress") or {}
            status_line = f"{result['status']}: executed {progress.get('executed', 0)}/{progress.get('total', 0)}, graded {progress.get('graded', 0)}"
            if on_status and status_line != last:
                on_status(status_line)
                last = status_line
            if result["status"] in TERMINAL:
                return result
            if remaining <= 0:
                raise ClientError(f"run {run_id} did not finish within {timeout_s:.0f}s (status {result['status']})")

    def run_and_wait(self, timeout_s: float = 3600, on_status: Callable[[str], None] | None = None, **kwargs) -> dict[str, Any]:
        brief = self.start_run(**kwargs)
        result = self.wait(brief["run_id"], timeout_s=timeout_s, on_status=on_status)
        verdict = result.get("verdict")
        if result["status"] != "completed" or not verdict:
            verdict = {"verdict": "error", "reasons": [result.get("error") or f"run {result['status']}"], "notes": []}
        return {"run_id": brief["run_id"], "verdict": verdict, "summary": result.get("summary"), "url": result.get("url")}

    def promote(self, run_id: str, name: str, suite_name: str | None = None) -> dict[str, Any]:
        return self.request("PUT", f"/api/v1/baselines/{name}", json={"run_id": run_id, "suite": suite_name})

    def get(self, path: str, **params) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self.request("POST", path, json=body or {})
