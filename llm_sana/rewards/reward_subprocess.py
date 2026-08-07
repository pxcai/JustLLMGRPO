"""Importable subprocess helper for the LLM-Sana reward.

verl loads custom reward files by path without registering them as importable
modules. multiprocessing spawn needs an importable target, so the worker lives
in this package module and imports the actual reward code inside the child
after CUDA_VISIBLE_DEVICES has been set.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback
from typing import Any


def _worker_main(conn, cuda_visible_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    os.environ["LLMSANA_REWARD_CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    from llm_sana.rewards.llm_sana_online_reward import _compute_score_batch_inprocess, _compute_score_inprocess

    while True:
        payload = conn.recv()
        if payload is None:
            break
        try:
            if isinstance(payload, list):
                result = _compute_score_batch_inprocess(payload)
            else:
                result = _compute_score_inprocess(**payload)
            conn.send({"ok": True, "result": result})
        except Exception:  # noqa: BLE001
            conn.send({"ok": False, "error": traceback.format_exc()})


class RewardSubprocessClient:
    def __init__(self, cuda_visible_devices: str):
        self.cuda_visible_devices = str(cuda_visible_devices)
        self._lock = threading.Lock()
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        try:
            self._process = ctx.Process(
                target=_worker_main,
                args=(child_conn, self.cuda_visible_devices),
                daemon=True,
            )
            self._process.start()
        finally:
            if old_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
        self._conn = parent_conn

    def compute(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._send(payload)

    def compute_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._send(payloads)

    def _send(self, payload: dict[str, Any] | list[dict[str, Any]]):
        with self._lock:
            if not self._process.is_alive():
                raise RuntimeError("LLM-Sana reward subprocess is not alive.")
            self._conn.send(payload)
            message = self._conn.recv()
        if message.get("ok"):
            return message["result"]
        raise RuntimeError(message.get("error", "Unknown LLM-Sana reward subprocess error."))
