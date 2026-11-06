"""Explicit bridge to the separately installed, pinned public ACT checkpoint.

This source uses corner4 RGB, a black wrist placeholder and native 4D robot
state. The public dataset's x action convention reverses native Meta-World.
It is not a language-conditioned model or a live two-camera system.
"""
import json
import subprocess
from pathlib import Path

import numpy as np


def source_action_to_native(actions):
    result = np.asarray(actions).copy()
    if result.shape[-1] != 4 or not np.isfinite(result).all():
        raise ValueError("Need finite 4D source actions")
    result[..., 0] *= -1
    return np.clip(result, -1, 1)


def source_image(native_rgb):
    image = np.asarray(native_rgb)
    if image.shape != (512, 512, 3) or image.dtype != np.uint8:
        raise ValueError("Need native uint8 RGB512")
    return np.ascontiguousarray(image[::-1])


class ACTClient:
    """One direct chunk request, no hidden action queue or persistent history."""
    def __init__(self, interpreter, worker, checkpoint, scratch):
        self.scratch = Path(scratch).resolve()
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.error_log = (self.scratch / "worker_stderr.log").open("w")
        self.process = subprocess.Popen(
            [str(interpreter), str(worker), "--checkpoint", str(checkpoint)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.error_log, text=True, bufsize=1)
        self.ready = self._response()
        if self.ready.get("status") != "ready":
            raise RuntimeError(self.ready)

    def _response(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ACT worker ended; inspect worker_stderr.log")
        return json.loads(line)

    def predict(self, state, base, wrist=None):
        if wrist is None:
            wrist = np.zeros_like(base)
        source = self.scratch / "request.npz"
        target = self.scratch / "response.npy"
        np.savez(source, state=state, base=base, wrist=wrist)
        self.process.stdin.write(json.dumps({"input": str(source), "output": str(target)})+"\n")
        self.process.stdin.flush()
        response = self._response()
        if response.get("status") != "ok":
            raise RuntimeError(response)
        return np.load(target, allow_pickle=False), response

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.error_log.close()
        if self.process.returncode != 0:
            raise RuntimeError(f"ACT worker exit {self.process.returncode}")


class ACTCamera:
    def __init__(self, contact_env):
        import mujoco
        # Only the rendering buffer dimensions change; no physics changes.
        contact_env.env.model.vis.global_.offwidth = 512
        contact_env.env.model.vis.global_.offheight = 512
        self.renderer = mujoco.Renderer(contact_env.env.model, height=512, width=512)

    def image(self, contact_env):
        self.renderer.update_scene(contact_env.env.data, camera="corner4")
        return source_image(self.renderer.render())

    def close(self):
        self.renderer.close()
