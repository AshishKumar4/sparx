"""A program run on simulated CPU devices, in a process of its own."""

import json
import os
import subprocess
import sys


def run_on(devices: int, program: str, timeout: float) -> dict:
    """The JSON object `program` prints last, run with `devices` CPU devices."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                          timeout=timeout)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])
