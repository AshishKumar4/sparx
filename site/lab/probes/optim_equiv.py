"""The shared training loop's optimizer, built from dew's OptimConfig, against the optax chain the lab
used before it: 200 updates of each, compared bit for bit, at five learning rates and lengths.

    site/lab/armada/run.sh HEAD optim-equiv site/lab/probes/optim_equiv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from training import optimizer

key = jax.random.key(0)
params = {"w": jax.random.normal(key, (64, 64)), "b": jnp.zeros(64)}
for lr, steps in [(2e-3, 4000), (1e-3, 3000), (7e-4, 3000), (1e-3, 20), (5e-4, 5)]:
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, min(100, steps // 10), steps, lr / 20)
    old = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    new = optimizer(lr, steps)
    so, sn, po, pn = old.init(params), new.init(params), params, params
    for t in range(min(steps, 200)):
        noise = jax.random.fold_in(key, t)
        g = jax.tree.map(lambda p, noise=noise: 3 * jax.random.normal(noise, p.shape), params)
        uo, so = old.update(g, so, po)
        un, sn = new.update(g, sn, pn)
        po, pn = optax.apply_updates(po, uo), optax.apply_updates(pn, un)
    same = all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(po), jax.tree.leaves(pn), strict=True))
    layout = jax.tree.structure(so) == jax.tree.structure(sn)
    print(f"lr {lr} steps {steps}: {min(steps, 200)} updates bitwise equal {same},",
          f"state layout equal {layout}")
