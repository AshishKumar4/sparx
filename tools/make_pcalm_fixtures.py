"""Write predictive coding and PC-ALM inference and weight updates from Seely and Gould's own code.

Runs the JAX reference of Augmented Lagrangian Predictive Coding (Seely and
Gould, arXiv 2605.31022; github.com/SakanaAI/pc-alm) on two of its residual
MLPs, in float64, and records for each method what a weight update reads:

- `bp`: backpropagation's gradient of their loss, half the squared error;
- `pc`: predictive coding, the activity relaxed for `budget` steps from the
  forward pass, and the gradient of the energy there;
- `pcalm`: PC-ALM, with the multipliers accumulating each layer's residual
  between activity steps, in both of their weight-credit timings and with
  one and two activity steps per multiplier step.

The networks are their `small_case` of `tests/test_formulation.py` (tanh,
depth 4) and a deeper ReLU network at their headline budget `T = 2L`, with
their initialization (`init_params`), scales (`model_scales`) and skips
(`skip_mask`). Saves the inputs, targets, weights (as `[in, out]` kernels),
settled activities, multipliers and gradients to
`tests/fixtures/pcalm.npz`. The committed fixture came from pc-alm
660747f61a8a7e547c0ecd2c48c8883380a7d1f6 with jax 0.11.2:

    git clone https://github.com/SakanaAI/pc-alm ../ref-pc-alm
    python tools/make_pcalm_fixtures.py [path to the cloned repository]
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "pcalm.npz"
CASES = {
    # depth, width, inputs, outputs, batch, activation, budget, state_lr
    "tanh": (4, 5, 3, 2, 7, "tanh", 6, 0.1),
    "relu": (8, 6, 4, 3, 5, "relu", 16, 0.25),
}
PRE, POST = "pre_dual_energy", "post_dual_energy"
METHODS = {
    "pc": {"family": "pc"},
    "pcalm": {"family": "pcalm", "alpha": 1.0, "inner_steps": 1, "weight_credit_timing": PRE},
    "pcalm_post": {"family": "pcalm", "alpha": 1.0, "inner_steps": 1, "weight_credit_timing": POST},
    "pcalm_inner": {"family": "pcalm", "alpha": 0.5, "inner_steps": 2, "weight_credit_timing": PRE},
}


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ref-pc-alm"
    sys.path.insert(0, str(repo))
    jax.config.update("jax_enable_x64", val=True)
    from pcalm.inference import Schedule, infer_for_schedule, method_grad
    from pcalm.model import activation_fn, init_params, model_scales, skip_mask

    saved = {"meta/jax": np.array(jax.__version__)}
    for name, (depth, width, inputs, outputs, batch, activation, budget, state_lr) in CASES.items():
        params = init_params(jax.random.PRNGKey(0), depth=depth, width=width, input_dim=inputs,
                             output_dim=outputs, dtype=jnp.float64)
        scales, skips, phi = model_scales(width, depth, inputs), skip_mask(depth), activation_fn(activation)
        x = jax.random.normal(jax.random.PRNGKey(1), (batch, inputs), jnp.float64)
        y = jax.nn.one_hot(jnp.arange(batch) % outputs, outputs, dtype=jnp.float64)
        case = {"x": x, "y": y, "scales": np.array(scales), "budget": np.array(budget),
                "state_lr": np.array(state_lr), "rho": np.array(1.0)}
        case |= {f"kernel_{layer}": w.T for layer, w in enumerate(params)}
        bp = method_grad(params, scales, skips, x, y, Schedule(family="bp", budget=0), state_lr=state_lr,
                         rho=1.0, phi=phi)
        case |= {f"bp/grad_{layer}": g.T for layer, g in enumerate(bp)}
        for method, fields in METHODS.items():
            schedule = Schedule(budget=budget, **fields)
            free, duals = infer_for_schedule(params, scales, skips, x, y, schedule, state_lr=state_lr,
                                             rho=1.0, phi=phi)
            grads = method_grad(params, scales, skips, x, y, schedule, state_lr=state_lr, rho=1.0, phi=phi)
            case |= {f"{method}/alpha": np.array(fields.get("alpha", 0.0)),
                     f"{method}/inner_steps": np.array(fields.get("inner_steps", 1))}
            case |= {f"{method}/free_{layer}": z for layer, z in enumerate(free)}
            case |= {f"{method}/dual_{layer}": d for layer, d in enumerate(duals)}
            case |= {f"{method}/grad_{layer}": g.T for layer, g in enumerate(grads)}
            ours = np.concatenate([np.ravel(g) for g in grads])
            theirs = np.concatenate([np.ravel(g) for g in bp])
            cosine = ours @ theirs / (np.linalg.norm(ours) * np.linalg.norm(theirs))
            print(f"{name} {method}: gradient cosine to BP {cosine:.4f}")
        saved |= {f"{name}/{key}": np.asarray(value) for key, value in case.items()}
    np.savez_compressed(OUT, **saved)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
