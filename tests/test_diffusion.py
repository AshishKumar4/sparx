from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference

from sparx.dynamics import Sparse
from sparx.learn import RNeuralNet, reward_diffusion

FIXTURES = Path(__file__).parent / "fixtures"


def original():
    """The fixture of the original program's run, and the network it ran, rebuilt."""
    f = np.load(FIXTURES / "rneuralnet.npz")
    net = RNeuralNet.wire(f["threshold"], f["pre"], f["post"], f["weight"], f["myelin"],
                          inputs=int(f["inputs"]), outputs=f["outputs"])
    return f, net


def replay(net, x, reward, **learning):
    """Run `net` over `x` in chunks that end at each reward, learning from each; every tick's outputs,
    and each reward's local rewards and weights."""
    outputs, credits, weights, state, start = [], [], [], None, 0
    for t in [*np.flatnonzero(reward), len(x) - 1]:
        out, state = net.run(jnp.asarray(x[start:t + 1]), state)
        outputs.append(np.asarray(out))
        if reward[t]:
            net, diffusion = net.learn(out[-1], reward[t], **learning)
            credits.append(np.asarray(diffusion.credit))
            weights.append(np.asarray(net.wiring.weight))
        start = t + 1
    return np.concatenate(outputs), np.asarray(credits), np.asarray(weights)


def test_rneuralnet_runs_as_the_original_program():
    f, net = original()
    assert str(f["commit"]) == "d4b7803a5bbe87747d27a7137cc05a756bef42f7"
    outputs, credits, weights = replay(net, f["x"], f["reward"])
    units = net.feeder  # the feeder sends nothing, so the original records no output for it
    # Observed 7.2e-7 over outputs as large as 9.4, both sides of the neurons' thresholds.
    np.testing.assert_allclose(outputs[:, :units], f["activity"], rtol=1e-6, atol=1e-6)
    above = outputs[:, :net.neurons] >= np.asarray(net.cell.inner.threshold)[:net.neurons]
    assert above.sum() > 100 and (~above).sum() > 100
    # Global_Renew never clears the input neurons' local rewards, which nothing reads, so they add up
    # over rewards in the original; the internal neurons' and the feeder's are each reward's.
    read = np.r_[np.ones(net.neurons, bool), np.zeros(net.inputs, bool), True]
    # Observed 6.0e-8.
    np.testing.assert_allclose(credits[:, read], f["credit"][:, read], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(weights, f["weights"], rtol=0, atol=1e-7)  # observed 0
    assert np.abs(weights[-1] - f["weight"]).max() > 1e-3


def test_the_pass_order_sets_each_connections_delay():
    # A connection whose source sends after its target in the pass reaches the target a tick later:
    # without that tick, the network is another one.
    f, net = original()
    myelin_only = net.wiring.replace(delay=jnp.asarray(f["myelin"] + 1, jnp.int32))
    other = net.replace(cell=net.cell.replace(wiring=myelin_only))
    outputs, _, _ = replay(other, f["x"], np.zeros_like(f["reward"]))
    assert np.abs(outputs[:, :net.feeder] - f["activity"]).max() > 0.1


def test_spreading_over_every_path_is_another_rule():
    f, net = original()
    _, credits, _ = replay(net, f["x"], f["reward"], paths="all", discount=0.9)
    assert np.abs(credits[0, :net.neurons] - f["credit"][0, :net.neurons]).max() > 0.01


def random_graph(seed, size=30, edges=90):
    rng = np.random.default_rng(seed)
    pairs = rng.choice(size * size, edges, replace=False)
    pre, post = pairs // size, pairs % size
    keep = pre != post
    pre, post = pre[keep], post[keep]
    order = rng.permutation(len(pre))
    activity = rng.normal(0.0, 2.0, size)
    return pre[order], post[order], activity, size


@pytest.mark.parametrize("seed", range(4))
def test_the_first_visit_spread_is_the_originals_recursion(seed):
    pre, post, activity, size = random_graph(seed)
    root = int(post[0])
    wiring = Sparse(jnp.asarray(pre), jnp.asarray(post), jnp.zeros(len(pre)), size)
    diffusion = reward_diffusion(wiring, jnp.asarray(activity, jnp.float32), 1.5, root=root)
    change, credit, share = reference.reward_spread(pre, post, activity, 1.5, root, size)
    # Observed 1.8e-7 on local rewards up to 1.7, 1.9e-9 on the changes and 1.5e-7 on the shares.
    np.testing.assert_allclose(diffusion.credit, credit, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(diffusion.change, change, rtol=1e-5, atol=1e-8)
    # The original takes the shares of the units it enters; it never reads the rest.
    entered = share != 0
    np.testing.assert_allclose(np.asarray(diffusion.share)[entered], share[entered], rtol=1e-5, atol=1e-7)
    assert (credit != 0).sum() > size // 2


def test_the_first_visit_spread_depends_on_the_order_of_the_connections():
    # C=0 -> B=1 -> A=2 -> F=3, and B -> F: F hands A and B half each. Entering A first gives B its
    # half before B is entered, so C receives all of it; entering B first sends B's half on, and the
    # half A gives B later stays there.
    activity = jnp.zeros(4)
    for edges, expected in (([(2, 3), (1, 3), (1, 2), (0, 1)], 1.0), ([(1, 3), (2, 3), (1, 2), (0, 1)], 0.5)):
        pre, post = (jnp.asarray(e) for e in zip(*edges, strict=True))
        wiring = Sparse(pre, post, jnp.zeros(4), 4)
        assert float(reward_diffusion(wiring, activity, 1.0, root=3).credit[0]) == expected
        every = reward_diffusion(wiring, activity, 1.0, root=3, paths="all", discount=0.5)
        np.testing.assert_allclose(every.credit[0], 0.5 * 0.5**2 + 0.5 * 0.5**3, rtol=1e-6)


@pytest.mark.parametrize("seed", range(3))
def test_spreading_over_every_path_solves_the_discounted_sum(seed):
    pre, post, activity, size = random_graph(seed)
    root = int(post[0])
    wiring = Sparse(jnp.asarray(pre), jnp.asarray(post), jnp.zeros(len(pre)), size)
    diffusion = reward_diffusion(wiring, jnp.asarray(activity, jnp.float32), -2.0, root=root, paths="all",
                                 discount=0.8)
    passes = np.zeros((size, size))
    np.add.at(passes, (pre, post), 0.8 * np.asarray(diffusion.share, np.float64))
    expected = np.linalg.solve(np.eye(size) - passes, -2.0 * np.eye(size)[root])
    np.testing.assert_allclose(diffusion.credit, expected, rtol=1e-5, atol=1e-6)  # observed 4.8e-7 of 2.3


def test_the_rule_is_blind_to_signs_and_to_the_rewards_zero():
    # Sources of +1 and -1 into an output that feeds the reward feeder (units 0, 1, 2, 3). Raising the
    # output needs the first weight up and the second down; both move up, by the same amount.
    wiring = Sparse(jnp.asarray([0, 1, 2]), jnp.asarray([2, 2, 3]), jnp.asarray([0.3, 0.3, 1.0]), 4)
    activity = jnp.asarray([1.0, -1.0, 0.0, 0.0])
    change = reward_diffusion(wiring, activity, 1.0, root=3).change
    assert float(change[0]) == float(change[1]) > 0
    # Adding 100 to every reward orders outcomes the same, yet changes the weights 101 times as much.
    raised = reward_diffusion(wiring, activity, 101.0, root=3).change
    np.testing.assert_allclose(raised, 101 * change, rtol=1e-6)
    assert not np.any(reward_diffusion(wiring, activity, 0.0, root=3).change)


def test_a_positive_reward_raises_a_weight_that_should_fall():
    # The notes' example: y = w at w = 2.5 toward 2 earns R = 1 - (y - 2)^2 = 0.75, and the rule raises
    # w to 2.5075, where R is 0.7424.
    wiring = Sparse(jnp.asarray([0, 1]), jnp.asarray([1, 2]), jnp.asarray([2.5, 1.0]), 3)
    change = reward_diffusion(wiring, jnp.asarray([1.0, 2.5, 0.0]), 0.75, root=2).change
    w = 2.5 + float(change[0])
    assert w == pytest.approx(2.5075)
    assert 1 - (w - 2) ** 2 == pytest.approx(0.7424, abs=1e-4)


def test_every_path_needs_a_discount_below_one():
    wiring = Sparse(jnp.asarray([0]), jnp.asarray([1]), jnp.ones(1), 2)
    with pytest.raises(ValueError, match="discount"):
        reward_diffusion(wiring, jnp.zeros(2), 1.0, root=1, paths="all")


def test_the_rule_runs_under_jit_and_vmap():
    pre, post, activity, size = random_graph(5)
    wiring = Sparse(jnp.asarray(pre), jnp.asarray(post), jnp.zeros(len(pre)), size)
    activities = jnp.asarray(np.stack([activity, -activity, 2 * activity]), jnp.float32)
    rewards = jnp.asarray([1.0, -1.0, 0.5])

    def spread(a, r):
        return reward_diffusion(wiring, a, r, root=int(post[0])).credit

    batched = jax.jit(jax.vmap(spread))(activities, rewards)
    for a, r, credit in zip(activities, rewards, batched, strict=True):
        np.testing.assert_allclose(credit, spread(a, r), rtol=1e-6, atol=1e-7)


def test_random_draws_the_network_neuralnet_init_draws():
    neurons, inputs, outputs = 300, 4, 3
    net = RNeuralNet.random(0, neurons, inputs, outputs)
    w = net.wiring
    pre, post, weight, delay = (np.asarray(a) for a in (w.pre, w.post, w.weight, w.delay))
    internal = post < neurons
    inner = internal & (pre < neurons)
    assert np.all(pre != post) and len(set(zip(pre, post, strict=True))) == len(pre)
    assert np.bincount(pre[inner], minlength=neurons).max() == 16
    assert np.bincount(post[internal], minlength=neurons).max() <= 16 + 1  # one more from an input neuron
    fed = post[pre >= neurons]
    assert len(fed) == inputs * 5 and len(set(fed)) == len(fed)
    assert np.array_equal(np.sort(pre[post == net.feeder]), np.asarray(net.outputs))
    assert not set(fed) & set(np.asarray(net.outputs).tolist())
    # A connection's delay is its Myelin, 0 to 19, plus 1, plus 1 more when its source sends after its target.
    myelin = delay[inner] - 1 - (pre[inner] > post[inner])
    assert myelin.min() == 0 and myelin.max() == 19
    assert np.all(delay[pre >= neurons] == 3)
    assert np.std(weight[inner]) == pytest.approx(0.25, rel=0.05)
    threshold = np.asarray(net.cell.inner.threshold)
    assert np.mean(threshold[:neurons]) == pytest.approx(2.0, abs=0.1)
    assert np.std(threshold[:neurons]) == pytest.approx(0.5, rel=0.15)


def test_wire_refuses_a_unit_that_would_send_twice_a_tick():
    # NeuralNet_init makes the output neurons' connections to the feeder last of all.
    f, _ = original()
    feeder = len(f["threshold"]) + int(f["inputs"])
    last = np.argsort(f["post"] == feeder, stable=True)
    with pytest.raises(ValueError, match="one after another"):
        RNeuralNet.wire(f["threshold"], f["pre"][last], f["post"][last], f["weight"][last], f["myelin"][last],
                        inputs=int(f["inputs"]), outputs=f["outputs"])
