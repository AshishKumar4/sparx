import jax

from sparx.dynamics import Delta, LeakyIntegrateAndFire, PairSTDP, Receptor
from sparx.graph import FixedProbability, Network, PoissonInput, Population, Projection, simulate

stdp = PairSTDP(tau_plus=16.8, tau_minus=33.7, lambda_=0.005, alpha=0.55, mu_plus=0.0, mu_minus=0.0,
                w_max=1.0)                                  # NEST's stdp_synapse, additive
neuron = LeakyIntegrateAndFire(tau_m=10.0, e_l=0.0, v_th=20.0, v_reset=0.0, t_ref=1.0)
network = Network(
    populations=(Population("in", 400, LeakyIntegrateAndFire(), {"ampa": Receptor(Delta())}),
                 Population("out", 1, neuron, {"ampa": Receptor(Delta())})),
    projections=(Projection("in", "out", FixedProbability(1.0), weight=0.475, delay=1.0,
                            receptor="ampa", plasticity=stdp),),
    inputs=(PoissonInput("in", rate=40.0, weight=30.0, receptor="ampa"),),
    dt=1.0,
)
result = simulate(network, network.init(jax.random.key(0)), duration=2000.0, key=jax.random.key(1))
pre, post, weight, delay = network.connections(result.variables)["in->out:ampa"]
print(weight.min(), weight.max())                         # the weights as STDP left them
