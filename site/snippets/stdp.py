import jax

from sparx.dynamics import (
                            Delta,
                            LeakyIntegrateAndFire,
                            PairSTDP,
                            Receptor,
)
from sparx.graph import (
                            FixedProbability,
                            Network,
                            PoissonInput,
                            Population,
                            Projection,
                            simulate,
)

# NEST's stdp_synapse with additive bounds
stdp = PairSTDP(tau_plus=16.8, tau_minus=33.7, lambda_=0.005,
                alpha=0.55, mu_plus=0.0, mu_minus=0.0, w_max=1.0)
delta = {"ampa": Receptor(Delta())}
listener = LeakyIntegrateAndFire(tau_m=10.0, e_l=0.0, v_th=20.0,
                                 v_reset=0.0, t_ref=1.0)
network = Network(
    populations=(Population("in", 400, LeakyIntegrateAndFire(), delta),
                 Population("out", 1, listener, delta)),
    projections=(Projection("in", "out", FixedProbability(1.0),
                            weight=0.475, delay=1.0, receptor="ampa",
                            plasticity=stdp),),
    inputs=(PoissonInput("in", rate=40.0, weight=30.0,
                         receptor="ampa"),),
    dt=1.0,
)
result = simulate(network, network.init(jax.random.key(0)),
                  duration=2000.0, key=jax.random.key(1))
connections = network.connections(result.variables)
weight = connections["in->out:ampa"].weight
print(weight.min(), weight.max())    # as STDP left them
