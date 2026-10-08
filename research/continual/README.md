# Continual learning: a modular core with selective fast plasticity

The owner's research notes on bio-inspired continual learning propose an agent whose recurrent core keeps three kinds of memory apart: its activity, fast weights on a few connections that change within an experience, and slow weights that learn across experiences. This directory builds the first pieces they recommend, on sparx and dew's public API alone:

- `core.py`, the recurrent core of the notes' table of initial choices: modules of leaky tanh units, dense within each module, a few delayed connections between modules, time constants that differ by module, and fast plasticity on a few incoming connections of each unit (`sparx.dynamics.FastWeights(connections=...)`, which keeps traces for those alone).
- `task.py`, a session of trials under a hidden mapping from switches to doors, the notes' test of learning within a new situation: each session draws a new mapping, and only pressing switches and seeing which door opens reveals it.
- `train.py`, the comparison the notes ask for: the core with fast plasticity, the same core without it, and a conventional recurrent network given the same observations and feedback, each trained by REINFORCE through whole sessions on dew's `Trainer`, with the gradient flowing through the recurrent state and the fast weights.

```bash
python research/continual/train.py --agent plastic    # or core, or rnn
```

Each run ends by printing its test accuracy on each trial of a session, pressing the highest-scoring switch. The first trial can only be a guess, and the trials after it show how much the agent learned from what it saw.

## Results

Measured on 8 October 2026 on a 4-core CPU with the script's defaults: seed 0, sessions of 8 trials of 3 steps, batch 32, Adam at 1e-3, an entropy bonus of 0.1, and each choice credited with its own trial's reward (`--discount 0`). Each row is the accuracy on each trial of 1,024 test sessions, pressing the highest-scoring switch. Its standard error is up to 1.6 points. On the first trial no agent can do better than guess, and the six runs score within 2.3 standard errors of chance. `plastic` and `core` are 4 modules of 64 units with 16 inputs per unit from other modules (20,480 connections); `plastic` adds fast weights on 16 inputs of each unit (4,096 connections) under `ModulatedTrace`. `rnn` is 256 tanh units connected all to all (65,536 connections). The last row is a learner that remembers every pair it has seen (`SwitchDoor.ideal`).

Two doors, 50,000 training sessions:

| Agent | Parameters | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plastic` | 28,163 | 0.498 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 624 s |
| `core` | 23,298 | 0.502 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 297 s |
| `rnn` | 68,354 | 0.517 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 26 s |
| remembers every pair | | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | |

Four doors, 100,000 training sessions:

| Agent | Parameters | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plastic` | 30,213 | 0.281 | 0.515 | 0.688 | 0.802 | 0.897 | 0.946 | 0.963 | 0.981 | 1,254 s |
| `core` | 25,348 | 0.229 | 0.481 | 0.625 | 0.644 | 0.652 | 0.624 | 0.628 | 0.675 | 595 s |
| `rnn` | 70,404 | 0.249 | 0.493 | 0.721 | 0.705 | 0.720 | 0.739 | 0.761 | 0.732 | 45 s |
| remembers every pair | | 0.250 | 0.500 | 0.688 | 0.828 | 0.910 | 0.954 | 0.977 | 0.988 | |

```bash
python research/continual/train.py --agent plastic --doors 2 --sessions 50000
python research/continual/train.py --agent plastic --doors 4 --sessions 100000
```

Accuracy on the last trial at each evaluation during training on four doors:

| Agent | 25,000 sessions | 50,000 | 75,000 | 100,000 |
| --- | ---: | ---: | ---: | ---: |
| `plastic` | 31.2% | 74.0% | 94.7% | 98.1% |
| `core` | 28.0% | 49.7% | 58.8% | 67.5% |
| `rnn` | 29.5% | 53.0% | 63.4% | 73.2% |

## What the runs so far say

- Two doors need one fact carried from the first trial, which switch opened which door, and every agent learns that: after one trial each is right every time.
- Four doors separate them. The core with fast weights follows the curve of the learner that remembers every pair it has seen, to 98.1% on the last trial against that learner's 98.8%. Without fast weights the same core levels off near 65% from the third trial on, and the dense network, with 2.8 times the parameters, near 73%. Each uses what its first two trials showed and little of what came after. Here the fast weights hold the pairs a session reveals, which neither network's activity learned to hold in the same training.
- The fast weights also learned faster: halfway through training the plastic core was right on 74.0% of last trials, the core without them on 49.7% and the dense network on 53.0%.
- These are single runs at seed 0, no agent had stopped improving at 100,000 sessions, and one task cannot say how far the advantage carries. A training step of the plastic core takes 385 ms on this CPU, the core's 185 ms and the dense network's 11 ms.

## Next

- The notes' size, 16 modules of 256 units with 16 plastic inputs per unit (`--modules 16 --units 256`): 1.11M connections against the small core's 20,480.
- Credit for exploration: with `--discount` above 0 a press is credited with what it teaches later trials, which the notes ask for and which a variance-reducing baseline (a learned value) would need first.
- Replay and consolidation, and PC-ALM against BPTT for the slow weights, the notes' later stages.
- A task whose dynamics change without announcement, the notes' first suggestion for an agent that must retain earlier skills.
- A faster recurrence over the sparse wiring: the cores' steps take 17 and 35 times the dense network's.
