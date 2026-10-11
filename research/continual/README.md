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
- A faster recurrence over the sparse wiring: since its gathers take rows (`docs/performance.md`), the cores' training steps take 9 and 25 times the dense network's.

## A second study: forgetting under local rules

When one network learns a stream of tasks in turn, with no replay, does a local learning rule forget less
of the earlier tasks than backpropagation through time? `stream.py` runs it on SHD with one network for
every learner: 700 inputs into 256 recurrent LIF neurons into 20 leaky readouts. `queue.sh` runs every
learner, stream and seed in turn on the workstation's RTX 4080.

- Streams: `classes`, SHD's 20 classes in 5 tasks of 4, 20 epochs each, one 20-way readout; and
  `permuted`, all 20 classes in every task with each task's 700 channels permuted (the first task's
  not), 5 epochs each.
- Learners: `bptt`, backpropagation through time with Adam at 1e-3; `ewc`, the same with elastic weight
  consolidation (Kirkpatrick et al. 2017) at three strengths; `eprop`, e-prop (`sparx.learn.eprop`,
  Bellec et al. 2020) with feedback through the readout's weights, and `eprop-random` through fixed
  random ones; `joint`, BPTT on every task at once for as many epochs, the reference.
- Measures, over 3 seeds: the accuracy on every task's test set after each task; the final accuracy over
  all 20 classes (class-incremental, chance 5%) and over each task's own classes (task-incremental, chance
  25% for `classes`); forgetting, a task's best accuracy before the last task minus its last; and each
  task's accuracy right after it was learned (for `joint`, the first task's).

### First results

`results/unclipped/`, 42 runs at commit e8bfab3 (`stream.py` as at 0d2b416), means ± sample deviations over seeds 0, 1 and 2
(`summarize.py`):

Classes in turn:

| Learner | Seeds | All 20 classes, % | Its own classes, % | Forgetting, points | Just learned, % |
| --- | --- | --- | --- | --- | --- |
| joint | 3 | 59.3 ± 20.3 | 84.6 ± 8.0 | none | 72.4 ± 8.7 |
| bptt | 3 | 15.5 ± 3.9 | 37.1 ± 6.4 | 89.7 ± 3.0 | 87.3 ± 5.9 |
| eprop | 3 | 17.1 ± 1.6 | 36.4 ± 3.0 | 90.0 ± 1.7 | 89.1 ± 2.5 |
| eprop-random | 3 | 16.2 ± 3.6 | 36.5 ± 1.7 | 89.1 ± 3.3 | 87.5 ± 1.8 |
| ewc1000 | 3 | 16.6 ± 1.9 | 51.0 ± 5.3 | 83.4 ± 9.5 | 83.3 ± 7.6 |
| ewc10000 | 3 | 16.1 ± 2.7 | 60.8 ± 9.9 | 87.9 ± 2.4 | 86.4 ± 0.7 |
| ewc100000 | 3 | 16.8 ± 1.9 | 77.3 ± 8.7 | 87.7 ± 3.9 | 87.0 ± 1.4 |

Permuted channels:

| Learner | Seeds | All 20 classes, % | Its own classes, % | Forgetting, points | Just learned, % |
| --- | --- | --- | --- | --- | --- |
| joint | 3 | 38.5 ± 7.9 | 38.5 ± 7.9 | none | 42.3 ± 6.4 |
| bptt | 3 | 31.3 ± 2.6 | 31.3 ± 2.6 | 14.9 ± 2.2 | 43.1 ± 2.7 |
| eprop | 3 | 45.9 ± 4.7 | 45.9 ± 4.7 | 33.7 ± 6.4 | 72.8 ± 0.5 |
| eprop-random | 3 | 36.5 ± 0.2 | 36.5 ± 0.2 | 21.5 ± 6.1 | 53.7 ± 4.9 |
| ewc1000 | 3 | 27.2 ± 0.6 | 27.2 ± 0.6 | 5.7 ± 0.7 | 30.4 ± 1.7 |
| ewc10000 | 3 | 19.8 ± 3.1 | 19.8 ± 3.1 | 2.9 ± 3.0 | 21.2 ± 2.6 |
| ewc100000 | 3 | 15.1 ± 2.7 | 15.1 ± 2.7 | 1.8 ± 1.5 | 16.5 ± 1.5 |

What the runs show:

- Learning classes in turn, every sequential learner ends at 15 to 17% of the 20 classes and forgets 83
  to 90 points of each earlier task: after the fifth task it answers almost only with the fifth task's
  classes. e-prop and random e-prop forget as much as BPTT. Joint training reaches 59 ± 20%.
- Within each task's own classes, e-prop (36%) and BPTT (37%) keep the same share. EWC keeps more as its
  strength grows, 51, 61 and 77%, without lifting the class-incremental score.
- With permuted channels, e-prop ends highest, 46 ± 5% against BPTT's 31 ± 3%. It learns each task
  better in 5 epochs, 73% right after learning it against 43%, and forgets more, 34 points against 15.
  EWC forgets 2 to 6 points and learns each new task to only 17 to 30%.
- BPTT trains this network less steadily than e-prop. Its loss rose by up to 1.9 nats from one epoch to
  the next within a task, and joint training's final accuracy ranges from 36% to 73% across seeds;
  e-prop's loss never rose by more than 0.14 within a task. Its gradient's median global norm, recorded
  in the clipped runs below, is 400 to 1,200 a batch for BPTT and 4 to 24 for e-prop.

### Clipping, an A/B

`results/clipped/` repeats every run with the gradient clipped to a global norm of 1 before Adam
(`--clip 1`, commit ca472b6), from the same initializations, data and batch order. It did not steady BPTT:
its largest rise in loss within a task grew to 7.2 nats. It lifted the task-incremental score slightly
for BPTT and e-prop (41 and 46% against 37 and 36%) and the permuted stream's final score for BPTT (36%
against 31%). It stalled EWC: the penalty's gradient, scaled by the Fisher information of BPTT's large
gradients, reached global norms of 1e7 to 1e14, so the clipped step followed the penalty alone, and
each new permuted task reached only 8%. Clipping is therefore off by default (`--clip 0`).

### Next

BPTT's instability makes it a weak baseline here. The next runs train BPTT at lower learning rates and
with a narrower surrogate, choose each learner's rate on held-out data from the first task, and run the
permuted stream for 20 epochs a task, so every learner reaches its own best on each task before the
next arrives.
