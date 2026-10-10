# The racer: a better driver, and whether a spiking one is worth it

The racer on sparxml.dev drives a car around closed tracks from an event camera's output and its own
speed, trained end to end by gradients through its spikes, the car and the camera's pixels
(`site/lab/racer.py`). This directory holds the work to make it better and to compare it with
non-spiking networks: the sweeps' configurations, the scripts that run and summarize them, and their
results. Models and raw logs stay out of git; each sweep's are attached to a GitHub release named after it.

```bash
research/racer/sweep.sh research/racer/configs/<sweep>.json <label> <out dir> [pool]   # on armada
python research/racer/run.py research/racer/configs/<sweep>.json <out dir>              # on a GPU
research/racer/queue.sh research/racer/configs/<sweep>.json <out dir> <results checkout> <first> <last>
python research/racer/summarize.py <out dir> research/racer/results/<sweep>.json
```

`configs/phase2.json` is the comparison: the seven arms below, A, B, C, the 2- and 4-bit arms, sigma-delta
and the dendritic arm, five seeds each, at Phase 0's recipe (truncated at 50 steps of a 300-step drive, the
curriculum from bends of 0.3 to 1.3, a reverse penalty of 2, 3,000 steps), each run keeping the parameters
that drove farthest on 32 held-out tracks, scored every 250 steps (`--select`). Its runs are ordered seed
by seed, all seven arms of seed 1 first, so any prefix compares every arm. It runs on the workstation's RTX
4080 through `queue.sh`, one run at a time, each committing the sweep's summary before the next begins.

Phase 2 stopped after 9 runs because arm C, seed 1, never moved: its frames reached the network as raw log
brightness, and from the first step the speed readout's sigmoid sat at zero, with gradient norms falling to
1e-12. The graded arms on events started slow too, at 0.08 to 0.48 m/s on the first step against the
spiking arms' 1.9 to 2.1, because a graded unit's summing membrane, with no reset to bound it, grows its
activity at each layer. From commit 290780a frames are standardized per image and the graded units'
membranes average their input (`configs/phase2b.json`). That restart stopped too: C seed 1 started
healthy, its random readout spinning the cars backwards at 1.27 m/s, and within 50 steps learned to stand
still, where neither the off-road nor the reverse penalty costs anything and a still camera's frame never
changes; by step 250 its gradient norm had fallen below 1e-3, and it finished no track. C seed 2, from
another random readout, finished 197 of 200. So from commit 673c337 every arm's readout starts at zero
(`configs/phase2c.json`, the whole comparison again): each car starts straight at 3 m/s, every arm's first
step has a mean speed of 2.14 m/s at seed 1 and 1.89 at seed 2 and a gradient norm of 54 to 79
(`configs/phase2-start.json`), and C at seeds 1 and 2 drove at 4.3 to 5.6 m/s from step 50 on. Phase 2
and 2b's results stay in `results/phase2.json` and `results/phase2b.json` on the `racer-phase2` branch.

## Why the first published racer was weak

The racer on the site was sweep 2's seed 1 until Phase 0b; it is now Phase 0b's seed 1, the seed whose
parameters drove farthest on their held-out tracks (`site/public/racer/`). On the evaluation's 200 tracks
it finished every one in a median lap of 9.5 s, never left the road, and on the harder sets finished
100, 100, 100 and 99.5% by bend, its one failure of 800 on a bend tighter than a metre. What follows is
about the first.

Training is unstable. Gradients carried back through 150 steps of car, camera and network explode.
Sweep 2 (`results/sweep2.json`) ran four configurations: seeds 1 and 2 of the first recipe, seed 1
at a learning rate of 7e-4, and seed 3 at a batch of 64. The run that learned, seed 1 of the first
recipe, had a median gradient norm of 97 and a largest of 1.2e7. The three that failed had medians of
6.8e6, 5.3e3 and 2.8e4, and largest norms up to 2.6e15. Clipping to norm 1 holds the step's size, but its
direction comes from whichever path exploded. Seed 2 learned to drive backwards: a mean speed of -1.6 m/s
at step 750, and 45% of its time off the road. The run at 7e-4 settled on crawling at 0.4 m/s. One run of
four learned to drive.

The camera is coarse and aliased. Each of its 24 × 12 pixels samples one point of the ground. The columns
are 0.05 m apart 0.6 m ahead, 0.18 m apart at 2.3 m and 0.55 m apart at 7 m, so the 0.16 m centre dash is
narrower than a column past 2.3 m, and at 7 m the whole road is 2.9 columns wide. Sampling one point makes
pixels flicker, and 123 of the 288 fire each step.

It fails on bends tighter than a metre. On 200 unseen tracks at each of four bend strengths, 800 in all
(`results/difficulty-published.json`), the first published network finished 100% of tracks whose tightest bend
has a radius of 1.5 m or more, 99.3% between 1 and 1.5 m, and 89.9% below 1 m. 35 of its 36 failures are
on bends tighter than a metre.

## The plan

1. Phase 0, on armada's CPUs: truncated gradients (6 s of driving, the gradient cut every 50 steps), a
   curriculum over the tracks' bends, and a penalty on driving backwards, on the 24 × 12 camera and
   dense network, five seeds against the first recipe's five (`configs/phase0.json`).
2. A camera of 64 × 32 pixels, each the mean of 2 × 2 points of the ground, and a convolutional network
   over it (`Vision` in `site/lab/stack.py`): two convolutions of stride 2 into 16 and 32 channels, a
   dense layer of 128 units and two leaky-integrator readouts, 0.53 million parameters.
3. The comparison, at the same budget, procedure and five seeds each: A, that network spiking, on
   events; B, the same network with graded units (the same leaky membrane read through a ReLU), on
   events; C, B on an ordinary camera's frames. Reported: the success rate over seeds and over 200 unseen
   tracks, the harder sets by bend, lap time, parameters, and multiply-adds per step counted two ways
   (below).

4. Spike precision, on A's network, budget and seeds: binary spikes (A); spikes of 2 and 4 bits, as Loihi
   2's graded spikes carry, where a spike says how many thresholds the membrane reached, up to 3 or 15,
   and resets it (`--bits`, `FewBit` in `site/lab/stack.py`); and sigma-delta units, B's graded units that
   send the change in their activation once it reaches 0.1 and receivers that add the changes up
   (`--neuron sigma-delta`), sigma-delta coding as Lava's `SigmaDelta` neurons do it on Loihi 2, though
   with a leaky membrane and unrounded changes. Sigma-delta at a threshold of 0 is B. Reported beside the
   success rate: events per step, the bits each event carries, and how fast the network steers back when
   its car is moved 0.3 m sideways 2 s into a drive (`response` in `site/lab/racer.py`): the time until
   the steering's difference from the same drive unnudged reaches a tenth of its largest, and half. The
   first published racer reaches half in a median of 180 ms on 99 tracks of 100, the current one in 20 ms on
   all 200 (`site/public/racer/evaluation.json`).
5. Active dendrites. sparx has no multi-compartment neuron: each of its models is one compartment,
   `Serial` chains models one way, and `GapJunction` couples two populations' membranes both ways only in
   a simulated `Network`, not in a layer trained by gradients. What the racer can train is Poirazi,
   Brannon and Mel's (2003) two-layer neuron, spiking (`--neuron dendritic`, `Dendrites`): each of the
   128 hidden neurons has 4 branches, each a LIF membrane fed by its own random quarter of the inputs,
   and a branch's spike reaches the soma through a learned coupling, with as many input weights as the
   dense layer it replaces. It lacks current flowing back from the soma into the branches, the long
   plateaus of NMDA spikes (Schiller et al. 2000), and a cable between compartments. A model in
   `sparx.dynamics` with a soma and dendrites coupled both ways would add them.

Multiply-adds per step: *dense* counts every connection, which is what a GPU or CPU computes; *triggered*
counts only those from inputs and units that are not zero, which is what event-driven hardware would
compute, each input or unit charged the connections it actually has (a pixel at the image's edge reaches
fewer of a convolution's outputs). An event arm's inputs are its events; a frame arm's are all its pixels,
every step, since a frame camera reads every pixel out whatever its value. A smaller triggered count is not a speed on a GPU.

The loss's band on firing rates (`--rate-low`, `--rate-high`) trains the spiking arms, A, the few-bit
arms and the dendritic arm, through their spikes' surrogate gradients. The graded arms, B, C and
sigma-delta, report the share of their units that are active or send, but the band has no gradient
through them and does not train them, as a ReLU network is ordinarily trained. The dendritic arm's GPU computes each
input's product for every branch, four times its dense count, of which three are zeros.

The browser runs the convolutional network as sparx does (`site/src/engines/vision.ts`): on small
networks at their random start, spiking on events and graded on frames, every event and every unit's
output matches sparx's float64 run, and the car's position, heading and speed agree within 1e-14
(`site/test/racer-conv.test.ts`).

## Results

| Run | Seeds that learned | Unseen tracks finished | Notes |
| --- | --- | --- | --- |
| sweep 1: 1,500 steps, seed 1 | 1 of 1 | 187 of 200 | `results/sweep1.json` |
| sweep 2: 3,000 steps, horizon 150 | 1 of 4 runs | 200, 2, 0 and 2 of 200 | `results/sweep2.json`: seeds 1 and 2, seed 1 at a learning rate of 7e-4, seed 3 at a batch of 64; seed 1 was the first published racer |
| Phase 0, the first recipe: seeds 3, 4 and 5 | 0 of 3 | 3, 94 and 25 of 200 | `results/phase0.json`, runs 0 to 2 |
| Phase 0, truncated at 50 steps, curriculum, reverse penalty: seeds 1 to 5 | 3 of 5 | 200, 200, 134, 163 and 200 of 200 | `results/phase0.json`, runs 3 to 7 |
| Phase 0b, the same with the parameters chosen on held-out tracks: seeds 1 to 5 | 5 of 5 | 200 of 200 each | `results/phase0b.json` |

Phase 0 ran on armada's CPUs at commit 6db9d24, 3,000 steps each at a learning rate of 1e-3 and a batch of
32. With seeds 1 and 2 from sweep 2, the first recipe has driven 1 of 5 seeds to finish its unseen
tracks. The new recipe drives 3 of 5 to finish every one, and the other two to 67% and 82%. Its median
gradient norms over training (every 50th step) were 0.1, 21, 11, 38 and 9, and its largest 3.3e3, against
medians of 1.6e3 to 8.4e3 and largest norms up to 3.2e14 for the first recipe's three seeds here. Its
three best seeds also drive faster: a median lap of 9.5, 16.2 and 10.7 s against the first published
racer's 20.6 s (sweep 2's seed 1 in `results/sweep2.json`). On the harder sets, by bend strength 0.9, 1.2, 1.5 and 1.8 (`difficulty` in each run), seeds 1
and 5 finished 99.5 to 100% of every set, seed 2 95.5 to 99%, and seeds 3 and 4 36.5 to 75%. The
first published racer finished 100, 99.5, 94 and 88.5% of the same sets (`results/difficulty-published.json`).
Each run of the new recipe took 72 to 93 minutes on one container.

Two of the new recipe's seeds drove well halfway through training and worse at its end: seeds 3 and 4 had
mean speeds of 5.35 and 5.04 m/s on their training tracks at step 1,500, once the curriculum reached its
hardest bends, and 2.19 and 3.50 m/s at step 3,000. Phase 0b (commit 3aa82a4) trains the same five seeds
the same way and keeps, of the parameters at every 250th step, those that drove farthest in 20 s on 32
tracks drawn as training's hardest are, never the evaluation's (`--select 250`). Every seed then
finished all 200 unseen tracks, with median laps of 9.5 to 10.9 s, and 96.5 to 100% of each harder set.
The parameters kept were those of steps 3,000, 1,250, 750, 1,500 and 2,000. Training is unchanged, so seed
1, whose best was its last, is Phase 0's run 3 again. The kept networks steer back from a 0.3 m nudge in a
median of 20 to 110 ms to half their response, against the first published racer's 180 ms.

One training step of each arm, batch 32 and 300 steps of driving cut every 50, on an RTX 4080, each arm in
a process of its own: 0.37 to 0.38 s for the conv arms, 0.40 s for the dendritic arm and 0.039 s for the
dense stack, peaking at 1.0 to 1.4 GB (`results/timing-rtx4080.json`). The step's time grows with
the batch, 0.88 s at 64 and 2.39 s at 160, so training several seeds in one batch would save nothing on
this GPU. On an 8-core CPU on armada a conv step takes 35 s against the dense stack's 1.9 s.
