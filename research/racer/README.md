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
python research/racer/compare.py research/racer/results/<sweep>.json research/racer/results/<sweep>-arms.json
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
changes; by step 300 its gradient norm had fallen below 1e-3, and it finished no track. C seed 2, from
another random readout, finished 197 of 200. So from commit 673c337 every arm's readout starts at zero
(`configs/phase2c.json`, the whole comparison again): each car steers straight with a target speed of
3 m/s, every arm's first step has a mean speed of 2.14 m/s at seed 1 and 1.89 at seed 2 and a gradient norm
of 54 to 79 (`configs/phase2-start.json`). Trained for about 900 steps on the 4080, C at seeds 1 and 2 had
mean speeds of 4.3 and 5.0 m/s at step 50 and 5.35 and 5.17 at step 250, never fell below 1.9 m/s at a
logged step, and scored 1.83 and 1.77 laps on its held-out tracks at step 250. Phase 2
and 2b's results stay in `results/phase2.json` and `results/phase2b.json` on the `racer-phase2` branch.

## Why the first published racer was weak

The racer on the site was sweep 2's seed 1, then Phase 0b's seed 1 (a median lap of 9.5 s, every one of
the 200 tracks, 100, 100, 100 and 99.5% of the harder sets by bend). It is now Phase 2's run 7, arm A at
seed 2, the spiking convolutional network whose parameters drove farthest on their held-out tracks of
the five A seeds (`site/public/racer/`): every one of the 200 tracks in a median lap of 9.85 s, never off
the road, and 100, 100, 100 and 99% of the harder sets by bend, its two failures of 800 on bends tighter
than a metre. What follows is about the first.

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

The browser runs the spiking convolutional network as sparx does (`site/src/engines/vision.ts`): on the
published network and a small one at its random start, every event and every spike matches sparx's float64
run, the readouts agree to 1e-15 of their size, and the car stays within the bound float64's rounding allows
(`site/lab/car_exact.py`).
The graded and frame arms run only in sparx.

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

## Phase 2: spiking against graded, and the spike variants

Seven arms, five seeds each, one recipe (`configs/phase2c.json`, commit 673c337), trained one after another
on the workstation's RTX 4080, about 18 minutes a run. Each arm is the convolutional network of the plan
on the 64 × 32 camera; only its units change. `results/phase2c.json` holds every run, and
`results/phase2c-arms.json` each arm's seeds with their means and standard deviations
(`compare.py`). Means ± sample standard deviations over the five seeds. A seed's lap is its median over
the tracks it finished, so the 4-bit arm's lap is over the four seeds that finished any, its seed 2 having
finished none:

| Arm | Seeds that finished every unseen track | Unseen tracks finished, % | Bend 1.8, % | Lap, s | Steers back, ms |
| --- | --- | --- | --- | --- | --- |
| A, spiking, on events | 5 of 5 | 100.0 ± 0.0 | 95.5 ± 3.5 | 9.8 ± 0.2 | 20 ± 0 |
| B, graded, on events | 4 of 5 | 99.9 ± 0.2 | 96.1 ± 6.2 | 10.8 ± 1.6 | 170 ± 143 |
| C, graded, on frames | 4 of 5 | 98.6 ± 3.1 | 89.7 ± 11.6 | 12.4 ± 5.8 | 216 ± 84 |
| spikes of 2 bits | 3 of 5 | 87.9 ± 25.4 | 79.8 ± 36.6 | 10.7 ± 1.2 | 70 ± 35 |
| spikes of 4 bits | 1 of 5 | 74.7 ± 42.2 | 64.3 ± 37.8 | 14.7 ± 5.8 | 96 ± 61 |
| sigma-delta | 5 of 5 | 100.0 ± 0.0 | 96.3 ± 4.2 | 10.0 ± 0.2 | 112 ± 11 |
| dendritic | 5 of 5 | 100.0 ± 0.0 | 98.1 ± 4.0 | 9.8 ± 0.2 | 20 ± 0 |

| Arm | Parameters | Units | Units sending a step | Bits sent a step | Dense MACs a step | Triggered MACs a step |
| --- | --- | --- | --- | --- | --- | --- |
| A, spiking, on events | 530,258 | 12,416 | 1,161 ± 78 | 1,161 | 1,464,992 | 151,920 ± 6,466 |
| B, graded, on events | 530,258 | 12,416 | 6,608 ± 916 | 211,444 | 1,464,992 | 629,750 ± 64,858 |
| C, graded, on frames | 529,858 | 12,416 | 5,545 ± 449 | 177,438 | 1,271,568 | 652,209 ± 63,466 |
| spikes of 2 bits | 530,258 | 12,416 | 1,782 ± 371 | 3,563 | 1,464,992 | 246,133 ± 67,222 |
| spikes of 4 bits | 530,258 | 12,416 | 1,899 ± 1,231 | 7,595 | 1,464,992 | 259,023 ± 155,781 |
| sigma-delta | 530,258 | 12,416 | 1,244 ± 241 | 39,820 | 1,464,992 | 152,430 ± 21,363 |
| dendritic | 531,154 | 12,928 | 1,258 ± 93 | 1,258 | 1,465,504 | 151,696 ± 10,021 |

Bits sent a step count one bit per binary spike, 2 or 4 per few-bit spike, and 32 per graded or sigma-delta
value, which these arms send as float32. The camera's events are the same kind of input for every arm on
events, 485 to 936 a step.

What the comparison says:

- In these five runs under this recipe, the spiking arm finished as many unseen tracks as any arm, every
  one at every seed, as did sigma-delta and the dendritic arm. The graded arm on events was one track
  short at one seed; on the harder sets its mean is above A's, 96.1 against 95.5%, inside either arm's
  deviation. A was the most consistent from seed to seed: its laps were 9.8 ± 0.2 s against B's
  10.8 ± 1.6 and C's 12.4 ± 5.8. The graded arm on frames was the weakest of the three: one seed finished
  186 of 200 with a median lap of 22.7 s, and of A, B and C it did worst on the sharpest bends.
- On a GPU there is no saving. The arms on events have the same dense multiply-adds to within 0.1%; C has
  13% fewer, 1.27 million against 1.46, because a frame has one channel where events have two. A training
  step took 0.37 to 0.40 s for every arm on the RTX 4080 (`results/timing-rtx4080.json`).
- On event-driven hardware the spiking arm would do about a quarter of the graded arm's work: 152,000
  triggered multiply-adds a step against 630,000, because about 9% of its units send in a step against
  53% of B's, and it would send 1 bit where B sends 32, 1,161 bits a step against 211,000. That is a count
  of operations, not a measurement on such hardware.
- A and the dendritic arm steered back from a sideways nudge within one step, 20 ms, at every seed; the
  few-bit arms took 70 and 96 ms, sigma-delta 112, and the graded arms 170 and 216 ms to reach half
  their response. These runs do not isolate why. One candidate is the units: a graded unit passes on its
  membrane, which follows a change in its input over its time constant of 3 steps, while a LIF neuron
  fires as soon as its membrane crosses threshold. As an idealized illustration only, three linear leaky
  stages of 3 steps in a row pass half of a step change after a median of about 160 ms; the network's
  readout, its ReLUs and thresholds, and the car's own response all add to or change that.
- More bits per spike made things worse, not better: the 2- and 4-bit arms sent more bits, did more
  triggered work and were the least reliable, the 2-bit arm finishing 85 of 200 at seed 2 and the 4-bit
  arm none (see the open questions below). Sigma-delta matched the spiking arm's reliability and work, but
  sends real values, 32 bits each, and responded in 112 ms. Active dendrites matched or beat the plain spiking arm at every
  measure of driving, and trained the most stably of all, with a median gradient norm of 0.1 at four seeds
  of five.

A homeostatic arm trained afterwards (`configs/phase2-homeostatic.json`, `results/phase2-homeostatic.json`):
A's network with each neuron's threshold drifting toward 3% of steps (`sparx.nn.Homeostatic`, the default
rate η = 0.001 a spike). It drove as A did, every track at every seed, laps of 9.8 ± 0.2 s, 95.9 ± 3.2% at
bend 1.8 and a 20 ms response, and it did not change the network's activity: 1,157 ± 173 units sent a
step, 9.3% of them, against A's 1,161 ± 78. At η = 0.001 and about 9% of steps firing, a threshold rises by
about 0.1 over a 30 s drive, too little to bring the rate toward 3%.

Five seeds an arm make these coarse: an arm's deviation is often larger than the difference between two
arms, and only the few-bit arms' failures and the latency gap stand clear of it.

## Open questions the comparison raised

Spikes of a few bits trained unstably under the common recipe. At seed 2 the 2-bit arm drove at 5.6 m/s
on its training tracks by step 100, slowed to 0.16 m/s by step 250, and then trained with gradient norms
of 1e3 to 7e4, ending with 85 of 200 unseen tracks finished. The 4-bit arm at seed 2 collapsed by step
100 and finished none, with gradient norms of 3e3 to 1e4 for the rest of its run. The spiking arm A,
from the same start, finished all 200 at that seed. A few-bit spike is the sum of one surrogate step per
level (`FewBitCell`), so its derivative is the sum of their surrogates: a membrane near several levels
passes several times a binary spike's gradient, and nothing scales it down. A candidate for the next study:
the same arms with each level's surrogate scaled by the number of levels, or one surrogate of the
rounded membrane, against the binary spike. This comparison keeps the recipe as it is.
