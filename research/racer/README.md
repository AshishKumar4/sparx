# The racer: a better driver, and whether a spiking one is worth it

The racer on sparxml.dev drives a car around closed tracks from an event camera's output and its own
speed, trained end to end by gradients through its spikes, the car and the camera's pixels
(`site/lab/racer.py`). This directory holds the work to make it better and to compare it with
non-spiking networks: the sweeps' configurations, the scripts that run and summarize them, and their
results. Models and raw logs stay out of git; each sweep's are attached to a GitHub release named after it.

```bash
research/racer/sweep.sh research/racer/configs/<sweep>.json <label> <out dir> [pool]   # on armada
python research/racer/summarize.py <out dir> research/racer/results/<sweep>.json
```

## Why the published racer is weak

Training is unstable. Gradients carried back through 150 steps of car, camera and network explode.
In sweep 2 (`results/sweep2.json`) the one seed that learned had a median gradient norm of 97 and a
largest of 1.2e7. The three that failed had medians of 6.8e6, 5.3e3 and 2.8e4, and largest norms up to
2.6e15. Clipping to norm 1 holds the step's size, but its direction comes from whichever path exploded.
Seed 2 learned to drive backwards: a mean speed of -1.6 m/s at step 750, and 45% of its time off the
road. The run at a learning rate of 7e-4 settled on crawling at 0.4 m/s. One seed of four learned to
drive.

The camera is coarse and aliased. Each of its 24 × 12 pixels samples one point of the ground. The columns
are 0.05 m apart 0.6 m ahead, 0.18 m apart at 2.3 m and 0.55 m apart at 7 m, so the 0.16 m centre dash is
narrower than a column past 2.3 m, and at 7 m the whole road is 2.9 columns wide. Sampling one point makes
pixels flicker, and 123 of the 288 fire each step.

It fails on bends tighter than a metre. On 200 unseen tracks at each of four bend strengths, 800 in all
(`results/difficulty-published.json`), the published network finished 100% of tracks whose tightest bend
has a radius of 1.5 m or more, 99.3% between 1 and 1.5 m, and 89.9% below 1 m. 35 of its 36 failures are
on bends tighter than a metre.

## The plan

1. Phase 0, on armada's CPUs: truncated gradients (6 s of driving, the gradient cut every 50 steps), a
   curriculum over the tracks' bends, and a penalty on driving backwards, on the published camera and
   network, five seeds against the published recipe's five (`configs/phase0.json`).
2. A camera of 64 × 32 pixels, each the mean of 2 × 2 points of the ground, and a convolutional network
   over it (`Vision` in `site/lab/stack.py`): two convolutions of stride 2 into 16 and 32 channels, a
   dense layer of 128 units and two leaky-integrator readouts, 0.53 million parameters.
3. The comparison, at the same budget, procedure and five seeds each: A, that network spiking, on
   events; B, the same network with graded units (the same leaky membrane read through a ReLU), on
   events; C, B on an ordinary camera's frames. Reported: the success rate over seeds and over 200 unseen
   tracks, the harder sets by bend, lap time, parameters, and multiply-adds per step counted two ways
   (below).

Multiply-adds per step: *dense* counts every connection, which is what a GPU or CPU computes; *triggered*
counts only those from inputs and units that are not zero, which is what event-driven hardware would
compute. A smaller triggered count is not a speed on a GPU.

## Results

| Run | Seeds that learned | Unseen tracks finished | Notes |
| --- | --- | --- | --- |
| sweep 1: 1,500 steps, seed 1 | 1 of 1 | 187 of 200 | `results/sweep1.json` |
| sweep 2: 3,000 steps, horizon 150 | 1 of 4 | 200, 2, 0 and 2 of 200 | `results/sweep2.json`; seed 1 is the published racer |

One training step of each arm, batch 32 and 300 steps of driving cut every 50, on an RTX 4080: 0.42 s for
each conv arm, 0.042 s for the published network (`results/timing-rtx4080.json`).
