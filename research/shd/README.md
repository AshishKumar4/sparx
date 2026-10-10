# SHD: sparx against the official SNN-delays code

Hammouamri et al.'s recipe ("Learning Delays in Spiking Neural Networks using Dilated Convolutions with
Learnable Spacings", ICLR 2024, arXiv 2306.17670) for the Spiking Heidelberg Digits: a 140-256-256-20
network of LIF neurons with learned delays, 150 epochs. sparx's implementation and the authors' code ran
the same recipe, three seeds each, on the same kind of GPU, through `tools/shd_comparison.py`, whose
docstring gives the commands and environments. `runs/<code>-<protocol>-<seed>/result.json` holds each
run's accuracy at every epoch, its timing and conditions; `summary.json` the means and deviations.

| Code and protocol | Last epoch, seeds 0, 1, 2 | Last epoch, mean ± sd | Best epoch on test, mean ± sd |
| --- | --- | --- | --- |
| official SNN-delays, all of the training set | 93.63, 93.90, 94.14% | 93.89 ± 0.26% | 95.17 ± 0.61% |
| sparx, all of the training set | 93.73, 94.30, 93.95% | 93.99 ± 0.29% | 94.96 ± 0.89% |
| sparx, a tenth of the training set held out | 94.21, 93.20, 94.79% | 94.07 ± 0.81% | 95.30 ± 0.37% |

On the protocol both codes share, training on every training recording and scoring the test set after
each epoch, sparx's last epoch is 93.99 ± 0.29% and the official code's 93.89 ± 0.26%, means and sample
standard deviations over three seeds. The paper reports 95.07 ± 0.24% over ten runs, a 95% confidence
interval from the t-distribution, of the best test accuracy over the epochs, which chooses the epoch on
the test set; the official code's best epoch here averages 95.17%, with a standard deviation of 0.61%. A number that does not use the test
set to choose comes from sparx's held-out protocol: the epoch with the best accuracy on the held-out
tenth scores 93.55, 93.60 and 95.27% on test, 94.14 ± 0.98%.

Conditions: an NVIDIA A100-SXM4-40GB on Colab for every run. sparx at commit 70005ec (sparxml 0.1.0,
dewml 0.1.0, jax 0.11.2.post3, the build `constraints.txt` pins, flax 0.12.10, optax 0.2.8, Python 3.13); the official code at SNN-delays
d169b4e3 on SpikingJelly 6fbee6ed, torch 2.11.0+cu130 and dcls 0.1.1, their `best_config_SHD.py`. An
epoch, with evaluation, data loading and start-up spread over the run, took 2.6 to 2.9 s in sparx and
10.6 s in the official code. These are three seeds a code, so the deviations are rough.
