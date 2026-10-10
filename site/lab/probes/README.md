# Probes

Short runs of the site's engines or sparx that a chapter's numbers come from, kept so each number can be
made again. A TypeScript probe runs on armada against a commit's engines; a Python one runs with sparx:

```bash
site/lab/armada/bun.sh HEAD <label> site/lab/probes/<probe>.ts
site/lab/armada/run.sh HEAD <label> site/lab/probes/<probe>.py
```

| Probe | Backs | Output |
| --- | --- | --- |
| `regimes.ts`, `chaos.ts` | chapter 9: Brunel's states at the figure's sizes, and how fast nudged copies part | `results/regimes*.txt`, `results/chaos.txt` |
| `twins-j1.ts`, `twins-dbg.ts` | chapter 9: the twin networks with 1 mV synapses, and the nudge crossing threshold | printed |
| `hh-ramp.ts`, `hh-ramp2.ts` | chapter 10: Hodgkin-Huxley's onset, its lowest rate and its two states between 630 and 1,040 pA | printed |
| `events-rate.ts` | chapter 12: the event camera's events a second by fan speed and threshold | printed |
| `recovery.ts`, `recovery2.ts` | chapter 13: the three recoveries' arrival times, spikes and rotor saturation | printed |
| `optim_equiv.py` | the lab's training loop built on dew's OptimConfig, bitwise equal to the optax chain before it | printed |

`results/conductance.txt` and `results/hodgkin.txt` are the outputs of chapter 10's snippets
(`site/snippets/conductance.py`, `hodgkin.py`) as the chapter quotes them.
