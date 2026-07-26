# Compounding Error, Recovery, and Coordination Overhead in Long-Horizon Agentic AI Systems

A simulation-based reliability analysis of long-horizon agentic AI task execution.

## What this is

A controlled Monte Carlo simulation framework that isolates and quantifies three structural
mechanisms in long-horizon LLM agent task execution:

1. **Compounding per-step error** — how task success probability decays as the number of
   sequential steps grows, even at high per-step reliability.
2. **Checkpoint-and-rollback recovery** — how much a verify/rollback strategy can offset that
   decay, and at what execution-cost overhead.
3. **Multi-agent coordination overhead** — how splitting a task across multiple coordinating
   agents affects reliability, separately from ordinary step-level error.

...plus a fourth mechanism, their **interaction**: whether within-agent recovery can compensate
for inter-agent coordination failure (short answer, per Experiments E/F: no — they behave as
largely separable design levers).

This is **not** a benchmark of real LLM agents. It is a structural, parameterized simulation
intended to give first-principles intuition for reliability tradeoffs that are hard to isolate
cleanly from noisy real-world agent traces, and a reference model that can be calibrated against
real trajectories in future work. See the paper's Limitations section for a full discussion of
what this model does and does not claim.

## Repository structure

```
.
├── code/
│   ├── simulation.py       # All Monte Carlo experiments (A–F), fixed seed, fully reproducible
│   ├── make_figures.py     # Regenerates every figure in the paper from results.json
│   └── results.json        # Raw output of simulation.py (already generated, checked in)
├── figures/                # All 9 figures used in the paper (PNG)
├── paper/
│   ├── paper.tex           # LaTeX source (arXiv-style, compiles standalone with figures/ copied alongside)
│   ├── paper.pdf           # Compiled PDF
│   └── Agentic_AI_Reliability_Paper.docx   # Word version (same content, alternate format)
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducing the results

```bash
pip install -r requirements.txt
cd code
python3 simulation.py       # regenerates results.json (fixed seed 20260726 — bit-for-bit reproducible)
python3 make_figures.py     # regenerates all figures from results.json
```

To recompile the LaTeX paper:

```bash
cd paper
cp ../figures/*.png .
pdflatex paper.tex
pdflatex paper.tex   # run twice for cross-references / TOC
```

## Headline findings

- Task success probability decays **multiplicatively** with horizon length: at a 97%-reliable
  step, a 150-step task succeeds only ~1.1% of the time with no recovery mechanism.
- Checkpoint-and-rollback recovery can **more than triple** success probability at long horizons,
  and its benefit is driven far more by **detection quality** than by checkpoint granularity.
- Multi-agent decomposition is reliability-neutral **only under perfect coordination**; a modest
  10% per-handoff coordination-failure rate costs an 8-agent system ~15 percentage points of
  absolute success probability relative to a single agent.
- Within-segment recovery substantially raises absolute reliability in multi-agent systems but
  does **not** specifically close the coordination-overhead gap — validated across a 3×3 grid of
  coordination-failure and detection-quality settings (Experiment F).

Full details, all parameter settings, and honest limitations are in `paper/paper.pdf`.

## Status / how to use this

This is an independent research report / preprint draft. Before submitting anywhere:

1. **Re-run the code yourself** and confirm you reproduce every number in the paper.
2. **Read it critically** — the Limitations section is not boilerplate; it lists real
   simplifying assumptions (i.i.d. step failures, no adversarial error model, a bounded
   parameter grid for the separability claim, etc.) that matter for how far these results
   generalize.
3. **Decide on a venue.** As a pure simulation study with no real LLM trace data, this is best
   suited to an arXiv preprint or a workshop submission on agent reliability/evaluation
   methodology — not a top-tier main-track empirical paper, unless extended with real-trace
   calibration (see "Future Work" in the paper).

## License

MIT — see `LICENSE`.
