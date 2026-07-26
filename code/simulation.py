"""
A Simulation-Based Study of Error Compounding, Recovery Strategies, and
Coordination Overhead in Long-Horizon Agentic AI Systems.

This script implements a Monte Carlo reliability model of an LLM-based
agent executing a task composed of a sequence of discrete "steps"
(e.g., a tool call, a reasoning hop, a sub-goal). It is a controlled,
parameterized simulation -- not a replay of real LLM traces -- designed
to isolate and quantify structural effects (compounding error,
checkpoint/rollback recovery, and multi-agent coordination overhead)
that are difficult to isolate cleanly from noisy real-world traces.

All numbers reported in the paper are produced by this script and are
reproducible with the fixed random seed below.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

RNG_SEED = 20260726
N_TRIALS = 20000  # Monte Carlo trials per configuration

rng = np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------------
# Strategy 1: Baseline (no recovery). Task succeeds iff every one of n
# steps succeeds. This is the simplest possible model of "compounding
# error": step failures are treated as terminal.
# ---------------------------------------------------------------------------
def simulate_baseline(n_steps, p_step, n_trials=N_TRIALS, rng=rng):
    successes = 0
    total_cost = 0  # number of step-executions consumed (proxy for $ / LLM calls)
    for _ in range(n_trials):
        step_outcomes = rng.random(n_steps) < p_step
        total_cost += n_steps  # baseline always executes exactly n steps once
        if step_outcomes.all():
            successes += 1
    return successes / n_trials, total_cost / n_trials


# ---------------------------------------------------------------------------
# Strategy 2: Checkpoint + rollback recovery. Every `checkpoint_interval`
# steps, a verifier checks the trajectory so far. A failed step is caught
# with probability `detect_prob` (imperfect verification, mirroring
# real-world silent failures). On detection, the agent rolls back to the
# last checkpoint and retries the segment, up to `max_retries_per_segment`
# times, at an extra cost of `rollback_cost` steps per retry (re-reading
# state, re-planning). Undetected failures propagate forward silently,
# exactly as in the baseline.
# ---------------------------------------------------------------------------
def simulate_checkpoint_rollback(
    n_steps,
    p_step,
    checkpoint_interval,
    detect_prob,
    rollback_cost,
    max_retries_per_segment=2,
    n_trials=N_TRIALS,
    rng=rng,
):
    successes = 0
    total_cost = 0
    for _ in range(n_trials):
        step = 0
        cost = 0
        task_failed_silently = False
        while step < n_steps:
            segment_end = min(step + checkpoint_interval, n_steps)
            segment_len = segment_end - step
            attempt = 0
            segment_ok = False
            while attempt <= max_retries_per_segment:
                outcomes = rng.random(segment_len) < p_step
                cost += segment_len
                attempt += 1
                if outcomes.all():
                    segment_ok = True
                    break
                else:
                    # a failure occurred in this segment; is it caught?
                    if rng.random() < detect_prob:
                        # caught: pay rollback cost and retry the segment
                        cost += rollback_cost
                        continue
                    else:
                        # silent failure: propagate forward uncorrected
                        segment_ok = False
                        task_failed_silently = True
                        break
            if not segment_ok:
                # either retries exhausted or silent failure -> task fails
                break
            step = segment_end
        else:
            pass
        task_succeeded = (step >= n_steps) and not task_failed_silently
        if task_succeeded:
            successes += 1
        total_cost += cost
    return successes / n_trials, total_cost / n_trials


# ---------------------------------------------------------------------------
# Strategy 3: Multi-agent decomposition. The task is split across m agents
# (m-1 handoffs). Each handoff introduces an additional independent
# coordination-failure probability q (miscommunication / context loss
# between agents), on top of ordinary per-step error. Per-agent segments
# otherwise follow the baseline (no-recovery) model. Total steps held
# fixed at n across all m, so we isolate the coordination cost specifically.
# ---------------------------------------------------------------------------
def simulate_multi_agent(n_steps, p_step, m_agents, coord_fail_prob, n_trials=N_TRIALS, rng=rng):
    base_len = n_steps // m_agents
    remainder = n_steps % m_agents
    seg_lengths = [base_len + (1 if i < remainder else 0) for i in range(m_agents)]

    successes = 0
    total_cost = 0
    for _ in range(n_trials):
        ok = True
        cost = 0
        for i, seg_len in enumerate(seg_lengths):
            outcomes = rng.random(seg_len) < p_step
            cost += seg_len
            if not outcomes.all():
                ok = False
                break
            if i < m_agents - 1:  # a handoff follows this segment
                if rng.random() < coord_fail_prob:
                    ok = False
                    break
        total_cost += cost
        if ok:
            successes += 1
    return successes / n_trials, total_cost / n_trials


# ---------------------------------------------------------------------------
# Experiment A: success rate vs. horizon length (compounding error),
# baseline vs. checkpoint/rollback, at a fixed realistic per-step
# reliability.
# ---------------------------------------------------------------------------
def experiment_a():
    p_step = 0.97
    horizons = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    rows = []
    for n in horizons:
        succ_base, cost_base = simulate_baseline(n, p_step)
        succ_ckpt, cost_ckpt = simulate_checkpoint_rollback(
            n, p_step, checkpoint_interval=10, detect_prob=0.7, rollback_cost=3
        )
        rows.append(dict(n=n, p_step=p_step,
                          success_baseline=succ_base, cost_baseline=cost_base,
                          success_checkpoint=succ_ckpt, cost_checkpoint=cost_ckpt))
    return rows


# ---------------------------------------------------------------------------
# Experiment B: effect of per-step reliability p on the horizon at which
# success probability crosses below 0.5 ("50%-reliable horizon"),
# analogous in spirit (not scale) to METR's task-horizon metric.
# ---------------------------------------------------------------------------
def experiment_b():
    p_values = [0.999, 0.99, 0.98, 0.97, 0.95, 0.90]
    horizons = list(range(1, 4000, 1))
    rows = []
    for p in p_values:
        # analytic p^n is exact for the baseline (i.i.d. steps) -- use closed
        # form here for precision, cross-checked against simulation below.
        crossing_n = None
        for n in horizons:
            if p ** n < 0.5:
                crossing_n = n
                break
        rows.append(dict(p_step=p, horizon_50pct=crossing_n))
    # cross-check closed form vs. simulation at one setting
    p_check = 0.97
    n_check = 23  # near the analytic crossing point
    sim_succ, _ = simulate_baseline(n_check, p_check, n_trials=50000)
    analytic_succ = p_check ** n_check
    cross_check = dict(p_step=p_check, n=n_check,
                        analytic=analytic_succ, simulated=sim_succ,
                        abs_diff=abs(analytic_succ - sim_succ))
    return rows, cross_check


# ---------------------------------------------------------------------------
# Experiment C: multi-agent coordination overhead. Fixed total step
# budget n=60; vary number of agents m and coordination failure prob q.
# ---------------------------------------------------------------------------
def experiment_c():
    n_steps = 60
    p_step = 0.98
    m_values = [1, 2, 3, 4, 5, 6, 8]
    q_values = [0.0, 0.02, 0.05, 0.10]
    rows = []
    for q in q_values:
        for m in m_values:
            succ, cost = simulate_multi_agent(n_steps, p_step, m, q)
            rows.append(dict(n_steps=n_steps, p_step=p_step, m_agents=m,
                              coord_fail_prob=q, success=succ, cost=cost))
    return rows


# ---------------------------------------------------------------------------
# Experiment D: reliability-cost Pareto frontier across strategies.
# For a fixed horizon, sweep checkpoint interval and detection quality to
# show the accuracy/cost tradeoff surface of the recovery strategy.
# ---------------------------------------------------------------------------
def experiment_d():
    n_steps = 150
    p_step = 0.97
    intervals = [5, 10, 20, 30, 50]
    detect_probs = [0.3, 0.5, 0.7, 0.9]
    rows = []
    # baseline point for reference
    succ_base, cost_base = simulate_baseline(n_steps, p_step)
    rows.append(dict(strategy="baseline", checkpoint_interval=None,
                      detect_prob=None, success=succ_base, cost=cost_base))
    for ci in intervals:
        for dp in detect_probs:
            succ, cost = simulate_checkpoint_rollback(
                n_steps, p_step, checkpoint_interval=ci, detect_prob=dp, rollback_cost=3
            )
            rows.append(dict(strategy="checkpoint_rollback", checkpoint_interval=ci,
                              detect_prob=dp, success=succ, cost=cost))
    return rows


# ---------------------------------------------------------------------------
# Multi-seed replication wrapper: re-runs a simulation function with K
# independent random-number generators (distinct seeds) and reports the
# across-seed mean and standard deviation of the success-rate estimate.
# This directly estimates Monte Carlo sampling variance (as opposed to the
# single-seed point estimates used in Experiments A-D), per the reporting
# norms discussed in Section 4 (error bars / std. dev. across independent
# runs, not just a single point estimate).
# ---------------------------------------------------------------------------
def replicate(sim_fn, kwargs, n_seeds=5, base_seed=RNG_SEED, n_trials_per_seed=5000):
    succs, costs = [], []
    for k in range(n_seeds):
        local_rng = np.random.default_rng(base_seed + 1000 * (k + 1))
        s, c = sim_fn(**kwargs, n_trials=n_trials_per_seed, rng=local_rng)
        succs.append(s)
        costs.append(c)
    succs = np.array(succs)
    costs = np.array(costs)
    return dict(
        success_mean=float(succs.mean()), success_std=float(succs.std(ddof=1)),
        cost_mean=float(costs.mean()), cost_std=float(costs.std(ddof=1)),
        n_seeds=n_seeds, n_trials_per_seed=n_trials_per_seed,
    )


# ---------------------------------------------------------------------------
# Experiment A2: same sweep as Experiment A, but replicated across 5
# independent seeds to attach standard-deviation error bars to the
# headline success-probability curve (baseline vs. checkpoint/rollback).
# ---------------------------------------------------------------------------
def experiment_a_replicated():
    p_step = 0.97
    horizons = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    rows = []
    for n in horizons:
        base = replicate(simulate_baseline, dict(n_steps=n, p_step=p_step))
        ckpt = replicate(simulate_checkpoint_rollback,
                          dict(n_steps=n, p_step=p_step, checkpoint_interval=10,
                               detect_prob=0.7, rollback_cost=3))
        rows.append(dict(n=n, p_step=p_step,
                          success_baseline_mean=base["success_mean"], success_baseline_std=base["success_std"],
                          success_checkpoint_mean=ckpt["success_mean"], success_checkpoint_std=ckpt["success_std"]))
    return rows


# ---------------------------------------------------------------------------
# Experiment C2: same sweep as Experiment C, replicated across 5 seeds for
# error bars on the multi-agent coordination-overhead curve.
# ---------------------------------------------------------------------------
def experiment_c_replicated():
    n_steps = 60
    p_step = 0.98
    m_values = [1, 2, 3, 4, 5, 6, 8]
    q_values = [0.0, 0.05, 0.10]
    rows = []
    for q in q_values:
        for m in m_values:
            r = replicate(simulate_multi_agent,
                          dict(n_steps=n_steps, p_step=p_step, m_agents=m, coord_fail_prob=q))
            rows.append(dict(n_steps=n_steps, p_step=p_step, m_agents=m, coord_fail_prob=q,
                              success_mean=r["success_mean"], success_std=r["success_std"]))
    return rows


# ---------------------------------------------------------------------------
# Experiment E: interaction between recovery and multi-agent decomposition.
# Does adding checkpoint/rollback recovery *within each agent's segment*
# rescue the coordination-overhead losses documented in Experiment C? We
# hold total steps n=60 fixed, vary m_agents, apply per-segment recovery
# (checkpoint_interval=10, detect_prob=0.7) within each agent's segment,
# and compare against the no-recovery multi-agent results from Experiment C
# at a fixed, realistic coordination-failure probability q=0.05.
# ---------------------------------------------------------------------------
def simulate_multi_agent_with_recovery(n_steps, p_step, m_agents, coord_fail_prob,
                                        checkpoint_interval, detect_prob, rollback_cost,
                                        n_trials=N_TRIALS, rng=rng):
    base_len = n_steps // m_agents
    remainder = n_steps % m_agents
    seg_lengths = [base_len + (1 if i < remainder else 0) for i in range(m_agents)]

    successes = 0
    total_cost = 0
    for _ in range(n_trials):
        ok = True
        cost = 0
        for i, seg_len in enumerate(seg_lengths):
            # run this agent's segment as its own checkpoint/rollback sub-task
            step = 0
            seg_failed = False
            while step < seg_len:
                sub_end = min(step + checkpoint_interval, seg_len)
                sub_len = sub_end - step
                attempt = 0
                sub_ok = False
                while attempt <= 2:
                    outcomes = rng.random(sub_len) < p_step
                    cost += sub_len
                    attempt += 1
                    if outcomes.all():
                        sub_ok = True
                        break
                    elif rng.random() < detect_prob:
                        cost += rollback_cost
                        continue
                    else:
                        sub_ok = False
                        seg_failed = True
                        break
                if not sub_ok:
                    break
                step = sub_end
            if step < seg_len or seg_failed:
                ok = False
                break
            if i < m_agents - 1:
                if rng.random() < coord_fail_prob:
                    ok = False
                    break
        total_cost += cost
        if ok:
            successes += 1
    return successes / n_trials, total_cost / n_trials


def experiment_e():
    n_steps = 60
    p_step = 0.98
    q = 0.05
    m_values = [1, 2, 3, 4, 5, 6, 8]
    rows = []
    for m in m_values:
        succ_norecovery, cost_norecovery = simulate_multi_agent(n_steps, p_step, m, q)
        succ_recovery, cost_recovery = simulate_multi_agent_with_recovery(
            n_steps, p_step, m, q, checkpoint_interval=10, detect_prob=0.7, rollback_cost=3
        )
        rows.append(dict(m_agents=m, coord_fail_prob=q,
                          success_no_recovery=succ_norecovery, cost_no_recovery=cost_norecovery,
                          success_with_recovery=succ_recovery, cost_with_recovery=cost_recovery))
    return rows


def ci95(mean, std, n_seeds):
    """Approximate 95% CI half-width from across-seed mean/std (normal approx)."""
    return 1.96 * std / np.sqrt(n_seeds)


# ---------------------------------------------------------------------------
# Experiment F: wider-grid robustness check of the Experiment E finding.
# Experiment E showed, at ONE setting (n=60, p=0.98, q=0.05, checkpoint
# interval=10, detect=0.7), that the *relative* coordination penalty
# (m=1 vs m=8) is nearly unchanged whether or not within-segment recovery
# is used (~29% either way). Experiment F tests whether this
# "multiplicative independence" pattern holds across a grid of
# coordination-failure probabilities q and recovery detection qualities d,
# directly addressing the future-work item raised after Experiment E.
# ---------------------------------------------------------------------------
def experiment_f():
    n_steps = 60
    p_step = 0.98
    q_values = [0.02, 0.05, 0.10]
    detect_values = [0.5, 0.7, 0.9]
    m_lo, m_hi = 1, 8
    rows = []
    for q in q_values:
        for d in detect_values:
            succ_lo_nr, _ = simulate_multi_agent(n_steps, p_step, m_lo, q, n_trials=15000)
            succ_hi_nr, _ = simulate_multi_agent(n_steps, p_step, m_hi, q, n_trials=15000)
            succ_lo_r, _ = simulate_multi_agent_with_recovery(
                n_steps, p_step, m_lo, q, checkpoint_interval=10, detect_prob=d,
                rollback_cost=3, n_trials=15000)
            succ_hi_r, _ = simulate_multi_agent_with_recovery(
                n_steps, p_step, m_hi, q, checkpoint_interval=10, detect_prob=d,
                rollback_cost=3, n_trials=15000)
            rel_penalty_no_recovery = (succ_lo_nr - succ_hi_nr) / succ_lo_nr if succ_lo_nr > 0 else None
            rel_penalty_with_recovery = (succ_lo_r - succ_hi_r) / succ_lo_r if succ_lo_r > 0 else None
            rows.append(dict(
                q=q, detect_prob=d,
                success_m1_no_recovery=succ_lo_nr, success_m8_no_recovery=succ_hi_nr,
                success_m1_with_recovery=succ_lo_r, success_m8_with_recovery=succ_hi_r,
                relative_penalty_no_recovery=rel_penalty_no_recovery,
                relative_penalty_with_recovery=rel_penalty_with_recovery,
                penalty_difference=(rel_penalty_no_recovery - rel_penalty_with_recovery)
                    if (rel_penalty_no_recovery is not None and rel_penalty_with_recovery is not None) else None,
            ))
    return rows


if __name__ == "__main__":
    results = {}
    results["experiment_a"] = experiment_a()
    horizon50, crosscheck = experiment_b()
    results["experiment_b_horizon50"] = horizon50
    results["experiment_b_crosscheck"] = crosscheck
    results["experiment_c"] = experiment_c()
    results["experiment_d"] = experiment_d()
    print("Running replicated Experiment A (error bars)...")
    results["experiment_a_replicated"] = experiment_a_replicated()
    print("Running replicated Experiment C (error bars)...")
    results["experiment_c_replicated"] = experiment_c_replicated()
    print("Running Experiment E (recovery x multi-agent interaction)...")
    results["experiment_e"] = experiment_e()
    print("Running Experiment F (wider-grid robustness check of Experiment E)...")
    results["experiment_f"] = experiment_f()

    # attach 95% CIs to the replicated experiments for direct reporting
    for r in results["experiment_a_replicated"]:
        r["success_baseline_ci95"] = ci95(r["success_baseline_mean"], r["success_baseline_std"], 5)
        r["success_checkpoint_ci95"] = ci95(r["success_checkpoint_mean"], r["success_checkpoint_std"], 5)
    for r in results["experiment_c_replicated"]:
        r["success_ci95"] = ci95(r["success_mean"], r["success_std"], 5)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Wrote results.json")
    print("Cross-check (analytic vs simulated p^n):", crosscheck)
