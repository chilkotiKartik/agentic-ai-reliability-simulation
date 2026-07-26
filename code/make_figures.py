import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

with open("results.json") as f:
    R = json.load(f)

# ---- Figure 1: success rate vs horizon (compounding error) ----
a = R["experiment_a"]
ns = [r["n"] for r in a]
sb = [r["success_baseline"] for r in a]
sc = [r["success_checkpoint"] for r in a]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(ns, sb, "o-", label="No recovery (baseline)", color="#c0392b")
ax.plot(ns, sc, "s-", label="Checkpoint + rollback\n(interval=10, detect=0.7)", color="#2471a3")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% reliability")
ax.set_xlabel("Task horizon (number of sequential steps, n)")
ax.set_ylabel("Task success probability")
ax.set_title("Figure 1: Compounding Error vs. Task Horizon (p=0.97/step)")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig1_horizon_success.png")
plt.close(fig)

# ---- Figure 2: cost vs horizon ----
cb = [r["cost_baseline"] for r in a]
cc = [r["cost_checkpoint"] for r in a]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(ns, cb, "o-", label="No recovery (baseline)", color="#c0392b")
ax.plot(ns, cc, "s-", label="Checkpoint + rollback", color="#2471a3")
ax.set_xlabel("Task horizon (number of sequential steps, n)")
ax.set_ylabel("Expected cost (step-executions consumed)")
ax.set_title("Figure 2: Expected Execution Cost vs. Task Horizon")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig2_horizon_cost.png")
plt.close(fig)

# ---- Figure 3: 50%-reliability horizon vs per-step reliability p ----
b = R["experiment_b_horizon50"]
ps = [r["p_step"] for r in b]
h50 = [r["horizon_50pct"] for r in b]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(ps, h50, "o-", color="#117a65")
for x, y in zip(ps, h50):
    ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
ax.set_xlabel("Per-step reliability (p)")
ax.set_ylabel("50%-reliability horizon (steps)")
ax.set_title("Figure 3: Sensitivity of the 50%-Reliability Horizon\nto Per-Step Reliability")
ax.invert_xaxis()
fig.tight_layout()
fig.savefig("fig3_reliability_sensitivity.png")
plt.close(fig)

# ---- Figure 4: multi-agent coordination overhead ----
c = R["experiment_c"]
qs = sorted(set(r["coord_fail_prob"] for r in c))
fig, ax = plt.subplots(figsize=(6, 4.2))
colors = ["#1a5276", "#117864", "#b9770e", "#943126"]
for q, col in zip(qs, colors):
    sub = [r for r in c if r["coord_fail_prob"] == q]
    sub.sort(key=lambda r: r["m_agents"])
    ax.plot([r["m_agents"] for r in sub], [r["success"] for r in sub],
            "o-", label=f"q={q}", color=col)
ax.set_xlabel("Number of agents (m), fixed total steps n=60")
ax.set_ylabel("Task success probability")
ax.set_title("Figure 4: Multi-Agent Coordination Overhead\n(per-handoff coordination-failure probability q)")
ax.legend(title="Coord. fail. prob.", fontsize=8)
fig.tight_layout()
fig.savefig("fig4_multiagent_coordination.png")
plt.close(fig)

# ---- Figure 5: reliability-cost Pareto frontier ----
d = R["experiment_d"]
fig, ax = plt.subplots(figsize=(6.2, 4.5))
baseline = [r for r in d if r["strategy"] == "baseline"][0]
ax.scatter([baseline["cost"]], [baseline["success"]], marker="*", s=220,
           color="#c0392b", zorder=5, label="Baseline (no recovery)")
ckpt = [r for r in d if r["strategy"] == "checkpoint_rollback"]
dp_vals = sorted(set(r["detect_prob"] for r in ckpt))
markers = ["o", "s", "^", "D"]
colors = ["#aed6f1", "#5dade2", "#2471a3", "#154360"]
for dp, mk, col in zip(dp_vals, markers, colors):
    sub = [r for r in ckpt if r["detect_prob"] == dp]
    sub.sort(key=lambda r: r["checkpoint_interval"])
    ax.plot([r["cost"] for r in sub], [r["success"] for r in sub], mk + "-",
             color=col, label=f"detect_prob={dp}", markersize=6)
ax.set_xlabel("Expected cost (step-executions consumed)")
ax.set_ylabel("Task success probability")
ax.set_title("Figure 5: Reliability-Cost Pareto Frontier\n(n=150 steps, p=0.97/step, varying checkpoint interval)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("fig5_pareto_frontier.png")
plt.close(fig)

# ---- Figure 1b: success rate vs horizon WITH error bars (replicated, 5 seeds) ----
a2 = R["experiment_a_replicated"]
ns2 = [r["n"] for r in a2]
sbm = [r["success_baseline_mean"] for r in a2]
sbs = [r["success_baseline_std"] for r in a2]
scm = [r["success_checkpoint_mean"] for r in a2]
scs = [r["success_checkpoint_std"] for r in a2]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.errorbar(ns2, sbm, yerr=sbs, fmt="o-", capsize=3, label="No recovery (baseline)", color="#c0392b")
ax.errorbar(ns2, scm, yerr=scs, fmt="s-", capsize=3, label="Checkpoint + rollback", color="#2471a3")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% reliability")
ax.set_xlabel("Task horizon (number of sequential steps, n)")
ax.set_ylabel("Task success probability (mean +/- 1 s.d., 5 seeds)")
ax.set_title("Figure 1b: Compounding Error vs. Horizon\nwith Across-Seed Variability (p=0.97/step)")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig1b_horizon_success_errorbars.png")
plt.close(fig)

# ---- Figure 4b: multi-agent coordination overhead WITH error bars ----
c2 = R["experiment_c_replicated"]
qs2 = sorted(set(r["coord_fail_prob"] for r in c2))
fig, ax = plt.subplots(figsize=(6, 4.2))
colors2 = ["#1a5276", "#b9770e", "#943126"]
for q, col in zip(qs2, colors2):
    sub = [r for r in c2 if r["coord_fail_prob"] == q]
    sub.sort(key=lambda r: r["m_agents"])
    ax.errorbar([r["m_agents"] for r in sub], [r["success_mean"] for r in sub],
                yerr=[r["success_std"] for r in sub], fmt="o-", capsize=3, label=f"q={q}", color=col)
ax.set_xlabel("Number of agents (m), fixed total steps n=60")
ax.set_ylabel("Task success probability (mean +/- 1 s.d., 5 seeds)")
ax.set_title("Figure 4b: Multi-Agent Coordination Overhead\nwith Across-Seed Variability")
ax.legend(title="Coord. fail. prob.", fontsize=8)
fig.tight_layout()
fig.savefig("fig4b_multiagent_errorbars.png")
plt.close(fig)

# ---- Figure 6: Experiment E -- recovery x multi-agent interaction ----
e = R["experiment_e"]
ms = [r["m_agents"] for r in e]
snr = [r["success_no_recovery"] for r in e]
swr = [r["success_with_recovery"] for r in e]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(ms, snr, "o-", label="No within-segment recovery", color="#c0392b")
ax.plot(ms, swr, "s-", label="With within-segment recovery\n(interval=10, detect=0.7)", color="#1e8449")
ax.set_xlabel("Number of agents (m), fixed total steps n=60, q=0.05")
ax.set_ylabel("Task success probability")
ax.set_title("Figure 6: Does Per-Segment Recovery Rescue\nMulti-Agent Coordination Losses?")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig6_recovery_multiagent_interaction.png")
plt.close(fig)

# ---- Figure 9: Experiment F -- relative coordination penalty across grid ----
f = R["experiment_f"]
qs3 = sorted(set(r["q"] for r in f))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
colors3 = ["#1a5276", "#117864", "#b9770e"]
for q, col in zip(qs3, colors3):
    sub = [r for r in f if r["q"] == q]
    sub.sort(key=lambda r: r["detect_prob"])
    axes[0].plot([r["detect_prob"] for r in sub], [r["relative_penalty_no_recovery"] for r in sub],
                 "o--", color=col, alpha=0.5, label=f"q={q} (no recovery)")
    axes[0].plot([r["detect_prob"] for r in sub], [r["relative_penalty_with_recovery"] for r in sub],
                 "s-", color=col, label=f"q={q} (with recovery)")
axes[0].set_xlabel("Recovery detection probability (d)")
axes[0].set_ylabel("Relative coordination penalty\n(m=1 vs m=8, fractional loss)")
axes[0].set_title("(a) Penalty vs. detection quality, by q")
axes[0].legend(fontsize=6.5, ncol=1)

for q, col in zip(qs3, colors3):
    sub = [r for r in f if r["q"] == q]
    sub.sort(key=lambda r: r["detect_prob"])
    diffs = [r["penalty_difference"] for r in sub]
    axes[1].plot([r["detect_prob"] for r in sub], diffs, "D-", color=col, label=f"q={q}")
axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
axes[1].set_xlabel("Recovery detection probability (d)")
axes[1].set_ylabel("Penalty difference\n(no-recovery minus with-recovery)")
axes[1].set_title("(b) Does recovery change the penalty?")
axes[1].legend(fontsize=7)
fig.suptitle("Figure 9: Robustness of the Multiplicative-Independence Finding (Experiment F)", fontsize=11)
fig.tight_layout()
fig.savefig("fig9_experiment_f_grid.png")
plt.close(fig)

print("Figures written.")
