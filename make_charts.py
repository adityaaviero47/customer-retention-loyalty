"""Charts for README from results/results.json and results/cohort_retention.csv."""
import json, matplotlib, pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
s = R["second_order"]
vals = [(1 - s["repeat_within_365d"]) * 100, (1 - s["third_within_365d_given_two"]) * 100]
labs = ["After the 1st order", "After the 2nd order"]
ax[0, 0].bar(labs, vals, color=["#E07A2F", "#0052CC"], width=.55)
for i, v in enumerate(vals): ax[0, 0].text(i, v + .8, f"{v:.0f}%", ha="center")
ax[0, 0].set_ylabel("No further order within a year (%)"); ax[0, 0].set_ylim(0, 32)
ax[0, 0].set_title("Loyalty journey: 1 in 4 never place a 2nd order")

t = R["loyalty_tiers"]
x = range(len(t))
ax[0, 1].bar([i - .2 for i in x], [r["customers"] / sum(z["customers"] for z in t) * 100 for r in t], .4, color="#9AA5AD", label="Share of customers")
ax[0, 1].bar([i + .2 for i in x], [r["revenue_share"] * 100 for r in t], .4, color="#0052CC", label="Share of revenue")
ax[0, 1].set_xticks(list(x)); ax[0, 1].set_xticklabels([r["tier"].replace(" (", "\n(") for r in t], fontsize=8)
ax[0, 1].set_ylabel("%"); ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("Loyalty tiers: top 5% bring 50% of 12-month revenue")

g = sorted(R["rfm"], key=lambda r: -r["revenue_share"])
ax[1, 0].barh([r["segment"] for r in g][::-1], [r["revenue_share"] * 100 for r in g][::-1],
              color=["#E07A2F" if r["segment"] == "At risk (high value)" else "#0052CC" for r in g][::-1])
for i, r in enumerate(g[::-1]): ax[1, 0].text(r["revenue_share"] * 100 + 1, i, f"{r['customers']:,} customers", va="center", fontsize=8)
ax[1, 0].set_xlabel("Share of lifetime revenue (%)"); ax[1, 0].set_xlim(0, 90); ax[1, 0].set_title("RFM segments: 680 high-value customers have gone quiet")

d = R["dormant"]
ax[1, 1].bar(["Dormant,\nordered once", "Dormant,\nordered 2+ times"], [d["natural_return_rate_12m_one_order"] * 100, d["natural_return_rate_12m_multi_order"] * 100], color=["#9AA5AD", "#0052CC"])
for i, v in enumerate([d["natural_return_rate_12m_one_order"] * 100, d["natural_return_rate_12m_multi_order"] * 100]): ax[1, 1].text(i, v + 1, f"{v:.0f}%", ha="center")
ax[1, 1].set_ylabel("Came back within 12 months (%)"); ax[1, 1].set_ylim(0, 60)
ax[1, 1].set_title("Reactivation: repeat buyers are the ones who return")
for a in ax.flat: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/summary.png", dpi=150)
