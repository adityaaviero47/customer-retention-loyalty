"""Customer retention, loyalty and reactivation on a real online retailer (UCI Online Retail II, 2009-2011).
Cohort retention, the first-to-second order drop, RFM segments, the dormant base, a loyalty tier design,
a next-90-day purchase model for personalised offers, and a reactivation business case.
Reproduces every number in README.md:  python run_analysis.py"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

R = {}
o = pd.read_csv("data/orders.csv.gz", parse_dates=["date"]).sort_values("date")
END = o.date.max().normalize() + pd.Timedelta(days=1)       # day after the last order
o["n"] = o.groupby("customer_id").cumcount() + 1
c = o.groupby("customer_id").agg(first=("date", "min"), last=("date", "max"), orders=("invoice", "size"), revenue=("value", "sum"))
R["data"] = dict(orders=int(len(o)), customers=int(len(c)), revenue_gbp=float(o.value.sum()),
                 start=str(o.date.min().date()), end=str(o.date.max().date()), avg_order_value=float(o.value.mean()))

# ---- 1. cohort retention: share of each monthly first-purchase cohort buying again in month k ----
o["cohort"] = o.customer_id.map(c["first"]).dt.to_period("M")
o["k"] = (o.date.dt.to_period("M") - o.cohort).apply(lambda x: x.n)
size = o[o.k == 0].groupby("cohort").customer_id.nunique()
act = o.groupby(["cohort", "k"]).customer_id.nunique().unstack(fill_value=0)
ret = act.div(size, axis=0)
curve = {}
for k in [1, 2, 3, 6, 12]:
    ok = [p for p in ret.index if (pd.Period(END - pd.Timedelta(days=1), "M") - p).n > k]   # cohorts old enough to observe month k
    curve[k] = float((act.loc[ok, k].sum()) / size.loc[ok].sum())
R["cohort_retention"] = dict(month_k_active_share=curve, cohorts=int(len(size)))
ret.to_csv("results/cohort_retention.csv")

# ---- 2. the first-to-second order drop ----
second = o[o.n == 2].set_index("customer_id").date
c["gap2"] = (pd.Series(c.index.map(second), index=c.index) - c["first"]).dt.days
win = c[c["first"] <= END - pd.Timedelta(days=365)]          # had a full year to come back
R["second_order"] = dict(
    repeat_rate_all=float((c.orders > 1).mean()),
    repeat_within_90d=float((win.gap2 <= 90).mean()), repeat_within_365d=float((win.gap2 <= 365).mean()),
    never_returned_within_365d=float(1 - (win.gap2 <= 365).mean()), customers_with_full_year=int(len(win)),
    median_days_to_second=float(c.gap2.median()))
# once a customer has 2 orders, how likely is a 3rd? (same 1-year window, measured from the 2nd order)
third = o[o.n == 3].set_index("customer_id").date
two = c[c.index.isin(second.index)].copy(); two["second"] = pd.Series(two.index.map(second), index=two.index)
two = two[two.second <= END - pd.Timedelta(days=365)]
R["second_order"]["third_within_365d_given_two"] = float(((pd.Series(two.index.map(third), index=two.index) - two.second).dt.days <= 365).mean())

# ---- 3. revenue concentration and loyalty tiers (trailing 12 months) ----
L = o[o.date >= END - pd.Timedelta(days=365)].groupby("customer_id").value.sum().sort_values(ascending=False)
cum = L.cumsum() / L.sum()
R["concentration"] = dict(active_12m=int(len(L)), top10_share=float(cum.iloc[int(len(L) * .1) - 1]),
                          top20_share=float(cum.iloc[int(len(L) * .2) - 1]), bottom50_share=float(1 - cum.iloc[int(len(L) * .5) - 1]))
q = L.rank(pct=True, ascending=False)
tier = pd.cut(q, [0, .05, .20, .50, 1], labels=["Platinum (top 5%)", "Gold (next 15%)", "Silver (next 30%)", "Member (bottom 50%)"])
oc = o[o.date >= END - pd.Timedelta(days=365)].groupby("customer_id").invoice.size()
T = pd.DataFrame(dict(revenue=L, tier=tier, orders=oc.reindex(L.index))).groupby("tier", observed=True).agg(
    customers=("revenue", "size"), revenue=("revenue", "sum"), min_spend=("revenue", "min"), orders_per_customer=("orders", "mean"))
T["revenue_share"] = T.revenue / T.revenue.sum()
R["loyalty_tiers"] = T.round(4).reset_index().to_dict("records")

# ---- 4. RFM segments at the end of the data ----
c["recency"] = (END - c["last"]).dt.days
r = pd.qcut(c.recency, 5, labels=[5, 4, 3, 2, 1]).astype(int)
f = pd.qcut(c.orders.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
m = pd.qcut(c.revenue, 5, labels=[1, 2, 3, 4, 5]).astype(int)
fm = (f + m) / 2
def seg(ri, fmi):
    if ri >= 4 and fmi >= 4: return "Champions"
    if ri >= 3 and fmi >= 3: return "Loyal"
    if ri >= 4: return "New / promising"
    if ri <= 2 and fmi >= 3: return "At risk (high value)"
    if ri == 3: return "Needs attention"
    return "Hibernating"
c["segment"] = [seg(a, b) for a, b in zip(r, fm)]
S = c.groupby("segment").agg(customers=("orders", "size"), revenue=("revenue", "sum"), avg_orders=("orders", "mean"),
                             median_recency_days=("recency", "median"))
S["customer_share"] = S.customers / len(c); S["revenue_share"] = S.revenue / S.revenue.sum()
R["rfm"] = S.sort_values("revenue", ascending=False).round(4).reset_index().to_dict("records")
c[["recency", "orders", "revenue", "segment"]].to_csv("results/customer_segments.csv")

# ---- 5. dormant base: no order in the last 180 days ----
dorm = c[c.recency > 180]
multi = dorm[dorm.orders > 1]
R["dormant"] = dict(customers=int(len(dorm)), share=float(len(dorm) / len(c)), lifetime_revenue=float(dorm.revenue.sum()),
                    multi_order_customers=int(len(multi)), multi_order_lifetime_revenue=float(multi.revenue.sum()),
                    at_risk_high_value=int((c.segment == "At risk (high value)").sum()),
                    at_risk_high_value_revenue=float(c.revenue[c.segment == "At risk (high value)"].sum()))
# natural return rate: customers dormant 180+ days one year before the end; share who ordered again in the next 12 months
cut = END - pd.Timedelta(days=365)
past = o[o.date < cut].groupby("customer_id").agg(last=("date", "max"), orders=("invoice", "size"), revenue=("value", "sum"))
pd_ = past[(cut - past["last"]).dt.days > 180]
back = o[(o.date >= cut)].groupby("customer_id").value.sum()
R["dormant"]["natural_return_rate_12m"] = float(pd_.index.isin(back.index).mean())
R["dormant"]["natural_return_rate_12m_multi_order"] = float(pd_[pd_.orders > 1].index.isin(back.index).mean())
R["dormant"]["natural_return_rate_12m_one_order"] = float(pd_[pd_.orders == 1].index.isin(back.index).mean())
R["dormant"]["returners_spend_next_12m"] = float(back.reindex(pd_.index).dropna().mean())

# ---- 6. next-90-day purchase model (personalised offers): train on an earlier cut-off, test on a later one ----
def frame(cutoff):
    h = o[o.date < cutoff]; y = o[(o.date >= cutoff) & (o.date < cutoff + pd.Timedelta(days=90))].customer_id.unique()
    g = h.groupby("customer_id")
    X = pd.DataFrame(dict(
        recency=(cutoff - g.date.max()).dt.days, tenure=(cutoff - g.date.min()).dt.days, orders=g.invoice.size(),
        revenue=g.value.sum(), aov=g.value.mean(), items=g["items"].sum(),
        orders_90d=h[h.date >= cutoff - pd.Timedelta(days=90)].groupby("customer_id").invoice.size(),
        orders_365d=h[h.date >= cutoff - pd.Timedelta(days=365)].groupby("customer_id").invoice.size(),
        revenue_365d=h[h.date >= cutoff - pd.Timedelta(days=365)].groupby("customer_id").value.sum(),
        uk=g.country.first().eq("United Kingdom").astype(int))).fillna(0)
    gaps = h.groupby("customer_id").date.apply(lambda d: d.diff().dt.days.mean())
    X["avg_gap"] = gaps.reindex(X.index).fillna(-1)
    return X, X.index.isin(y).astype(int)
test_cut = END - pd.Timedelta(days=90); train_cut = test_cut - pd.Timedelta(days=90)
Xtr, ytr = frame(train_cut); Xte, yte = frame(test_cut)
mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=7).fit(Xtr, ytr)
p = mdl.predict_proba(Xte)[:, 1]
def cap(s, frac):
    k = int(len(s) * frac); idx = np.argsort(-s)[:k]; return float(yte[idx].sum() / yte.sum())
base = -Xte.recency.to_numpy().astype(float)                  # simple rule: most recent buyers first
R["propensity"] = dict(train_cutoff=str(train_cut.date()), test_cutoff=str(test_cut.date()), test_customers=int(len(yte)),
                       base_rate=float(yte.mean()), auc=float(roc_auc_score(yte, p)), top20_capture=cap(p, .2),
                       top20_buy_rate=float(yte[np.argsort(-p)[: int(len(p) * .2)]].mean()),
                       recency_rule_auc=float(roc_auc_score(yte, base)), recency_rule_top20_capture=cap(base, .2))

# ---- 7. reactivation business case (assumptions labelled) ----
# Target: dormant customers (180+ days) who ordered at least twice. Assumed campaign: an email/WhatsApp offer.
A = dict(incremental_reactivation_rate=0.05, discount=0.10, cost_per_contact_gbp=0.10)   # ASSUMPTIONS, to be tested with a holdout
spend = R["dormant"]["returners_spend_next_12m"]
N = len(multi); won = N * A["incremental_reactivation_rate"]
gross = won * spend; disc = won * R["data"]["avg_order_value"] * A["discount"]; cost = N * A["cost_per_contact_gbp"]
R["reactivation_case"] = dict(assumptions=A, target_customers=int(N), customers_won=float(won),
                              revenue_12m_gbp=float(gross), discount_cost_gbp=float(disc), contact_cost_gbp=float(cost),
                              revenue_less_campaign_cost_gbp=float(gross - disc - cost))
# holdout sizing: customers per arm to detect the assumed lift over the natural return rate (alpha 5%, power 80%)
p0 = R["dormant"]["natural_return_rate_12m_multi_order"] * 90 / 365     # rough 90-day natural return (12-month rate spread evenly)
p1 = p0 + A["incremental_reactivation_rate"]
n_arm = (1.96 + 0.84) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2
R["reactivation_case"]["holdout"] = dict(p0_90d=float(p0), p1_90d=float(p1), customers_per_arm=int(np.ceil(n_arm)))

json.dump(R, open("results/results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
