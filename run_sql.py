"""Run sql/retention_metrics.sql on the order data in SQLite and check each result against results/results.json.
python run_sql.py"""
import json, re, sqlite3, pandas as pd

o = pd.read_csv("data/orders.csv.gz", parse_dates=["date"])
o["date"] = o.date.dt.strftime("%Y-%m-%d %H:%M:%S")
con = sqlite3.connect(":memory:")
o.to_sql("orders", con, index=False)
queries = re.findall(r"-- name: (\w+)\n(.*?)(?=\n-- name: |\Z)", open("sql/retention_metrics.sql").read(), re.S)
R = json.load(open("results/results.json"))
out = {}
for name, q in queries:
    df = pd.read_sql_query(q, con)
    out[name] = df.to_dict("records")
    print(f"\n== {name}\n{df.to_string(index=False)}")

checks = {
    "second_order_within_365d": (out["second_order_within_365d"][0]["repeat_within_365d"], R["second_order"]["repeat_within_365d"]),
    "third_order_given_two": (out["third_order_given_two"][0]["third_within_365d_given_two"], R["second_order"]["third_within_365d_given_two"]),
    "platinum_revenue_share": ([t for t in out["tier_revenue_share"] if t["tier"].startswith("Platinum")][0]["revenue_share"],
                               [t for t in R["loyalty_tiers"] if t["tier"].startswith("Platinum")][0]["revenue_share"]),
    "dormant_customers": (out["dormant_base"][0]["dormant_customers"], R["dormant"]["customers"]),
    "dormant_repeat_buyers": (out["dormant_base"][0]["dormant_repeat_buyers"], R["dormant"]["multi_order_customers"]),
}
print("\nSQL vs Python:")
ok = True
for k, (sql, py) in checks.items():
    match = abs(float(sql) - float(py)) < 1e-3
    ok &= match
    print(f"  {k:28s} SQL {float(sql):.4f}   Python {float(py):.4f}   {'match' if match else 'DIFFERENT'}")
json.dump(out, open("results/sql_results.json", "w"), indent=2, default=str)
print("\nAll SQL results match the Python analysis." if ok else "\nSome SQL results differ from the Python analysis.")
