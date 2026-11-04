"""Download Online Retail II and write the order-level file the analysis reads (data/orders.csv.gz).
Only needed if data/ is empty:  python get_data.py"""
import os, pandas as pd

URL = "https://raw.githubusercontent.com/omidreza-amrollahi/online_retail_data/main/online_retail_II.csv"  # public mirror of UCI dataset 502

def build(src=URL):
    d = pd.read_csv(src, encoding="latin1", dtype={"Invoice": str, "StockCode": str})
    d = d.dropna(subset=["Customer ID"])
    d = d[~d.Invoice.str.startswith("C") & (d.Quantity > 0) & (d.Price > 0)]   # drop cancellations, returns and zero-price lines
    d["value"] = d.Quantity * d.Price
    o = d.groupby(["Invoice", "Customer ID", "Country"], as_index=False).agg(date=("InvoiceDate", "min"), value=("value", "sum"), items=("Quantity", "sum"))
    o["Customer ID"] = o["Customer ID"].astype(int)
    o = o.rename(columns={"Customer ID": "customer_id", "Invoice": "invoice", "Country": "country"})
    os.makedirs("data", exist_ok=True)
    o.to_csv("data/orders.csv.gz", index=False, compression="gzip")
    print(f"data/orders.csv.gz  {len(o):,} orders")

if __name__ == "__main__":
    build()
