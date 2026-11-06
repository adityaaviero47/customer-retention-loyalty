-- Retention metrics in SQL (SQLite 3.25+): the same numbers run_analysis.py computes in pandas.
-- Table: orders(invoice, customer_id, country, date, value, items), one row per order.
-- run_sql.py loads data/orders.csv.gz into SQLite, runs each named query below and checks it against results/results.json.

-- name: second_order_within_365d
-- Of customers whose first order was at least 365 days before the end of the data,
-- the share who placed a 2nd order within 365 days of their first.
WITH ranked AS (
  SELECT customer_id, date,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY date, invoice) AS n
  FROM orders
),
firsts AS (
  SELECT r1.customer_id, r1.date AS first_date, r2.date AS second_date
  FROM ranked r1 LEFT JOIN ranked r2 ON r2.customer_id = r1.customer_id AND r2.n = 2
  WHERE r1.n = 1
),
ends AS (SELECT DATE(MAX(date), '+1 day') AS end_day FROM orders)
SELECT AVG(CASE WHEN second_date IS NOT NULL
                 AND CAST(julianday(second_date) - julianday(first_date) AS INTEGER) <= 365
            THEN 1.0 ELSE 0.0 END) AS repeat_within_365d,
       COUNT(*) AS customers_with_full_year
FROM firsts, ends
WHERE julianday(first_date) <= julianday(end_day) - 365;

-- name: third_order_given_two
-- Of customers whose 2nd order was at least 365 days before the end, the share with a 3rd order within 365 days of the 2nd.
WITH ranked AS (
  SELECT customer_id, date,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY date, invoice) AS n
  FROM orders
),
seconds AS (
  SELECT r2.customer_id, r2.date AS second_date, r3.date AS third_date
  FROM ranked r2 LEFT JOIN ranked r3 ON r3.customer_id = r2.customer_id AND r3.n = 3
  WHERE r2.n = 2
),
ends AS (SELECT DATE(MAX(date), '+1 day') AS end_day FROM orders)
SELECT AVG(CASE WHEN third_date IS NOT NULL
                 AND CAST(julianday(third_date) - julianday(second_date) AS INTEGER) <= 365
            THEN 1.0 ELSE 0.0 END) AS third_within_365d_given_two
FROM seconds, ends
WHERE julianday(second_date) <= julianday(end_day) - 365;

-- name: tier_revenue_share
-- Customers active in the last 365 days ranked by 12-month spend; share of revenue by loyalty tier.
WITH ends AS (SELECT DATE(MAX(date), '+1 day') AS end_day FROM orders),
spend AS (
  SELECT customer_id, SUM(value) AS revenue
  FROM orders, ends
  WHERE julianday(date) >= julianday(end_day) - 365
  GROUP BY customer_id
),
ranked AS (
  SELECT customer_id, revenue,
         ROW_NUMBER() OVER (ORDER BY revenue DESC) * 1.0 / COUNT(*) OVER () AS top_share
  FROM spend
)
SELECT CASE WHEN top_share <= 0.05 THEN 'Platinum (top 5%)'
            WHEN top_share <= 0.20 THEN 'Gold (next 15%)'
            WHEN top_share <= 0.50 THEN 'Silver (next 30%)'
            ELSE 'Member (bottom 50%)' END AS tier,
       COUNT(*) AS customers,
       SUM(revenue) / (SELECT SUM(revenue) FROM spend) AS revenue_share
FROM ranked
GROUP BY tier
ORDER BY revenue_share DESC;

-- name: dormant_base
-- Customers with no order in the last 180 days (whole days since the last order > 180), and how many had ordered more than once.
WITH ends AS (SELECT DATE(MAX(date), '+1 day') AS end_day FROM orders),
c AS (
  SELECT customer_id, MAX(date) AS last_date, COUNT(*) AS orders
  FROM orders GROUP BY customer_id
)
SELECT SUM(CASE WHEN CAST(julianday(end_day) - julianday(last_date) AS INTEGER) > 180 THEN 1 ELSE 0 END) AS dormant_customers,
       SUM(CASE WHEN CAST(julianday(end_day) - julianday(last_date) AS INTEGER) > 180 AND orders > 1 THEN 1 ELSE 0 END) AS dormant_repeat_buyers
FROM c, ends;
