-- Mixed read-only transaction for pgbench (-f), with work multiplier.
--
-- Pass scale via:  pgbench -D scale=N -f this_file ...
-- scale=1 matches the original 03_run_benchmark.sql work amounts.
-- Scale multiplies LIMIT/slice widths (≈ linear CPU on a cached working set).
--
-- Session GUCs (jit/work_mem/max_parallel_workers_per_gather) must be set
-- once on the database/role — not per transaction — so serial vs pipeline
-- comparisons are not dominated by extra SET round-trips.

\set aid random(1, 50000)
\set oid random(1, 250000)
\set pid random(1, 5000)
\set region_i random(0, 4)
-- :scale comes from pgbench -D scale=N (integer work multiplier).
\set recent_lim 40 * :scale
\set cohort_lim 800 * :scale
\set slice_width 12000 * :scale
\set topn_lim 25 * :scale

SELECT md5(string_agg(x, '|' ORDER BY x)) AS checksum
FROM (
    WITH params AS (
        SELECT
            :aid::int AS customer_id,
            :oid::int AS order_id,
            :pid::int AS product_id,
            (ARRAY['us-east', 'us-west', 'eu-west', 'asia', 'sa'])[1 + :region_i]
                AS region
    ),
    cust AS (
        SELECT c.*
        FROM ro_cpu_customer c
        JOIN params p ON c.customer_id = p.customer_id
    ),
    recent AS (
        SELECT o.order_id, o.status, o.ordered_at, o.meta, o.note, o.customer_id
        FROM ro_cpu_order o
        JOIN params p ON o.customer_id = p.customer_id
        WHERE o.status IN ('paid', 'shipped', 'done')
        ORDER BY o.ordered_at DESC
        LIMIT :recent_lim
    ),
    lines AS (
        SELECT
            r.order_id,
            r.status,
            i.product_id,
            i.qty,
            i.line_total,
            pr.category,
            pr.attrs,
            pr.unit_price
        FROM recent r
        JOIN ro_cpu_order_item i ON i.order_id = r.order_id
        JOIN ro_cpu_product pr ON pr.product_id = i.product_id
    ),
    ranked AS (
        SELECT
            l.*,
            sum(l.line_total) OVER (PARTITION BY l.order_id) AS order_sum,
            row_number() OVER (
                PARTITION BY l.category
                ORDER BY l.line_total DESC, l.product_id
            ) AS cat_rank
        FROM lines l
    ),
    q1 AS (
        SELECT
            'q1' AS tag,
            count(*)::text
                || ':' || coalesce(sum(line_total), 0)::text
                || ':' || coalesce(sum(order_sum) FILTER (WHERE cat_rank = 1), 0)::text
                || ':' || md5(string_agg(attrs ->> 'tier', ',' ORDER BY product_id))
                AS dig
        FROM ranked
    ),
    cohort AS (
        SELECT c.customer_id, c.profile, c.email
        FROM ro_cpu_customer c
        JOIN params p ON c.region = p.region
        WHERE (c.profile ->> 'plan') IN ('pro', 'ent')
        LIMIT :cohort_lim
    ),
    cohort_orders AS (
        SELECT o.order_id, o.status, o.customer_id, o.meta
        FROM ro_cpu_order o
        JOIN cohort c ON c.customer_id = o.customer_id
        WHERE o.status <> 'cancel'
          AND (o.meta ->> 'channel') IN ('web', 'api')
    ),
    cohort_agg AS (
        SELECT
            co.status,
            count(*) AS n_orders,
            sum(i.line_total) AS revenue,
            count(DISTINCT i.product_id) AS n_products
        FROM cohort_orders co
        JOIN ro_cpu_order_item i ON i.order_id = co.order_id
        GROUP BY co.status
    ),
    q2 AS (
        SELECT
            'q2' AS tag,
            string_agg(
                status || '=' || n_orders || '/' || revenue::text || '/' || n_products,
                ';' ORDER BY status
            ) AS dig
        FROM cohort_agg
    ),
    slice AS (
        SELECT o.order_id, o.note, o.meta, o.status, o.ordered_at
        FROM ro_cpu_order o
        JOIN params p ON true
        WHERE o.order_id BETWEEN p.order_id AND p.order_id + :slice_width
    ),
    scanned AS (
        SELECT
            count(*) AS n,
            count(*) FILTER (
                WHERE note ~ 'email=user[0-9]+@example\.com'
            ) AS n_email,
            count(*) FILTER (
                WHERE note ~* 'path=/var/log/orders/[0-9]+\.log'
            ) AS n_path,
            count(*) FILTER (
                WHERE (meta -> 'flags' ->> 'rush')::boolean
            ) AS n_rush,
            sum(length(md5(note || (meta ->> 'trace')))) AS hash_work,
            sum(((meta ->> 'trace') IS NOT NULL)::int) AS n_trace
        FROM slice
    ),
    q3 AS (
        SELECT
            'q3' AS tag,
            n::text || ':' || n_email || ':' || n_path || ':' || n_rush
                || ':' || hash_work || ':' || n_trace AS dig
        FROM scanned
    ),
    prod AS (
        SELECT pr.*
        FROM ro_cpu_product pr
        JOIN params p ON pr.product_id = p.product_id
    ),
    top_buyers AS (
        SELECT
            i.order_id,
            sum(i.qty) AS qty,
            sum(i.line_total) AS spent
        FROM ro_cpu_order_item i
        JOIN prod pr ON pr.product_id = i.product_id
        GROUP BY i.order_id
        ORDER BY spent DESC, i.order_id
        LIMIT :topn_lim
    ),
    q4 AS (
        SELECT
            'q4' AS tag,
            coalesce(sum(qty), 0)::text || ':' || coalesce(sum(spent), 0)::text
                || ':' || coalesce(string_agg(order_id::text, ',' ORDER BY spent DESC), '')
                AS dig
        FROM top_buyers
    )
    SELECT dig AS x FROM q1
    UNION ALL
    SELECT dig FROM q2
    UNION ALL
    SELECT dig FROM q3
    UNION ALL
    SELECT dig FROM q4
) s;
