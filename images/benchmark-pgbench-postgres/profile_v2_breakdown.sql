-- Dev-only calibration helper for ro_cpu_txn.sql (not shipped in the Docker
-- image; run manually against a freshly loaded ro_cpu_* schema to re-tune
-- block widths after changing the schema or query).
--
-- Usage:
--   psql ... -f ro_cpu_setup.sql
--   psql ... -f profile_v2_breakdown.sql
--
-- Prints one EXPLAIN per non-btree access path (confirms Hash Join / Merge
-- Join / GIN / BRIN are actually chosen, not just hoped for) followed by
-- per-block timed-loop averages (40 iters each) so block weights in
-- ro_cpu_txn.sql can be re-balanced if any one of them starts to dominate.
--
-- plan_cache_mode = force_custom_plan is required below: the timed loops use
-- PL/pgSQL variables (not literals) for LIMIT/width params, and plpgsql
-- switches to a genericized cached plan after 5 calls -- which mis-estimates
-- a variable LIMIT badly enough to turn q_array's ~8ms bounded join back
-- into the same "seq-scan the full 750k-row order_item table" plan q_array
-- was specifically redesigned to avoid (271ms measured without this GUC).
-- This is purely an artifact of this PL/pgSQL harness: real pgbench (simple
-- query protocol) substitutes :variables as literal text before planning
-- every time, so ro_cpu_txn.sql itself is not affected (confirmed via a
-- direct `pgbench -f ro_cpu_txn.sql` run: 84ms avg latency, 0 failures).

SET jit = off;
SET work_mem = '64MB';
SET max_parallel_workers_per_gather = 0;
SET plan_cache_mode = force_custom_plan;

\set aid 12346
\set oid 100000
\set pid 2500
\set hj_start_sec 50000

-- Current ro_cpu_txn.sql weights (scale=1) -- keep in sync when re-tuning.
\set regex_width 3600
\set hj_window_sec 1500
\set fts_lim 40
\set array_slice_width 48000
\set array_lim 200
\set stats_width 8000
\set toast_n 700

\echo === q_hashjoin plan ===
EXPLAIN (COSTS OFF)
WITH fact_slice AS (
    SELECT o.order_id
    FROM ro_cpu_order o
    WHERE o.ordered_at >= timestamptz '2022-01-01 00:00:00+00' + make_interval(secs => :hj_start_sec)
      AND o.ordered_at < timestamptz '2022-01-01 00:00:00+00' + make_interval(secs => :hj_start_sec + :hj_window_sec)
),
fact_items AS (
    SELECT fs.order_id, i.product_id, i.qty, i.line_total
    FROM fact_slice fs JOIN ro_cpu_order_item i ON i.order_id = fs.order_id
)
SELECT pr.category, pr.attrs ->> 'tier' AS tier, count(*), sum(fi.qty), sum(fi.line_total)
FROM fact_items fi JOIN ro_cpu_product pr ON pr.product_id = fi.product_id
GROUP BY pr.category, pr.attrs ->> 'tier';

\echo === q_fts plan (want Bitmap/GIN index scan) ===
EXPLAIN (COSTS OFF)
SELECT o.order_id, o.status, ts_rank(o.search_doc, to_tsquery('simple', 'urgent & priority')) AS rank
FROM ro_cpu_order o
WHERE o.search_doc @@ to_tsquery('simple', 'urgent & priority')
ORDER BY rank DESC
LIMIT :fts_lim;

\echo === q_array plan (want GIN index scan on tags, bounded join) ===
EXPLAIN (COSTS OFF)
WITH arr AS (
    SELECT pr.product_id, pr.category FROM ro_cpu_product pr WHERE pr.tags @> ARRAY['urgent', 'north']
    LIMIT :array_lim
),
arr_slice AS (
    SELECT i.product_id, i.qty, i.line_total FROM ro_cpu_order_item i
    WHERE i.order_id BETWEEN :oid AND :oid + :array_slice_width
)
SELECT a.category, count(*), sum(s.qty), sum(s.line_total)
FROM arr a JOIN arr_slice s ON s.product_id = a.product_id
GROUP BY a.category;

\echo === q_toast plan ===
EXPLAIN (COSTS OFF)
SELECT pr.product_id, length(pr.spec_blob), md5(pr.spec_blob)
FROM ro_cpu_product pr
WHERE pr.product_id BETWEEN 1 AND :toast_n;

\echo === timed loops (40 iters, wall clock) ===
DO $$
DECLARE
  i int; t0 timestamptz; dummy text; niter int := 40; ms numeric;
  v_aid int := 12346; v_oid int := 100000; v_pid int := 2500;
  v_tag1 text := 'urgent'; v_tag2 text := 'north';
  v_fts1 text := 'urgent'; v_fts2 text := 'priority';
  v_hj_start int := 50000; v_hj_window int := 1500;
  v_regex_width int := 3600; v_fts_lim int := 40;
  v_array_slice_width int := 48000; v_array_lim int := 200;
  v_stats_width int := 8000; v_toast_n int := 700;
BEGIN
  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT count(*)::text || ':' || coalesce(sum(line_total),0)::text
        || ':' || coalesce(sum(order_sum) FILTER (WHERE cat_rank=1),0)::text
        || ':' || md5(string_agg(attrs->>'tier', ',' ORDER BY product_id))
    INTO dummy
    FROM (
        WITH recent AS (
            SELECT o.order_id FROM ro_cpu_order o
            WHERE o.customer_id = v_aid AND o.status IN ('paid','shipped','done')
        ),
        lines AS (
            SELECT r.order_id, i.product_id, i.line_total, pr.category, pr.attrs
            FROM recent r
            JOIN ro_cpu_order_item i ON i.order_id = r.order_id
            JOIN ro_cpu_product pr ON pr.product_id = i.product_id
        )
        SELECT l.*, sum(l.line_total) OVER (PARTITION BY l.order_id) AS order_sum,
            row_number() OVER (PARTITION BY l.category ORDER BY l.line_total DESC, l.product_id) AS cat_rank
        FROM lines l
    ) ranked;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_idx avg_ms=%', ms;

  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT string_agg(category||'/'||tier||'='||n_lines||':'||qty||':'||revenue::text, ';' ORDER BY category, tier)
    INTO dummy
    FROM (
        WITH fact_slice AS (
            SELECT o.order_id FROM ro_cpu_order o
            WHERE o.ordered_at >= timestamptz '2022-01-01 00:00:00+00' + make_interval(secs => v_hj_start)
              AND o.ordered_at < timestamptz '2022-01-01 00:00:00+00' + make_interval(secs => v_hj_start + v_hj_window)
        ),
        fact_items AS (
            SELECT fs.order_id, i.product_id, i.qty, i.line_total
            FROM fact_slice fs JOIN ro_cpu_order_item i ON i.order_id = fs.order_id
        )
        SELECT pr.category, pr.attrs->>'tier' AS tier, count(*) AS n_lines, sum(fi.qty) AS qty, sum(fi.line_total) AS revenue
        FROM fact_items fi JOIN ro_cpu_product pr ON pr.product_id = fi.product_id
        GROUP BY pr.category, pr.attrs->>'tier'
    ) hj;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_hashjoin avg_ms=%', ms;

  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT count(*) + count(*) FILTER (WHERE note ~ 'email=user[0-9]+@example\.com')
         + count(*) FILTER (WHERE note ~* 'path=/var/log/orders/[0-9]+\.log')
         + count(*) FILTER (WHERE (meta->'flags'->>'rush')::boolean)
         + sum(length(md5(note || (meta->>'trace'))))
    INTO dummy
    FROM ro_cpu_order WHERE order_id BETWEEN v_oid AND v_oid + v_regex_width;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_regex avg_ms=%', ms;

  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT count(*)::text || ':' || coalesce(sum(rank),0)::text
    INTO dummy
    FROM (
        SELECT ts_rank(o.search_doc, to_tsquery('simple', v_fts1 || ' & ' || v_fts2)) AS rank
        FROM ro_cpu_order o
        WHERE o.search_doc @@ to_tsquery('simple', v_fts1 || ' & ' || v_fts2)
        ORDER BY rank DESC
        LIMIT v_fts_lim
    ) fts;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_fts avg_ms=%', ms;

  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT coalesce(string_agg(category||'='||n_rows||':'||qty||':'||revenue::text, ';' ORDER BY category), '')
    INTO dummy
    FROM (
        WITH arr AS (
            SELECT pr.product_id, pr.category FROM ro_cpu_product pr WHERE pr.tags @> ARRAY[v_tag1, v_tag2]
            LIMIT v_array_lim
        ),
        arr_slice AS (
            SELECT i.product_id, i.qty, i.line_total FROM ro_cpu_order_item i
            WHERE i.order_id BETWEEN v_oid AND v_oid + v_array_slice_width
        )
        SELECT a.category, count(*) AS n_rows, sum(s.qty) AS qty, sum(s.line_total) AS revenue
        FROM arr a JOIN arr_slice s ON s.product_id = a.product_id
        GROUP BY a.category
    ) aj;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_array avg_ms=%', ms;

  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT coalesce(string_agg(category||'='||n||':'||round(coalesce(median_total,0)::numeric,2)||':'||round(coalesce(sd_total,0)::numeric,2)||':'||round(coalesce(qty_total_corr,0)::numeric,4), ';' ORDER BY category), '')
    INTO dummy
    FROM (
        SELECT pr.category, count(*) AS n,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY i.line_total) AS median_total,
            stddev_samp(i.line_total) AS sd_total,
            corr(i.qty::float8, i.line_total::float8) AS qty_total_corr
        FROM ro_cpu_order_item i JOIN ro_cpu_product pr ON pr.product_id = i.product_id
        WHERE i.order_id BETWEEN v_oid AND v_oid + v_stats_width
        GROUP BY pr.category
    ) sa;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_stats avg_ms=%', ms;

  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT count(*)::text || ':' || sum(blob_len)::text
    INTO dummy
    FROM (
        SELECT length(pr.spec_blob) AS blob_len FROM ro_cpu_product pr WHERE pr.product_id BETWEEN 1 AND v_toast_n
    ) tp;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_toast avg_ms=%', ms;

  t0 := clock_timestamp();
  FOR i IN 1..niter LOOP
    SELECT string_agg(category||'='||n||':'||round(avg_price,4)||':'||round(sum_price,4), ';' ORDER BY category)
    INTO dummy
    FROM (
        SELECT category, count(*) AS n, avg(unit_price) AS avg_price, sum(unit_price) AS sum_price
        FROM ro_cpu_product GROUP BY category
    ) sq;
  END LOOP;
  ms := round((1000.0*extract(epoch FROM (clock_timestamp()-t0))/niter)::numeric,3);
  RAISE NOTICE 'q_seqscan avg_ms=%', ms;
END $$;
