-- Mixed read-only transaction for pgbench (-f), with work multiplier.
--
-- Pass scale via:  pgbench -D scale=N -f this_file ...
-- Scale multiplies LIMIT/slice widths (~linear CPU on a cached working set).
--
-- Session GUCs (jit/work_mem/max_parallel_workers_per_gather) must be set
-- once on the database/role -- not per transaction -- so serial vs pipeline
-- comparisons are not dominated by extra SET round-trips.
--
-- Design goal: spread CPU across distinct Postgres subsystems so no single
-- one dominates (earlier version spent ~70% of its time in regex alone; the
-- whole script never touched a non-btree index AM, a hash/merge join, full
-- text search, arrays, ordered-set aggregates, or TOAST). Each block below
-- targets code the old script skipped entirely; weights were calibrated
-- empirically in a local Docker Postgres 18 (see profile_v2_breakdown.sql)
-- so no block exceeds ~30% of total transaction time at scale=1:
--   q_idx ~0.1ms  q_hashjoin ~24ms  q_regex ~15ms  q_fts ~12ms
--   q_array ~8ms  q_stats ~13ms  q_toast ~4ms  q_seqscan ~2ms  (total ~78ms)
--
--   q_idx       btree index scan, nested loop, window agg      (nodeIndexscan.c, nodeWindowAgg.c)
--   q_hashjoin  time-windowed order slice -> hash join against
--               the product dimension -> hash aggregate.       (nodeHash.c, nodeHashjoin.c)
--               Verified via EXPLAIN: at the calibrated window width the
--               planner also throws in a Merge Join (nodeMergejoin.c) for
--               the outer product join; the time-range scan itself picks
--               either the BRIN index or a btree skip-scan over
--               (status, ordered_at) depending on window width -- not
--               force-pinned, since a single monolithic statement can't
--               apply per-block planner GUCs without affecting every block.
--   q_regex     regex + md5 over a small order slice            (utils/adt/regexp.c) -- shrunk, no longer dominant
--   q_fts       full text search via tsvector/GIN/ts_rank        (utils/adt/tsvector_op.c, access/gin)
--   q_array     array containment via GIN, bounded join          (utils/adt/arrayfuncs.c, access/gin)
--   q_stats     ordered-set + statistical aggregates              (percentile_cont/stddev/corr in nodeAgg.c, numeric.c)
--   q_toast     out-of-line TOAST fetch + decompression           (access/heap/heaptoast.c)
--   q_seqscan   plain seq scan + aggregate, no predicate/index     (nodeSeqscan.c)

\set aid random(1, 50000)
\set oid random(1, 250000)
\set pid random(1, 5000)
\set region_i random(0, 4)
\set tag1_i random(0, 9)
\set tag2_i random(0, 4)
\set fts1_i random(0, 9)
\set fts2_i random(0, 2)

-- :scale comes from pgbench -D scale=N (integer work multiplier). Weights
-- below were calibrated on a local Docker Postgres 18 so no block exceeds
-- ~30% of total transaction time at scale=1 (see profile_v2_breakdown.sql):
--   q_idx ~0.1ms  q_hashjoin ~24ms  q_regex ~15ms  q_fts ~12ms
--   q_array ~8ms  q_stats ~13ms  q_toast ~4ms  q_seqscan ~2ms  (total ~78ms)
\set regex_width 3600 * :scale
\set hj_window_sec 1500 * :scale
\set hj_start_sec random(0, 250000 - :hj_window_sec)
-- Precompute sums here (pgbench arithmetic). Under -M prepared, writing
-- `:a + :b` in SQL becomes `$N + $M` with unknown types and fails with
-- "operator is not unique: unknown + unknown".
\set hj_end_sec :hj_start_sec + :hj_window_sec
\set fts_lim 40 * :scale
\set array_slice_width 48000 * :scale
\set array_slice_end :oid + :array_slice_width
\set array_lim 200
\set stats_width 8000 * :scale
\set stats_end :oid + :stats_width
\set toast_n 700 * :scale
\set regex_end :oid + :regex_width

SELECT md5(string_agg(x, '|' ORDER BY x)) AS checksum
FROM (
    WITH params AS (
        SELECT
            :aid::int AS customer_id,
            :oid::int AS order_id,
            :pid::int AS product_id,
            (ARRAY['us-east', 'us-west', 'eu-west', 'asia', 'sa'])[1 + :region_i]
                AS region,
            (ARRAY['urgent', 'fragile', 'giftwrap', 'backorder', 'expedite',
                   'digital', 'bulk', 'sample', 'recurring', 'warranty'])[1 + :tag1_i]
                AS tag1,
            (ARRAY['north', 'south', 'east', 'west', 'central'])[1 + :tag2_i]
                AS tag2,
            (ARRAY['urgent', 'fragile', 'giftwrap', 'backorder', 'expedite',
                   'digital', 'bulk', 'sample', 'recurring', 'warranty'])[1 + :fts1_i]
                AS fts_term1,
            (ARRAY['priority', 'standard', 'economy'])[1 + :fts2_i]
                AS fts_term2
    ),

    --------------------------------------------------------------------------
    -- q_idx: btree index scan -> nested loop -> window agg over one
    -- customer's paid/shipped/done orders (always exactly 3 of 5, see
    -- ro_cpu_setup.sql status-generation note).
    --------------------------------------------------------------------------
    recent AS (
        SELECT o.order_id, o.customer_id
        FROM ro_cpu_order o
        JOIN params p ON o.customer_id = p.customer_id
        WHERE o.status IN ('paid', 'shipped', 'done')
    ),
    lines AS (
        SELECT
            r.order_id,
            i.product_id,
            i.line_total,
            pr.category,
            pr.attrs
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
    q_idx AS (
        SELECT
            'q_idx' AS tag,
            count(*)::text
                || ':' || coalesce(sum(line_total), 0)::text
                || ':' || coalesce(sum(order_sum) FILTER (WHERE cat_rank = 1), 0)::text
                || ':' || md5(string_agg(attrs ->> 'tier', ',' ORDER BY product_id))
                AS dig
        FROM ranked
    ),

    --------------------------------------------------------------------------
    -- q_hashjoin: time-windowed order slice -> join against the full
    -- order_item + product tables with no customer/status filter, so the
    -- planner has no cheap narrowing index into order_item and hashes the
    -- (small) product side instead of a per-row nested loop.
    --------------------------------------------------------------------------
    fact_slice AS (
        SELECT o.order_id
        FROM ro_cpu_order o, params p
        WHERE o.ordered_at >= timestamptz '2022-01-01 00:00:00+00'
                + make_interval(secs => :hj_start_sec::double precision)
          AND o.ordered_at < timestamptz '2022-01-01 00:00:00+00'
                + make_interval(secs => :hj_end_sec::double precision)
    ),
    fact_items AS (
        SELECT fs.order_id, i.product_id, i.qty, i.line_total
        FROM fact_slice fs
        JOIN ro_cpu_order_item i ON i.order_id = fs.order_id
    ),
    hj_agg AS (
        SELECT
            pr.category,
            pr.attrs ->> 'tier' AS tier,
            count(*) AS n_lines,
            sum(fi.qty) AS qty,
            sum(fi.line_total) AS revenue
        FROM fact_items fi
        JOIN ro_cpu_product pr ON pr.product_id = fi.product_id
        GROUP BY pr.category, pr.attrs ->> 'tier'
    ),
    q_hashjoin AS (
        SELECT
            'q_hashjoin' AS tag,
            string_agg(
                category || '/' || tier || '=' || n_lines || ':' || qty || ':' || revenue::text,
                ';' ORDER BY category, tier
            ) AS dig
        FROM hj_agg
    ),

    --------------------------------------------------------------------------
    -- q_regex: regex + md5 over a small order slice (shrunk from the
    -- original 12000-row width so it no longer dominates the transaction).
    --------------------------------------------------------------------------
    slice AS (
        SELECT o.order_id, o.note, o.meta
        FROM ro_cpu_order o, params p
        WHERE o.order_id BETWEEN p.order_id AND :regex_end::int
    ),
    scanned AS (
        SELECT
            count(*) AS nrows,
            count(*) FILTER (
                WHERE note ~ 'email=user[0-9]+@example\.com'
            ) AS n_email,
            count(*) FILTER (
                WHERE note ~* 'path=/var/log/orders/[0-9]+\.log'
            ) AS n_path,
            count(*) FILTER (
                WHERE (meta -> 'flags' ->> 'rush')::boolean
            ) AS n_rush,
            sum(length(md5(note || (meta ->> 'trace')))) AS hash_work
        FROM slice
    ),
    q_regex AS (
        SELECT
            'q_regex' AS tag,
            nrows::text || ':' || n_email || ':' || n_path || ':' || n_rush
                || ':' || hash_work AS dig
        FROM scanned
    ),

    --------------------------------------------------------------------------
    -- q_fts: full text search. GIN index on search_doc, `@@` match + ts_rank
    -- ordering -- inverted-index posting-list code, distinct from regex.
    --------------------------------------------------------------------------
    fts AS (
        SELECT
            o.order_id,
            o.status,
            ts_rank(o.search_doc, tq.q) AS rank
        FROM params p
        CROSS JOIN LATERAL (
            SELECT to_tsquery('simple', p.fts_term1 || ' & ' || p.fts_term2) AS q
        ) tq
        JOIN ro_cpu_order o ON o.search_doc @@ tq.q
        ORDER BY rank DESC
        LIMIT :fts_lim
    ),
    q_fts AS (
        SELECT
            'q_fts' AS tag,
            count(*)::text || ':' || coalesce(sum(rank), 0)::text
                || ':' || coalesce(string_agg(status, ',' ORDER BY rank DESC), '') AS dig
        FROM fts
    ),

    --------------------------------------------------------------------------
    -- q_array: array containment (`@>`) via GIN on product.tags -- exercises
    -- the array opclass/AM, previously unused. Joined against a *bounded*
    -- order_id slice of order_item (not the full 750k-row table): an
    -- unbounded join here made the planner pick the same
    -- Seq-Scan-order_item + Hash Join plan as q_hashjoin, silently
    -- duplicating that block's cost instead of isolating the array/GIN path.
    --------------------------------------------------------------------------
    arr AS (
        SELECT pr.product_id, pr.category
        FROM ro_cpu_product pr, params p
        WHERE pr.tags @> ARRAY[p.tag1, p.tag2]
        LIMIT :array_lim
    ),
    arr_slice AS (
        SELECT i.product_id, i.qty, i.line_total
        FROM ro_cpu_order_item i
        WHERE i.order_id BETWEEN :oid::int AND :array_slice_end::int
    ),
    arr_join AS (
        SELECT a.category, count(*) AS n, sum(s.qty) AS qty, sum(s.line_total) AS revenue
        FROM arr a
        JOIN arr_slice s ON s.product_id = a.product_id
        GROUP BY a.category
    ),
    q_array AS (
        SELECT
            'q_array' AS tag,
            coalesce(string_agg(
                category || '=' || n || ':' || qty || ':' || revenue::text,
                ';' ORDER BY category
            ), '') AS dig
        FROM arr_join
    ),

    --------------------------------------------------------------------------
    -- q_stats: ordered-set + statistical aggregates (percentile_cont needs a
    -- per-group sort; stddev/corr use multi-value transition states) --
    -- previously only plain sum/count were used anywhere in this script.
    --------------------------------------------------------------------------
    stats_slice AS (
        SELECT i.qty, i.line_total, pr.category
        FROM ro_cpu_order_item i
        JOIN ro_cpu_product pr ON pr.product_id = i.product_id
        WHERE i.order_id BETWEEN :oid::int AND :stats_end::int
    ),
    stats_agg AS (
        SELECT
            category,
            count(*) AS n,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY line_total) AS median_total,
            stddev_samp(line_total) AS sd_total,
            corr(qty::float8, line_total::float8) AS qty_total_corr
        FROM stats_slice
        GROUP BY category
    ),
    q_stats AS (
        SELECT
            'q_stats' AS tag,
            coalesce(string_agg(
                category || '=' || n
                    || ':' || round(coalesce(median_total, 0)::numeric, 2)
                    || ':' || round(coalesce(sd_total, 0)::numeric, 2)
                    || ':' || round(coalesce(qty_total_corr, 0)::numeric, 4),
                ';' ORDER BY category
            ), '') AS dig
        FROM stats_agg
    ),

    --------------------------------------------------------------------------
    -- q_toast: fetch+decompress a >2KB out-of-line column for a bounded
    -- number of rows via a PK range probe -- previously no column in this
    -- schema was ever large enough to be TOASTed.
    --------------------------------------------------------------------------
    toast_probe AS (
        SELECT pr.product_id, length(pr.spec_blob) AS blob_len, md5(pr.spec_blob) AS blob_hash
        FROM ro_cpu_product pr
        WHERE pr.product_id BETWEEN 1 AND :toast_n::int
    ),
    q_toast AS (
        SELECT
            'q_toast' AS tag,
            count(*)::text || ':' || sum(blob_len)::text
                || ':' || md5(string_agg(blob_hash, ',' ORDER BY product_id)) AS dig
        FROM toast_probe
    ),

    --------------------------------------------------------------------------
    -- q_seqscan: full-table aggregate with no predicate/index -- the only
    -- plain SeqScan in the script (everything else is index/bitmap/hash-driven).
    --------------------------------------------------------------------------
    seq_agg AS (
        SELECT category, count(*) AS n, avg(unit_price) AS avg_price, sum(unit_price) AS sum_price
        FROM ro_cpu_product
        GROUP BY category
    ),
    q_seqscan AS (
        SELECT
            'q_seqscan' AS tag,
            string_agg(
                category || '=' || n || ':' || round(avg_price, 4) || ':' || round(sum_price, 4),
                ';' ORDER BY category
            ) AS dig
        FROM seq_agg
    )

    SELECT dig AS x FROM q_idx
    UNION ALL SELECT dig FROM q_hashjoin
    UNION ALL SELECT dig FROM q_regex
    UNION ALL SELECT dig FROM q_fts
    UNION ALL SELECT dig FROM q_array
    UNION ALL SELECT dig FROM q_stats
    UNION ALL SELECT dig FROM q_toast
    UNION ALL SELECT dig FROM q_seqscan
) s;
