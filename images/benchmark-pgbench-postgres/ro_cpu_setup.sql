-- Cached read-only CPU/SQL benchmark schema.
-- Goal: exercise a *broad, evenly-weighted* cross-section of Postgres's
-- executor/access-method/type subsystems on a dataset that fits entirely
-- in shared_buffers (disk is not the bottleneck; CPU dominates).
--
-- Deliberately covers subsystems the original design skipped entirely:
--   - non-btree index AMs: GIN (tsvector, array), BRIN (time-correlated range)
--   - hash join / hash aggregate (fact-table joins with no narrowing index)
--   - full text search (tsvector/tsquery/ts_rank) instead of only regex
--   - array containment (text[] + GIN)
--   - ordered-set / statistical aggregates (percentile_cont, stddev, corr)
--   - TOAST compression + out-of-line fetch (>2 KB column, rarely touched)
--   - plain seq scan + aggregate (no supporting predicate/index)
-- See ro_cpu_txn.sql for how each block maps to a query tag (q_idx, q_hashjoin, ...).
--
-- Approximate footprint: ~260–320 MB data + indexes (well under typical SB).

BEGIN;

DROP TABLE IF EXISTS ro_cpu_order_item CASCADE;
DROP TABLE IF EXISTS ro_cpu_order CASCADE;
DROP TABLE IF EXISTS ro_cpu_customer CASCADE;
DROP TABLE IF EXISTS ro_cpu_product CASCADE;

CREATE TABLE ro_cpu_product (
    product_id   integer PRIMARY KEY,
    sku          text        NOT NULL,
    category     text        NOT NULL,
    attrs        jsonb       NOT NULL,
    tags         text[]      NOT NULL,
    unit_price   numeric(12, 4) NOT NULL,
    -- >2KB, low intra-value redundancy (concatenated distinct md5s) so it
    -- resists PGLZ compression and exercises real TOAST fetch, not just
    -- cheap inline decompression.
    spec_blob    text        NOT NULL
);

CREATE TABLE ro_cpu_customer (
    customer_id  integer PRIMARY KEY,
    email        text        NOT NULL,
    region       text        NOT NULL,
    profile      jsonb       NOT NULL,
    created_at   timestamptz NOT NULL
);

CREATE TABLE ro_cpu_order (
    order_id     integer PRIMARY KEY,
    customer_id  integer     NOT NULL REFERENCES ro_cpu_customer (customer_id),
    status       text        NOT NULL,
    ordered_at   timestamptz NOT NULL,
    meta         jsonb       NOT NULL,
    note         text        NOT NULL,
    tags_text    text        NOT NULL,
    -- Full text search corpus; GIN-indexed below. Kept separate from `note`
    -- so FTS and regex blocks scan comparable-but-distinct content.
    search_doc   tsvector GENERATED ALWAYS AS (to_tsvector('simple', tags_text)) STORED
);

CREATE TABLE ro_cpu_order_item (
    order_id     integer     NOT NULL REFERENCES ro_cpu_order (order_id),
    line_no      smallint    NOT NULL,
    product_id   integer     NOT NULL REFERENCES ro_cpu_product (product_id),
    qty          integer     NOT NULL,
    line_total   numeric(12, 4) NOT NULL,
    PRIMARY KEY (order_id, line_no)
);

-- ~20k products (long-tail catalog: only the first 5k ever appear in an
-- order_item line below, so seq-scan/TOAST blocks touch cold rows too —
-- mirrors real shops where most SKUs rarely sell).
INSERT INTO ro_cpu_product (product_id, sku, category, attrs, tags, unit_price, spec_blob)
SELECT
    g,
    format('SKU-%s', lpad(g::text, 6, '0')),
    (ARRAY['compute', 'storage', 'network', 'gpu', 'misc'])[1 + (g % 5)],
    jsonb_build_object(
        'tier', (ARRAY['S', 'M', 'L', 'XL'])[1 + (g % 4)],
        'tags', jsonb_build_array('bench', md5(g::text), (g % 17)::text),
        'specs', jsonb_build_object(
            'cores', 1 + (g % 64),
            'mem_gb', 1 + (g % 512),
            'nested', jsonb_build_object('path', format('root.item[%s]', g % 100))
        )
    ),
    ARRAY[
        (ARRAY['urgent', 'fragile', 'giftwrap', 'backorder', 'expedite',
               'digital', 'bulk', 'sample', 'recurring', 'warranty'])[1 + (g % 10)],
        (ARRAY['north', 'south', 'east', 'west', 'central'])[1 + ((g / 3) % 5)]
    ],
    (10 + (g % 990))::numeric / 4,
    (SELECT string_agg(md5(g::text || '-' || s::text), '') FROM generate_series(1, 140) AS s)
FROM generate_series(1, 20000) AS g;

-- ~50k customers
INSERT INTO ro_cpu_customer (customer_id, email, region, profile, created_at)
SELECT
    g,
    format('user%s@example.com', g),
    (ARRAY['us-east', 'us-west', 'eu-west', 'asia', 'sa'])[1 + (g % 5)],
    jsonb_build_object(
        'plan', (ARRAY['free', 'pro', 'ent'])[1 + (g % 3)],
        'score', (g % 1000)::float8 / 10.0,
        'prefs', jsonb_build_object('newsletter', g % 2 = 0, 'locale', 'en')
    ),
    timestamptz '2021-01-01 00:00:00+00' + make_interval(mins => g)
FROM generate_series(1, 50000) AS g;

-- ~250k orders (=5 per customer).
-- Status: NOTE the naive `1 + (g % 5)` formula is a trap — with 50000
-- customers (a multiple of 5), every order for a given customer lands on
-- the same residue, so a customer's 5 orders always share ONE status
-- (verified empirically: `status IN ('paid','shipped','done')` returned 0
-- rows for ~40% of customers and 5 for the rest, breaking q_idx). Mixing in
-- the order's sequence-within-customer fixes this: residues 0..4 rotate
-- through all 5 statuses exactly once per customer.
INSERT INTO ro_cpu_order (order_id, customer_id, status, ordered_at, meta, note, tags_text)
SELECT
    g,
    1 + ((g - 1) % 50000) AS customer_id,
    (ARRAY['new', 'paid', 'shipped', 'done', 'cancel'])[
        1 + ((1 + ((g - 1) % 50000)) + ((g - 1) / 50000)) % 5
    ] AS status,
    timestamptz '2022-01-01 00:00:00+00' + make_interval(secs => g),
    jsonb_build_object(
        'channel', (ARRAY['web', 'api', 'partner'])[1 + (g % 3)],
        'trace', md5(g::text),
        'flags', jsonb_build_object('rush', g % 11 = 0, 'gift', g % 23 = 0)
    ),
    format(
        'order=%s email=user%s@example.com path=/var/log/orders/%s.log pad=%s',
        g,
        1 + ((g - 1) % 50000),
        lpad((g % 1000)::text, 4, '0'),
        repeat(md5((g / 10)::text), 2)
    ),
    (ARRAY['urgent', 'fragile', 'giftwrap', 'backorder', 'expedite',
           'digital', 'bulk', 'sample', 'recurring', 'warranty'])[1 + (g % 10)]
        || ' ' || (ARRAY['north', 'south', 'east', 'west', 'central'])[1 + ((g / 7) % 5)]
        || ' ' || (ARRAY['priority', 'standard', 'economy'])[1 + ((g / 13) % 3)]
FROM generate_series(1, 250000) AS g;

-- ~750k line items (3 per order); only references the first 5k products.
INSERT INTO ro_cpu_order_item (order_id, line_no, product_id, qty, line_total)
SELECT
    o.order_id,
    l AS line_no,
    1 + ((o.order_id * 17 + l * 31) % 5000),
    1 + ((o.order_id + l) % 9),
    ((1 + ((o.order_id + l) % 9)) * (10 + ((o.order_id * 17 + l * 31) % 5000) % 990))::numeric / 4
FROM ro_cpu_order o
CROSS JOIN generate_series(1, 3) AS l;

-- Index access paths (PK indexes already exist).
CREATE INDEX ro_cpu_customer_region_idx ON ro_cpu_customer (region);
CREATE INDEX ro_cpu_customer_email_idx ON ro_cpu_customer (email);
CREATE INDEX ro_cpu_order_customer_idx ON ro_cpu_order (customer_id);
CREATE INDEX ro_cpu_order_status_ordered_idx ON ro_cpu_order (status, ordered_at);
CREATE INDEX ro_cpu_order_item_product_idx ON ro_cpu_order_item (product_id);
CREATE INDEX ro_cpu_product_category_idx ON ro_cpu_product (category);
-- Expression / JSON helpers used by the scan-heavy query.
CREATE INDEX ro_cpu_order_meta_channel_idx ON ro_cpu_order ((meta ->> 'channel'));
CREATE INDEX ro_cpu_product_tier_idx ON ro_cpu_product ((attrs ->> 'tier'));

-- Non-btree access methods (previously entirely absent from this schema).
CREATE INDEX ro_cpu_order_search_doc_gin ON ro_cpu_order USING gin (search_doc);
CREATE INDEX ro_cpu_product_tags_gin ON ro_cpu_product USING gin (tags);
-- ordered_at is inserted in strictly increasing order (== physical order),
-- so a BRIN index is near-ideal here and much smaller than a btree would be.
CREATE INDEX ro_cpu_order_ordered_at_brin ON ro_cpu_order USING brin (ordered_at);

ANALYZE ro_cpu_product;
ANALYZE ro_cpu_customer;
ANALYZE ro_cpu_order;
ANALYZE ro_cpu_order_item;

GRANT SELECT ON ro_cpu_product, ro_cpu_customer, ro_cpu_order, ro_cpu_order_item TO PUBLIC;

COMMIT;

-- Size report (informational)
SELECT
    relname AS table,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_with_indexes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND relname LIKE 'ro_cpu_%'
  AND relkind = 'r'
ORDER BY pg_total_relation_size(c.oid) DESC;
