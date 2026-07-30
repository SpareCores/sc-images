-- Cached read-only CPU/SQL benchmark schema.
-- Goal: real table I/O paths (index + heap + seq scan) on a dataset that fits
-- entirely in shared_buffers so disk is not the bottleneck; CPU (parse/plan/
-- execute, joins, aggregates, text/json expressions) dominates.
--
-- Approximate footprint: ~120–180 MB data + indexes (well under typical SB).
-- Inspired by Postgres regress patterns (WITH/CTE, joins, aggs, window, text ops)
-- but sized for a ~50–100 ms mixed pgbench transaction.

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
    unit_price   numeric(12, 4) NOT NULL
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
    note         text        NOT NULL
);

CREATE TABLE ro_cpu_order_item (
    order_id     integer     NOT NULL REFERENCES ro_cpu_order (order_id),
    line_no      smallint    NOT NULL,
    product_id   integer     NOT NULL REFERENCES ro_cpu_product (product_id),
    qty          integer     NOT NULL,
    line_total   numeric(12, 4) NOT NULL,
    PRIMARY KEY (order_id, line_no)
);

-- ~5k products
INSERT INTO ro_cpu_product (product_id, sku, category, attrs, unit_price)
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
    (10 + (g % 990))::numeric / 4
FROM generate_series(1, 5000) AS g;

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

-- ~250k orders (≈5 per customer)
INSERT INTO ro_cpu_order (order_id, customer_id, status, ordered_at, meta, note)
SELECT
    g,
    1 + ((g - 1) % 50000),
    (ARRAY['new', 'paid', 'shipped', 'done', 'cancel'])[1 + (g % 5)],
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
    )
FROM generate_series(1, 250000) AS g;

-- ~750k line items (3 per order)
INSERT INTO ro_cpu_order_item (order_id, line_no, product_id, qty, line_total)
SELECT
    o.order_id,
    l AS line_no,
    1 + ((o.order_id * 17 + l * 31) % 5000),
    1 + ((o.order_id + l) % 9),
    ((1 + ((o.order_id + l) % 9)) * (10 + ((o.order_id * 17 + l * 31) % 5000) % 990))::numeric / 4
FROM ro_cpu_order o
CROSS JOIN generate_series(1, 3) AS l;

-- Index access paths (PK indexes already exist)
CREATE INDEX ro_cpu_customer_region_idx ON ro_cpu_customer (region);
CREATE INDEX ro_cpu_customer_email_idx ON ro_cpu_customer (email);
CREATE INDEX ro_cpu_order_customer_idx ON ro_cpu_order (customer_id);
CREATE INDEX ro_cpu_order_status_ordered_idx ON ro_cpu_order (status, ordered_at);
CREATE INDEX ro_cpu_order_item_product_idx ON ro_cpu_order_item (product_id);
CREATE INDEX ro_cpu_product_category_idx ON ro_cpu_product (category);
-- Expression / JSON helpers used by the scan-heavy query
CREATE INDEX ro_cpu_order_meta_channel_idx ON ro_cpu_order ((meta ->> 'channel'));
CREATE INDEX ro_cpu_product_tier_idx ON ro_cpu_product ((attrs ->> 'tier'));

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
