import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import _parse_tpch_score  # noqa: E402

HAMMERDB_4X_SAMPLE = """\
Vuser 1:query 12 completed in 5.271 seconds

Vuser 1:Completed 1 query set(s) in 369 seconds

Vuser 1:Geometric mean of query times returning rows (22) is 10.06547

Vuser 1:FINISHED SUCCESS

ALL VIRTUAL USERS COMPLETE
"""


def test_parse_tpch_score_from_geometric_mean():
    score = _parse_tpch_score(HAMMERDB_4X_SAMPLE)
    assert score == 358


def test_parse_tpch_score_from_qphh_line():
    out = "Score (TPROC-H) = 12345"
    assert _parse_tpch_score(out) == 12345
