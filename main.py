"""
End-to-end NFL QB grading pipeline.

    python main.py                # full 2025 run
    python main.py --season 2024  # a different season
    python main.py --skip-load    # reuse the data already in nfl_qb.db

Steps:
    1. pull season stats + contracts from nflverse
    2. grade on-field performance (A+..F), splice onto the stats   -> qb_performance
    3. grade performance vs contract value, splice onto contracts  -> qb_value

Both tables carry a ``season`` column and span every season graded so far; a
re-run replaces just that season's rows.
"""

import argparse

import polars as pl
import sqlalchemy as sa

from lib.grade_qb import (
    _PERF_GRADE_COLS,
    _VALUE_GRADE_COLS,
    grade_qb_performance,
    grade_qb_value,
    join_contracts_and_value,
    join_stats_and_grades,
)
from lib.load_qb_data import load_player_contracts, load_qb_stats
from lib.sqlite import DB_PATH, upsert_season

# Season graded when none is given on the CLI or to run().
DEFAULT_SEASON = 2025


def _print_performance(graded: pl.DataFrame) -> None:
    for row in graded.iter_rows(named=True):
        print(
            f"{row['letter_grade']:>2}  {row['composite_z']:+.2f}  "
            f"{row['player_display_name']} ({row['recent_team']})"
        )


def _print_value(value: pl.DataFrame) -> None:
    for row in value.iter_rows(named=True):
        print(
            f"{row['value_tier']:<15}  {row['value_resid']:+.2f}  "
            f"{row['player_display_name']} ({row['recent_team']})  "
            f"grade {row['letter_grade']}, {row['apy_cap_pct']:.1%} of cap"
        )


def run(
    season: int = DEFAULT_SEASON,
    *,
    skip_load: bool = False,
    min_attempts: int = 200,
    db_path: str = DB_PATH,
) -> dict[str, pl.DataFrame]:
    """
    Run the whole pipeline and return the two resulting frames keyed by table name
    (``qb_performance``, ``qb_value``).

    Args:
        season: NFL season to grade.
        skip_load: reuse the season's rows already in ``qb_performance`` /
            ``qb_value`` instead of pulling fresh from nflverse.
        min_attempts: passed through to ``grade_qb_performance``.
        db_path: SQLite file to read from / write to.
    """
    engine = sa.create_engine(f"sqlite:///{db_path}")

    if skip_load:
        if not sa.inspect(engine).has_table("qb_performance"):
            raise RuntimeError(
                f"no data for season {season}; run without --skip-load first"
            )
        prev_perf = pl.read_database(
            "SELECT * FROM qb_performance WHERE season = :s",
            engine,
            execute_options={"parameters": {"s": season}},
        )
        prev_value = pl.read_database(
            "SELECT * FROM qb_value WHERE season = :s",
            engine,
            execute_options={"parameters": {"s": season}},
        )
        if prev_perf.height == 0:
            raise RuntimeError(
                f"no data for season {season}; run without --skip-load first"
            )
        qb_stats = prev_perf.drop(
            "season", *[c for c in _PERF_GRADE_COLS if c in prev_perf.columns]
        )
        contracts = prev_value.drop(
            "season", *[c for c in _VALUE_GRADE_COLS if c in prev_value.columns]
        )
    else:
        qb_stats = load_qb_stats(seasons=[season])
        contracts = load_player_contracts(season)

    performance = grade_qb_performance(qb_stats, min_attempts=min_attempts)
    value = grade_qb_value(performance, contracts)

    perf_out = join_stats_and_grades(qb_stats, performance).with_columns(
        pl.lit(season).alias("season")
    )
    value_out = join_contracts_and_value(contracts, value).with_columns(
        pl.lit(season).alias("season")
    )
    upsert_season(perf_out, "qb_performance", season, db_path)
    upsert_season(value_out, "qb_value", season, db_path)

    print(f"\n=== {season} performance grades ===")
    _print_performance(performance)
    print(f"\n=== {season} contract-value grades ===")
    _print_value(value)

    return {"qb_performance": perf_out, "qb_value": value_out}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="re-grade the season's rows already in nfl_qb.db instead of pulling from nflverse",
    )
    parser.add_argument("--min-attempts", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(season=args.season, skip_load=args.skip_load, min_attempts=args.min_attempts)
