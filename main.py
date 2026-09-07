"""
End-to-end NFL QB grading pipeline.

    python main.py                # full 2025 run
    python main.py --season 2024  # a different season
    python main.py --skip-load    # reuse the data already in nfl_qb.db

Steps:
    1. pull season stats + contracts from nflverse           -> qb_stats_<yr>, qb_contracts_<yr>
    2. grade on-field performance (A+..F)                    -> qb_performance_grades_<yr>
    3. grade performance against contract value (tiers)      -> qb_value_grades_<yr>
"""

import argparse

import polars as pl
import sqlalchemy as sa

from lib.grade_qb import grade_qb_performance, grade_qb_value
from lib.load_qb_data import load_player_contracts, load_qb_stats
from lib.sqlite import DB_PATH, write_to_sqlite


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
    season: int = 2025,
    *,
    skip_load: bool = False,
    min_attempts: int = 200,
    db_path: str = DB_PATH,
) -> dict[str, pl.DataFrame]:
    """
    Run the whole pipeline and return the four resulting frames keyed by table name.

    Args:
        season: NFL season to grade.
        skip_load: reuse ``qb_stats_<season>`` / ``qb_contracts_<season>`` already
            in the DB instead of pulling fresh from nflverse.
        min_attempts: passed through to ``grade_qb_performance``.
        db_path: SQLite file to read from / write to.
    """
    engine = sa.create_engine(f"sqlite:///{db_path}")
    stats_table = f"qb_stats_{season}"
    contracts_table = f"qb_contracts_{season}"

    if skip_load:
        qb_stats = pl.read_database(f"SELECT * FROM {stats_table}", engine)
        contracts = pl.read_database(f"SELECT * FROM {contracts_table}", engine)
    else:
        qb_stats = load_qb_stats(seasons=[season])
        contracts = load_player_contracts()
        write_to_sqlite(qb_stats, stats_table, db_path)
        write_to_sqlite(contracts, contracts_table, db_path)

    performance = grade_qb_performance(qb_stats, min_attempts=min_attempts)
    write_to_sqlite(performance, f"qb_performance_grades_{season}", db_path)

    value = grade_qb_value(performance, contracts)
    write_to_sqlite(value, f"qb_value_grades_{season}", db_path)

    print(f"\n=== {season} performance grades ===")
    _print_performance(performance)
    print(f"\n=== {season} contract-value grades ===")
    _print_value(value)

    return {
        stats_table: qb_stats,
        contracts_table: contracts,
        f"qb_performance_grades_{season}": performance,
        f"qb_value_grades_{season}": value,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="reuse data already in nfl_qb.db instead of pulling from nflverse",
    )
    parser.add_argument("--min-attempts", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(season=args.season, skip_load=args.skip_load, min_attempts=args.min_attempts)
