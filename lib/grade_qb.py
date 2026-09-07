import polars as pl
import sqlalchemy as sa

from lib.sqlite import DB_PATH, write_to_sqlite

# Component weights for the composite. Keys are the derived-metric names built in
# `grade_qb_performance`; values need not sum to 1.0 (they're re-normalized).
_DEFAULT_WEIGHTS = {
    "epa_per_db": 0.35,   # EPA per dropback -- the core efficiency signal
    "cpoe": 0.18,         # completion % over expected (accuracy vs difficulty)
    "sack_rate": 0.10,    # sacks / dropback (pocket management) -- lower is better
    "to_rate": 0.17,      # (INT + fumbles lost) / dropback -- lower is better
    "fd_rate": 0.10,      # passing first downs / attempt (moving the chains)
    "rush_epa_pg": 0.10,  # rushing EPA per game (mobility value add)
}

# Components where a smaller raw value is better; their z-score is negated so that
# "higher composite = better QB" holds for every term.
_INVERTED = {"sack_rate", "to_rate"}

# Absolute composite-z cutoffs, checked high-to-low. Not a curve: a weak QB pool
# would simply produce few A's.
_GRADE_CUTS = [
    (1.50, "A+"),
    (1.00, "A"),
    (0.70, "A-"),
    (0.40, "B+"),
    (0.15, "B"),
    (-0.15, "B-"),
    (-0.40, "C+"),
    (-0.70, "C"),
    (-1.00, "C-"),
    (-1.50, "D"),
]  # anything below the last cut is "F"


def grade_qb_performance(
    qb_stats: pl.DataFrame,
    min_attempts: int = 200,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """
    Grade quarterback season performance on an A+..F scale.

    The grade is a weighted composite of per-dropback efficiency and value metrics
    (EPA, CPOE, sack rate, turnover rate, first-down rate) plus a rushing term.
    Each metric is z-scored across the qualified QB pool, winsorized at +/-3 SD,
    weighted, and summed into ``composite_z``; that is bucketed into a letter grade
    via fixed absolute cutoffs.

    Args:
        qb_stats: One row per QB, as produced by ``load_qb_stats`` / the
            ``qb_stats_2025`` table.
        min_attempts: Minimum pass attempts to be graded. The default (200) drops
            deep backups whose tiny samples would distort the pool mean/SD.
        weights: Optional override of component weights. Keys must be a subset of
            the default component names; values are re-normalized to sum to 1.0.

    Returns:
        pl.DataFrame sorted by ``composite_z`` descending, with the per-component
        z-scores, ``composite_z``, ``score_0_100`` and ``letter_grade``.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(_DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weight keys: {sorted(unknown)}")
        w.update(weights)
    total = sum(w.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    w = {k: v / total for k, v in w.items()}

    qualified = qb_stats.filter(pl.col("attempts") >= min_attempts)
    if qualified.height < 2:
        raise ValueError(
            f"need >=2 QBs with attempts >= {min_attempts}, got {qualified.height}"
        )

    dropbacks = pl.col("attempts") + pl.col("sacks_suffered")
    metrics = qualified.with_columns(
        small_sample=pl.col("attempts") < 300,
        epa_per_db=pl.col("passing_epa") / dropbacks,
        cpoe=pl.col("passing_cpoe"),
        sack_rate=pl.col("sacks_suffered") / dropbacks,
        to_rate=(pl.col("passing_interceptions") + pl.col("fumbles_lost_total"))
        / dropbacks,
        fd_rate=pl.col("passing_first_downs") / pl.col("attempts"),
        rush_epa_pg=pl.col("rushing_epa") / pl.col("games"),
    )

    z_exprs = []
    for name in _DEFAULT_WEIGHTS:
        z = (pl.col(name) - pl.col(name).mean()) / pl.col(name).std()
        z = z.clip(-3.0, 3.0)
        if name in _INVERTED:
            z = -z
        z_exprs.append(z.alias(f"z_{name}"))
    metrics = metrics.with_columns(z_exprs)

    composite = pl.sum_horizontal(
        [pl.col(f"z_{name}") * weight for name, weight in w.items()]
    )
    metrics = metrics.with_columns(composite_z=composite).with_columns(
        score_0_100=(50 + 15 * pl.col("composite_z")).clip(0, 100)
    )

    grade = pl.lit("F")
    for cut, letter in reversed(_GRADE_CUTS):
        grade = pl.when(pl.col("composite_z") >= cut).then(pl.lit(letter)).otherwise(grade)
    metrics = metrics.with_columns(letter_grade=grade)

    out_cols = (
        ["player_display_name", "recent_team", "games", "attempts", "small_sample"]
        + [f"z_{name}" for name in _DEFAULT_WEIGHTS]
        + ["composite_z", "score_0_100", "letter_grade"]
    )
    return metrics.select(out_cols).sort("composite_z", descending=True)


if __name__ == "__main__":
    engine = sa.create_engine(f"sqlite:///{DB_PATH}")
    qb_stats = pl.read_database("SELECT * FROM qb_stats_2025", engine)

    graded = grade_qb_performance(qb_stats)

    write_to_sqlite(graded, "qb_performance_grades_2025")

    for row in graded.iter_rows(named=True):
        print(
            f"{row['letter_grade']:>2}  {row['composite_z']:+.2f}  "
            f"{row['player_display_name']} ({row['recent_team']})"
        )
