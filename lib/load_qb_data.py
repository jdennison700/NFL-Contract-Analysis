import polars as pl
import nflreadpy as nfl

# Column families that carry no signal for quarterback analysis: receiving
# production, defensive stats, kicking, punting, and return/special-teams work.
_DROP_PREFIXES = ("receiving_", "def_", "fg_", "pat_", "gwfg_", "pt_")

_DROP_EXACT = (
    # roster / meta noise
    "player_name",
    "position_group",
    "headshot_url",
    "season",
    "season_type",
    # receiving volume + efficiency
    "receptions",
    "targets",
    "racr",
    "target_share",
    "air_yards_share",
    "wopr",
    # returns / special teams
    "special_teams_tds",
    "punt_returns",
    "punt_return_yards",
    "kickoff_returns",
    "kickoff_return_yards",
)


def load_qb_stats(seasons: list[int], summary_level: str = "reg") -> pl.DataFrame:
    """
    Load quarterback stats from the NFL API for the specified seasons and summary level.

    Args:
        seasons (list[int]): List of seasons to load data for.
        summary_level (str): Summary level of the data ("reg" for regular season, "post" for postseason).

    Returns:
        pl.DataFrame: A Polars DataFrame containing the quarterback stats.
    """
    qb_stats = nfl.load_player_stats(seasons=seasons, summary_level=summary_level)
    qb_stats = qb_stats.filter(pl.col("position") == "QB")

    # drop non qb stats
    drop_cols = [
        c
        for c in qb_stats.columns
        if c in _DROP_EXACT or c.startswith(_DROP_PREFIXES)
    ]

    return qb_stats.drop(drop_cols)

def load_player_contracts(season: int) -> pl.DataFrame:
    """
    Load QB contracts and keep only the deal that governed each QB's ``season``,
    one row per player.

    "Active in ``season``" means the contract's ``season_history`` has an entry for
    that year with a real cap charge (``cap_number > 0``) -- the authoritative
    cap-sheet record. The top-level ``is_active`` flag is *not* used: it marks
    currently rostered deals as of the present league year, not ``season``.

    Args:
        season (int): The NFL season the contract must have been in force for.
    """
    contracts = nfl.load_contracts().filter(pl.col("position") == "QB")

    covers_season = (
        pl.col("season_history")
        .list.eval(
            (pl.element().struct.field("year") == str(season))
            & (pl.element().struct.field("cap_number") > 0)
        )
        .list.any()
    )

    # A later extension still lists ``season`` in its (duplicated) season_history,
    # so restrict to deals signed on or before ``season``.
    qb_contracts = contracts.filter(covers_season & (pl.col("year_signed") <= season))

    # Collapse to one row per QB: the most recent deal in force during ``season``.
    qb_contracts = (
        qb_contracts.sort(["year_signed", "value", "guaranteed"])
        .group_by("player", maintain_order=True)
        .last()
    )

    # gsis_id is the join key to the qb_stats frame (its player_id column).
    qb_contracts = qb_contracts.rename({"gsis_id": "player_id"}).drop(
        "position",
        "player_page",
        "otc_id",
        "height",
        "weight",
        "college",
        "draft_year",
        "draft_round",
        "draft_overall",
        "date_of_birth",
        "season_history",
        "contract_history",
    )

    return qb_contracts


if __name__ == "__main__":
    # Smoke-test the nflverse pull. The pipeline (main.py) does not stage these
    # frames -- they're spliced with their grades into qb_performance / qb_value.
    SEASON = 2025

    qb_stats = load_qb_stats(seasons=[SEASON])
    qb_contracts = load_player_contracts(SEASON)

    print(f"qb_stats:     {qb_stats.shape}  {qb_stats.columns}")
    print(f"qb_contracts: {qb_contracts.shape}  {qb_contracts.columns}")
