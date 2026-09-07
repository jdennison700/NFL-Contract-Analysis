import polars as pl
import nflreadpy as nfl

from lib.sqlite import write_to_sqlite

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

def load_player_contracts() -> pl.DataFrame:
    """
    Load QB contracts and keep only the deal that governed each QB's 2025 season,
    one row per player.

    "Active in 2025" means the contract's ``season_history`` has a 2025 entry with
    a real cap charge (``cap_number > 0``) -- the authoritative cap-sheet record.
    The top-level ``is_active`` flag is *not* used: it marks currently rostered
    deals as of the present league year, not 2025.
    """
    contracts = nfl.load_contracts().filter(pl.col("position") == "QB")

    covers_2025 = (
        pl.col("season_history")
        .list.eval(
            (pl.element().struct.field("year") == "2025")
            & (pl.element().struct.field("cap_number") > 0)
        )
        .list.any()
    )

    # A 2026 extension still lists 2025 in its (duplicated) season_history, so
    # restrict to deals signed on or before 2025.
    qb_contracts = contracts.filter(covers_2025 & (pl.col("year_signed") <= 2025))

    # Collapse to one row per QB: the most recent deal in force during 2025.
    qb_contracts = (
        qb_contracts.sort(["year_signed", "value", "guaranteed"])
        .group_by("player", maintain_order=True)
        .last()
    )

    # gsis_id is the join key to qb_stats_2025 (its player_id column).
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

    qb_stats = load_qb_stats(seasons=[2025])
    qb_contracts = load_player_contracts()

    write_to_sqlite(qb_stats, "qb_stats_2025")
    write_to_sqlite(qb_contracts, "qb_contracts_2025")
