"""
Streamlit dashboard for the NFL QB grades in ``nfl_qb.db``.

Run it with:

    uv run streamlit run dashboard.py

Reads the two tables written by ``main.py`` (``qb_performance`` and ``qb_value``)
read-only and presents four views: an Overview leaderboard, a per-QB Performance
deep-dive, a contract-Value analysis, and a Compare view. Grades are pool-relative
*within* a season, so everything is scoped to the single season chosen in the sidebar.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import sqlalchemy as sa
import streamlit as st

from lib.sqlite import DB_PATH

# Resolve the DB next to this file so the app works regardless of the launch cwd.
DB_FILE = Path(__file__).resolve().parent / DB_PATH
DB_URL = f"sqlite:///{DB_FILE}"

# --- Shared vocabulary -----------------------------------------------------

# Letter grades, best -> worst. Used to order axes/tables consistently.
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]

# Grade -> color, by tier family (an ordinal quality scale, good -> bad):
# A green, B blue, C amber, D orange, F red. Distinct hues so identity never
# relies on shade alone.
_GRADE_FAMILY = {
    "A+": "#0ca30c", "A": "#0ca30c", "A-": "#0ca30c",
    "B+": "#2a78d6", "B": "#2a78d6", "B-": "#2a78d6",
    "C+": "#eda100", "C": "#eda100", "C-": "#eda100",
    "D": "#eb6834",
    "F": "#d03b3b",
}

# Value tiers, best -> worst, as a 5-step diverging scale (green -> gray -> red).
VALUE_TIER_ORDER = ["Massive surplus", "Bargain", "Fair", "Overpaid", "Albatross"]
_VALUE_TIER_COLOR = {
    "Massive surplus": "#0ca30c",
    "Bargain": "#5bbf5b",
    "Fair": "#898781",
    "Overpaid": "#ec835a",
    "Albatross": "#d03b3b",
}

# Friendly labels for the six z-score components (all oriented higher = better).
COMPONENTS = {
    "z_epa_per_db": "EPA / dropback",
    "z_cpoe": "CPOE (accuracy)",
    "z_sack_rate": "Sack avoidance",
    "z_to_rate": "Ball security",
    "z_fd_rate": "First downs",
    "z_rush_epa_pg": "Rushing",
}


# --- Data loading ----------------------------------------------------------

@st.cache_resource
def _engine():
    return sa.create_engine(DB_URL)


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    """Read a whole grade table from SQLite (cached across reruns)."""
    return pd.read_sql_table(name, _engine())


@st.cache_data
def available_seasons() -> list[int]:
    df = pd.read_sql(
        "SELECT DISTINCT season FROM qb_performance ORDER BY season DESC", _engine()
    )
    return df["season"].tolist()


def season_frames(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (performance, value) frames scoped to one season."""
    perf = load_table("qb_performance")
    value = load_table("qb_value")
    perf = perf[perf["season"] == season].copy()
    value = value[value["season"] == season].copy()
    return perf, value


# --- Small helpers ---------------------------------------------------------

def grade_sort_key(series: pd.Series) -> pd.Series:
    """Categorical order so A+..F sorts as a quality ladder, not alphabetically."""
    return series.map({g: i for i, g in enumerate(GRADE_ORDER)})


def style_grades(df: pd.DataFrame):
    """Return a pandas Styler that tints the letter_grade / value_tier cells."""
    styler = df.style
    if "letter_grade" in df.columns:
        styler = styler.map(
            lambda g: f"background-color: {_GRADE_FAMILY.get(g, '#898781')}; color: white;",
            subset=["letter_grade"],
        )
    if "value_tier" in df.columns:
        styler = styler.map(
            lambda t: f"background-color: {_VALUE_TIER_COLOR.get(t, '#898781')}; color: white;",
            subset=["value_tier"],
        )
    return styler


# --- Views -----------------------------------------------------------------

def view_overview(perf: pd.DataFrame, value: pd.DataFrame, season: int) -> None:
    st.subheader(f"{season} season leaderboard")

    top = perf.sort_values("composite_z", ascending=False).iloc[0]
    best_value = value.sort_values("value_resid", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("QBs graded", len(perf))
    c2.metric(
        "Top performer",
        top["player_display_name"],
        f"{top['letter_grade']}  ({top['composite_z']:+.2f})",
    )
    c3.metric(
        "Best value",
        best_value["player_display_name"],
        f"{best_value['value_tier']}  ({best_value['value_resid']:+.2f})",
    )

    cols = [
        "player_display_name", "recent_team", "games", "attempts",
        "score_0_100", "composite_z", "letter_grade",
    ]
    table = (
        perf[cols]
        .sort_values("composite_z", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "player_display_name": "Player", "recent_team": "Team",
            "games": "G", "attempts": "Att", "score_0_100": "Score",
            "composite_z": "Composite z", "letter_grade": "Grade",
        })
    )
    st.dataframe(
        style_grades(table).format({"Score": "{:.1f}", "Composite z": "{:+.2f}"}),
        width="stretch",
        hide_index=True,
    )

    # Grade distribution.
    counts = (
        perf["letter_grade"].value_counts()
        .reindex(GRADE_ORDER, fill_value=0)
    )
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[_GRADE_FAMILY[g] for g in counts.index],
            text=counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Grade distribution",
        xaxis_title="Letter grade",
        yaxis_title="QBs",
        showlegend=False,
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig, width="stretch")


def view_performance(perf: pd.DataFrame, season: int) -> None:
    st.subheader(f"{season} performance deep-dive")

    ranked = perf.sort_values("composite_z", ascending=False).reset_index(drop=True)
    names = ranked["player_display_name"].tolist()
    name = st.selectbox("Quarterback", names)
    row = ranked[ranked["player_display_name"] == name].iloc[0]
    rank = int(ranked.index[ranked["player_display_name"] == name][0]) + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grade", row["letter_grade"])
    m2.metric("Score / 100", f"{row['score_0_100']:.1f}")
    m3.metric("Composite z", f"{row['composite_z']:+.2f}")
    m4.metric("Rank in season", f"{rank} / {len(ranked)}")
    if bool(row.get("small_sample", False)):
        st.caption("Small sample (< 300 attempts) — grade is less stable.")

    left, right = st.columns([3, 2])

    # Radar of the six components vs the pool average (zero line).
    labels = list(COMPONENTS.values())
    player_vals = [row[c] for c in COMPONENTS]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=player_vals + [player_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=name,
        line_color="#2a78d6",
    ))
    radar.add_trace(go.Scatterpolar(
        r=[0] * (len(labels) + 1),
        theta=labels + [labels[0]],
        name="Pool average",
        line=dict(color="#898781", dash="dot"),
    ))
    radar.update_layout(
        title="Component z-scores (higher = better)",
        polar=dict(radialaxis=dict(range=[-3, 3])),
        template="plotly_white",
        height=440,
    )
    left.plotly_chart(radar, width="stretch")

    # Score gauge.
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(row["score_0_100"]),
        title={"text": "Score / 100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": _GRADE_FAMILY.get(row["letter_grade"], "#2a78d6")},
            "steps": [
                {"range": [0, 40], "color": "#f3d6d6"},
                {"range": [40, 60], "color": "#efeeea"},
                {"range": [60, 100], "color": "#d7ecd7"},
            ],
        },
    ))
    gauge.update_layout(template="plotly_white", height=440)
    right.plotly_chart(gauge, width="stretch")


def view_value(value: pd.DataFrame, season: int) -> None:
    st.subheader(f"{season} contract value")
    st.caption(
        "Performance vs pay. Points above the market line beat their contract; "
        "below it, they lag. Both axes are standard deviations within the season pool."
    )

    fig = go.Figure()

    # The OLS market line (stored per-row as expected_perf_z).
    line = value.sort_values("cost_z")
    fig.add_trace(go.Scatter(
        x=line["cost_z"],
        y=line["expected_perf_z"],
        mode="lines",
        name="Market line",
        line=dict(color="#898781", dash="dash"),
        hoverinfo="skip",
    ))

    # One scatter trace per tier so identity comes from the legend, not shade alone.
    for tier in VALUE_TIER_ORDER:
        sub = value[value["value_tier"] == tier]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["cost_z"],
            y=sub["perf_z"],
            mode="markers",
            name=tier,
            marker=dict(size=11, color=_VALUE_TIER_COLOR[tier],
                        line=dict(width=1, color="white")),
            customdata=sub[["player_display_name", "apy", "apy_cap_pct",
                            "letter_grade", "value_resid"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "APY: $%{customdata[1]:.1f}M (%{customdata[2]:.1%} of cap)<br>"
                "Grade: %{customdata[3]}<br>"
                "Value resid: %{customdata[4]:+.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Cost vs performance",
        xaxis_title="Cost (z of APY cap %)",
        yaxis_title="Performance (z of composite)",
        template="plotly_white",
        height=520,
        legend_title="Value tier",
    )
    st.plotly_chart(fig, width="stretch")

    cols = [
        "player_display_name", "recent_team", "letter_grade",
        "apy", "apy_cap_pct", "value_resid", "value_tier",
    ]
    table = (
        value[cols]
        .sort_values("value_resid", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "player_display_name": "Player", "recent_team": "Team",
            "letter_grade": "Grade", "apy": "APY ($M)",
            "apy_cap_pct": "Cap %", "value_resid": "Value resid",
            "value_tier": "Tier",
        })
    )
    st.dataframe(
        style_grades(table).format(
            {"APY ($M)": "{:.1f}", "Cap %": "{:.1%}", "Value resid": "{:+.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )


def view_compare(perf: pd.DataFrame, value: pd.DataFrame, season: int) -> None:
    st.subheader(f"{season} head-to-head")

    ranked = perf.sort_values("composite_z", ascending=False)
    names = ranked["player_display_name"].tolist()
    default = names[:2]
    picked = st.multiselect("Quarterbacks (pick 2+)", names, default=default)
    if len(picked) < 2:
        st.info("Pick at least two quarterbacks to compare.")
        return

    sub = perf[perf["player_display_name"].isin(picked)]

    # Grouped bar of the six components, one color per QB (categorical identity).
    palette = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
    labels = list(COMPONENTS.values())
    fig = go.Figure()
    for i, name in enumerate(picked):
        r = sub[sub["player_display_name"] == name].iloc[0]
        fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=[r[c] for c in COMPONENTS],
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(
        title="Component z-scores (higher = better)",
        barmode="group",
        xaxis_title="Component",
        yaxis_title="z-score",
        template="plotly_white",
        height=460,
        legend_title="Quarterback",
    )
    st.plotly_chart(fig, width="stretch")

    # Summary table joining grade + value facts.
    merged = sub.merge(
        value[["player_id", "value_tier", "apy", "apy_cap_pct", "value_resid"]],
        on="player_id",
        how="left",
    )
    cols = [
        "player_display_name", "recent_team", "letter_grade", "score_0_100",
        "composite_z", "apy", "apy_cap_pct", "value_resid", "value_tier",
    ]
    table = (
        merged[cols]
        .sort_values("composite_z", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "player_display_name": "Player", "recent_team": "Team",
            "letter_grade": "Grade", "score_0_100": "Score",
            "composite_z": "Composite z", "apy": "APY ($M)",
            "apy_cap_pct": "Cap %", "value_resid": "Value resid",
            "value_tier": "Tier",
        })
    )
    st.dataframe(
        style_grades(table).format({
            "Score": "{:.1f}", "Composite z": "{:+.2f}", "APY ($M)": "{:.1f}",
            "Cap %": "{:.1%}", "Value resid": "{:+.2f}",
        }),
        width="stretch",
        hide_index=True,
    )


# --- App shell -------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="NFL QB Grades", page_icon="🏈", layout="wide")
    st.title("🏈 NFL QB Grades")

    if not DB_FILE.exists():
        st.error(
            f"Database not found at `{DB_FILE}`. Run the pipeline first, e.g. "
            "`python main.py --season 2025`."
        )
        return

    seasons = available_seasons()
    with st.sidebar:
        st.header("Filters")
        season = st.selectbox("Season", seasons, index=0)
        hide_small = st.toggle("Hide small-sample QBs", value=False)

    perf, value = season_frames(season)

    teams = sorted(perf["recent_team"].dropna().unique())
    with st.sidebar:
        picked_teams = st.multiselect("Teams", teams, default=[])
    if picked_teams:
        perf = perf[perf["recent_team"].isin(picked_teams)]
        value = value[value["recent_team"].isin(picked_teams)]
    if hide_small and "small_sample" in perf.columns:
        keep = perf.loc[~perf["small_sample"].astype(bool), "player_id"]
        perf = perf[perf["player_id"].isin(keep)]
        value = value[value["player_id"].isin(keep)]

    if perf.empty:
        st.warning("No quarterbacks match the current filters.")
        return

    tabs = st.tabs(["Overview", "Performance", "Value", "Compare"])
    with tabs[0]:
        view_overview(perf, value, season)
    with tabs[1]:
        view_performance(perf, season)
    with tabs[2]:
        if value.empty:
            st.warning("No contract-value rows match the current filters.")
        else:
            view_value(value, season)
    with tabs[3]:
        view_compare(perf, value, season)

    st.caption(
        "Grades are pool-relative within each season and are not comparable across "
        "seasons. Money figures are in millions of dollars."
    )


if __name__ == "__main__":
    main()
