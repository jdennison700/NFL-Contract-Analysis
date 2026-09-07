# NFL QB Grading

Grades NFL quarterback seasons two ways:

1. **Performance grade** (`A+` … `F`) — how well the QB actually played, from
   advanced per-dropback efficiency plus a rushing term.
2. **Contract-value grade** (`Massive surplus` … `Albatross`) — how that
   performance compares to what the QB is paid.

Data comes from [`nflreadpy`](https://github.com/nflverse/nflreadpy) (nflverse) and
is staged in a local SQLite database (`nfl_qb.db`). Everything is computed with
[Polars](https://pola.rs/).

---

## 1. Pipeline

`main.py` runs the whole thing end to end:

```bash
python main.py                 # full 2025 run: load -> grade -> write -> report
python main.py --season 2024   # a different season
python main.py --skip-load     # reuse data already in nfl_qb.db (no nflverse pull)
python main.py --min-attempts 150   # widen the qualifying pool
```

| Step | Code | Result |
|---|---|---|
| Load season stats + contracts | `load_qb_stats`, `load_player_contracts` (`lib/load_qb_data.py`) | in-memory frames (not staged) |
| Grade on-field performance, splice onto the stats | `grade_qb_performance` + `join_stats_and_grades` (`lib/grade_qb.py`) | `qb_performance` table |
| Grade performance vs contract value, splice onto contracts | `grade_qb_value` + `join_contracts_and_value` (`lib/grade_qb.py`) | `qb_value` table |

The pipeline writes exactly **two tables**: `qb_performance` (every `qb_stats`
column plus the performance-grade columns) and `qb_value` (every contract column
plus the contract-value-grade columns). Both carry a `season` column and span
every season graded so far.

The season is variable everywhere. `main.py` defaults to `DEFAULT_SEASON`
(currently 2025) — override with `--season` on the CLI or `run(season=...)`.
Re-running a season **upserts**: `upsert_season` deletes that season's rows and
re-inserts them, leaving other seasons untouched, so 2020–2025 can be built up one
run at a time. `--skip-load` re-grades a season straight from the rows already in
`qb_performance` / `qb_value` (no nflverse pull).

`main.run(season=DEFAULT_SEASON, skip_load=False, min_attempts=200)` is importable
and returns `{"qb_performance": ..., "qb_value": ...}`.

Each module also has its own `__main__` (with a local `SEASON = 2025`):
`python -m lib.load_qb_data` smoke-tests the nflverse pull; `python -m lib.grade_qb`
re-grades that season from the rows already in `qb_performance` / `qb_value`.

### 1.1 `load_qb_data.py`

- **`load_qb_stats(seasons, summary_level="reg")`** — `nfl.load_player_stats`
  filtered to `position == "QB"`, one row per QB for the season. Receiving,
  defensive, kicking, punting and return columns are dropped (no QB signal).
- **`load_player_contracts(season)`** — `nfl.load_contracts` filtered to QBs,
  reduced to **the one deal that governed each QB's `season`**:
  - keep contracts whose `season_history` has a `season` entry with
    `cap_number > 0` (an actual cap charge, i.e. the deal was really on the books
    that year);
  - `year_signed <= season` (a later extension still lists `season` in its
    duplicated history — exclude it);
  - if a QB still has more than one, take the most recently signed.
  - `gsis_id` is renamed to `player_id` so it joins to `qb_stats` / the grades.

### 1.2 Key columns used downstream

From the loaded stats frame (all of which land in `qb_performance`): `player_id`,
`player_display_name`, `recent_team`, `games`, `attempts`, `sacks_suffered`,
`passing_epa`, `passing_cpoe`, `passing_interceptions`, `fumbles_lost_total`,
`passing_first_downs`, `rushing_epa`.

From the loaded contracts frame (all of which land in `qb_value`): `player_id`,
`apy`, `apy_cap_pct`, `year_signed`, `years`.

---

## 2. Performance grade — the maths

Implemented in `grade_qb_performance(qb_stats, min_attempts=200, weights=None)`.

### 2.1 Qualifying pool

Only QBs with `attempts >= min_attempts` (default **200**, roughly half a season as
a primary starter) are graded. Everything below — means, standard deviations,
z-scores — is computed **within this pool**, so a grade is always relative to the
other starters that season, not to history. In 2025 this yields **36** QBs.

A `small_sample` flag marks `attempts < 300` (graded, but treat with caution).

### 2.2 Component metrics

Let

```
dropbacks = attempts + sacks_suffered
```

| Component (`name`) | Formula | Better when | Meaning |
|---|---|---|---|
| `epa_per_db`  | `passing_epa / dropbacks`                              | higher | expected points added per pass play — the core efficiency signal |
| `cpoe`        | `passing_cpoe` (already per-attempt, from nflverse)    | higher | completion % over expected given throw difficulty (accuracy) |
| `sack_rate`   | `sacks_suffered / dropbacks`                           | lower  | pocket management / pressure handling |
| `to_rate`     | `(passing_interceptions + fumbles_lost_total) / dropbacks` | lower | giveaways per play (picks **and** lost fumbles) |
| `fd_rate`     | `passing_first_downs / attempts`                       | higher | moving the chains through the air |
| `rush_epa_pg` | `rushing_epa / games`                                  | higher | rushing value added, per game (credits mobile QBs) |

### 2.3 Standardise each component

For component *x* with pool mean `μ_x` and sample standard deviation `σ_x`:

```
z_x = clip( (x - μ_x) / σ_x , -3, +3 )
```

- **Winsorising at ±3 SD** stops a single historic outlier (e.g. a 46-passing-TD
  season) from dominating the composite.
- For the two "lower is better" components (`sack_rate`, `to_rate`) the sign is
  **flipped** after standardising, so for every component *higher `z` = better*.

### 2.4 Weighted composite

```
composite_z = Σ_x  w_x · z_x
```

Default weights (`_DEFAULT_WEIGHTS`), chosen to lean on efficiency:

| Component | Weight |
|---|---|
| `epa_per_db`  | 0.35 |
| `cpoe`        | 0.18 |
| `to_rate`     | 0.17 |
| `sack_rate`   | 0.10 |
| `fd_rate`     | 0.10 |
| `rush_epa_pg` | 0.10 |

Pass `weights={...}` to override any subset; the full set is then **re-normalised
to sum to 1**, so the composite stays on a comparable scale. (e.g.
`weights={"rush_epa_pg": 0.0}` grades passing only.)

`composite_z` is itself a blend of z-scores, so it is centred near 0 with a spread
somewhat below 1 SD.

### 2.5 Readable score

```
score_0_100 = clip( 50 + 15 · composite_z , 0, 100 )
```

Purely cosmetic — a 0–100 restatement of `composite_z` for sorting/inspection.

### 2.6 Letter grade

Absolute cutoffs on `composite_z`, checked high → low (`_GRADE_CUTS`). This is
**not a curve**: a weak QB class simply produces fewer `A`s.

| Grade | `composite_z` |
|---|---|
| A+ | ≥ 1.50 |
| A  | 1.00 – 1.50 |
| A− | 0.70 – 1.00 |
| B+ | 0.40 – 0.70 |
| B  | 0.15 – 0.40 |
| B− | −0.15 – 0.15 |
| C+ | −0.40 – −0.15 |
| C  | −0.70 – −0.40 |
| C− | −1.00 – −0.70 |
| D  | −1.50 – −1.00 |
| F  | < −1.50 |

### 2.7 Output — `qb_performance` table

`grade_qb_performance` returns the grade columns `player_id`,
`player_display_name`, `recent_team`, `games`, `attempts`, `small_sample`,
`z_epa_per_db`, `z_cpoe`, `z_sack_rate`, `z_to_rate`, `z_fd_rate`,
`z_rush_epa_pg`, `composite_z`, `score_0_100`, `letter_grade` — sorted by
`composite_z` descending.

`join_stats_and_grades` then inner-joins that onto the full stats frame on
`player_id` (so only graded QBs survive) and `main.py` adds a `season` column
before the upsert. The stored `qb_performance` table therefore holds **every
`qb_stats` column** followed by `small_sample`, the six `z_*` columns,
`composite_z`, `score_0_100`, `letter_grade`, `season`.

---

## 3. Contract-value grade — the maths

Implemented in `grade_qb_value(perf_grades, contracts)`. Question answered: *did the
QB play better or worse than his pay grade?*

### 3.1 Join

Inner join `perf_grades` to `contracts` on `player_id`. QBs with no matching
contract row are dropped with a printed warning (0 of 36 in 2025).

### 3.2 The two axes, standardised over the joined pool

**Performance:**

```
perf_z = (composite_z - mean(composite_z)) / std(composite_z)
```

(re-standardised here so it is a clean unit-SD variable within the contract pool).

**Cost** — `apy_cap_pct` = average-per-year contract value as a share of that
season's salary cap. Chosen over raw `apy` because it is comparable across signing
years (a \$45M deal signed in 2020 ate far more cap than one signed in 2025):

```
cost_z = (apy_cap_pct - mean(apy_cap_pct)) / std(apy_cap_pct)
```

### 3.3 Fit the market line (ordinary least squares)

Regress performance on cost across the pool — the "market expectation" of how a QB
at a given pay level performs:

```
slope      b = Σ (cost_z - mean(cost_z)) · (perf_z - mean(perf_z))
               ───────────────────────────────────────────────────
                        Σ (cost_z - mean(cost_z))²

intercept  a = mean(perf_z) - b · mean(cost_z)

expected_perf_z = a + b · cost_z
```

### 3.4 Value residual

```
value_resid = perf_z - expected_perf_z
```

How many standard deviations of performance a QB sits **above** (beating his
contract) or **below** (not earning it) the pay-predicted line.

### 3.5 Value tiers

Absolute cutoffs on `value_resid`, high → low (`_VALUE_TIERS`):

| Tier | `value_resid` |
|---|---|
| Massive surplus | ≥ 1.25 |
| Bargain | 0.50 – 1.25 |
| Fair | −0.50 – 0.50 |
| Overpaid | −1.25 – −0.50 |
| Albatross | < −1.25 |

### 3.6 Output — `qb_value` table

`grade_qb_value` returns `player_id`, `player_display_name`, `recent_team`,
`letter_grade` (performance, for context), `composite_z`, `apy`, `apy_cap_pct`,
`year_signed`, `years`, `perf_z`, `cost_z`, `expected_perf_z`, `value_resid`,
`value_tier` — sorted by `value_resid` descending.

`join_contracts_and_value` then appends any remaining contract columns (`team`,
`value`, `guaranteed`, the `inflated_*` figures, …) by joining back to the
contracts frame on `player_id`, and `main.py` adds a `season` column before the
upsert.

### 3.7 How to read it (2025 fit)

The regression is weak: **slope `b` ≈ 0.40, R² ≈ 0.16** — pay explains only ~16% of
performance variance. Consequences:

- `expected_perf_z` is small for everyone, so `value_resid ≈ perf_z` with a modest
  tilt: expensive QBs are held to a slightly higher bar, cheap QBs to a slightly
  lower one.
- The tiers reliably flag **expensive underperformers** as `Overpaid` /
  `Albatross` (this is the signal to trust).
- A cheap QB who simply played badly (rookie on a minimum deal) can still land in
  `Albatross` — read that as "bad play", not "bad contract".

---

## 4. Tuning knobs

| Knob | Where | Effect |
|---|---|---|
| `min_attempts` | `grade_qb_performance` arg | who qualifies; also shifts the pool mean/SD every z-score is measured against |
| `weights` | `grade_qb_performance` arg | component emphasis (auto-renormalised); e.g. zero `rush_epa_pg` for passing-only |
| `_DEFAULT_WEIGHTS` | `lib/grade_qb.py` | default component emphasis |
| `_GRADE_CUTS` | `lib/grade_qb.py` | performance letter thresholds |
| `_VALUE_TIERS` | `lib/grade_qb.py` | value tier thresholds |
| winsor limit `±3` | `grade_qb_performance` | outlier clipping severity |
| cost metric | `grade_qb_value` (`apy_cap_pct`) | swap for `apy` / `inflated_apy` to change what "expensive" means |

---

## 5. Caveats

- **Single season, no opponent adjustment.** Grades reflect what happened, not
  context (schedule, supporting cast, weather, scheme).
- **Pool-relative.** Every z-score is measured against that season's qualifying
  starters, so grades are not directly comparable across seasons — even though
  `qb_performance` / `qb_value` now stack every season in one table, filter to a
  single `season` before ranking.
- **`summary_level="reg"`** — regular season only. Pass `"post"` to
  `load_qb_stats` for playoffs.
- **Contract snapshot** is the deal in force during 2025; restructures and
  extensions signed later are not reflected.
- Small OLS pool (36) — the market line is indicative, not precise.

---

## 6. Project layout

```
main.py             end-to-end pipeline + CLI (--season, --skip-load, --min-attempts)
lib/
  load_qb_data.py   data pull + contract reduction
  grade_qb.py       grade_qb_performance, grade_qb_value, join_stats_and_grades, join_contracts_and_value
  sqlite.py         write_to_sqlite / upsert_season helpers (JSON-encode nested columns)
nfl_qb.db           local SQLite store (git-ignored) -- holds qb_performance + qb_value
```

Requires Python ≥ 3.14. Dependencies in `pyproject.toml` (`uv sync`).
