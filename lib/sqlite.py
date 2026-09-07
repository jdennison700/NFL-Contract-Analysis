import polars as pl
import sqlalchemy as sa
import json



DB_PATH = "nfl_qb.db"


def _encode_nested(pdf):
    """
    JSON-encode any list/dict/ndarray-valued columns to text in place.

    List/Struct columns (e.g. the nested contract history from nflreadpy) can't
    be stored natively in SQLite.
    """
    for col in pdf.columns:
        is_nested = pdf[col].map(
            lambda v: isinstance(v, (list, dict)) or hasattr(v, "tolist")
        ).any()
        if is_nested:
            pdf[col] = pdf[col].map(
                lambda v: json.dumps(
                    v.tolist() if hasattr(v, "tolist") else v, default=str
                )
            )
    return pdf


def write_to_sqlite(df: pl.DataFrame, table_name: str, db_path: str = DB_PATH) -> None:
    """Write a Polars DataFrame to a SQLite table, replacing it if it exists."""
    pdf = _encode_nested(df.to_pandas())

    engine = sa.create_engine(f"sqlite:///{db_path}")
    pdf.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Wrote {len(pdf)} rows to {db_path}::{table_name}")


def upsert_season(
    df: pl.DataFrame, table_name: str, season: int, db_path: str = DB_PATH
) -> None:
    """
    Replace all rows for ``season`` in ``table_name`` with ``df``, leaving rows
    for other seasons untouched. Creates the table if it does not exist.

    ``df`` must already carry a ``season`` column.
    """
    pdf = _encode_nested(df.to_pandas())

    engine = sa.create_engine(f"sqlite:///{db_path}")
    table_exists = sa.inspect(engine).has_table(table_name)
    with engine.begin() as conn:
        if table_exists:
            conn.execute(
                sa.text(f'DELETE FROM "{table_name}" WHERE season = :s'), {"s": season}
            )
        pdf.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"Upserted {len(pdf)} rows into {db_path}::{table_name} for season {season}")
