import polars as pl
import sqlalchemy as sa
import json



DB_PATH = "nfl_qb.db"

def write_to_sqlite(df: pl.DataFrame, table_name: str, db_path: str = DB_PATH) -> None:
    """
    Write a Polars DataFrame to a SQLite table, replacing it if it exists.

    List/Struct columns (e.g. the nested contract history from nflreadpy) can't
    be stored natively in SQLite, so they're JSON-encoded to text first.
    """
    pdf = df.to_pandas()

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

    engine = sa.create_engine(f"sqlite:///{db_path}")
    pdf.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Wrote {len(pdf)} rows to {db_path}::{table_name}")