"""Prepara train/test para regresión de jugadores sin crear un target binario."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def owners_midpoint(value):
    numbers = [int(x) for x in re.findall(r"\d+", str(value).replace(",", ""))]
    return float(sum(numbers) / len(numbers)) if numbers else np.nan


def list_count(value):
    text = str(value).strip()
    return 0 if not text else text.count(",") + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROCESSED / "games_fixed_clean.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_path, test_path = PROCESSED / "games_train.csv", PROCESSED / "games_test.csv"
    if not args.overwrite and (train_path.exists() or test_path.exists()):
        raise FileExistsError("Ya existen games_train.csv/games_test.csv; usa --overwrite.")

    columns = [
        "appid", "name", "estimated_owners", "price", "required_age",
        "windows", "mac", "linux", "release_date", "supported_languages",
        "full_audio_languages", "categories", "genres", "tags",
    ]
    df = pd.read_csv(args.input, usecols=lambda c: c in columns, dtype=str, keep_default_na=False)
    owners = df["estimated_owners"].map(owners_midpoint)
    df = df.loc[owners.notna()].copy()
    df["target_players"] = owners.loc[df.index]
    date = pd.to_datetime(df["release_date"], errors="coerce")
    out = pd.DataFrame({
        "appid": df["appid"], "name": df["name"],
        "target_players": df["target_players"],
        "target_players_log": np.log1p(df["target_players"]),
        "price": pd.to_numeric(df["price"], errors="coerce"),
        "required_age": pd.to_numeric(df["required_age"], errors="coerce"),
        "windows": df["windows"].eq("true").astype("int8"),
        "mac": df["mac"].eq("true").astype("int8"),
        "linux": df["linux"].eq("true").astype("int8"),
        "release_year": date.dt.year, "release_month": date.dt.month,
    })
    for column in ("supported_languages", "full_audio_languages", "categories", "genres", "tags"):
        out[f"n_{column}"] = df[column].map(list_count).astype("int16")
    out = out.replace([np.inf, -np.inf], np.nan)
    train, test = train_test_split(out, test_size=0.2, random_state=42)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Preparados {len(train):,} train y {len(test):,} test.")


if __name__ == "__main__":
    main()
