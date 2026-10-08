"""Configuración de la limpieza de juegos y reseñas."""

import csv
import re

csv.field_size_limit(10_000_000)

MONTHS = dict(
    zip(
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(),
        range(1, 13),
    )
)

# Repara únicamente el caso conocido de K'iche' dentro de listas mal formadas.
KICHE = re.compile(r"(?<=[,\[])\s*K'iche'(?=\s*[,\]])")

CONFIG = {
    "games": {
        "file": "games_fixed.csv",
        "drop": {"movies"},
        "integers": set(
            "appid peak_ccu required_age discount dlc_count metacritic_score "
            "positive negative score_rank achievements recommendations "
            "average_playtime_forever average_playtime_two_weeks "
            "median_playtime_forever median_playtime_two_weeks".split()
        ),
        "decimals": {"price", "user_score"},
        "booleans": {"windows", "mac", "linux"},
        "lists": {"supported_languages", "full_audio_languages"},
    },
    "reviews": {
        "file": "steam_game_reviews.csv",
        "drop": {"release_date"},
        "integers": set(
            "appid word_count votes_up votes_funny timestamp_created "
            "author_playtime_forever".split()
        ),
        "decimals": {"price"},
        "booleans": {"voted_up"},
        "lists": set(),
    },
}
