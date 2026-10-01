import pandas as pd
from data_profiling import ProfileReport

# Generar reportes de Pandas Profiling para los datasets de reviews y games_fixed.

# PATHS

REVIEWS_FILE = 'data/raw/steam_game_reviews.csv'
FIXED_FILE = 'data/raw/games_fixed.csv'

REPORT_REVIEWS = 'reports/reporte_reviews.html'
REPORT_FIXED = 'reports/reporte_games_fixed.html'


# GENERAR REPORTE DE REVIEWS

print("\nGenerando reporte de reviews...")

df_review = pd.read_csv(REVIEWS_FILE)

print(df_review.iloc[:, :5].head())

profile_reviews = ProfileReport(
    df_review,
    title="Pandas Profiling Report - Reviews",
    explorative=True,
    minimal=True
)

profile_reviews.to_file(REPORT_REVIEWS)

print(f"Reporte generado: {REPORT_REVIEWS}")


# GENERAR REPORTE DE GAMES FIXED

print("\nGenerando reporte de games_fixed.csv...")

df_fixed = pd.read_csv(FIXED_FILE)

profile_fixed = ProfileReport(
    df_fixed,
    title="Pandas Profiling Report - Games Fixed",
    explorative=True,
    minimal=True
)

profile_fixed.to_file(REPORT_FIXED)

print(f"Reporte generado: {REPORT_FIXED}")


print("PROCESO TERMINADO")