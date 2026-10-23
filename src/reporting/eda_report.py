"""Gráficos y resumen del EDA de regresión, calculados solo sobre train."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TARGET = "owners_midpoint"
NON_FEATURES = {"appid", "name", TARGET, "owners_log", "price_log"}
CORR_COLUMNS = ["price", "required_age", "n_languages", "n_audio_languages",
                "release_year", "n_genres", "n_categories", "n_tags"]
SAMPLE_SIZE = 3000
RANDOM_STATE = 42


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path.name}", flush=True)


def _iqr_outliers(series: pd.Series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return low, high, int(((series < low) | (series > high)).sum())


class EDAReport:
    """Genera las figuras y un resumen numérico para redactar los insights."""

    def generate(self, train_file: Path, output_dir: Path,
                 reviews_file: Path | None = None) -> None:
        sns.set_theme(style="whitegrid")
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(train_file, dtype={"appid": str},
                         na_values=[""], keep_default_na=False)
        df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
        df = pd.concat([
            df,
            pd.DataFrame({
                "owners_log": np.log1p(df[TARGET]),
                "price_log": np.log1p(df["price"].clip(lower=0)),
            }, index=df.index),
        ], axis=1)
        features = [c for c in df.columns if c not in NON_FEATURES]
        print(f"EDA sobre train: {len(df):,} filas, {len(features)} features", flush=True)

        # 1. Distribución univariada
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df, x="owners_log", bins=50, kde=True, ax=ax)
        ax.set_xlabel("ln(1 + punto medio de propietarios estimados)")
        ax.set_title("Distribución del target de regresión (train)")
        _save(fig, output_dir / "01_target.png")

        p99 = df["price"].quantile(0.99)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(df.loc[df["price"] <= p99, "price"], bins=40, kde=True, ax=axes[0])
        axes[0].set_title(f"price (recortado a p99 = {p99:.2f}, solo visual)")
        sns.histplot(df["price_log"], bins=40, kde=True, ax=axes[1])
        axes[1].set_title("log(1 + price)")
        sns.boxplot(y=df["price"], ax=axes[2])
        axes[2].set_title("price (boxplot)")
        _save(fig, output_dir / "02_price.png")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, col in zip(axes, ["n_languages", "n_audio_languages", "n_tags"]):
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(col)
        _save(fig, output_dir / "03_boxplots.png")

        missing = df[features].isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=missing.values, y=missing.index, ax=ax)
            ax.set_title("Valores faltantes por columna")
            _save(fig, output_dir / "04_faltantes.png")
        else:
            print("  (sin valores faltantes: no se genera 04_faltantes.png)")

        # 2. Relación de variables con el target continuo
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col in zip(axes, ["price_log", "n_languages", "n_tags"]):
            sns.scatterplot(data=df, x=col, y="owners_log", alpha=0.35,
                            s=12, ax=ax)
            ax.set_ylabel("ln(1 + punto medio de propietarios)")
            ax.set_title(f"{col} vs. target")
        _save(fig, output_dir / "05_target_por_features.png")

        by_year = df.groupby("release_year")[TARGET].agg(["median", "size"])
        by_year = by_year[by_year["size"] >= 30]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x=by_year.index, y=np.log1p(by_year["median"]),
                     marker="o", ax=ax)
        ax.set_ylabel("ln(1 + mediana del punto medio de propietarios)")
        ax.set_title("Target por año de lanzamiento")
        _save(fig, output_dir / "06_target_por_anio.png")

        genre_cols = [c for c in df.columns if c.startswith("genre_")]
        genres = pd.DataFrame({
            "juegos": df[genre_cols].sum(),
            "mediana_owners_midpoint": [
                df.loc[df[c] == 1, TARGET].median() for c in genre_cols
            ],
        })
        genres.index = [c.removeprefix("genre_") for c in genre_cols]
        top = genres.nlargest(15, "juegos")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(x=top["juegos"], y=top.index, ax=axes[0])
        axes[0].set_title("Juegos por género (top 15)")
        sns.barplot(x=np.log1p(top["mediana_owners_midpoint"]),
                    y=top.index, ax=axes[1])
        axes[1].set_xlabel("ln(1 + mediana del punto medio de propietarios)")
        axes[1].set_title("Target mediano por género")
        _save(fig, output_dir / "07_target_por_genero.png")

        sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=sample, x="n_languages", y="owners_log",
                        hue="price_log", palette="viridis", alpha=0.5, ax=ax)
        ax.set_ylabel("ln(1 + punto medio de propietarios)")
        ax.set_title("Target vs idiomas (color = precio log)")
        _save(fig, output_dir / "08_scatter_target_idiomas.png")

        # 3. Multivariado
        for method in ("pearson", "spearman"):
            corr = df[CORR_COLUMNS + [TARGET]].corr(method=method)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, vmin=-1, vmax=1, ax=ax)
            ax.set_title(f"Correlación de {method.capitalize()} (triangular)")
            _save(fig, output_dir / f"09_corr_{method}.png")

        with_target = df[features].corrwith(df[TARGET], method="spearman").dropna()
        top_corr = with_target.reindex(
            with_target.abs().sort_values(ascending=False).index).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax)
        ax.set_title("Correlación de Spearman con el target (top 15)")
        _save(fig, output_dir / "10_corr_con_target.png")

        corr_all = df[features].corr()
        upper = corr_all.where(np.triu(np.ones(corr_all.shape, dtype=bool), k=1))
        pairs = upper.stack().rename("correlacion").reset_index()
        pairs.columns = ["variable_1", "variable_2", "correlacion"]
        pairs = pairs[pairs["correlacion"].abs() >= 0.8].sort_values(
            "correlacion", key=abs, ascending=False)
        pairs.to_csv(output_dir / "pares_correlacionados.csv", index=False)
        print("  pares_correlacionados.csv", flush=True)

        pair_data = sample[["price_log", "n_languages", "n_tags",
                            "release_year", "owners_log"]].dropna().head(2000)
        grid = sns.pairplot(pair_data, corner=True,
                            plot_kws={"alpha": 0.4, "s": 10})
        grid.savefig(output_dir / "11_pairplot.png", dpi=100)
        plt.close("all")
        print("  11_pairplot.png", flush=True)

        # 4. Reseñas (texto), si el archivo limpio existe
        if reviews_file is not None and reviews_file.exists():
            reviews = pd.read_csv(reviews_file, usecols=["voted_up", "word_count"])
            limit = reviews["word_count"].quantile(0.99)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            sns.countplot(x=reviews["voted_up"], ax=axes[0])
            axes[0].set_title("Reseñas positivas (True) vs negativas (False)")
            sns.histplot(reviews.loc[reviews["word_count"] <= limit, "word_count"],
                         bins=50, ax=axes[1])
            axes[1].set_title(f"Palabras por reseña (recortado a p99 = {limit:.0f})")
            _save(fig, output_dir / "12_resenas.png")
        else:
            print("  (sin archivo de reseñas limpio: se omite 12_resenas.png)")

        self._write_summary(df, features, missing, genres, with_target, pairs, output_dir)

    def _write_summary(self, df, features, missing, genres, with_target, pairs, out):
        numeric = ["price", "required_age", "n_languages", "n_audio_languages",
                   "release_year", "n_genres", "n_categories", "n_tags"]
        lines = ["# Resumen del EDA (train)", "",
                 f"- Filas: {len(df):,}. Features: {len(features)}.",
                 f"- Target (`{TARGET}`):", "```",
                 df[TARGET].describe(
                     percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]
                 ).round(2).to_string(), "```",
                 f"- Columnas con faltantes: {len(missing)}.", "",
                 "## Estadísticos", "", "```",
                 df[numeric].describe().round(2).to_string(), "```", "",
                 "## Asimetría (skew)", "", "```",
                 df[numeric].skew().round(2).to_string(), "```", "",
                 "## Outliers por IQR (informativo; los límites finales se "
                 "calculan con train en el preprocesamiento)", ""]
        for col in ["price", "n_languages", "n_audio_languages", "n_tags"]:
            low, high, n = _iqr_outliers(df[col].dropna())
            lines.append(f"- `{col}`: límites [{low:.2f}, {high:.2f}] -> "
                         f"{n:,} fuera ({n / len(df):.1%}).")
        expensive = df.nlargest(10, "price")[["appid", "name", "price"]]
        lines += ["", "## 10 precios más altos (investigar: ¿error o caso real?)", "",
                  "```", expensive.to_string(index=False), "```", "",
                  "## Géneros: juegos y mediana del target", "", "```",
                  genres.sort_values("juegos", ascending=False).round(3).to_string(), "```", "",
                  "## Top 15 correlaciones (Spearman) con el target", "", "```",
                  with_target.reindex(with_target.abs().sort_values(
                      ascending=False).index).head(15).round(3).to_string(), "```", "",
                  f"## Pares de variables con |r| >= 0.8: {len(pairs)}", "", "```",
                  pairs.head(15).round(3).to_string(index=False), "```", "",
                  "## Insights (completar uno por figura)", ""]
        lines += [f"- {name}: " for name in [
            "01_target", "02_price", "03_boxplots", "04_faltantes",
            "05_target_por_features", "06_target_por_anio", "07_target_por_genero",
            "08_scatter_target_idiomas", "09_corr_pearson/spearman",
            "10_corr_con_target", "11_pairplot", "12_resenas"]]
        (out / "resumen_eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("  resumen_eda.md", flush=True)