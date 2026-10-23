"""Gráficos y resumen del EDA de reseñas, calculados solo sobre train."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

TARGET = "voted_up"
LABELS = {0: "Negativa", 1: "Positiva"}
ORDER = list(LABELS.values())
MIN_GAME_REVIEWS = 30
TOP_TERMS = 15
# Proxy de idioma: reseñas con al menos una de estas palabras inglesas comunes.
ENGLISH_WORDS = r"\b(?:the|and|is|it|this|to|of|you)\b"


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path.name}", flush=True)


class ReviewsEDAReport:
    """Genera las figuras y un resumen numérico para redactar los insights."""

    def generate(self, train_file: Path, output_dir: Path) -> None:
        sns.set_theme(style="whitegrid")
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(train_file, dtype={"appid": str},
                         na_values=[""], keep_default_na=False)
        df["clase"] = df[TARGET].map(LABELS)
        print(f"EDA sobre train: {len(df):,} reseñas", flush=True)

        # 1. Target
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="clase", order=ORDER, ax=ax)
        ax.set_title("Distribución del target (train)")
        _save(fig, output_dir / "01_target.png")

        # 2. Longitud del texto
        p99 = df["word_count"].quantile(0.99)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data=df[df["word_count"] <= p99], x="word_count", hue="clase",
                     hue_order=ORDER, bins=50, element="step", ax=axes[0])
        axes[0].set_title(f"Palabras por reseña (recortado a p99 = {p99:.0f}, solo visual)")
        sns.boxplot(data=df, x="clase", y="word_count", order=ORDER, ax=axes[1])
        axes[1].set_yscale("log")
        axes[1].set_title("word_count por clase (escala log)")
        _save(fig, output_dir / "02_longitud.png")

        # 3. Sentimiento por juego
        per_game = df.groupby("appid").agg(n_resenas=("review", "size"),
                                           positivas=(TARGET, "mean"))
        per_game = per_game[per_game["n_resenas"] >= MIN_GAME_REVIEWS]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(per_game["positivas"], bins=30, ax=ax)
        ax.axvline(0.5, color="gray", linestyle="--")
        ax.set_title(f"Proporción de positivas por juego (>= {MIN_GAME_REVIEWS} reseñas)")
        _save(fig, output_dir / "03_positivas_por_juego.png")

        # 4. Términos más frecuentes por clase (aprendidos solo con train)
        vectorizer = CountVectorizer(stop_words="english", binary=True,
                                     min_df=50, max_features=20_000)
        matrix = vectorizer.fit_transform(df["review"])
        terms = vectorizer.get_feature_names_out()
        freq = {}
        for value, label in LABELS.items():
            mask = (df[TARGET] == value).to_numpy()
            freq[label] = matrix[mask].sum(axis=0).A1 / mask.sum()
        freq = pd.DataFrame(freq, index=terms)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, label in zip(axes, ORDER):
            top = freq[label].nlargest(TOP_TERMS)
            sns.barplot(x=top.values, y=top.index, ax=ax)
            ax.set_title(f"Términos más frecuentes: {label} (proporción de reseñas)")
            ax.set_ylabel("")
        _save(fig, output_dir / "04_terminos.png")

        self._write_summary(df, freq, per_game, output_dir)

    def _write_summary(self, df, freq, per_game, out):
        by_class = df.groupby("clase")["word_count"].describe(
            percentiles=[0.5, 0.95, 0.99]).round(1)
        ratio = np.log((freq["Positiva"] + 1e-4) / (freq["Negativa"] + 1e-4))
        english = df["review"].str.contains(ENGLISH_WORDS, case=False, regex=True).mean()
        spearman = df["word_count"].corr(df[TARGET], method="spearman")

        lines = [
            "# Resumen del EDA de reseñas (train)", "",
            f"- Filas: {len(df):,}. Juegos distintos: {df['appid'].nunique():,}.",
            "- Distribución del target: "
            + ", ".join(f"{k}={v:,}" for k, v in df["clase"].value_counts().items()),
            f"- Reseñas con alguna palabra inglesa común (proxy de idioma): {english:.1%}.",
            f"- Reseñas con más de 256 palabras: {(df['word_count'] > 256).mean():.1%}; "
            f"con más de 512: {(df['word_count'] > 512).mean():.1%} (se truncan en BERT).",
            f"- Spearman entre word_count y target: {spearman:.3f}.", "",
            "## word_count por clase", "", "```", by_class.to_string(), "```", "",
            f"## Reseñas por juego (juegos con >= {MIN_GAME_REVIEWS} reseñas)", "",
            "```", per_game.describe().round(3).to_string(), "```", "",
            "## Términos más propios de cada clase (log-ratio de frecuencias)", "",
            "Positivas:", "```", ratio.nlargest(TOP_TERMS).round(2).to_string(), "```", "",
            "Negativas:", "```", ratio.nsmallest(TOP_TERMS).round(2).to_string(), "```", "",
            "## Insights (completar uno por figura)", "",
        ]
        lines += [f"- {name}: " for name in [
            "01_target", "02_longitud", "03_positivas_por_juego", "04_terminos"]]
        (out / "resumen_eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("  resumen_eda.md", flush=True)