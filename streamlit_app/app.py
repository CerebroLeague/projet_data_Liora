"""
Démo PoC (Streamlit) — Quiz IA.

App multi-onglets pour la soutenance :
  📅 Suggestions du jour · ✨ Génération à la demande · 📈 Explication du score · 📊 DataViz

⚠️ Le modèle est CHARGÉ (jamais ré-entraîné en live) via st.cache_resource.
Lancement :  streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# rendre `src` importable depuis la racine du dépôt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src import taxonomy  # noqa: E402
from src.generator.generate import generate_ideas  # noqa: E402
from src.generator.retriever import Retriever  # noqa: E402

load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Quiz IA — Studio", page_icon="🎬", layout="wide")
FIG = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"


# ── chargement unique (pas de ré-entraînement) ─────────────────
@st.cache_resource(show_spinner="Chargement des modèles…")
def load_engine():
    retriever = Retriever()
    try:
        from src.generator.llm import LLMClient
        import os
        client = LLMClient() if os.getenv("LLM_API_KEY") else None
    except Exception:
        client = None
    return retriever, client


retriever, llm_client = load_engine()
USING_LLM = llm_client is not None


def score_color(s: float) -> str:
    return "#3fb950" if s >= 0.3 else "#e3b341" if s >= -0.2 else "#ff7b72"


def render_ideas(ideas: list[dict]):
    for rank, it in enumerate(ideas, 1):
        fam = taxonomy.family_label(it["family"]) if it.get("family") else "—"
        s = it.get("predicted_score", 0.0)
        c1, c2 = st.columns([0.82, 0.18])
        with c1:
            st.markdown(f"**#{rank} · {it['title']}**")
            st.caption(f"🏷️ {fam}" + (f" — {it.get('description','')}" if it.get("description") else ""))
        with c2:
            st.markdown(
                f"<div style='text-align:center;font-size:1.4rem;font-weight:800;color:{score_color(s)}'>"
                f"{s:+.2f}</div><div style='text-align:center;font-size:.7rem;color:#888'>viralité</div>",
                unsafe_allow_html=True)
        st.divider()


# ── En-tête ────────────────────────────────────────────────────
st.title("🎬 Quiz IA — Studio")
st.caption("Génération & scoring d'idées de quiz vidéo virales bilingues (FR/EN) — démo PoC")
st.info(
    ("🟢 LLM connecté — génération réelle" if USING_LLM
     else "🟡 Mode démo (sans clé LLM) : idées générées en *mock* à partir du RAG ; le scoring est réel."),
    icon="ℹ️")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📅 Suggestions du jour", "✨ Génération à la demande", "📈 Explication du score", "📊 DataViz"])

# ── Onglet 1 — Suggestions du jour ─────────────────────────────
with tab1:
    st.subheader("Menu du jour — 5 FR + 5 EN classées par potentiel viral")
    cat = st.selectbox("Catégorie", taxonomy.CATEGORIES,
                       format_func=lambda c: taxonomy.CATEGORY_LABELS.get(c, c), key="menu_cat")
    if st.button("🍳 Générer le menu du jour", type="primary"):
        cols = st.columns(2)
        for col, lang in zip(cols, taxonomy.LANGUAGES):
            with col:
                st.markdown(f"### {'🇫🇷' if lang=='FR' else '🇬🇧'} {lang}")
                with st.spinner(f"Génération {lang}…"):
                    ideas = generate_ideas(cat, lang, n=5, retriever=retriever,
                                           client=llm_client, mock=not USING_LLM)
                render_ideas(ideas)

# ── Onglet 2 — Génération à la demande ─────────────────────────
with tab2:
    st.subheader("Génération à la demande")
    c1, c2, c3 = st.columns(3)
    cat2 = c1.selectbox("Catégorie", taxonomy.CATEGORIES,
                        format_func=lambda c: taxonomy.CATEGORY_LABELS.get(c, c), key="gen_cat")
    lang2 = c2.selectbox("Langue", taxonomy.LANGUAGES, key="gen_lang")
    n2 = c3.slider("Nombre d'idées", 3, 10, 5)
    theme = st.text_input("Thème (optionnel, recherche sémantique)", "")
    if st.button("✨ Générer", type="primary"):
        with st.spinner("Génération…"):
            ideas = generate_ideas(cat2, lang2, n=n2, theme=theme or None,
                                   retriever=retriever, client=llm_client, mock=not USING_LLM)
        render_ideas(ideas)

# ── Onglet 3 — Explication du score (SHAP) ─────────────────────
with tab3:
    st.subheader("Pourquoi ce score ? (interprétabilité SHAP)")
    metrics_path = REPORTS / "scoring_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        best = m.get("best")
        res = m.get("results", {}).get(best, {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Modèle", best)
        c2.metric("Spearman (test)", res.get("spearman"))
        c3.metric("MAE", res.get("mae"))
    shap_img = FIG / "07_shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="Facteurs de viralité prédite (SHAP)")
    imp = REPORTS / "shap_importance.csv"
    if imp.exists():
        st.dataframe(pd.read_csv(imp).head(15), use_container_width=True)

# ── Onglet 4 — DataViz ─────────────────────────────────────────
with tab4:
    st.subheader("Exploration des données (Rendu 1)")
    figs = sorted(FIG.glob("0[1-6]_*.png"))
    if not figs:
        st.warning("Aucune figure. Lance `python -m src.features.run_features`.")
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i:i + 2]):
            col.image(str(f), caption=f.stem, use_container_width=True)
