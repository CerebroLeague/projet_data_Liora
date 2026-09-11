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

# ── Charte graphique « Liora » (fond blanc · titres Lora serif · accent orange) ──
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300..700&family=Lora:wght@500;600;700&display=swap');
      html, body, [class*="css"], .stMarkdown, p, span, label, li { font-family:'DM Sans','Segoe UI',system-ui,sans-serif; }
      h1,h2,h3,h4,h5 { font-family:'Lora',Georgia,serif !important; color:#1a1a1a !important; letter-spacing:-.01em; }
      .liora-mark { font-family:'Lora',serif; font-weight:700; color:#d9480f; font-size:1.35rem; margin:0 0 -.2rem; }
      .liora-mark sup { font-size:.5rem; vertical-align:super; font-weight:500; }
      .liora-note { font-size:.86rem; padding:11px 15px; border-radius:6px; margin:.4rem 0 .2rem; }
      .liora-note.ok   { background:#ebfbee; border:1px solid #c3e6cb; color:#2b6b34; }
      .liora-note.warn { background:#fff4e6; border:1px solid #ffe0bf; color:#8a4b12; }
      .stButton>button[kind="primary"] { border-radius:6px; font-weight:600; }
      div[data-testid="stMetricValue"] { font-family:'Lora',serif; color:#d9480f; }
      hr { border-color:#e6e4e1; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    return "#2f9e44" if s >= 0.3 else "#e8590c" if s >= -0.2 else "#e03131"


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
st.markdown('<p class="liora-mark">Liora<sup>®</sup></p>', unsafe_allow_html=True)
st.title("Quiz IA — Studio")
st.caption("Génération et scoring d'idées de quiz vidéo virales bilingues (FR/EN) — démonstration")
if USING_LLM:
    st.markdown('<div class="liora-note ok">LLM connecté — génération réelle.</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="liora-note warn">Mode démo (sans clé LLM) : les idées sont générées en repli à '
        'partir du RAG ; le scoring, lui, reste réel.</div>', unsafe_allow_html=True)

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
