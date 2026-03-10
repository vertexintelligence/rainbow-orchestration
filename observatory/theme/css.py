"""
GENX Observatory CSS

Returns the unified V4.2 glass panel stylesheet.
"""

import streamlit as st


def inject_stylesheet() -> None:
    """Inject the GENX Observatory V4.2 stylesheet into the Streamlit page."""
    st.markdown(_STYLESHEET, unsafe_allow_html=True)


_STYLESHEET = """
<style>
/* ======================================================
   GENX OBSERVATORY V4.2 — UNIFIED GLASS PANEL SYSTEM
   ====================================================== */

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(0,255,200,0.12), transparent 24%),
        radial-gradient(circle at 100% 0%, rgba(255,196,0,0.08), transparent 22%),
        radial-gradient(circle at 50% 100%, rgba(90,120,255,0.10), transparent 28%),
        linear-gradient(180deg, #03101b 0%, #061522 45%, #081726 100%);
    color: #eaf4ff;
}

.block-container {
    max-width: 1520px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    color: #f5fbff !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
}

p, li, label, .stCaption, .stMarkdown {
    color: #9fb4c8 !important;
}

/* ---- Section Divider ---- */
.genx-section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,255,200,0.18), transparent);
    margin: 2rem 0;
}

/* ---- Section Titles — Tier System ---- */
.genx-section-wrap {
    margin-top: 0.25rem;
    margin-bottom: 0.8rem;
}

.genx-section-title {
    color: #f5fbff;
    font-size: 1.55rem;
    font-weight: 700;
    margin-bottom: 0.18rem;
}

.genx-section-caption {
    color: #8da4b9;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* Primary tier — command-level sections */
.genx-section-wrap--primary {
    margin-top: 0.35rem;
    margin-bottom: 1rem;
    padding-left: 0.9rem;
    border-left: 4px solid rgba(0,255,200,0.35);
}

.genx-section-wrap--primary .genx-section-title {
    font-size: 2.1rem;
    font-weight: 850;
}

.genx-section-wrap--primary .genx-section-caption {
    font-size: 1rem;
    color: #9fb4c8;
}

/* ---- Hero ---- */
.genx-hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 1.5rem 1.6rem 1.2rem 1.6rem;
    background:
        linear-gradient(135deg, rgba(0,255,200,0.10), transparent 35%),
        linear-gradient(135deg, rgba(255,196,0,0.08), transparent 62%),
        rgba(7, 18, 30, 0.88);
    border: 1px solid rgba(0,255,200,0.16);
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.02) inset,
        0 24px 60px rgba(0,0,0,0.35);
    margin-bottom: 1.4rem;
}

.genx-hero-title {
    color: #f8fcff;
    font-size: 2.25rem;
    font-weight: 850;
    letter-spacing: 0.01em;
    margin-bottom: 0.3rem;
}

.genx-hero-title span {
    color: #7bf0dc;
}

.genx-hero-subtitle {
    color: #9bbbc4;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* ---- Pills ---- */
.genx-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 0.4rem;
}

.genx-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    font-size: 0.92rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(8,16,24,0.82);
    box-shadow: 0 10px 22px rgba(0,0,0,0.16);
}

.genx-pill-label {
    color: #8ea6bc;
    font-weight: 600;
}

.genx-pill-value {
    color: #f5fbff;
    font-weight: 700;
}

/* ---- Tone System ---- */
.genx-tone-good {
    border-color: rgba(0,255,200,0.22) !important;
    box-shadow: 0 0 18px rgba(0,255,200,0.08);
}

.genx-tone-warn {
    border-color: rgba(255,196,0,0.25) !important;
    box-shadow: 0 0 18px rgba(255,196,0,0.08);
}

.genx-tone-bad {
    border-color: rgba(255,90,90,0.25) !important;
    box-shadow: 0 0 18px rgba(255,90,90,0.08);
}

.genx-tone-neutral {
    border-color: rgba(110,168,255,0.18) !important;
}

/* ---- Expander ---- */
details {
    background: rgba(7,15,25,0.95);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 0.35rem 0.7rem;
}

summary {
    color: #eef7ff !important;
    font-weight: 700 !important;
}

/* ---- Signals / warning ---- */
div[data-baseweb="notification"] {
    border-radius: 18px !important;
    border: 1px solid rgba(255,196,0,0.18) !important;
    background:
        linear-gradient(90deg, rgba(70,52,8,0.95), rgba(48,35,6,0.96)) !important;
    color: #ffe8a9 !important;
    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
}

/* ---- JSON viewer ---- */
div[data-testid="stJson"] {
    background: rgba(8,16,27,0.94);
    border: 1px solid rgba(0,255,200,0.14);
    border-radius: 20px;
    padding: 0.35rem;
    box-shadow: 0 14px 35px rgba(0,0,0,0.24);
}

/* ---- Signal cards ---- */
.genx-signal-card {
    background: linear-gradient(90deg, rgba(70,52,8,0.95), rgba(48,35,6,0.96));
    border: 1px solid rgba(255,196,0,0.18);
    border-radius: 18px;
    color: #ffe8a9;
    padding: 1rem 1.05rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
    font-size: 1rem;
    line-height: 1.5;
}

/* ---- Info cards ---- */
.genx-info-card {
    border-radius: 20px;
    padding: 1rem 0.95rem;
    background: rgba(8,16,27,0.94);
    border: 1px solid rgba(98,224,255,0.14);
    box-shadow: 0 14px 35px rgba(0,0,0,0.22);
    min-height: 120px;
}

.genx-info-card-title {
    color: #7fb7cc;
    font-size: 0.86rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.genx-info-card-body {
    color: #f4fbff;
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1.5;
}

/* ---- Metric cards ---- */
.genx-metric-card {
    background: linear-gradient(180deg, rgba(9,18,30,0.94), rgba(7,16,26,0.98));
    border: 1px solid rgba(0, 224, 255, 0.14);
    border-radius: 22px;
    padding: 1rem;
    min-height: 120px;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.02) inset,
        0 14px 34px rgba(0,0,0,0.24);
    overflow: hidden;
}

.genx-metric-card.compact {
    min-height: 96px;
}

.genx-metric-label {
    color: #7fb7dc;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.genx-metric-value {
    color: #f4fbff;
    font-size: 1.15rem;
    font-weight: 800;
    line-height: 1.3;
    word-break: break-word;
    overflow-wrap: anywhere;
}

/* ---- Glass cards ---- */
.genx-glass-card {
    background: linear-gradient(180deg, rgba(10,18,28,0.82), rgba(7,13,22,0.94));
    border: 1px solid rgba(0, 224, 255, 0.12);
    border-radius: 24px;
    padding: 1rem;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
    margin-bottom: 1rem;
}

.genx-mini-title {
    color: #f4fbff;
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

/* ---- KV Grid (CSS grid layout) ---- */
.genx-kv-grid {
    display: grid;
    gap: 0.6rem;
}

.genx-kv-card {
    background: rgba(11, 23, 38, 0.82);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 0.9rem;
    min-height: 86px;
}

.genx-kv-label {
    color: #7db7dd;
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}

.genx-kv-value {
    color: #f7fbff;
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.35;
    word-break: break-word;
    overflow-wrap: anywhere;
}

/* ---- Chips ---- */
.genx-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.35rem;
}

.genx-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 0.8rem;
    border-radius: 999px;
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    border: 1px solid transparent;
}

.genx-tone-good.genx-chip {
    background: rgba(20,194,120,0.14);
    color: #8ff0bf;
    border-color: rgba(20,194,120,0.30);
}

.genx-tone-warn.genx-chip {
    background: rgba(255,184,0,0.14);
    color: #ffd57a;
    border-color: rgba(255,184,0,0.28);
}

.genx-tone-neutral.genx-chip {
    background: rgba(96,165,250,0.12);
    color: #b7d8ff;
    border-color: rgba(96,165,250,0.22);
}

.genx-muted {
    color: #8aa9bb;
    font-size: 0.95rem;
}

/* ---- Command Rail ---- */
.genx-rail {
    border-radius: 26px;
    padding: 1.2rem;
    background: linear-gradient(180deg, rgba(8,18,30,0.94), rgba(7,14,24,0.96));
    border: 1px solid rgba(0, 255, 200, 0.12);
    box-shadow: 0 18px 40px rgba(0,0,0,0.28);
    margin-bottom: 1.25rem;
}

.genx-rail-title {
    color: #f6fbff;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}

.genx-rail-subtitle {
    color: #8fb2c8;
    font-size: 0.96rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.genx-rail-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
}

.genx-rail-item {
    border-radius: 16px;
    padding: 0.85rem 1rem;
    background: rgba(10, 20, 34, 0.88);
    border: 1px solid rgba(96, 165, 250, 0.14);
    flex: 1 1 140px;
    min-width: 140px;
}

.genx-rail-label {
    color: #7fb6e9;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

.genx-rail-value {
    color: #f4fbff;
    font-size: 1.02rem;
    font-weight: 700;
    line-height: 1.35;
    word-break: break-word;
}

/* ---- Glass panel ---- */
.genx-glass-panel {
    background: linear-gradient(180deg, rgba(10,22,36,0.90), rgba(8,18,30,0.94));
    border: 1px solid rgba(0,255,200,0.12);
    border-radius: 24px;
    padding: 1rem;
    box-shadow: 0 16px 34px rgba(0,0,0,0.24);
    margin-bottom: 1rem;
}

/* ---- Radar grid ---- */
.genx-radar-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.85rem;
}

.genx-radar-card {
    border-radius: 18px;
    padding: 0.9rem 1rem;
    min-height: 96px;
    background: linear-gradient(180deg, rgba(10,20,34,0.96), rgba(8,16,28,0.98));
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
}

.genx-radar-label {
    color: #85b6d8;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}

.genx-radar-value {
    color: #f7fbff;
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1.2;
    word-break: break-word;
}

/* ---- Styled HTML tables ---- */
.genx-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
}

.genx-table-th {
    color: #9fc8e6;
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.75rem 0.85rem;
    text-align: left;
    background: rgba(0,0,0,0.18);
    border-bottom: 2px solid rgba(0,224,255,0.14);
}

.genx-table-td {
    color: #f4fbff;
    font-size: 0.92rem;
    font-weight: 600;
    padding: 0.65rem 0.85rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    line-height: 1.4;
}

.genx-table tbody tr:nth-child(even) .genx-table-td {
    background: rgba(255,255,255,0.02);
}

.genx-table tbody tr:last-child .genx-table-td {
    border-bottom: none;
}

.genx-table tbody tr:hover .genx-table-td {
    background: rgba(96, 165, 250, 0.06);
}

/* Tone-aware table cells */
.genx-table-td--good {
    color: #8ff0bf !important;
}

.genx-table-td--warn {
    color: #ffd57a !important;
}

.genx-table-td--bad {
    color: #ff8a8a !important;
}

/* ---- Responsive ---- */
@media (max-width: 1200px) {
    .genx-radar-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 700px) {
    .genx-radar-grid {
        grid-template-columns: 1fr;
    }
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
