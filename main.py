import io, os, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ──────────────────────────────────────────────
# CSV 읽기 (인코딩 이중 시도)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv(file):
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()

# ──────────────────────────────────────────────
# 열 이름 표준화 (KeyError 방지)
# ──────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") \
                + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ──────────────────────────────────────────────
# 그래프 헬퍼 함수
# ──────────────────────────────────────────────
def line_fig(df, xcol, ycol, title, color="#2E86AB", height=340):
    if isinstance(xcol, pd.Index):
        x_vals = xcol.values
        x_name = str(xcol.name) if xcol.name else "시간"
    else:
        x_vals = df[xcol].values
        x_name = str(xcol)
    y_vals = df[ycol].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines", name=ycol,
        line=dict(color=color, width=1.8),
        connectgaps=False,
    ))
    fig.update_layout(
        title=title, height=height, margin=dict(t=40, b=30, l=40, r=10),
        xaxis_title=x_name, yaxis_title=ycol,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#D0E4F0"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
                    font=dict(color="#B0C4D8")),
        hovermode="x unified",
        plot_bgcolor="rgba(11,28,42,0.82)",
        paper_bgcolor="rgba(11,28,42,0.82)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
    )
    fig.update_xaxes(tickformat="%m-%d %H:%M", tickangle=0,
                     color="#8AA8C0")
    fig.update_yaxes(color="#8AA8C0")
    return fig

def hist_fig(series, title, color="#A23B72", height=300):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=series.dropna().values, nbinsx=20,
        marker_color=color, opacity=0.82, name="분포",
        marker_line_color="rgba(255,255,255,0.6)", marker_line_width=1,
    ))
    fig.update_layout(
        title=title, height=height,
        xaxis_title="생중량 (g)", yaxis_title="개체 수",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#D0E4F0"),
        bargap=0.08, margin=dict(t=40, b=30, l=40, r=10),
        plot_bgcolor="rgba(11,28,42,0.82)",
        paper_bgcolor="rgba(11,28,42,0.82)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
    )
    return fig

def scatter_fig(x, y, title, color="#2E86AB"):
    df0 = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df0) < 2:
        return None, None
    x_v, y_v = df0["x"].values, df0["y"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_v, y=y_v, mode="markers",
        marker=dict(color=color, size=8, opacity=0.75,
                    line=dict(width=1, color="rgba(255,255,255,0.8)")),
        name="데이터",
    ))
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_v, y_v)
    xs = np.linspace(x_v.min(), x_v.max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept,
        mode="lines", name=f"추세선 (R²={r_val**2:.3f})",
        line=dict(color="crimson", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=title, height=340,
        xaxis_title=x.name, yaxis_title=y.name,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#D0E4F0"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
                    font=dict(color="#B0C4D8")),
        margin=dict(t=40, b=30, l=40, r=10),
        plot_bgcolor="rgba(11,28,42,0.82)",
        paper_bgcolor="rgba(11,28,42,0.82)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
    )
    return fig, (slope, intercept, r_val**2)

def bar_compare_fig(labels, top_vals, bot_vals, title,
                    color_top="#2E86AB", color_bot="#E84545"):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="상위 25%", x=labels, y=top_vals,
                        marker_color=color_top, opacity=0.85))
    fig.add_trace(go.Bar(name="하위 25%", x=labels, y=bot_vals,
                        marker_color=color_bot, opacity=0.85))
    fig.update_layout(
        title=title, height=320, barmode="group",
        xaxis_title="지표", yaxis_title="평균값",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#D0E4F0"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
                    font=dict(color="#B0C4D8")),
        margin=dict(t=40, b=30, l=40, r=10),
        plot_bgcolor="rgba(11,28,42,0.82)",
        paper_bgcolor="rgba(11,28,42,0.82)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                   linecolor="rgba(255,255,255,0.14)",
                   title_font=dict(color="#B0C4D8")),
    )
    return fig

# ──────────────────────────────────────────────
# wide mode + 북극 배경 + 한글 폰트 CSS
# ──────────────────────────────────────────────
st.set_page_config(page_icon="🌱", layout="wide")

# 북극 동물 배경 SVG (극곰·펭귄·물개·이글루 타일)
ARCTIC_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 500 500' preserveAspectRatio='xMidYMid slice'>"
    "<defs>"
    "<radialGradient id='sky' cx='50%' cy='28%' r='85%'>"
    "<stop offset='0%'   stop-color='#152E44'/>"
    "<stop offset='100%' stop-color='#0B1F2D'/>"
    "</radialGradient>"
    "<linearGradient id='ice' x1='0' y1='0' x2='0' y2='1'>"
    "<stop offset='0%'   stop-color='rgba(255,255,255,0.10)'/>"
    "<stop offset='50%'  stop-color='rgba(255,255,255,0.04)'/>"
    "<stop offset='100%' stop-color='rgba(255,255,255,0.10)'/>"
    "</linearGradient>"
    "</defs>"
    "<rect width='500' height='500' fill='url(%23sky)'/>"
    "<ellipse cx='250' cy='300' rx='500' ry='220' fill='url(%23ice)'/>"
    # 별
    + "".join(
        f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='rgba(255,255,255,{a})'/>"
        for cx, cy, r, a in [
            (40,40,1.2,0.55),(80,90,1.0,0.42),(150,30,1.3,0.48),
            (220,55,1.0,0.36),(300,35,1.2,0.55),(380,70,1.0,0.42),
            (450,40,1.3,0.48),(490,90,1.0,0.36),(60,160,1.0,0.36),
            (430,150,1.2,0.48),(110,75,0.9,0.38),(340,120,1.1,0.44),
            (470,110,0.9,0.38),
        ]
    )
    # 펭귄 1행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🐧</text></g>"
        for x, y, s in [(60,80,14),(220,110,12),(380,70,15),
                        (460,150,11),(130,170,13),(290,65,11)]
    )
    # 물개 1행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🦭</text></g>"
        for x, y, s in [(100,150,13),(300,90,11),(410,160,12),
                        (170,200,10),(60,120,10)]
    )
    # 이글루 A/B/C/D
    + "".join(
        f"<g transform='translate({x},{y})'>"
        f"<ellipse cx='24' cy='32' rx='24' ry='7' fill='rgba(255,255,255,0.50)'/>"
        f"<ellipse cx='24' cy='20' rx='24' ry='20' "
        f"fill='rgba(255,255,255,0.16)' stroke='rgba(255,255,255,0.55)' stroke-width='2.5'/>"
        f"<ellipse cx='24' cy='20' rx='6' ry='5' fill='rgba(0,0,0,0.42)'/>"
        f"</g>"
        for x, y in [(140,200),(320,190),(440,200)]
    )
    + "<g transform='translate(250,290) scale(0.65)'>"
    + "<ellipse cx='24' cy='32' rx='24' ry='7' fill='rgba(255,255,255,0.50)'/>"
    + "<ellipse cx='24' cy='20' rx='24' ry='20' "
    + "fill='rgba(255,255,255,0.16)' stroke='rgba(255,255,255,0.55)' stroke-width='2.5'/>"
    + "<ellipse cx='24' cy='20' rx='6' ry='5' fill='rgba(0,0,0,0.42)'/>"
    + "</g>"
    # 펭귄 2행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🐧</text></g>"
        for x, y, s in [(30,240,11),(170,250,13),(360,230,12),
                        (90,310,10),(210,340,11),(430,350,10)]
    )
    # 물개 2행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🦭</text></g>"
        for x, y, s in [(240,250,13),(420,260,11),(60,330,12),
                        (470,320,10),(160,320,9)]
    )
    # 극곰 1행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🐻‍❄️</text></g>"
        for x, y, s in [(200,130,24),(400,140,19)]
    )
    # 펭귄 3행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🐧</text></g>"
        for x, y, s in [(70,270,12),(260,280,10),(440,270,11)]
    )
    # 극곰 2행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🐻‍❄️</text></g>"
        for x, y, s in [(70,280,21),(440,280,17)]
    )
    # 물개 3행
    + "".join(
        f"<g transform='translate({x},{y})'><text x='0' y='0' font-size='{s}' "
        f"text-anchor='middle' dominant-baseline='central'>🦭</text></g>"
        for x, y, s in [(30,300,12),(340,310,13),(200,370,10)]
    )
    # 이글루 E/F
    + "<g transform='translate(30,360) scale(0.60)'>"
    + "<ellipse cx='24' cy='32' rx='24' ry='7' fill='rgba(255,255,255,0.50)'/>"
    + "<ellipse cx='24' cy='20' rx='24' ry='20' "
    + "fill='rgba(255,255,255,0.16)' stroke='rgba(255,255,255,0.55)' stroke-width='2.5'/>"
    + "<ellipse cx='24' cy='20' rx='6' ry='5' fill='rgba(0,0,0,0.42)'/>"
    + "</g>"
    + "<g transform='translate(490,370) scale(0.55)'>"
    + "<ellipse cx='24' cy='32' rx='24' ry='7' fill='rgba(255,255,255,0.50)'/>"
    + "<ellipse cx='24' cy='20' rx='24' ry='20' "
    + "fill='rgba(255,255,255,0.16)' stroke='rgba(255,255,255,0.55)' stroke-width='2.5'/>"
    + "<ellipse cx='24' cy='20' rx='6' ry='5' fill='rgba(0,0,0,0.42)'/>"
    + "</g>"
    # 극곰 3행 (작게)
    + "<g transform='translate(270,360) scale(0.78)'>"
    + "<text x='0' y='0' font-size='26' text-anchor='middle' dominant-baseline='central'>🐻‍❄️</text></g>"
    + "<g transform='translate(460,380) scale(0.68)'>"
    + "<text x='0' y='0' font-size='26' text-anchor='middle' dominant-baseline='central'>🐻‍❄️</text></g>"
).replace("\n", "")

st.markdown(
    f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

.arctic-bg {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
    background: url("{ARCTIC_SVG}") no-repeat;
    background-size: 1200px 1200px;
    background-repeat: repeat;
}}
.arctic-overlay {{
    position: fixed;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background: rgba(7,16,24,0.55);
}}
.arctic-content {{
    position: relative;
    z-index: 2;
    min-height: 100vh;
}}

html, body {{
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    background: transparent;
    margin: 0;
    min-height: 100vh;
}}
.stApp,
[class*="css"] > div[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

/* ── 메인 제목 ── */
h1, .stTitle {{
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #5DB8C4 0%, #A2D5DA 50%, #D4ECF4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    display: inline-block;
    text-shadow: 0 2px 14px rgba(93,184,196,0.30);
}}
h1::after {{
    content: '';
    display: block;
    width: 96px;
    height: 3px;
    background: linear-gradient(90deg, #5DB8C4, #A2D5DA);
    border-radius: 3px;
    margin-top: 6px;
}}
.stTitle::after {{ display: none; }}

/* ── 섹션 헤더 ── */
h2, .stSubtitle,
[class*="css"] h2 {{
    border-left: 4px solid #5DB8C4;
    padding-left: 12px;
    font-weight: 700 !important;
    color: #DDE9F2 !important;
}}

/* ── 사이드바 ── */
[class*="css"] section[data-testid="stSidebar"] {{
    background: rgba(13,26,38,0.90) !important;
    border-right: 1px solid rgba(93,184,196,0.30) !important;
}}

/* ── divider ── */
.st-divider {{
    border-top: 1px solid #5DB8C4 !important;
    opacity: 0.40 !important;
}}

/* ── 메인 버튼 ── */
.stButton > button {{
    background: #2E86AB !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}}
.stButton > button:hover {{
    background: #226688 !important;
}}

/* ── 다운로드 버튼 ── */
button[kind="secondary"] {{
    background: #2E86AB !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}}
button[kind="secondary"]:hover {{
    background: #226688 !important;
}}

/* ── alert 박스 (info/success/warning) ── */
.stAlert, .stInfo, .stSuccess, .stWarning {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    color: #DDE9F2 !important;
    border-radius: 8px !important;
}}
.stSuccess {{
    background: rgba(46,134,171,0.18) !important;
    border-color: rgba(46,134,171,0.45) !important;
}}
.stWarning {{
    background: rgba(232,70,70,0.13) !important;
    border-color: rgba(232,70,70,0.40) !important;
    color: #FFC8C8 !important;
}}
.stInfo {{
    background: rgba(93,184,196,0.11) !important;
    border-color: rgba(93,184,196,0.30) !important;
}}

/* ── metric 카드 ── */
[class*="css"] .metric-card,
[class*="css"] [data-testid="stVerticalBlock"] {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}}

/* ── caption ── */
.stCaption {{
    color: #88A8BE !important;
}}

/* ── Plotly hover 박스 ── */
.js-plotly-plot .plotly .hoverlayer .hovertext {{
    background: rgba(46,134,171,0.15) !important;
    border: 1px solid rgba(46,134,171,0.50) !important;
    border-radius: 6px !important;
    color: #DDE9F2 !important;
}}

/* ── tabs ── */
[class*="css"] button[data-baseweb="tab"] {{
    border-radius: 6px !important;
    font-weight: 500 !important;
    color: #90AEC0 !important;
}}
[class*="css"] button[data-baseweb="tab"][aria-selected="true"] {{
    background: rgba(46,134,171,0.30) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(46,134,171,0.55) !important;
}}

/* ── dataframe 테이블 ── */
div[data-testid="stDataFrame"] table {{
    background: rgba(11,28,42,0.70) !important;
    color: #DDE9F2 !important;
    border-color: rgba(93,184,196,0.30) !important;
}}
div[data-testid="stDataFrame"] th {{
    background: rgba(46,134,171,0.25) !important;
    color: #DDE9F2 !important;
}}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="arctic-bg"></div>
       <div class="arctic-overlay"></div>
       <div class="arctic-content"></div>""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
st.sidebar.title("📂  데이터 업로드")
st.sidebar.caption("학교마다 열 이름이 달라도 괜찮아요 — 자동으로 맞춰 분석합니다.")
env_file    = st.sidebar.file_uploader("🌡️  환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌿  생육 결과 CSV",   type="csv")

env_df    = std_cols(read_csv(env_file))
growth_df = std_cols(read_csv(growth_file))

def extract_school(fname):
    if not fname:
        return "우리 학교"
    base = os.path.basename(fname)
    for suf in ("_환경데이터.csv", "_생육결과데이터.csv",
                "_환경.csv", "_생육.csv", ".csv"):
        base = base.replace(suf, "")
    return base

school = extract_school(
    env_file.name if env_file else (growth_file.name if growth_file else None)
)

st.sidebar.divider()
st.sidebar.header("📏  생육 측정 항목")
growth_cols = [
    c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                "지상부 질량", "지하부 질량"]
    if c in growth_df.columns
]
sel_col = st.sidebar.selectbox(
    "🔍  산점도로 볼 지표 (탭3)",
    options=growth_cols if growth_cols else ["지표 없음"],
    index=0 if growth_cols else 0,
)
st.sidebar.divider()
st.sidebar.info(
    "💡  어떤 파일이든 업로드하면\n"
    "   인코딩 · 열 이름 · 날짜 형식을\n"
    "   자동으로 맞춰 분석합니다.",
    icon="🧊",
)

# ──────────────────────────────────────────────
# 제목
# ──────────────────────────────────────────────
st.title(f"{school} × 극지연구소  콜라보레이션! 🌱")
st.caption("극지식물 개체별 생육 차이 분석 대시보드  ·  인천여고 × 극지연구소")

tabs = st.tabs([
    "📋  연구 개요 및 스마트팜 환경",
    "📊  개체별 생육 차이 분석",
    "🔗  생육 지표와 생중량의 상관관계 및 예측",
])

# ──────────────────────────────────────────────
# 탭 1 ─ 연구 개요 및 환경
# ──────────────────────────────────────────────
with tabs[0]:
    try:
        st.success(
            "극지연구소와 함께 극지식물의 생육 비밀을 파헤치는 연구입니다.",
            icon="🧊",
        )
        col_info = st.columns(2)
        col_info[0].info(
            "📌  연구 목적\n"
            "• 같은 스마트팜에서 자란 극지식물 개체들 사이에\n"
            "   잎 수·길이·무게 차이가 왜 생기는지 알아봅니다.",
            icon="🎯",
        )
        col_info[1].info(
            "📌  연구 가설\n"
            "• 잎 수가 많고 지상부·지하부가 길수록\n"
            "   생중량(전체 무게)도 무거울 것이다.",
            icon="💡",
        )
        st.divider()

        if env_df.empty:
            st.info("🌡️  환경 데이터 CSV를 사이드바에서 업로드해주세요.",
                    icon="☁️")
        else:
            st.header("🌡️  스마트팜 환경 데이터 요약")
            cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in env_df.columns]
            if not cols:
                st.info("환경 지표 열이 하나도 없어 요약 카드를 표시할 수 없어요.")
            else:
                card_cols = st.columns(min(len(cols), 4))
                labels_map = {
                    "temperature": "평균 온도",
                    "humidity":    "평균 습도",
                    "ph":          "평균 pH",
                    "ec":          "평균 EC",
                }
                units_map  = {
                    "temperature": "°C",
                    "humidity":    "%",
                    "ph":          "",
                    "ec":          "mS/cm",
                }
                for i, cn in enumerate(cols):
                    s = env_df[cn].dropna()
                    if len(s) == 0:
                        continue
                    avg, mx, mn = s.mean(), s.max(), s.min()
                    card_cols[i].metric(
                        labels_map.get(cn, cn),
                        f"{avg:.1f}{units_map.get(cn, '')}",
                        delta=f"{mn:.1f} ~ {mx:.1f}{units_map.get(cn, '')}",
                    )
                st.caption(
                    "※  평균: 모든 값을 더해 개수로 나눈 값  ·  "
                    "표준편차: 값들이 평균에서 얼마나 흩어져 있는지  ·  "
                    "변동계수: 표준편차를 평균으로 나눈 값 (단위가 다른 것도 비교 가능)"
                )

            if "time" in env_df.columns \
               and {"temperature", "humidity"} <= set(env_df.columns):
                st.divider()
                st.subheader("📅  날짜별 온도·습도 변화")
                tmp = env_df[["time", "temperature", "humidity"]].dropna()
                if len(tmp) >= 2:
                    tmp = tmp.sort_values("time")
                    daily_t = tmp.set_index("time")["temperature"].resample("D").mean()
                    daily_h = tmp.set_index("time")["humidity"].resample("D").mean()
                    ddf = pd.DataFrame({
                        "temperature": daily_t.values,
                        "humidity":    daily_h.values,
                    }, index=daily_t.index).dropna()
                    if len(ddf) >= 1:
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(
                            line_fig(ddf, ddf.index, "temperature",
                                     "📈  날짜별 평균 온도 변화 (°C)"),
                            width="stretch",
                        )
                        c2.plotly_chart(
                            line_fig(ddf, ddf.index, "humidity",
                                     "💧  날짜별 평균 습도 변화 (%)", color="#57B4E4"),
                            width="stretch",
                        )
                    else:
                        st.info("날짜별 평균을 계산할 데이터가 충분하지 않아요.")
                else:
                    st.info("온도·습도 데이터가 부족해서 그래프를 표시할 수 없어요.")
            else:
                st.info("time 열이나 temperature·humidity 열이 없어 환경 그래프가 표시되지 않아요.")
    except Exception as e:
        st.warning(f"⚠️  이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ──────────────────────────────────────────────
# 탭 2 ─ 개체별 생육 차이 분석
# ──────────────────────────────────────────────
with tabs[1]:
    try:
        if growth_df.empty:
            st.info("🌿  생육 결과 CSV를 사이드바에서 업로드해주세요.", icon="☁️")
        elif "생중량" not in growth_df.columns:
            st.info("생중량(g) 열이 없어 생육 분석을 표시할 수 없어요.")
        else:
            st.header(f"🔬  {school}  극지식물 개체별 생육 차이 분석")
            fresh = growth_df["생중량"].dropna()
            n = len(fresh)
            if n < 2:
                st.info("생중량 데이터가 부족해 분석을 표시할 수 없어요.")
            else:
                mean_v = fresh.mean()
                std_v  = fresh.std()
                cv_v   = (std_v / mean_v * 100) if mean_v != 0 else np.nan

                st.caption(
                    "📖  용어 설명  |  "
                    "평균: 모든 개체 생중량을 더해 개수로 나눈 값  ·  "
                    "표준편차: 값들이 평균에서 얼마나 흩어져 있는지  ·  "
                    "변동계수(CV): 표준편차를 평균으로 나눈 백분율 "
                    "(단위가 다른 지표 비교용)"
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("🌱  개체 수",      f"{n}개")
                mc2.metric("⚖️  평균 생중량",  f"{mean_v:.2f} g")
                mc3.metric("📊  표준편차",      f"{std_v:.2f} g")
                mc4.metric("📈  변동계수 (CV)", f"{cv_v:.1f} %")

                st.divider()
                st.subheader("📊  개체별 생중량 분포 (히스토그램)")
                fig_h = hist_fig(fresh, "개체별 생중량 분포 (g)", color="#A23B72")
                st.plotly_chart(fig_h, width="stretch")

                q75 = fresh.quantile(0.75)
                q25 = fresh.quantile(0.25)
                indicators  = ["잎 수", "지상부 길이", "지하부 길이"]
                ind_present = [ind for ind in indicators if ind in growth_df.columns]
                if ind_present:
                    top_means, bot_means = [], []
                    for ind in ind_present:
                        valid = growth_df[[ind, "생중량"]].dropna()
                        if len(valid) < 2:
                            top_means.append(np.nan)
                            bot_means.append(np.nan)
                            continue
                        t_idx = valid[valid["생중량"] >= q75].index
                        b_idx = valid[valid["생중량"] <= q25].index
                        top_means.append(valid.loc[t_idx, ind].mean() if len(t_idx) else np.nan)
                        bot_means.append(valid.loc[b_idx, ind].mean() if len(b_idx) else np.nan)

                    units_map = {"잎 수": "장", "지상부 길이": "mm", "지하부 길이": "mm"}
                    disp_names = [
                        f"{ind}\n({units_map.get(ind, '')})"
                        for ind in ind_present
                    ]
                    fig_b = bar_compare_fig(
                        disp_names,
                        [v if not np.isnan(v) else 0 for v in top_means],
                        [v if not np.isnan(v) else 0 for v in bot_means],
                        f"생중량 상위 25%(≥{q75:.1f}g)  vs  하위 25%(≤{q25:.1f}g)  비교",
                    )
                    st.plotly_chart(fig_b, width="stretch")
                    st.caption(
                        f"※  상위 25% 기준: ≥ {q75:.1f} g  ·  하위 25% 기준: ≤ {q25:.1f} g\n"
                        "각 막대는 해당 집단에서 지표별 평균값을 나타냅니다."
                    )
                else:
                    st.info("잎 수·길이 지표 열이 없어 비교 막대그래프를 표시할 수 없어요.")
    except Exception as e:
        st.warning(f"⚠️  이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ──────────────────────────────────────────────
# 탭 3 ─ 상관관계 및 예측
# ──────────────────────────────────────────────
with tabs[2]:
    try:
        if growth_df.empty:
            st.info("🌿  생육 결과 CSV를 사이드바에서 업로드해주세요.", icon="☁️")
        elif "생중량" not in growth_df.columns:
            st.info("생중량(g) 열이 없어 상관관계 분석을 표시할 수 없어요.")
        else:
            st.header("🔗  생육 지표와 생중량의 상관관계 및 예측")
            st.caption(
                "📖  용어 설명  |  "
                "상관계수(r): 두 지표가 함께 움직이는 정도 (−1 ~ 1, 1에 가까울수록 강한 양의 관계)  ·  "
                "결정계수(R²): 회귀식이 데이터를 얼마나 잘 설명하는지 "
                "(0 ~ 1, 1에 가까울수록 예측 정확도 높음)"
            )
            predictors = [c for c in ["잎 수", "지상부 길이", "지하부 길이"]
                          if c in growth_df.columns and "생중량" in growth_df.columns]
            if not predictors:
                st.info("잎 수·지상부 길이·지하부 길이 열이 없어 상관 분석을 표시할 수 없어요.")
            else:
                valid_all = growth_df[["생중량"] + predictors].dropna()
                if len(valid_all) < 2:
                    st.info("유효 데이터가 부족해 상관분석을 표시할 수 없어요.")
                else:
                    rows = []
                    for pred in predictors:
                        sub = valid_all[[pred, "생중량"]].dropna()
                        if len(sub) >= 2:
                            r_val, _ = stats.pearsonr(sub[pred], sub["생중량"])
                            rows.append({"예측 지표": pred, "상관계수 (r)": round(r_val, 4)})
                    rdf = pd.DataFrame(rows)
                    if len(rdf) == 0:
                        st.info("상관계수를 계산할 데이터가 충분하지 않아요.")
                    else:
                        st.subheader("📐  지표별 생중량 상관계수 비교")
                        st.dataframe(rdf, width="stretch")

                        st.divider()
                        c_left, c_right = st.columns(2)

                        if "잎 수" in growth_df.columns:
                            fig_l, (sl, ic, r2) = scatter_fig(
                                growth_df["잎 수"], growth_df["생중량"],
                                "🍃  잎 수 vs 생중량  (산점도 + 추세선)",
                                color="#5DB8C4",
                            )
                            if fig_l:
                                c_left.plotly_chart(fig_l, width="stretch")
                                st.markdown(
                                    f"**🌿  잎 수로 생중량 예측 회귀식**\n\n"
                                    f"$$\n"
                                    f"생중량(g) = {sl:.3f} \\times \\text{{잎 수}} + {ic:.3f}"
                                    f"\\qquad R² = {r2:.3f}\n$$\n\n"
                                    f"※  기울기({sl:.3f}): 잎 수가 1장 늘어날 때 생중량이 평균 {sl:.3f}g 증가  ·  "
                                    f"절편({ic:.3f}): 잎 수가 0일 때의 예측값  ·  "
                                    f"R²={r2:.3f}: 이 식이 데이터를 {r2*100:.1f}% 설명"
                                )
                            else:
                                c_left.info("잎 수와 생중량 데이터가 부족해 산점도를 표시하기 어려워요.")

                        if sel_col in growth_df.columns and sel_col != "잎 수":
                            fig_s, (sl2, ic2, r22) = scatter_fig(
                                growth_df[sel_col], growth_df["생중량"],
                                f"📐  {sel_col} vs 생중량  (산점도 + 추세선)",
                                color="#E84545",
                            )
                            if fig_s:
                                c_right.plotly_chart(fig_s, width="stretch")
                                st.markdown(
                                    f"**📐  {sel_col}으로 생중량 예측 회귀식**\n\n"
                                    f"$$\n"
                                    f"생중량(g) = {sl2:.3f} \\times {sel_col} + {ic2:.3f}"
                                    f"\\qquad R² = {r22:.3f}\n$$\n\n"
                                    f"※  기울기({sl2:.3f}): {sel_col}이 1단위 증가할 때 "
                                    f"생중량 평균 {sl2:.3f}g 변화"
                                )
                            else:
                                c_right.info(f"{sel_col}과 생중량 데이터가 부족해 "
                                             f"산점도를 표시하기 어려워요.")
                        else:
                            c_right.info("사이드바에서 다른 생육 지표를 선택해보세요.")

                        st.divider()
                        st.subheader("📁  생육 데이터 다운로드 (UTF-8-SIG)")
                        csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="⬇️  생육 결과 CSV 다운로드",
                            data=csv_bytes,
                            file_name=f"{school}_생육결과.csv",
                            mime="text/csv",
                        )
    except Exception as e:
        st.warning(f"⚠️  이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
