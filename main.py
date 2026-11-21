import io, os, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

# ──────────────────────────────────────────────
# wide mode + 아이콘  (theme= 인자는 이 버전에서 미지원 → 제거)
# ──────────────────────────────────────────────
st.set_page_config(page_icon="🌱", layout="wide")

# ──────────────────────────────────────────────
# 커스텀 CSS — 제목·헤더·버튼·Plotly hover 색 통일
# ──────────────────────────────────────────────
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
    :root {
        --p: #2E86AB;      /* primary */
        --p2: #5DB8C4;     /* secondary accent */
        --p3: #A2D5DA;     /* light accent */
    }
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    }
    /* ── 메인 제목: 청록 그라데이션 ── */
    h1, .stTitle {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, var(--p) 0%, var(--p2) 50%, var(--p3) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        display: inline-block;
    }
    h1::after {
        content: '';
        display: block;
        width: 90px;
        height: 3px;
        background: linear-gradient(90deg, var(--p), var(--p2));
        border-radius: 3px;
        margin-top: 6px;
    }
    .stTitle::after { display: none; }
    /* ── 섹션 헤더: 왼쪽 청록 구분선 ── */
    h2, .stSubtitle, [class*="css"] h2 {
        border-left: 4px solid var(--p);
        padding-left: 12px;
        font-weight: 700 !important;
        color: #1A2332 !important;
    }
    /* ── sidebar 배경 ── */
    [class*="css"] section[data-testid="stSidebar"] {
        background: #EEF3F8 !important;
    }
    /* ── divider ── */
    .st-divider {
        border-top: 1px solid var(--p) !important;
        opacity: 0.30 !important;
    }
    /* ── 메인 버튼: primary 색으로 ── */
    .stButton > button {
        background: var(--p) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: #226688 !important;
    }
    /* ── 다운로드 버튼(secondary) ── */
    button[kind="secondary"] {
        background: var(--p) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover {
        background: #226688 !important;
    }
    /* ── Plotly hover 박스 ── */
    .js-plotly-plot .plotly .hoverlayer .hovertext {
        background: rgba(46,134,171,0.06) !important;
        border: 1px solid rgba(46,134,171,0.25) !important;
        border-radius: 6px !important;
    }
    /* ── tabs 레이블 ── */
    [class*="css"] button[data-baseweb="tab"] {
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# CSV 읽기 (utf-8-sig → cp949 이중 시도)
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
# 열 이름 표준화 (학교마다 다른 열 이름 → KeyError 방지)
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
        x_vals, x_name = df[xcol].values, str(xcol)
    y_vals = df[ycol].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines", name=ycol,
        line=dict(color=color, width=2), connectgaps=False,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#1A2332")),
        height=height, margin=dict(t=50, b=30, l=40, r=10),
        xaxis_title=x_name, yaxis_title=ycol,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#4A5568"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
                    font=dict(size=12)),
        hovermode="x unified",
        plot_bgcolor="rgba(255,255,255,0.6)",
        paper_bgcolor="rgba(255,255,255,0.6)",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
    )
    fig.update_xaxes(tickformat="%m-%d %H:%M", tickangle=0)
    return fig

def hist_fig(series, title, color="#A23B72", height=300):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=series.dropna().values, nbinsx=20,
        marker_color=color, opacity=0.82, name="분포",
        marker_line_color="white", marker_line_width=1,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#1A2332")),
        height=height,
        xaxis_title="생중량 (g)", yaxis_title="개체 수",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#4A5568"),
        bargap=0.08, margin=dict(t=50, b=30, l=40, r=10),
        plot_bgcolor="rgba(255,255,255,0.6)",
        paper_bgcolor="rgba(255,255,255,0.6)",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
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
        marker=dict(color=color, size=9, opacity=0.72,
                    line=dict(width=1.2, color="white")),
        name="데이터",
    ))
    slope, intercept, r_val, *_ = stats.linregress(x_v, y_v)
    xs = np.linspace(x_v.min(), x_v.max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines",
        name=f"추세선  (R²={r_val**2:.3f})",
        line=dict(color="crimson", width=2.5, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#1A2332")),
        height=340,
        xaxis_title=x.name, yaxis_title=y.name,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#4A5568"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
                    font=dict(size=12)),
        margin=dict(t=50, b=30, l=40, r=10),
        plot_bgcolor="rgba(255,255,255,0.6)",
        paper_bgcolor="rgba(255,255,255,0.6)",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
    )
    return fig, (slope, intercept, r_val**2)

def bar_compare_fig(labels, top_vals, bot_vals, title,
                    color_top="#2E86AB", color_bot="#E84545"):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="상위 25%", x=labels, y=top_vals,
                        marker_color=color_top, opacity=0.85,
                        marker_line_color="white", marker_line_width=0.8))
    fig.add_trace(go.Bar(name="하위 25%", x=labels, y=bot_vals,
                        marker_color=color_bot, opacity=0.85,
                        marker_line_color="white", marker_line_width=0.8))
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#1A2332")),
        height=320, barmode="group",
        xaxis_title="지표", yaxis_title="평균값",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color="#4A5568"),
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1,
                    font=dict(size=12)),
        margin=dict(t=50, b=30, l=40, r=10),
        plot_bgcolor="rgba(255,255,255,0.6)",
        paper_bgcolor="rgba(255,255,255,0.6)",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E0",
                   title_font=dict(size=13, color="#4A5568")),
    )
    return fig

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
                labels = {"temperature": "평균 온도", "humidity": "평균 습도",
                          "ph": "평균 pH", "ec": "평균 EC"}
                units  = {"temperature": "°C", "humidity": "%", "ph": "", "ec": "mS/cm"}
                for i, cn in enumerate(cols):
                    s = env_df[cn].dropna()
                    if len(s) == 0:
                        continue
                    avg, mx, mn = s.mean(), s.max(), s.min()
                    card_cols[i].metric(
                        labels.get(cn, cn),
                        f"{avg:.1f}{units.get(cn, '')}",
                        delta=f"{mn:.1f} ~ {mx:.1f}{units.get(cn, '')}",
                    )
                st.caption(
                    "※  평균: 모든 값을 더해 개수로 나눈 값  ·  "
                    "표준편차: 값들이 평균에서 얼마나 흩어져 있는지  ·  "
                    "변동계수: 표준편차를 평균으로 나눈 값 (단위가 다른 것도 비교 가능)"
                )

            if "time" in env_df.columns and {"temperature", "humidity"} <= set(env_df.columns):
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
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

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
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

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
                                color="#2E86AB",
                            )
                            if fig_l:
                                c_left.plotly_chart(fig_l, width="stretch")
                                st.markdown(
                                    f"**잎 수로 생중량 예측 회귀식**\n\n"
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
                                    f"**{sel_col}으로 생중량 예측 회귀식**\n\n"
                                    f"$$\n"
                                    f"생중량(g) = {sl2:.3f} \\times {sel_col} + {ic2:.3f}"
                                    f"\\qquad R² = {r22:.3f}\n$$\n\n"
                                    f"※  기울기({sl2:.3f}): {sel_col}이 1단위 증가할 때 "
                                    f"생중량 평균 {sl2:.3f}g 변화"
                                )
                            else:
                                c_right.info(f"{sel_col}과 생중량 데이터가 부족해 산점도를 표시하기 어려워요.")
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
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
