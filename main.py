import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import streamlit as st


@st.cache_data
def read_file_bytes(file):
    return file.getvalue()


def read_csv_flexible(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


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
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


def extract_school(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    return base.split("_")[0] if "_" in base else base


def explain_mean(label):
    return ("**평균(mean) — " + label + "** : 모든 값을 더한 뒤 개수로 나눈 "
            "대표값입니다. 데이터가 일반적으로 어느 수준에 모여 있는지 보여줍니다.")


def explain_std(label):
    return ("**표준편차(standard deviation) — " + label + "** : 값들이 평균에서 "
            "얼마나 흩어져 있는지 나타냅니다. 클수록 개체 간 차이가 큽니다.")


def explain_corr(x, y):
    return ("**상관계수(Pearson r) — " + x + " vs " + y + "** : 두 수치가 함께 "
            "오르고 내리는 정도입니다. 1에 가까울수록 강한 양의 관계, -1에 가까울수록 "
            "강한 음의 관계, 0이면 뚜렷한 선형 관계가 없습니다.")


st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 인천여고 극지식물 개체별 생육 차이 분석")

env_file = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌱 생육 결과 CSV", type="csv")

env_df = None
growth_df = None
school_name = "우리 학교"

if env_file is not None:
    env_df = read_csv_flexible(env_file)
    if not env_df.empty:
        env_df = std_cols(env_df)
        sn = extract_school(env_file.name)
        if school_name == "우리 학교":
            school_name = sn

if growth_file is not None:
    growth_df = read_csv_flexible(growth_file)
    if not growth_df.empty:
        growth_df = std_cols(growth_df)
        sn = extract_school(growth_file.name)
        if school_name == "우리 학교":
            school_name = sn

if env_file is None and growth_file is None:
    st.info("왼쪽 사이드바에서 CSV 파일 2개를 업로드해 주세요.")
    st.stop()

tabs = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ── Tab 1: 실험 개요 ──────────────────────────────────────────────
with tabs[0]:
    try:
        st.header("🔍 연구 배경 및 목적")
        st.markdown(
            "극지에서 자라는 식물(극지식물)은 아주 낮은 온도, 건조한 흙, 적은 양분 같은 "
            "혹독한 환경에서 자랍니다. 같은 학교 학생들이 직접 키운 극지식물 개체들을 비교하면, "
            "어떤 개체가 더 잘 자랐는지, 잎 수나 줄기 길이 같은 측정값이 생중량(전체 무게)과 "
            "어떤 관계가 있는지 알 수 있습니다.\n\n"
            "이 대시보드에서는 **각 개체의 생중량 차이**, **잎 수와 줄기 길이가 생중량과 "
            "어떤 관계인지(상관계수)**, 그리고 **생중량이 큰 상위 25%와 작은 하위 25% 개체의 "
            "차이**를 살펴봅니다."
        )
        st.markdown("---")
        st.header("📋 데이터 요약")

        data_ok = False
        if env_df is not None and not env_df.empty and "time" in env_df.columns:
            tvals = env_df["time"].dropna()
            if len(tvals) > 0:
                tmin = tvals.min()
                tmax = tvals.max()
                if pd.notna(tmin) and pd.notna(tmax):
                    date_min = tmin.strftime("%Y-%m-%d")
                    date_max = tmax.strftime("%Y-%m-%d")
                    st.markdown("| 항목 | 값 |")
                    st.markdown("|---|---|")
                    if growth_df is not None and not growth_df.empty and "개체번호" in growth_df.columns:
                        st.markdown("| 총 개체 수 | " + str(growth_df["개체번호"].nunique()) + "개체 |")
                    else:
                        st.markdown("| 총 개체 수 | (생육 CSV를 업로드해 주세요) |")
                    st.markdown("| 측정 기간 | " + date_min + " ~ " + date_max + " |")
                    env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                    st.markdown("| 환경 측정 항목 | " + ", ".join(env_cols) + " |")
                    if growth_df is not None and not growth_df.empty:
                        gr_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]
                        st.markdown("| 생육 측정 항목 | " + ", ".join(gr_cols) + " |")
                    else:
                        st.markdown("| 생육 측정 항목 | (생육 CSV를 업로드해 주세요) |")
                    data_ok = True

        if not data_ok:
            st.info("데이터 요약표를 표시하려면 환경 CSV와 생육 CSV를 모두 업로드해 주세요.")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if growth_df is not None and not growth_df.empty and "개체번호" in growth_df.columns:
                st.metric("총 개체 수", growth_df["개체번호"].nunique())
            else:
                st.metric("총 개체 수", "—")

        with c2:
            if env_df is not None and not env_df.empty and "temperature" in env_df.columns:
                t = env_df["temperature"].dropna()
                if len(t) >= 2:
                    st.metric("평균 온도", str(round(t.mean(), 1)) + " °C")
                else:
                    st.info("온도 데이터가 부족해요")
            else:
                st.metric("평균 온도", "—")

        with c3:
            if env_df is not None and not env_df.empty and "humidity" in env_df.columns:
                h = env_df["humidity"].dropna()
                if len(h) >= 2:
                    st.metric("평균 습도", str(round(h.mean(), 1)) + " %")
                else:
                    st.info("습도 데이터가 부족해요")
            else:
                st.metric("평균 습도", "—")

        with c4:
            if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    st.metric("평균 생중량", str(round(gw.mean(), 2)) + " g")
                else:
                    st.info("생중량 데이터가 부족해요")
            else:
                st.metric("평균 생중량", "—")
    except Exception as e:
        st.warning("이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))

# ── Tab 2: 환경 데이터 ────────────────────────────────────────────
with tabs[1]:
    try:
        st.header("🌡️ 환경 데이터 요약")
        if env_df is not None and not env_df.empty:
            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if len(env_num_cols) == 0:
                st.info("환경 수치 열(temperature, humidity, ph, ec) 중 맞는 열이 없어요.")
            else:
                summary_rows = []
                for col in env_num_cols:
                    vals = env_df[col].dropna()
                    if len(vals) >= 2:
                        summary_rows.append({
                            "측정 항목": col,
                            "평균": round(vals.mean(), 2),
                            "최소": round(vals.min(), 2),
                            "최대": round(vals.max(), 2),
                            "측정 개수": int(len(vals)),
                        })
                if len(summary_rows) > 0:
                    st.dataframe(pd.DataFrame(summary_rows), width="stretch")
                else:
                    st.info("수치 데이터가 충분하지 않아요.")

            if "time" in env_df.columns:
                env_sorted = env_df.sort_values("time")
                daily_t = env_sorted.set_index("time")["temperature"].resample("D").mean().dropna()
                if len(daily_t) >= 2:
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(x=daily_t.index, y=daily_t.values, mode="lines+markers",
                                               name="일별 평균 온도",
                                               line=dict(color="#E74C3C", width=2),
                                               marker=dict(size=5)))
                    fig_t.update_layout(
                        title={"text": "날짜별 평균 온도 변화", "x": 0.05, "xanchor": "left"},
                        xaxis_title="날짜",
                        yaxis_title="온도 (°C)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=300,
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_t, width="stretch")

                daily_h = env_sorted.set_index("time")["humidity"].resample("D").mean().dropna()
                if len(daily_h) >= 2:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(x=daily_h.index, y=daily_h.values, mode="lines+markers",
                                               name="일별 평균 습도",
                                               line=dict(color="#3498DB", width=2),
                                               marker=dict(size=5)))
                    fig_h.update_layout(
                        title={"text": "날짜별 평균 습도 변화", "x": 0.05, "xanchor": "left"},
                        xaxis_title="날짜",
                        yaxis_title="습도 (%)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=300,
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_h, width="stretch")

                if "ph" in env_df.columns:
                    daily_ph = env_sorted.set_index("time")["ph"].resample("D").mean().dropna()
                    if len(daily_ph) >= 2:
                        fig_ph = go.Figure()
                        fig_ph.add_trace(go.Scatter(x=daily_ph.index, y=daily_ph.values, mode="lines+markers",
                                                    name="일별 평균 pH",
                                                    line=dict(color="#27AE60", width=2),
                                                    marker=dict(size=5)))
                        fig_ph.update_layout(
                            title={"text": "날짜별 평균 pH 변화", "x": 0.05, "xanchor": "left"},
                            xaxis_title="날짜",
                            yaxis_title="pH",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            height=300,
                            margin=dict(l=40, r=20, t=50, b=40),
                        )
                        st.plotly_chart(fig_ph, width="stretch")

                if "ec" in env_df.columns:
                    daily_ec = env_sorted.set_index("time")["ec"].resample("D").mean().dropna()
                    if len(daily_ec) >= 2:
                        fig_ec = go.Figure()
                        fig_ec.add_trace(go.Scatter(x=daily_ec.index, y=daily_ec.values, mode="lines+markers",
                                                    name="일별 평균 EC",
                                                    line=dict(color="#8E44AD", width=2),
                                                    marker=dict(size=5)))
                        fig_ec.update_layout(
                            title={"text": "날짜별 평균 EC 변화", "x": 0.05, "xanchor": "left"},
                            xaxis_title="날짜",
                            yaxis_title="EC (mS/cm)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            height=300,
                            margin=dict(l=40, r=20, t=50, b=40),
                        )
                        st.plotly_chart(fig_ec, width="stretch")
        else:
            st.info("환경 CSV를 업로드해 주세요.")

        with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
            if env_df is not None and not env_df.empty:
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSV 다운로드", csv_bytes,
                                   file_name="환경데이터_다운로드.csv",
                                   mime="text/csv")
            else:
                st.info("환경 데이터가 없어요.")
    except Exception as e:
        st.warning("이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))

# ── Tab 3: 생육 결과 ──────────────────────────────────────────────
with tabs[2]:
    try:
        st.header("🥇 핵심 결과")
        if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
            gw_vals = growth_df["생중량"].dropna()
            if len(gw_vals) >= 2:
                mean_gw = gw_vals.mean()
                max_idx = gw_vals.idxmax()
                min_idx = gw_vals.idxmin()
                max_id = growth_df.loc[max_idx, "개체번호"] if "개체번호" in growth_df.columns else "?"
                min_id = growth_df.loc[min_idx, "개체번호"] if "개체번호" in growth_df.columns else "?"
                st.metric("평균 생중량", str(round(mean_gw, 2)) + " g")
                st.info("🏆 생중량 최대 개체: " + str(max_id) + " (" + str(round(gw_vals.max(), 2)) + " g)"
                        + "  |  ⚠️ 생중량 최소 개체: " + str(min_id) + " (" + str(round(gw_vals.min(), 2)) + " g)")
            else:
                st.info("생중량 데이터가 충분하지 않아요.")
        else:
            st.info("생육 CSV를 업로드하거나 생중량 열이 필요해요.")

        st.markdown("---")
        st.header("📊 개체별 생육 비교")

        gr_cols_plot = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                        if c in growth_df.columns] if (growth_df is not None and not growth_df.empty) else []
        if len(gr_cols_plot) == 0:
            st.info("생육 데이터에서 표시할 수치 열(생중량, 잎 수, 지상부 길이, 지하부 길이)을 찾을 수 없어요.")
        else:
            label_map = {
                "생중량": "생중량 (g) — 개체의 전체 무게",
                "잎 수": "잎 수 (장) — 잎이 몇 개 달렸는지",
                "지상부 길이": "지상부 길이 (mm) — 땅 위 줄기 길이",
                "지하부 길이": "지하부 길이 (mm) — 땅속 뿌리 길이",
            }
            color_map = {
                "생중량": "#E74C3C",
                "잎 수": "#27AE60",
                "지상부 길이": "#3498DB",
                "지하부 길이": "#8E44AD",
            }
            n_plot = len(gr_cols_plot)
            fig_all = make_subplots(rows=n_plot, cols=1,
                                    subplot_titles=[label_map.get(c, c) for c in gr_cols_plot],
                                    vertical_spacing=0.12)
            for i, col in enumerate(gr_cols_plot):
                sub = growth_df[["개체번호", col]].dropna()
                if len(sub) == 0:
                    continue
                sub = sub.sort_values("개체번호")
                x = [str(v) for v in sub["개체번호"].tolist()]
                y = sub[col].tolist()
                try:
                    import colorsys
                    colors = [colorsys.hsv_to_rgb(j / max(len(x), 1), 0.65, 0.55) for j in range(len(x))]
                    colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]
                except Exception:
                    colors = [color_map.get(col, "#4C72B0")] * len(x)
                fig_all.add_trace(go.Bar(x=x, y=y, marker_color=colors,
                                         name=label_map.get(col, col), showlegend=False),
                                  row=i + 1, col=1)
                fig_all.update_xaxes(title_text="개체번호", row=i + 1, col=1)
                fig_all.update_yaxes(title_text=label_map.get(col, col).split("—")[0].strip(),
                                     row=i + 1, col=1)
            fig_all.update_layout(
                height=180 * n_plot + 40,
                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_all, width="stretch")

        st.markdown("---")
        st.header("📦 생중량 분포")
        if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
            gw_vals = growth_df["생중량"].dropna()
            if len(gw_vals) >= 3:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=gw_vals, name="생중량",
                                         fillcolor="rgba(231, 76, 60, 0.35)",
                                         line=dict(color="#E74C3C", width=1.5),
                                         boxpoints="outliers"))
                fig_box.update_layout(
                    title={"text": "생중량 분포 (박스플롯) — 상자: 가운데 50% 범위, 선: 중앙값, 점: 이상치",
                           "x": 0.05, "xanchor": "left"},
                    yaxis_title="생중량 (g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=320,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_box, width="stretch")

                q1 = gw_vals.quantile(0.25)
                q3 = gw_vals.quantile(0.75)
                gw_sorted = growth_df.dropna(subset=["생중량"]).sort_values("생중량")
                gw_sorted = gw_sorted.copy()
                gw_sorted["분위"] = np.where(gw_sorted["생중량"] >= q3, "상위 25%",
                                              np.where(gw_sorted["생중량"] <= q1, "하위 25%", "중간 50%"))
                top_ids = gw_sorted[gw_sorted["분위"] == "상위 25%"]["개체번호"].tolist()
                bot_ids = gw_sorted[gw_sorted["분위"] == "하위 25%"]["개체번호"].tolist()
                st.markdown("**생중량 상위 25% 개체:** " + ", ".join(str(x) for x in top_ids))
                st.markdown("**생중량 하위 25% 개체:** " + ", ".join(str(x) for x in bot_ids))
            else:
                st.info("생중량 데이터가 3개 미만이에요. 분포를 보여주려면 더 많은 개체가 필요해요.")
        else:
            st.info("생중량 열이 없어요.")

        st.markdown("---")
        st.header("🔗 상관관계 분석")
        if growth_df is not None and not growth_df.empty:
            pairs = [("잎 수", "생중량"), ("지상부 길이", "생중량")]
            for xcol, ycol in pairs:
                if xcol in growth_df.columns and ycol in growth_df.columns:
                    st.subheader(xcol + " vs " + ycol)
                    sub = growth_df[[xcol, ycol]].dropna()
                    sub = sub[(np.abs(sub[xcol]) < 1e9) & (np.abs(sub[ycol]) < 1e9)]
                    if len(sub) >= 2:
                        r, p = stats.pearsonr(sub[xcol], sub[ycol])
                        slope, intercept = np.polyfit(sub[xcol], sub[ycol], 1)
                        xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
                        ys = slope * xs + intercept
                        fig_sc = go.Figure()
                        fig_sc.add_trace(go.Scatter(x=sub[xcol], y=sub[ycol], mode="markers",
                                                    marker=dict(size=9, color="#3498DB",
                                                               line=dict(width=1, color="white")),
                                                    name="개체",
                                                    hovertemplate="x: %{x}<br>y: %{y}<extra></extra>"))
                        fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                                    line=dict(color="#E74C3C", width=2, dash="dash"),
                                                    name="추세선"))
                        fig_sc.update_layout(
                            title={"text": "산점도 — " + explain_corr(xcol, ycol),
                                   "x": 0.05, "xanchor": "left"},
                            xaxis_title=xcol + " 측정값",
                            yaxis_title=ycol + " (g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            height=400,
                            margin=dict(l=40, r=20, t=70, b=40),
                        )
                        st.plotly_chart(fig_sc, width="stretch")
                        st.markdown("| 통계 | 값 | 설명 |")
                        st.markdown("|---|---|---|")
                        st.markdown("| 상관계수 r | " + f"{r:.3f}" + " | " + explain_corr(xcol, ycol) + " |")
                        st.markdown("| p값 | " + f"{p:.4f}" + " | 0.05보다 작으면 우연이 아닌 관계일 가능성이 높아요 |")
                        st.markdown("| 기울기 | " + f"{slope:.3f}" + " | " + xcol + "이 1 증가할 때 "
                                    + ycol + "는 평균 " + f"{slope:.3f}" + " g 증가 |")
                    else:
                        st.info(xcol + "과 " + ycol + "의 공통 데이터가 2개 미만이에요.")
                else:
                    miss = [c for c in [xcol, ycol] if c not in growth_df.columns]
                    st.info("열이 없어요: " + ", ".join(miss))

        with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
            if growth_df is not None and not growth_df.empty:
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSV 다운로드", csv_bytes,
                                   file_name="생육결과_다운로드.csv",
                                   mime="text/csv")
            else:
                st.info("생육 데이터가 없어요.")
    except Exception as e:
        st.warning("이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))
