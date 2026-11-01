"""沪深300双模型方向预测 · 可部署网页版本."""
from __future__ import annotations
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
import streamlit as st
import torch

from core import CNNStock, FEATURES, indicators, make_images

ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")
LOGO_PATH = ROOT / "assets" / "assist_logo.svg"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="HS300 双模型方向预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp {background:#0a1422; color:#edf3ff;}
.block-container {padding-top:1.45rem; max-width:1320px;}
[data-testid="stSidebar"] {background:#0c1b2d;}
h1,h2,h3 {letter-spacing:-.02em;}
[data-testid="stMetric"] {background:#12233a; border:1px solid #28405d; border-radius:16px; padding:16px 18px;}
[data-testid="stMetricValue"] {color:#f8fbff;}
.stButton>button, .stDownloadButton>button {border-radius:10px;}
.hero-wrap {padding:14px 18px 18px; background:linear-gradient(135deg,#132c4d,#0e2232); border:1px solid #36567a; border-radius:18px; margin-bottom:14px;}
.hero-tag {letter-spacing:.18em; font-size:11px; color:#8ed9ff; font-weight:700;}
.hero-title {font-size:30px; line-height:1.25; font-weight:800; margin:6px 0 8px; color:#ffffff;}
.hero-sub {color:#c4d4e6; font-size:14px; line-height:1.75;}
.hero-note {display:inline-block; margin-top:8px; font-size:12px; color:#cce7ff; background:#173958; border-radius:999px; padding:4px 10px;}
.note {font-size:13px; color:#9db2c9;}
.footer-box {padding:14px 16px; background:#0f1d30; border:1px solid #263f5b; border-radius:14px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=900, show_spinner=False)
def online_quotes() -> pd.DataFrame:
    """Fetch CSI 300 daily data from Eastmoney."""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": "1.000300",
        "klt": "101",
        "fqt": "0",
        "lmt": "2200",
        "end": "20500101",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    resp = requests.get(
        url,
        params=params,
        timeout=15,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/",
            "Accept": "application/json",
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    klines = (payload.get("data") or {}).get("klines") or []
    if not klines:
        raise RuntimeError("东方财富接口没有返回日线数据。")
    rows = []
    for item in klines:
        cols = item.split(",")
        if len(cols) < 6:
            continue
        rows.append(
            {
                "date": cols[0],
                "open": cols[1],
                "close": cols[2],
                "high": cols[3],
                "low": cols[4],
                "volume": cols[5],
            }
        )
    if not rows:
        raise RuntimeError("在线行情返回为空，无法解析。")
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def models():
    dma = joblib.load(ROOT / "models" / "dma_classifier.joblib")
    cnn = CNNStock()
    cnn.load_state_dict(torch.load(ROOT / "models" / "cnn_stock.pt", map_location="cpu", weights_only=True))
    cnn.eval()
    return cnn, dma


@st.cache_data(show_spinner=False)
def read_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_market():
    local = pd.read_csv(ROOT / "data" / "hs300_2021_2026.csv", encoding="utf-8-sig")
    try:
        live = online_quotes()
        merged = pd.concat([local, live], ignore_index=True)
        out = indicators(merged)
        newest = pd.to_datetime(live["date"]).max().date()
        return out, True, newest, None
    except Exception as err:  # noqa: BLE001
        return indicators(local), False, None, str(err)


def predictor(data: pd.DataFrame):
    last = data.iloc[-1]
    if last[FEATURES].isna().any():
        raise ValueError("有效历史数据不足，无法构造 DMA 特征。")
    cnn, dma = models()
    p_dma = float(dma.predict_proba(data.iloc[[-1]][FEATURES])[0, 1])
    xx, idx = make_images(data)
    if not len(idx) or int(idx[-1]) != len(data) - 1:
        raise ValueError("有效历史数据不足，无法构造 CNN 图像输入。")
    with torch.inference_mode():
        p_cnn = float(torch.sigmoid(cnn(torch.from_numpy(xx[-1:]))).item())
    return {"cnn": p_cnn, "dma": p_dma}


def decision(prob: float) -> str:
    return "上涨 ↑" if prob >= 0.5 else "下跌 ↓"


def verdict(prob: float, actual_up: bool) -> str:
    return "正确 ✓" if (prob >= 0.5) == actual_up else "错误 ×"


def db_connection():
    dbpath = Path(os.environ.get("PREDICTION_DB", str(ROOT / "predictions.sqlite3")))
    dbpath.parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(str(dbpath), timeout=20)
    cx.execute("PRAGMA busy_timeout=20000")
    cx.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            asof TEXT PRIMARY KEY,
            target_date TEXT NOT NULL,
            published_at TEXT NOT NULL,
            cnn_prob REAL NOT NULL,
            dma_prob REAL NOT NULL,
            data_last TEXT NOT NULL,
            model_version TEXT NOT NULL,
            data_source TEXT NOT NULL
        )
        """
    )
    cx.commit()
    return cx


def next_exchange_day(day):
    import exchange_calendars as xcals

    cal = xcals.get_calendar("XSHG")
    t = pd.Timestamp(day)
    if cal.is_session(t):
        return str(cal.next_session(t).date())
    return str(cal.date_to_session(t, direction="next").date())


def safe_to_publish(last_date, online_ok, online_last):
    if not online_ok or last_date != online_last:
        return False, "只有在线行情成功读取到最新收盘数据时，系统才允许写入正式公示记录。"
    now = dt.datetime.now(TZ)
    if last_date > now.date():
        return False, "行情日期晚于当前日期，禁止发布。"
    if last_date == now.date() and now.time() < dt.time(15, 20):
        return False, "北京时间 15:20 前不视为当日收盘数据确认完成，禁止发布。"
    if (now.date() - last_date).days > 9:
        return False, "在线行情超过 9 天未更新，禁止按实时预测公示。"
    return True, ""


def archive_auto(latest, market, online_ok, online_last):
    ok, why = safe_to_publish(latest, online_ok, online_last)
    if not ok:
        return False, why
    try:
        target = next_exchange_day(latest)
    except Exception as err:  # noqa: BLE001
        return False, f"无法确认交易所日历：{err}"
    values = predictor(market)
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with db_connection() as cx:
        cx.execute("BEGIN IMMEDIATE")
        exists = cx.execute("SELECT asof FROM predictions WHERE asof=?", (str(latest),)).fetchone()
        if exists:
            return True, "本期预测此前已公示，系统按唯一键保护，不会覆盖原记录。"
        cx.execute(
            """
            INSERT INTO predictions
            (asof, target_date, published_at, cnn_prob, dma_prob, data_last, model_version, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(latest), target, ts, values["cnn"], values["dma"], str(latest), "course-v2", "Eastmoney: 1.000300"),
        )
        cx.commit()
    return True, "本期预测已自动公示并写入存档。"


def read_archive() -> pd.DataFrame:
    with db_connection() as cx:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY asof DESC", cx)


def comparison(row, lookup: pd.DataFrame):
    target = row["target_date"]
    result = {
        "预测依据日": row["asof"],
        "目标交易日": target,
        "公示时间 UTC": row["published_at"],
        "CNN预测": decision(row["cnn_prob"]),
        "DMA预测": decision(row["dma_prob"]),
    }
    if target in lookup.index and row["asof"] in lookup.index:
        base = float(lookup.loc[row["asof"], "close"])
        final = float(lookup.loc[target, "close"])
        rate = (final / base - 1) * 100
        up = rate >= 0
        result.update(
            {
                "实际涨跌幅": f"{rate:+.2f}%",
                "真实方向": "上涨" if up else "下跌",
                "CNN核验": verdict(row["cnn_prob"], up),
                "DMA核验": verdict(row["dma_prob"], up),
            }
        )
    else:
        result.update(
            {
                "实际涨跌幅": "尚未产生",
                "真实方向": "待验证",
                "CNN核验": "待验证",
                "DMA核验": "待验证",
            }
        )
    return result


def render_header():
    c1, c2 = st.columns([1.1, 5.2], vertical_alignment="center")
    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
    with c2:
        st.markdown(
            """
<div class="hero-wrap">
  <div class="hero-tag">TIME SERIES · FINAL PROJECT · GROUP 7</div>
  <div class="hero-title">沪深300双模型方向预测系统</div>
  <div class="hero-sub">
    首尔科学综合大学院大学（aSSIST University）课程作业网页版<br/>
    模型一：多源图像 CNN_Stock　｜　模型二：DMA 趋势分类模型<br/>
    仅预测下一交易日方向；系统先公示预测，再等待实际行情自动核验；不提供买卖建议。
  </div>
  <div class="hero-note">支持发布为外部公网可访问 URL · 也提供静态预览页</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar():
    with st.sidebar:
        st.subheader("系统控制台")
        st.caption("第七组 · 时间序列期末考核作业")
        if st.button("↻ 刷新在线行情", use_container_width=True):
            online_quotes.clear()
            st.rerun()
        st.divider()
        st.write("**网页部署说明**")
        st.write("- 本地启动：双击 `启动网页版.bat`")
        st.write("- 公网部署：推荐 Streamlit Cloud / Render")
        st.write("- 静态展示：可直接打开 `离线界面预览_双击打开.html`")
        st.divider()
        st.write("**模型说明**")
        st.write("① CNN：OHLC + MACD + 成交量代理图像")
        st.write("② DMA：双均线差值及衍生特征分类器")
        st.divider()
        st.caption("正式公示记录只会在在线行情成功获取并确认收盘后自动写入。")


render_sidebar()
render_header()
market, is_online, live_last, live_err = load_market()
latest = market["date"].iloc[-1].date()
lookup = market.assign(date=market["date"].dt.strftime("%Y-%m-%d")).set_index("date")
metrics = read_metrics()

auto_ok, auto_msg = archive_auto(latest, market, is_online, live_last)
archive = read_archive()

if is_online:
    st.success(f"在线行情已连接：最新在线记录 {live_last} · 合并后的总样本 {len(market)} 条")
else:
    st.warning("当前为离线演示模式：无法获取实时行情，因此不会写入新的公示记录。失败原因：" + str(live_err))

m1, m2, m3, m4 = st.columns(4)
m1.metric("沪深300最新收盘", f"{market['close'].iloc[-1]:,.2f}")
m2.metric("当日涨跌幅", f"{market['pct_change_recomputed'].iloc[-1]:+.2f}%")
m3.metric("最新数据日期", str(latest))
m4.metric("运行模式", "LIVE" if is_online else "DEMO")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 次交易日预测",
    "🕘 历史日期回测",
    "🔐 公示与自动核验",
    "📊 模型效果",
    "🚀 公网部署说明",
])

with tab1:
    left, right = st.columns([1.8, 1], vertical_alignment="top")
    with left:
        st.subheader("最近 260 个交易日收盘走势")
        st.line_chart(market.set_index("date")["close"].tail(260), height=340)
        st.caption("数据标的：沪深300（000300.SH）。")
    with right:
        st.subheader("最新预测公示")
        st.caption(auto_msg)
        if auto_ok:
            current = archive[archive["asof"] == str(latest)]
            if len(current):
                row = current.iloc[0]
                st.info(f"预测依据日：{row['asof']}\n\n目标交易日：{row['target_date']}")
                c1, c2 = st.columns(2)
                c1.metric("CNN", decision(float(row["cnn_prob"])))
                c2.metric("DMA", decision(float(row["dma_prob"])))
                st.caption(f"CNN上涨概率：{float(row['cnn_prob']):.1%} · DMA上涨概率：{float(row['dma_prob']):.1%}")
                st.caption("公示时间（UTC）：" + row["published_at"])
        else:
            st.info("当前条件不满足正式公示；仍可查看历史回测和已有公示存档。")
        st.warning("本系统只输出涨跌方向分类结果，不构成任何投资建议。")

with tab2:
    st.subheader("历史日期回测（演示页面）")
    st.caption("这里是按当前模型回放某个历史交易日 T 收盘后，对下一交易日 T+1 的方向判断；该结果不是当日真实事前公示记录。")
    valid_dates = market.loc[market['date'].dt.date >= dt.date(2021, 5, 1), 'date'].dt.date.tolist()
    idx_default = max(0, len(valid_dates) - 31)
    choice = st.selectbox("选择预测依据交易日", valid_dates, index=idx_default)
    if st.button("计算该日期的双模型结果", type="primary"):
        i = int(market.index[market['date'].dt.date == choice][-1])
        chosen = market.iloc[: i + 1].copy()
        try:
            pred = predictor(chosen)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("CNN 模型", decision(pred['cnn']))
            cc2.metric("DMA 模型", decision(pred['dma']))
            if i + 1 < len(market):
                next_day = market.iloc[i + 1]
                delta = (float(next_day['close']) / float(chosen['close'].iloc[-1]) - 1) * 100
                actual_up = delta >= 0
                cc3.metric("次交易日真实方向", "上涨" if actual_up else "下跌")
                st.write(
                    {
                        "预测依据日": str(choice),
                        "目标交易日": str(next_day['date'].date()),
                        "CNN上涨概率": round(pred['cnn'], 4),
                        "DMA上涨概率": round(pred['dma'], 4),
                        "次日实际涨跌幅": round(delta, 4),
                        "CNN判断": verdict(pred['cnn'], actual_up),
                        "DMA判断": verdict(pred['dma'], actual_up),
                    }
                )
            else:
                cc3.metric("次交易日真实方向", "尚未产生")
        except Exception as err:  # noqa: BLE001
            st.error(f"该日期无法完成推断：{err}")

with tab3:
    st.subheader("预测公示存档与自动核验")
    st.caption("只有‘在线行情可达 + 收盘确认 + 交易所日历有效’时，系统才会自动公示该期预测。真实行情出现后，系统自动对比核验。")
    if archive.empty:
        st.info("目前还没有正式公示记录。")
    else:
        table = pd.DataFrame([comparison(r, lookup) for _, r in archive.iterrows()])
        st.dataframe(table, use_container_width=True, hide_index=True)
        csv = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载公示记录 CSV", data=csv, file_name="hs300_prediction_archive.csv", mime="text/csv")
    st.markdown(
        """
<div class="footer-box">
<b>防作假机制说明</b><br/>
1. 系统只允许在在线最新收盘数据确认后写入一条正式记录；<br/>
2. 数据库以 <code>asof</code> 为主键，正常页面访问不会覆盖既有公示；<br/>
3. 历史回测与正式公示分开呈现，避免把回测结果伪装成事前预测；<br/>
4. 真实涨跌依据相邻收盘价自动计算，平盘按“上涨”处理。
</div>
        """,
        unsafe_allow_html=True,
    )

with tab4:
    st.subheader("模型训练结果（来自本工程内 metrics.json）")
    st.caption("说明：本次 CNN 为按原思路重新训练的版本，不等同于上次小组作业中未保留下来的原始训练权重。")
    d1, d2 = st.columns(2)
    for box, key, title in [(d1, 'dma', 'DMA 趋势分类模型'), (d2, 'cnn', '多源图像 CNN_Stock')]:
        with box:
            st.markdown(f"### {title}")
            for split_key, split_title in [('validation', '验证集'), ('test', '测试集')]:
                s = metrics[key][split_key]
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{split_title}准确率", f"{s['accuracy']*100:.2f}%")
                c2.metric(f"{split_title}平衡准确率", f"{s['balanced_accuracy']*100:.2f}%")
                c3.metric(f"{split_title}MCC", f"{s['mcc']:.4f}")
                st.caption(f"{split_title}样本数：{s['n']}")
    st.write({
        '数据区间': f"{metrics['data_start']} ~ {metrics['data_end']}",
        '训练截止': metrics['split_train_end'],
        '验证开始': metrics['split_val_start'],
        '测试开始': metrics['split_test_start'],
        '备注': metrics['notes'],
    })

with tab5:
    st.subheader("如何发布为外部公网可访问 URL")
    st.markdown(
        """
**推荐方式 A：Streamlit Community Cloud（最快）**
1. 把本工程整体上传到 GitHub 仓库；
2. 登录 Streamlit Community Cloud，新建 App；
3. Main file 选择 `app.py`；
4. 部署完成后即可得到公网 URL；
5. 若需要长期保留 `predictions.sqlite3`，免费环境不够稳妥，建议改用 Render 或云服务器。  

**推荐方式 B：Render（更适合作业演示 URL）**
1. 把工程上传到 GitHub；
2. 在 Render 新建 Web Service；
3. 选择本仓库，平台会自动读取 `render.yaml`；
4. 成功部署后，Render 会分配 `https://xxx.onrender.com` 公网地址。  

**如果老师只看页面效果**
- 可直接打开工程里的 `离线界面预览_双击打开.html`；
- 但静态页只能展示界面和历史示意，不能完成实时预测与自动核验。  

**重要提醒**
- 老师的要求包含“程序必须能在线取数”；因此正式提交时，建议优先提交已部署好的 Streamlit / Render URL。
- 为了保证老师随时点开可用，建议在提交前一天完成部署并做一次收盘后联调。
        """
    )
    st.code(
        "streamlit run app.py --server.port $PORT --server.address 0.0.0.0",
        language="bash",
    )

st.caption("页头学校标识：aSSIST / 首尔科学综合大学院大学；本工程同时提供本地运行、静态预览和公网部署方案。")
