# 沪深300双模型方向预测系统（第七组 · 课程考核发布版）

## 本版新增内容
- 已在网页头部加入 **aSSIST / 首尔科学综合大学院大学** 标识。
- 同时提供：
  - 本地可运行的 Streamlit 网页工程；
  - 可直接双击查看的静态预览页；
  - 可发布到公网 URL 的部署文件（Render / Streamlit Cloud / Docker）。

## 工程文件结构
- `app.py`：主网页（五个页面：实时预测、历史回测、公示核验、模型效果、部署说明）
- `core.py`：特征工程与 CNN 图像构造
- `models/`：已训练模型与 `metrics.json`
- `data/`：历史行情基础数据
- `assets/assist_logo.svg`：学校页头标识
- `离线界面预览_双击打开.html`：静态界面预览页
- `启动网页版.bat`：Windows 一键启动
- `render.yaml` / `Procfile` / `Dockerfile`：公网部署文件
- `DEPLOY_PUBLIC_URL.md`：公网部署详细说明

## 本地运行
### Windows
1. 安装 Python 3.11 或 3.12。
2. 双击 `启动网页版.bat`。
3. 浏览器访问 `http://127.0.0.1:8501`。

### 命令行
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 公网发布
请优先阅读 `DEPLOY_PUBLIC_URL.md`。

## 作业要求对应关系
- **至少两个模型**：已包含 CNN_Stock 与 DMA 两个模型；
- **只预测方向**：只输出上涨 / 下跌；
- **先预测，后核验**：正式公示记录写入 SQLite，再等待真实行情自动比对；
- **老师点开即可运行**：支持发布到 Streamlit Cloud / Render 得到公网 URL；
- **静态页面**：可作为备用演示，但不能替代正式在线系统。

## 重要说明
- 目前 CNN 的测试表现较弱，DMA 略优；系统已如实展示各项指标。
- 原 CSV 中的 `pct_change` 未直接用作标签；实际方向以收盘价重新计算。
- 如果需要“老师随时点开可用”，建议在提交前实际部署一次，并把公网 URL 和本压缩包一起提交。
