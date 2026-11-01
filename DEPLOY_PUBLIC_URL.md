# 公网部署说明（提交给老师的正式 URL）

## 方案一：Streamlit Community Cloud（最省事）
1. 把本文件夹整体上传到 GitHub 仓库。
2. 登录 https://share.streamlit.io/ 。
3. New app -> 选择仓库 -> Main file 选择 `app.py`。
4. 部署成功后得到公网 URL，可直接发给老师。

> 注意：免费环境可能会休眠，SQLite 预测记录也可能因重启而丢失。适合课程展示，但不适合长期严肃留痕。

## 方案二：Render（更适合稳定公网访问）
1. 同样先上传到 GitHub。
2. 登录 https://render.com/ -> New -> Web Service。
3. 连接仓库后，Render 会自动读取 `render.yaml`。
4. 首次部署完成后得到 `https://xxxx.onrender.com` 公网地址。
5. 在 Render 的磁盘或持久化目录中保存 `predictions.sqlite3`，这样老师后续查看仍能看到历史公示记录。

## 方案三：云服务器 / 宝塔 / Docker
- 直接使用包内 `Dockerfile` 构建镜像，或在 Linux 中执行：
  `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
- 再使用 Nginx 做反向代理即可形成正式公网 URL。

## 静态页面的用途
- `离线界面预览_双击打开.html` 只能作为界面预览或备用展示；
- 它不能完成老师要求的“在线取数 + 自动核验 + 正式公示留痕”。

## 正式提交建议
- **优先提交：已部署成功的公网 URL**
- 同时附上：
  1. GitHub 仓库地址（如有）
  2. 网页静态预览页
  3. 压缩包源码作为备份
