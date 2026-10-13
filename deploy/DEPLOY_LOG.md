# 部署记录 · 腾讯云 CVM

| 项 | 值 |
|---|---|
| 主机 | `ubuntu@106.54.192.87`（VM-0-7-ubuntu，Ubuntu 24.04，Python 3.12.3） |
| 应用目录 | `/opt/ttascope` |
| 运行环境 | `/opt/ttascope/.venv`（腾讯 PyPI 镜像） |
| 进程 | systemd `ttascope`（已 enable，开机自启） |
| 上游端口 | `127.0.0.1:8511`（uvicorn，不对外暴露） |
| 对外地址 | `http://106.54.192.87:8501/tta/` |
| nginx | `/etc/nginx/conf.d/hybridscope.conf` 内插入 `location /tta/` |
| 安全组 | 无变更，复用已放行的 8501 |

## 执行步骤

1. `bash deploy/upload.sh ubuntu@106.54.192.87` —— tar over SSH，上传
   `detector/ app/ models/ deploy/ requirements.txt README.md`，共 11 MB。
   实验语料、训练脚本、论文草稿均未上传。
2. `sudo bash /opt/ttascope/deploy/install.sh` —— 建 venv、装依赖、写
   systemd、插入 nginx location、`nginx -t` 通过后重载。全程 29 s。
3. `python tests/test_remote.py http://106.54.192.87:8501/tta` —— 公网冒烟。

`install.sh` 会先把 `hybridscope.conf` 备份为 `*.bak.<时间戳>` 再改。
`location /tta/` 单独关闭站点口令，便于外部测试。

## 实测结果（24 条混合批次：12 条 AI 风格 + 12 条人类随笔）

| 特征集 | 选中变体 | 推论 1 预测 F1 | AI 占比 | 逐条命中 | 首包耗时 |
|---|---|---|---|---|---|
| 15D | em_l1.0 | 0.7697 | 0.25 | 18/24 | 149 ms |
| char 2-4 | em_l0.5 | 0.9976 | 0.50 | 24/24 | 114 ms |

- 诊断恒等式 Δ_T = Δ − Ξ_ai + Ξ_hu 的残差 `identity_resid` 两个特征集均为 0。
- 15D 命中偏低与论文一致：手工风格特征跨域稳健但单域精度低于字符级 TF-IDF。
- 本地与线上结果逐格一致（同模型、同随机性），说明部署未引入偏差。
- 批次 24 条 < 50，接口返回 `warn` 提示诊断量仅供参考，判定仍可用。

## 运维

```bash
systemctl status|restart|stop ttascope
journalctl -u ttascope -f
curl http://127.0.0.1:8511/health          # 本机
curl http://106.54.192.87:8501/tta/health  # 公网
```

### 2026-10-10 · 门槛按特征集动态化

`E31_small_batch`（n=10/20/30/40）显示 15D 在该区间期望增益为负，把固定
20 条门槛改为按特征集取值：char_2_4 = 20 / 15D = 50 / roberta = 50，
常量统一放在 `detector/model.py`，由 `/api/models` 下发给前端。服务端不达标
返回 400，前端同步禁用。交叉对照在对照方不达标时主动跳过并说明。

改动：`detector/model.py`（MIN_N/REC_N + min_n/rec_n + detect 校验）、
`app/server.py`（接口暴露门槛 + 400 校验）、`app/web/index.html`（动态门槛、
选项标注、切换刷新、对照降级）、`tests/test_smoke.py`（新增 `test_min_batch_gate`）、
`tests/browser_check.js`（门槛切换与对照跳过）。
浏览器实测 15 项全 PASS，本地冒烟 6 项通过。


### 2026-10-10 · 文件上传 + 报告导出

- 上传：`txt / md / csv / jsonl / log`，可多选，浏览器本地解析不上服务器。
  UTF-8→GBK 编码兜底、CSV 列名识别与表头跳过、JSONL 字段提取、>8 MB 跳过并提示。
- 导出：Markdown 报告（溯源 / 结论 / 漂移诊断 / 候选变体 / **逐段分析表** /
  高置信段落 / **逐段原文** / 适用边界）与逐段分析 CSV（带 BOM）。
  页面只渲染 500 条，导出时按原参数重取 `item_limit=0`，保证覆盖全部段落。
- 测试：`tests/test_report.js`（Node 单测解析器与报告结构，直接从 index.html
  抽函数）、`tests/fixtures/`（sample.txt/csv/jsonl）、`tests/browser_check.js`
  扩到 28 项（含上传、CSV 取列、导出落盘、报告章节、表格覆盖 48 段、CSV 行数）。
  `tests/gen_sample_report.js` 生成可读样例报告。


更新代码：`bash deploy/upload.sh` 后 `sudo systemctl restart ttascope`
（模型文件换了才需要重启；只改 `app/web/index.html` 不必重启）。

## 排障记录

**现象**：浏览器打开返回 `{"detail":"Not Found"}`。
**原因**：访问日志显示请求路径为 `/tta/**（免口令，可直接发给人测试）`
—— 地址栏把说明文字一并粘了进去，后端无此路由。服务本身一直正常
（同一时刻脚本与 curl 全部 200）。
**已加防呆**：`app/server.py` 注册 404 兜底，非 `/api` 路径返回提示页并
给出相对链接 `./`（反向代理剥前缀，跳转必须用相对路径），`/api/*` 仍返回
JSON 以免前端解析失败。

访问地址只写这一行，不带任何附加文字：

```
http://106.54.192.87:8501/tta/
```

## 第二轮排障：按钮无反应 + 长文检测（2026-10-09 晚）

**按钮全部无反应的根因**：`index.html` 里对象字面量写成了 `{15D:'…'}`——
`15D` 以数字开头，不是合法 JS 标识符，**整个 `<script>` 解析失败**，
所有按钮（含"载入示例"）从未绑定事件。修为 `{'15D':…}`，并用
`node --check` 把前端脚本纳入语法校验。

**长文检测**：新增切分方式（按行 / 按段落 / 按句成块 / 按字数切块 /
整篇一条，块大小默认 300 字）+ 整篇结论卡（AI 块占比、平均 P(AI)、
混合度）。接口地址改为由 `location.pathname` 推导的绝对前缀，不再依赖
相对路径解析；首页禁缓存（`Cache-Control: no-store`）。

**块数纪律（重要）**：批校正靠类条件均值估计，块数 < 20 时整篇结论波动
很大——页面黄条提示，结果仅供参考。实测 6 块曾把 AI 长文误判为人类；
增大块数（减小块大小）后恢复。单条短文无法检测，这是方法边界不是缺陷。

**验证**：`tests/browser_check.js`（puppeteer-core + 无头 Edge）实测
载入示例 → 运行检测 → KPI 渲染 → 按句切块 → 整篇结论，全链路通过、
JS 零报错；`tests/test_long.py` 验证线上长文链路。

## 第三轮：数量门槛 + 默认特征集纠偏（2026-10-09 夜）

**数量门槛（用户要求）**：条数/块数 < 20 时**按钮禁用、不给检测**，输入框
下方实时显示"不足 20 条，还差 N 条，暂不可检测"（红字），并提示补行或
减小块大小；20–49 条放行但黄条说明诊断量波动；≥ 50 条诊断量稳定。
示例扩充到 24 条（原 12 条本身不达标），长文示例扩到 1883 字（按句/60 字
切出 24 块，双击「载入示例」即可直接跑）。

**默认特征集 15D → char_2_4（重要纠偏）**：同一篇 AI 长文（24 块）实测，
15D 只判出 4/24 块、整篇误判为人类；char_2_4 判出 14/24、结论正确。
默认改为 char_2_4 后，24 条示例批次 AI 占比正好 50.0%（真值 12/24）。

**新增交叉对照**：整篇结论自动用另一个特征集再跑一遍，一致标 ✅，
不一致标 ⚠️ 并提示"本批次可能缺少两类对照（整篇单一来源时常见），
请以人工复核为准"——不再给出看似确定实则不可靠的单一答案。

实测（`tests/browser_check.js`，9 项全 PASS）：空输入禁用 / 10 条禁用并
提示差 10 条 / 24 条可检 / 结果渲染 / 长文 24 块 / 整篇判 AI 生成 /
默认 char_2_4 / 交叉对照完成 / 无 JS 报错。

## 第五轮：校正口径对齐论文统一实现（2026-10-13）

**背景**：论文的最终实现（exp/X5_unified_impl.py::tta_final）已移除整批
对齐 bnc，改为每轮迭代把批混合中心钉回 ½(μ0+μ1)；表 6 / S1 / S2 / X10 全部
按新族重跑。而线上产品仍用旧实现（开头一次 bnc 预步 + 每步扣位移批均值），
候选族里还留着 bnc（共 11 个变体）。评审若对着论文去线上复算会对不上，
且 auto 选参可能选中已被论文判为"损害排序质量"的 bnc（表 S5 / §5.3）。

**改动**：
- `detector/tta.py`：新增 `tta_final`（逐行对齐 X5 实现）；旧 `align` 降级为
  复现旧实验用，标注不进在线候选族。
- `detector/model.py::_candidates`：候选族 11 → 10（`none` + `bn` + `em`×2
  + `conf`×6），移除 `bnc`，全部走 `tta.tta_final`。
- 前端 `app/web/index.html`：下拉删除 bnc 选项；`README.md` 同步 10 变体清单
  并说明移除理由。
- `tests/test_smoke.py`：新增断言候选族恰为 10 个且不含 bnc、指定 `bnc` 应
  抛 KeyError；恒等式残差断言由绝对阈值改为相对阈值（合成语料两类各重复
  40 遍 → Δ≈1e17，绝对 1e-3 必然误报，实测残差/Δ≈3e-16）。
- `tests/test_remote.py`：从 `/api/models` 读 min_n，用例条数不足时跳过该
  特征集并打印说明，不再把 400 当成失败（15D 门槛 50 > 用例 24 条）。
- 新增 `tests/_local_e2e.py`：不联网、加载真实 joblib 跑端到端，校验候选族
  口径与判定命中率。

**验证**：
- 本地 `test_smoke.py` 6 项全通过。
- 本地端到端：char_2_4 命中 24/24（候选族 10 个、无 bnc）；15D 用例为重复
  构造的病态批，只校验口径不断言命中率——与线上旧实现同输入 A/B 对照，
  两者 diag（Ξ_ai=−6.493、Δ_T=9.4575）与选中变体完全一致，确认差异来自
  语料而非本次改动。
- 线上 `test_remote.py`：`http://106.54.192.87:8501/tta` PASS，char_2_4
  选中 em_l0.5、命中 24/24、候选族返回 10 个变体。
- 示例报告 `sample_report.md` 按新版管线重生成（10 变体、无 bnc 行）。

**部署**：上传前在服务器备份 `detector.bak.1791881655` / `app.bak.1791881655`
作回滚点（此前 install.sh 是覆盖式、无回滚）；`sudo systemctl restart ttascope`
后 health 正常、两模型装载。

## 第六轮：类先验前置拒绝（2026-10-13）

**背景**：批次里两类严重失衡（几乎全是 AI 或几乎全是人类）时，批校正会把
少数类的尾巴硬拉过去，给出看似确定的结论。E39 扫描给出可判据：未校正批的
硬判 AI 比例 `r_AI` 与真实混合比 π 单调相关，取 cut = 0.64 时可用区（π=0.5）
误触发 0.00，危险区（π≥0.9）平均拦下 0.71。

**改动**：
- `detector/diagnostics.py`：新增 `REJECT_R_AI = 0.64` 与 `prior_reject(gamma0)`。
- `detector/model.py`：detect 返回新增 `prior_gate`（`r_ai` / `cut` / `rejected` /
  `explicit` / `applied` / `original_pick`）；触发且 `mode='auto'` 时 `pick='none'`，
  即退回不校正、只报原始分数。
- `app/web/index.html`：结果区新增 `prior_gate` 提示块，报告导出同步带一行。
  文案三态：applied＝已按规则拒绝校正；explicit＝显式指定变体、按用户选择执行；
  否则＝auto 本就选 none / 未触发。

**验证**：`exp/_verify_gate.py` 跑真实语料——π=0.5 时 r_AI 落在 0.37–0.59，
全部放行（零误伤）；π≥0.9 时 r_AI 0.66–0.995，全部触发。已上传并
`systemctl restart`。

## 第七轮：限流与资源护栏（2026-10-13）

**背景**：检测是 CPU 密集、uvicorn 单 worker，公网地址又是可直接转发的，
没有任何速率约束时，一个连点脚本或超大请求就能把整站拖死。

**改动**：
- `deploy/nginx-ttascope-limits.conf`（新，http 级）：`tta_page` 10r/s、
  `tta_api` 2r/s、`tta_conn` 单 IP 并发 6，`limit_req_status/conn_status = 429`。
- `deploy/nginx-ttascope.conf`：`/tta/api/` 独立 location（前缀更长优先匹配，
  `proxy_pass …/api/`），带 `limit_req burst=10 nodelay`、`limit_conn`、
  `client_max_body_size 8m`；两个 location 都 `error_page 429 = @tta429`，
  由 named location 返回 JSON + `Retry-After: 2`。配置块用哨兵注释
  `# >>> ttascope begin/end` 标记，便于整段替换。
- `deploy/nginx_sync.py`（新）：幂等写入限流区与 location 片段，识别旧版块并
  整块替换（避免新旧两个 `/tta/` 并存触发 duplicate location），写入前备份。
  `install.sh` 改为调用它。
- `app/server.py`：应用层护栏 `MAX_ITEMS=2000`、`MAX_CHARS=2e6`、
  `MAX_ITEM_CHARS=20000`（超长单条截断并在响应里带 `truncated`）；
  `/health` 回传 `limits`。顺带修了 warmup：原来只有 10 条，过不了 15D 的
  50 条门槛，预热全程被拒绝、首包仍要付约 3 s 的惰性导入开销——现在按
  `max(min_n)+10` 拼出 60 条互不相同的文本。
- `app/web/index.html`：429 走专门文案（主检测、交叉对照各一处），报告导出
  沿用 `detail`。
- `tests/test_limits.py`（新）：`/health.limits`、条数超限 400、总量超限 400、
  单条超长截断、正常批次 200。本地 5 项全通过。

**验证**：
- 服务器本机连发 30 次 `/tta/api/models`：前 11 次 200，之后全部 429，
  响应体为 `{"detail":"请求过于频繁…"}`、`Retry-After: 2`。
- 公网连发 20 次：18 次 200 后转 429（curl 往返本身在补令牌，符合 2r/s）。
- 公网 POST 2100 条：400 `单批最多 2000 条…`。
- `tests/test_remote.py` PASS：char_2_4 选中 `em_l0.5`、命中 24/24、
  AI 占比 0.5、候选族 10 个且无 bnc；`/health` 回传 limits 三项。
- `tests/test_smoke.py` 7 项通过，`tests/test_limits.py` 5 项通过。

**回滚**：`detector.bak.1791891234` / `app.bak.1791891234`（本轮上传前），
nginx 侧为 `/etc/nginx/conf.d/hybridscope.conf.bak.1791891423` 与新建的
`ttascope-limits.conf`。
