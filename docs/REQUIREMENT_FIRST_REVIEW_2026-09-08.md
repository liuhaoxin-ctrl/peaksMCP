# 需求先行复查：peaksMCP 是否围绕"意图驱动 + 函数组合 + 规范输出 + consent 保存"最小成立

日期：2026-09-08（基线 HEAD 80043b3 + 工作区未提交 diff）
方法：从底层重新遍历文件树（Python ≈7.3k 行 / 测试 ≈6.5k 行 / 前端 TS 505 + dashboard 1.2k），
**不把现有结构当作设计前提**，只拿下面的核心需求当标尺，逐文件核对"这一层在需求里对应什么、是否多余"。

## 0. 标尺（唯一依据）

1. **函数访问**：LLM 通过 `search`/`get` 使用"黑箱内函数"（本项目函数 + peaks 原生函数）；
   任务目标由模型从用户意图理解，**组合逻辑写在 notebook 里**，模型负责调度、不重复实现已有功能。
2. **输出规范化**：调用已有函数 → 直接展示函数自身的标准输出；模型自写代码 → 规范简洁输出；
   降低人工 review 成本，不产生低价值噪音。
3. **结果保存**：默认不保存；**只有用户明确 consent 才保存**；consent 的前提是**结果先完整展示给用户确认**。

据此推导出"从需求出发的最小架构"，再拿现状比对。最小架构只有 5 个要素：

```
Jupyter notebook（用户的审查面，append-only 执行日志）
 └─ ① 函数目录：search(catalog) / get(detail)          [需求1]
 └─ ② 一次执行：写 cell → 扫描 → consent → 执行 → 返回"规范化输出" [需求1+2]
 └─ ③ 输出契约：函数原生输出进 notebook；模型文字简短规范       [需求2]
 └─ ④ 保存闸门：写盘动作 = 展示 → 用户确认 → 落盘           [需求3]
（进程/桥接等基础设施是让 ② 跑起来的必要条件，不是需求本身）
```

---

## 1. 现状底数（按最小架构四要素归类）

| 需求要素 | 现状模块 | 规模（近似行） |
|---|---|---|
| ① 目录 | `discovery/index.py`(AST+运行时双扫描、双 tier、指纹)、`signatures.py`、`api_overrides.yaml` | 736+179+YAML |
| ② 执行 | `backend/notebook_unsafe.py`(294) + `security/code_scanner.py`(768) + `api_provenance.py`(215) + `api_allowlists.py`(106) + `ipython_scanner.py`(50) + `consent.py`(23) + `audit.py`(32) + 前端 `index.ts` execute_code/add_cell + Comm 桥 | ≈1.7k |
| ② 工具面 | `core/tools.py`(277) 注册 14 工具；`metadata_baseline.yaml`；`prompts.yaml` | 14 工具 |
| ③ 输出契约 | `tools.py:_output_content`；各黑箱函数模块内自带的"一行 print + 返回 dict"约定（如 `slice_view.py`）；`prompts.yaml` server_instructions | 零散 |
| ④ 保存闸门 | 扫描器 SAVE001/SAVE002 + `require_consent` 开关 + 前端 consent dialog；`convert_pxt` 幂等写盘（豁免）；savefig 永久禁 | 散落 |
| 基础设施 | `app/runtime.py`(729)、`app/api.py`(346)+webapp、`cli.py`(659)、`app/kernel.py`、`app/profiles.py`、`observability/runfile.py`、`transport/stdio_proxy.py`、JupyterLab extension | ≈3k |

结论先行：**需求 1（目录+组合）基本成立且质量高；需求 2（输出规范化）的实现位置错了——规范化逻辑不在执行返回路径上，模型每次执行收到的恰恰是未规范化输出；需求 3（展示→确认→保存）没有成形的机制，现状是"默认不确认就放行写盘 + 图形永久禁存"两个极端。**

下面按需求逐条给证据与冗余判定。

---

## 2. 需求 1 核查：search/get + notebook 组合

现状已实现：override-first 双 tier 搜索（`search_index_tiered`），`peaks_get_api` 返回签名/docstring/note（override 条目隐藏 source_path 实现路径，符合"黑箱"）；写代码时 `write_with_api_check` 逐调用点校验 API 名（未知名 escalate）。方向上贴需求。

### 1.1 冗余：目录里混着"内部零件"，且索引收录 > 审定清单

- `api_overrides.yaml` 里 `project: true` 共 18 个，其中 `default_output_dir`、`index_from_path`、`register_l112_loader`、`batch_execution_lock` 是转换器/批处理的**内部零件**，不是用户意图的动词。模型不需要"知道输出目录怎么算"——它需要的是 `convert_path` 返回目标目录。这四个条目进目录等于把实现细节塞进黑箱面，且每个都多一份 L1 别名+docstring 维护成本。
- 更深一层：`build_index()` 用 AST 扫 `peaksMCP.{plotting,workflows,pxt_utils,batch}` 下**所有公开函数**（`scan_modules` 无 project 过滤），18 个审定条目只是其中被标记 override 的部分。`csv_translator` 顶层工具函数、`metadata` 各判定函数等即使不在清单里也能被搜到——审定面与可搜面不一致（有测试断言"声明↔暴露"双向一致，但那只是保护审定 18 个，不是保护"目录里只有审定 18 个"）。从需求看：目录应只含"该给模型看的动词"，其余降为私有。

### 1.2 冗余：写时校验与 search 是两套"名字→API"解析

写代码时每个调用点做 `index.search(target.leaf,...)` + scope 二次匹配（`notebook_unsafe.py`），配套 `api_provenance`(215)+`api_allowlists`(106)+`verified_peaks_names`/`unknown_api_attempts` 状态机（`base.py:57-64`）。而 search/get 本身已经是"名字→API"的权威。需求 1 只要求模型先 search/get；现状为防幻觉又建了第二张"名字解析网"（外加 AST 扫描器第三张网）。三张网各维护各的规则表，规则重复/漂移成本高：
- `code_scanner._FILE_WRITERS` 与 `api_allowlists.GENERIC_METHODS` 各自枚举文件写类方法；
- `notebook_unsafe._saves_figure` 与 scanner 的 `_is_savefig` 是同一条规则的两份实现（前者 `call_names` 只用于 savefig，等于为一个特例再解析一遍 AST）；
- 状态机部分只在模型"不守规矩"时触发，正常流程（先 get 再写）下是纯死重。

建议：以 search 的 scope/name 校验为唯一校验（已是权威索引），把 escalate 状态机简化为"写前必须成功 `peaks_get_api` 过（一次性 unlock 已有）"；`call_names`+savefig 特判并入 scanner（scanner 已经挡了）。

### 1.3 项目函数自身的重复/分层问号（需求 1 的"调用已有函数"审查）

- `load_metadata`(dict) 与 `read_meta`(加载+打印记录表)：`read_meta` 的增量是"打印一张表给用户看"。若该表正是函数标准输出（需求 2 第 1 句），`load_metadata` 反而是可省的一层；否则 `read_meta` 是包装层。**二选一**，别两个都进目录。
- `publication_grid` = `validate_arpes_metadata` + `plot_batch`（`workflows/publication.py:44-85`）。作为"发表级"入口说得通（dpi=300 等收敛），但 validate 列表默认值下经常为空数组、不产生动作；两层 faÃ§ade 叠加是否真比 `plot_batch(..., dpi=300)` 省事，缺使用证据。建议保留一个并让目录里写清"何时用它而不是 plot_batch"，避免模型两个都试。
- `register_l112_loader` 已在 `jupyter_mcp_extension._start()` 自动注册（extension 加载即生效），**模型根本不需要调用它**——它在目录里是纯噪音入口，应设为私有并从目录移除（运行时注册留在 extension）。
- 绘图四入口（plot_batch/plot_validation_pair/show_mapping_slice/publication_grid）各自动词可区分，属合理黑箱面；但目录检索词与 `prompts` 的 server_instructions 里"优先 façade"规则需要保持一致，目前文字层已有一致性（L1/L3/L5 分离），无新增冗余。

---

## 3. 需求 2 核查：输出规范化 —— 最大偏差在这里

### 3.1 【核心偏差】规范化实现不在执行返回路径上，模型每次执行收到的是"原始全量输出"

证据链：
- `write_with_api_check` → `execute_code` → `bridge.request("execute_code")`，前端 `index.ts:227-253` 返回 `executedJSON()`：**含该 cell 的全部 outputs（文本 + text/plain repr + base64 图片，图片上限单张 8MB/合计 16MB，`boundedOutputs`）**；`notebook_unsafe.py:291` 原样返回并附 `api_check`。→ 模型每条代码都收到整段原始输出（要抑制的恰恰是这些）。
- 规范化函数 `_output_content`（`tools.py:125-190`，行为正确：纯文本/markdown 不回传、图片只计数、错误透传）**只挂在一个独立工具 `notebook_read_active_cell_output` 上**（`tools.py:239`），且该工具读的是"用户光标所在 cell"的输出（`notebook.py:178-187` 按 `state.active_cell["id"]` 查缓存）——模型执行完还得 `move_cursor` 指向自己刚写的 cell 再读一遍。
- 结果：模型被淹没（16MB 级原始 JSON），而"正确的那份输出"要模型额外两步才拿得到。上一轮审查提过"把 `_output_content` 接到写工具返回值"，本轮实现走反了方向（规范化留在读工具上，写工具裸奔）。

**修复（需求 2 的头号动作）**：`write_with_api_check` 返回前调 `_output_content`（identity 由执行结果自带，天然正确），删 `notebook_read_active_cell_output`/`notebook_read_active_cell`/`notebook_move_cursor`/`notebook_read_content` 四个工具与对应前端 op（对模型执行循环不再有意义），只保留面向"用户手动 cell/变量上下文"的最小读取。

### 3.2 六个"读上下文"工具里四个是为补偿 3.1 而存在

`tools.py:231-243`：list_variables / read_variable / read_active_cell / read_active_cell_output / read_content / move_cursor。修好 3.1 后，模型侧组合循环真正需要的是 **read_variable（拿类型/结构好写下一段）**；list_variables 与 read_variable 可合并为一个（列表项已含 dims/shape）。四个定位/回读工具对"模型闭环"是死重（对用户手动浏览场景保留 read_active_cell 一个即可，若仍要）。

### 3.3 `summarize_xarray` 13 字段正面违反"不过度输出"

`notebook.py:107-126` 返回 type/name/dims/sizes/dtype/variables/coords/units/attrs/chunks/peaks_apis/lazy 等 13 组。人 review 时通常只需要 dims + shape + units（+ 是否 lazy）。逐字段数一下：`chunks`、`peaks_apis`（通过 `__mro__` 扫出来的 accessor 名集合）、`attrs` 全量 `_json_value` 都是高噪音低信息。需求 2 的"规范简洁"应落在这里：默认返回 4 字段，其余按需 `detail=` 打开。

### 3.4 状态工具三兄弟信息量为零，属"为存在而存在"

`notebook_server_status` / `notebook_kernel_status` / `notebook_wait_for_kernel`（`tools.py:241-243`）。执行是**同步**的（前端 `runCells` 完成后才回包），`wait_for_kernel` 几乎永远立刻 `ready`；前两个的信息（mode/api-count/comm/busy-since）模型没法据此改变行为。真正用 `notebook_server_status` 的是基础设施探测（`stdio_proxy.check_http_mcp_server`、`runtime.wait_ready`），不是模型。建议：探测走内部通道（如保留一个 `_server_status` 内部工具或直接查 dashboard /api/status），模型面删掉三兄弟。

### 3.5 输出"双通道 + 双缓存"重复

同一份 cell 输出：
- 通道：前端 `watchExecutionOutputs` push（`index.ts:180-201`，30 秒窗口）+ 请求/响应直接返回（3.1 已述）；
- 缓存：`SharedState.active_cell_output` 与 `cell_outputs`（`base.py:51-52`）两份同构数据，`active_cell_bridge.py:103-126` 与 `notebook.py:156-187` 两处各自维护同一套"写入+容量裁剪"逻辑（代码重复两份）。

3.1 修复后，模型路径不再需要 push 回读；`active_cell_output`/`cell_outputs` 合一（若保留"用户手动 cell 查看"功能）或整体删掉（若不留读输出工具）。

### 3.6 前端死 handler

前端 9 个 op（index.ts:204-285），Python 只发 7 个（consent/execute/add/read_active_cell/read_notebook/move_cursor + supervisor 注入的 save_notebook/restart_kernel）。`read_cell_at`（index.ts:268-279）**无任何 Python 调用方**，死代码（上一轮已删 delete_cell/apply_patch，这个漏网）。

### 3.7 "模型自写分析输出规范"没有落地机制

需求 2 第二句（模型自行生成的分析/处理结果要规范简洁）目前只靠 `prompts.yaml` 一句 general 话术。notebook 里模型可写任意长 markdown、可 print 任意调试文本（print 属于函数标准输出？——模型代码里的 print 是它自己的，不是"已有函数的标准输出"，按需求应该被规范约束）。建议：把"输出契约"写成 prompts 的显式规则 + 黑箱函数文档首行统一约定（`slice_view.py` 的"一行 print + 返回 dict、无进度条"已是好模板，推广到所有 project 函数并在 get 的 docstring_note 里固化），并给执行器一个"执行后单行回执"（如 `{executed_cell_index, outputs: [figure_count, error?]}`）作为默认形态。

### 3.8 遗留文档/规则冲突（小）

`code_scanner.py:83-86` 注释说 savefig 走 `requires_explicit_consent`（不硬禁、可批准），而 `_classify_call` 实际把 SAVE001 追加进 `issues` **硬禁**，`notebook_unsafe.write_with_api_check` 开头还有 `_saves_figure` 二次硬禁——注释与代码三层不一致（见 4.2）。另外工作区未提交 diff 还在改这段注释，说明本轮刚动过但方向是"说它软"，与代码"做它硬"相反，需要定夺。

---

## 4. 需求 3 核查：默认不保存 / 展示后 consent —— 机制缺口

### 4.1 三个落盘面，只有"图形"和"扫描器能看见的写"受控，且默认都不受控

| 落盘动作 | 现状 | 与需求 3 的关系 |
|---|---|---|
| 模型 cell 里 `xr.to_netcdf`/`np.save`/`pd.to_csv` 等（SAVE002） | 扫描器归入 `requires_explicit_consent`；**只在 `require_consent=True` 时弹窗**。默认 `require_consent=False`（`app/profiles.py:34`）→ **直接放行写盘，无确认** | 违背"只有 consent 才保存"。默认配置下分析结果可被静默落盘 |
| 模型 cell 里 `savefig` | 永久硬禁（SAVE001 + `notebook_unsafe._saves_figure` + 文案"没有用户确认保存图片的路径"） | 与"consent 后可保存"相反：**用户永远无法同意保存图形**，需求 3 的"consent 授权保存"在这里不存在 |
| `convert_pxt`/`convert_path`（函数内部 to_netcdf） | 幂等自动写盘，豁免（api_overrides docstring_note 明说"非 save-result、不需 consent"） | 语义上属于"处理原始 PXT 的第一步前置转换"（用户已给出输入文件与路径），可接受，**但"豁免"目前是靠文档承诺，不是机制**——见 4.3 |

即现状是需求 3 的两个极端同时存在：**无确认就放行**（默认）与**永远不许存**（savefig），中间缺的正是需求要的"完整展示 → 用户确认 → 落盘"。

### 4.2 保存机制的根问题：写盘意图的最终裁决权在"一个默认关掉的开关"

- `require_consent` 开关语义（AGENTS/代码一致）：默认 false = 扫描器硬禁 + 审计，其余全放行。但扫描器对 SAVE002 只判"require"，不判"是否用户明示"。
- 需求 3 要的是**面向落盘动作的强制前置**，与"每次执行都要弹窗"（开关 true 时的现状）是两件事。更贴需求的形态：`require_consent` 只管"是否每次执行都确认"；**检测到写盘意图（SAVE002/savefig/network）时无论开关如何都走一次"展示→确认"**（确认内容 = 将写入的路径 + 该 cell 已渲染的结果，可复用前端 consent dialog，把待保存对象本身放进去预览）。
- 图形"永久禁"应改为 SAVE002 同款（永不自动、可批准），否则无法满足"用户确认后保存结果"。

### 4.3 "豁免清单"是文档不是机制

converter 的写盘豁免只存在于 docstring_note 与 AGENTS 文字。写盘判断在扫描器按"调用点语法"（函数内部的 to_netcdf 在调用点不可见，`peaksMCP` 又是 compute 白名单模块），所以 `convert_path(..., output_dir=...)` 落到哪、写多大，扫描器不感知。需求 3 下建议：要么把 convert 家族也纳入 SAVE002 式确认（输入文件+输出目录已在参数里，可完整展示，非常符合需求 3 的形态），要么在机制里显式登记"预转换豁免"（如允许清单），别靠文档。

### 4.4 基础设施的自动落盘（政策问题，需裁定）

- 前端每次 execute_code/add_cell 后自动 `panel.context.save()`（index.ts:291-302）——append-only 日志要防 supervisor 重启丢失，动机正当；但严格讲这是"无 consent 的磁盘写"。若 notebook 文件本身算"结果"，它违背需求 3 字面；若算"执行日志/工作区"，它是持久化的必要设施。**裁定建议**：notebook 属日志（append-only 语义已保证可审计），落盘视为环境保真而非结果保存；但要在文档里写明边界，避免以后被人当"自动保存分析结果"引用。
- dashboard `snapshot`/`_flush_frontend_save`（api.py:32-51, 153-211）——operator 在控制台手动按按钮才触发，是"人的显式动作"，合规，保留。

---

## 5. 结构层：与需求无关的"同一件事多份"（对前几轮已删项的复核）

上一轮收敛（CLI/dashboard 数据入口、resources、Inspector、delete_cell/apply_patch、ExecutionMode、工具名清单×4→1）**已确认落地**：cli.py 无数据子命令、api.py 无数据端点、index.ts 无 delete/apply_patch、tools.py 只有 require_consent、工具名单源（metadata.py tool_names() + stdio_proxy 引用）。以下为**仍未收敛或新增**的项：

| # | 现象 | 位置 | 判定 |
|---|---|---|---|
| S1 | 模型工具面 14 个里 ≥8 个服务于"补偿式回读/状态"（见 §3），最小闭环 4~6 个工具即可 | tools.py | 删（见 §7 目标面） |
| S2 | 同一输出双缓存 + 双通道，且两处后端逻辑代码重复 | base.py:51-52 / bridge.py:103-126 / notebook.py:156-187 | 合一 |
| S3 | 前端 op 9 实现 vs 7 调用；`read_cell_at` 无调用方 | index.ts:268 | 删 |
| S4 | `is_stale()` 每次 search/write 全树 os.walk（peaks+peaksMCP 全部 .py stat）算指纹 | index.py:127-160,730-737；base.py:12-33 | 高频路径 CPU 浪费：改"适配文件 mtime 增量检查 + 低频率全树指纹" |
| S5 | 三张名字解析网规则重复（scanner / allowlists+provenance / search） | §1.2 | 收敛为一张 |
| S6 | 状态三兄弟工具 + dashboard /api/status + CLI status = 三个状态面 | §3.4 | 模型面删，探测走内部 |
| S7 | `load_metadata`/`read_meta`、`plot_batch`/`publication_grid` 疑重复 | §1.3 | 各收敛为一个（保留 read_meta/plot_batch，前者加 detail 参数） |
| S8 | 目录含 4 个内部零件 + `register_l112_loader` 运行时已自动注册 | §1.1 | 移除出目录 |
| S9 | 配置 5 个 YAML + prompt 6 层投递 | config/metadata_baseline.yaml / discovery/api_overrides.yaml / config/prompts.yaml / app/profiles.py / app/defaults | 文字层分离合理，维持；但 L5 注入的 docstring_note 与运行时 docstring 无自动校验，漂移风险仍在（有测试，缺"note 与真实签名"类全覆盖） |
| S10 | 生命周期过深：cli 14 子命令 + dashboard 5 控制端点 + 4 magics + runfile/SIGWINCH/stale 回收（runtime.py:729 + cli.py 八个 host 管理 helper） | cli.py/app | 面向"多个 host 实例接管"问题的工程，核心需求只需"能启动一个稳定环境"；若接受单工作流可砍一半。属基础设施，非数据面冗余，按需取舍，本轮不列为必须动作 |
| S11 | 测试重心：test_runtime(515)/test_profiles_cli(562)/test_security(627) 仍占大头；对"执行返回规范化输出"“展示→确认→保存”两条需求契约**无端到端断言**（现有 test_tools 只单测 `_output_content` 单元，不测写工具返回路径） | tests/ | 与需求 2/3 对齐补契约测试 |

---

## 6. 需求 2/3 的两条"契约测试"缺位（本轮最值得先补）

- 无测试断言 `write_with_api_check` 的返回**不含**原始文本输出/大图、**含**规范化摘要（3.1 的回归护栏）。
- 无测试断言"含写盘动作的 cell 在默认配置下被要求展示+确认，而不是直接执行"（4.2 的回归护栏）。
- 无测试断言"savefig 经用户确认后可保存"（若采纳 4.2 的改法）。
- 无测试断言前端 `read_cell_at` 无调用方（防死代码复现，可并入现有 frontend op 测试）。

---

## 7. 目标形态（需求先行收敛后，工具面 14 → ~6）

```
模型可见（≈6）：
  peaks_search_api / peaks_get_api        # 需求1：目录
  notebook_write_code                     # 需求1+2+3：写 cell→扫描→(写盘动作→展示+确认)→执行
                                          #   → 返回规范化回执（错误/图计数，无文本回声）
  notebook_add_markdown                   # 需求2：模型自写内容（规范由 prompts 输出契约约束）
  notebook_read_variable                  # 需求1：拿类型/结构写下一段（合并 list）
  notebook_read_notebook                  # （可选保留）用户手动 cell/整簿概览
基础设施（模型不可见）：
  server_status/探测  → stdio_proxy/dashboard 直接查 /api/status 或内部工具
  其余读取/状态/定位/cursor 工具全部删除

数据层：
  convert_pxt / convert_path / load_pxt / read_meta / translate_datasheet / plot_batch /
  plot_validation_pair / show_mapping_slice（+ publication_grid 与 plot_batch 收敛）→ 目录
  default_output_dir / index_from_path / register_l112_loader / batch_execution_lock /
  classify_data_format / is_gold_format / theta_offset_deg → 私有或并入宿主函数

保存（需求3）：
  写盘意图（SAVE002/savefig/convert 家族）→ 展示（路径+已渲染结果）→ 用户确认 → 落盘
  require_consent 只保留"每次执行都弹窗"的增强语义，不再承担保存闸门
```

---

## 8. 优先级建议

1. **P0（需求 2）**：规范化接到 `write_with_api_check` 返回值；删 4 个回读/定位工具 + 双缓存合一 + `read_cell_at`。一次改动同时消除 §3.1/3.2/3.5/3.6。
2. **P0（需求 3）**：写盘意图默认必须"展示→确认"（不再依赖默认 false 的开关）；savefig 从永久禁改为可确认；convert 家族豁免从文档变成机制或纳入确认。§4.1/4.2/4.3。
3. **P1（需求 1）**：目录只留审定动词，内部零件与自动注册项移出；`load_metadata`/`read_meta`、`plot_batch`/`publication_grid` 收敛。
4. **P1（契约测试）**：第 6 节四条护栏。
5. **P2（结构）**：指纹高频路径改增量检查（S4）；模型面状态三兄弟下线（S6）；三张名字解析网收敛（S5）——其中 S5 与 P0-1 结合后，`notebook_unsafe` 的逐调用点校验可改为"scanner + verified-name"两层即可。
