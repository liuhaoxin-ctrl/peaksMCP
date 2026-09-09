# peaksMCP 全项目冗余审查（2026-09-08）

判定标尺沿用你给出的 intent（不是项目文档里的自我描述）：

1. AI 通过 `search` / `get` 使用黑箱函数，必要时用 peaks 原生函数，**组合逻辑写在 notebook 里**；
2. notebook 输出**自动规范化**，用函数就输出函数输出，**不过度输出**；
3. **不轻易保存**，保存要 consent；
4. consent 的前提是**结果先展示给人看**。

规模基线：Python **8960 行**（42 个文件）、前端 TS **529 行**、dashboard 前端 **1375 行**、测试 **5807 行**（29 个文件）、工作区里 `peaksMCP/extensions` **479M**（node_modules 389M + .yarn）、`build/`+`dist/` 522M（未入库）。

---

# 第一部分 · 结构冗余（与 intent 无关的"同一件事有多份"）

## S1 【最重】同一个数据功能有三条前端入口

| 功能 | ① CLI | ② Dashboard HTTP | ③ notebook / MCP（intent 要求的唯一路径） |
|---|---|---|---|
| PXT → NetCDF 转换 | `peaksMCP convert`（`cli.py:611`） | `POST /api/convert`（`api.py:483`） | `convert_path` / `convert_pxt` |
| datasheet 翻译 | `peaksMCP metadata translate`（`cli.py:594`） | `POST /api/metadata/translate`（`api.py:471`） | `translate_datasheet` |
| 载入 notebook | `peaksMCP load`（`cli.py:619`） | `POST /api/notebook/load`（`api.py:509`） | `load_pxt` / native `load` |

按 intent 1，这三个操作都该是"模型在 notebook 里调黑箱函数"。现在它们各有一条绕开 notebook 的通道，而且 ①② 都不经过扫描器、API 校验和 consent。

顺带：`dashboard` 还内置了一个 **Inspector**（`api.py:33-38` 的 `_INSPECTOR_ALLOWED`，8 个只读工具 + `POST /api/mcp/tool` 代理），功能上是 Claude Desktop 的 read-only 复刻。

## S2 三条把代码送进 kernel 的路径

1. **MCP** `notebook_write_with_api_check`：扫描器 → API 校验 → consent → 执行（唯一合规路径）；
2. **Dashboard `load_into_notebook`**（`api.py:61-334`，**270 行**）：拼出 `from peaks import load; data = load(...)`，起一个后台线程通过 comm 桥把代码塞进 kernel 执行，**完全绕开扫描器 / API 校验 / consent**；
3. **`RuntimeSupervisor.execute_kernel`**（`runtime.py:434`）：进程管理用的直连 kernel client，`_flush_frontend_save` 就走它（`api.py:42`：注入代码调用 `b.request('save_notebook')`，即**自动保存 notebook 到磁盘**）。

路径 3 是"自动落盘"，路径 2 是"自动执行 + 自动落盘数据"。两者都在 intent 3/4 之外。

## S3 生命周期控制有四个入口

CLI 17 个子命令（`cli.py:645-723`：dash / open(隐藏别名) / _serve / status / stop / restart / logs / profiles / install-extension / install-kernel / uninstall-kernel / stdio-proxy / mcp-ping / metadata translate / convert / load / version）+ Dashboard 的 `/api/jupyter/{action}`、`/api/restart/{component}`、`/api/start-mcp`、`/api/mcp/stop` + **7 个 IPython magics**（`jupyter_mcp_extension.py:66-115`：start/stop/restart/status/safe/unsafe/dangerous）+ profile YAML。

724 行的 `cli.py` 里，真正管进程的部分被 `_ensure_host` / `_recover_discovery` / `_launch_readiness` / `_replace_host` / `_host_matches_request` / `_cleanup_stale_jupyter_tree` / `_listener_on_port` / `_jupyter_process_matches` 八个辅助函数铺开——这些都是为了"多个 host 实例谁该接管"这一个问题。

## S4 工具名清单有 4 份副本

`tools.py:294-308`（注册字典）、`config/metadata_baseline.yaml:3-83`（15 条 L1 描述）、`transport/stdio_proxy.py:14`（代理白名单）、`app/api.py:33-38`（dashboard 白名单）。前一份是源，后三份是手抄，且 `app/api.py` 那份只有 8 个、是另一个子集。AGENTS.md 的同步清单里没包括 `stdio_proxy.py` 和 `app/api.py`。

## S5 前端 11 个 comm handler，Python 只用 6 个

前端 `src/index.ts:205-322` 实现：`read_active_cell` / `read_notebook` / `move_cursor` / `request_consent` / `execute_code` / `add_cell` / **`save_notebook`** / **`read_cell_at`** / **`delete_cell`** / **`apply_patch`** / **`restart_kernel`**。

Python 侧实际发起的只有前 6 个中的 6 个（`grep bridge.request`）。剩下 5 个里：
- `delete_cell`（`:280`）和 `apply_patch`（`:300`）是**能改/删 cell 的死代码**——MCP 不暴露删除工具，但这条通道存在且能落盘（之前那轮审查记过这条：append-only 靠"没暴露"而非机制）；
- `read_cell_at` / `restart_kernel` / `save_notebook` 只被 dashboard 的注入代码用到。

## S6 前端源码与构建产物同时入库

`git ls-files peaksMCP/extensions` 显示：`src/index.ts`（源）+ `lib/index.js`、`lib/index.js.map`、`lib/index.d.ts` + `labextension/static/*.js`（构建产物）**全部跟踪**。24K 源码配 88K 产物，改一次源码就有三个产物需要同步，否则前端行为与源不一致。

## S7 输出状态有三份缓存，其中一份只写不读

`SharedState`（`base.py:66-68`）同时有 `active_cell_output`、`cell_outputs`、`last_execution_cell_id`：

- `last_execution_cell_id` 在 `active_cell_bridge.py:124` 被写入，**全仓无任何读取点**（`grep` 确认）→ 死状态；
- `active_cell_output` 和 `cell_outputs[cell_id]` 装同一份数据，`notebook.py:178-185` 先查 dict 再退回 list；
- 同时前端还在 **push** 执行结果（`index.ts:184-202` `publishExecution` / `watchExecutionOutputs`），和 **pull**（`read_active_cell`）两条路并存。

## S8 安全模式是"3 个 mode × 1 个开关"，且有 4 处可设

`ExecutionMode`（safe/unsafe/dangerous，`base.py:34-42`）+ `require_consent` 布尔（`profiles.py:33`）。文档自己承认"所有 15 个工具在每个模式下都暴露，mode 只改变 consent 严格度"，dangerous 唯一作用是"non-executing 的 add_cell 免 consent"。实际是 2 个布尔量的语义，却用了 3 值枚举 + 独立开关，并能从 profile / dashboard / magics 三处改。

## S9 测试重心错位

5807 行测试里最大的三块是 `test_security.py`(708)、`test_profiles_cli.py`(563)、`test_runtime.py`(517)——**生命周期 + CLI + dashboard 约 1800 行**，而真正的数据与工具路径是 `test_pxt.py`(486) + `test_tools.py`(265) + `test_discovery.py`(248) + `test_plotting.py`(107) + 两个 override 测试(197)。

也就是说：**给"启动/关闭/配置"写的测试，比给"分析数据"写的多**。这些测试是正确的（进程管理确实容易坏），但从 intent 看，它们保护的是可以被 S1–S3 收敛掉的那些路径。

## S10 工作区卫生

- `peaksMCP/extensions/jupyterlab/` 下 389M `node_modules` + 172K `yarn.lock`，**在 Python 包目录内**（`pyproject.toml:59` 显式 exclude，说明已经踩过一次）；
- `build/`(48 个 py) 与 `dist/`(whl+tar.gz) 是过期构建产物，未入库但占 522M；
- `peaksMCP/extensions/.DS_Store`、`app/webapp/.DS_Store` 存在（后者 6KB）。

---

# 第二部分 · 功能冗余（按 intent 逐条判定）

## F1 intent 1：黑箱函数 + notebook 里组合

| 冗余 | 说明 |
|---|---|
| **`mcp_list_resources` 的 6 个内联 matplotlib 模板** | `metadata_baseline.yaml:88-261`，每个都附整段可运行代码，并强制"画图前必须先读"（`notebook_unsafe.py:189`）。模型被引导去手写 `pcolormesh`，**绕开** `plot_batch` / `plot_validation_pair` / `show_mapping_slice`。这是"黑箱函数"最大的漏气口。 |
| **加载三入口** | native `load`（accessor）+ `load_pxt` + 计划新增的 `open_scan`。加载是 native 已经做好的事，`open_scan` 唯一增量是"自动注册 L112 loader"，该修在 `load_pxt` 里。 |
| **转换三入口** | `convert_pxt` / `convert_path` / 计划新增 `convert_experiment`（计划里三个都留）。 |
| **IO 细节进了暴露面** | 18 个项目 API 里，`default_output_dir` / `index_from_path` / `register_l112_loader` / `batch_execution_lock` 是转换器的内部零件，不该是模型可见的一等 API。 |
| **`inspect_experiment` 与 `read_meta`** | 计划新增的 façade 与已有 `read_meta` 输出同一张记录表。若 `inspect_experiment` 只做压缩，它值得；若只是换个返回类型，就是重复。 |

## F2 intent 2：输出规范化、不过度输出

| 冗余 | 说明 |
|---|---|
| **5 个"再看一眼"的读取工具** | `notebook_list_variables` / `notebook_read_variable` / `notebook_read_active_cell` / `notebook_read_active_cell_output` / `notebook_read_content`（+ `notebook_move_cursor` 用于先定位）。15 个工具里 6 个花在"读取上下文"上。 |
| **执行不返回输出 → 必须二次调用** | `write_with_api_check` 返回 bridge 原始结果，规范化逻辑 `_output_content()`（`tools.py:131`，写得很好：markdown→文本、图片只计数、widget 只留标记）却挂在**另一个工具**上。模型执行完还得单独调一次读输出。正确形态是**一次执行直接返回规范化输出**。 |
| **`summarize_xarray` 返回 13 个字段** | `notebook.py:107-123`：dims/sizes/dtype/variables/coords/units/attrs/chunks/peaks_apis/lazy… 与"不过度输出"正面冲突。人 review 时只需要 dims + shape + 单位。 |
| **三套 axis label 约定** | `layout.py:24` 生成 `kx [Å^-1]`，`validation.py:18` 生成 `$k_x$ ($\mathrm{\AA}^{-1}$)`，模板里写 `k$_{∥}$ (Å$^{-1}$)`。同一个包三种风格，"规范化输出"没有单一实现。 |
| **状态类工具三兄弟** | `notebook_server_status` / `notebook_kernel_status` / `notebook_wait_for_kernel`。前端 `execute_code` 是同步等执行完成再返回的，前两个基本无信息量；`wait_for_kernel` 返回体里就带着 kernel_status。 |

## F3 intent 3/4：不轻易保存、consent 前先展示

| 冗余 / 风险 | 说明 |
|---|---|
| **落盘有三条路，只有一条要 consent** | ① notebook cell 里写盘 → 扫描器 `SAVE002` → consent（合规）；② `convert_path` / `convert_pxt` 经 **CLI 或 dashboard** 直接写盘（无任何 consent）；③ `_flush_frontend_save`（`api.py:42`）自动保存 notebook。**需要收紧的是 ②，而不是新增 `save_result` façade。** |
| **`load_into_notebook` 的后台线程** | 270 行，注入代码执行，绕开扫描器与 consent（S2）。 |
| **前端 `delete_cell` / `apply_patch`** | 死 handler 但能改删 cell（S5），与"append-only 是机制而非约定"矛盾。 |
| **`askuserquestion`** | 返回 `{"status":"needs_input"}` 的空转工具，只是提示模型"去对话里问"。它能做的一切都能由 `prompts.yaml` 一行完成，占一个工具名额 + 一份 L1 描述。 |

---

# 第三部分 · 收敛建议

## 结构层（先砍这一层，功能层才清爽）

| 动作 | 对象 |
|---|---|
| **删** | CLI 的 `convert` / `metadata translate` / `load`、`open` 隐藏别名；dashboard 的 `/api/convert`、`/api/metadata/translate`、`/api/notebook/load`、`load_into_notebook`(270 行)、Inspector(`/api/mcp/tool` + `_INSPECTOR_ALLOWED`) |
| **删** | 前端 `delete_cell` / `apply_patch` / `read_cell_at` handler |
| **删** | `last_execution_cell_id`；`active_cell_output` 与 `cell_outputs` 合并为一个 |
| **删** | `ExecutionMode` 三值枚举 → 收敛成 `require_consent` 一个开关（profile 一处可设） |
| **合并** | 工具名清单 4 份 → 1 份（写个测试断言另外三处与源一致，比手抄可靠） |
| **合并** | 生命周期入口：保留 CLI（人用）+ profile（配置），删掉 magics 里的 safe/unsafe/dangerous，dashboard 只留进程控制 |
| **清理** | 前端 `lib/`、`labextension/static/` 移出 git（产物随构建生成）；`peaksMCP/extensions/jupyterlab/node_modules` 移到仓库外或彻底 gitignore；`build/`、`dist/`、两个 `.DS_Store` |

## 功能层（贴 intent）

1. **把 `_output_content` 接到 `write_with_api_check` 的返回值上**——一次执行 = 一份规范化输出。这条的收益高于本轮任何 schema 工作。
2. **`mcp_list_resources` 只留 `figure_conventions`**，6 个模板改成"façade 调用示例"一行；"画图前必须读"的 gate 保留，但它约束样式而不是代码。
3. **读取工具收敛**：`list_variables` + `read_variable` → 1 个；`read_active_cell` + `read_active_cell_output` → 1 个（合并进执行返回值后可直接删）；`server_status` / `kernel_status` / `wait_for_kernel` → 1 个。
4. **`summarize_xarray` 默认只返回 dims / shape / units / lazy**，详细字段按需开关。
5. **落盘唯一化**：所有写盘走 notebook cell（consent 链已完整），`convert_path` 不再由 CLI/dashboard 直接触发；`_flush_frontend_save` 改为显式调用（人或模型要求快照时）。
6. **暴露面收敛**：`default_output_dir` / `index_from_path` / `register_l112_loader` / `batch_execution_lock` 从"一等 API"降为内部函数，索引只从 `peaksMCP.overrides` 构建。

## 一个判断顺序上的提醒

如果先做 S1/S2（删掉 CLI 与 dashboard 的数据入口），F1 里的"转换三入口"、"加载三入口"会自动从三个变成一个——**计划里打算用 façade 分层解决的问题，有一半其实是结构层的路径太多造成的**。先收敛路径，再决定 façade 需要几个，能省掉大部分 schema 设计。
