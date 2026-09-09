# Override 黑盒层计划 — 冗余审查（2026-09-08）

审查对象：`# peaksMCP Override 黑盒层实施计划（修订版）`
审查标尺：需求方明确给出的四条 intent（不是项目文档里现有的 intent）

1. 模型通过 `search` / `get` 拿黑箱函数；必要时用 peaks 原生函数；**组合逻辑写在 notebook 里**。
2. notebook 输出**自动规范化**：用函数就输出函数输出，不要额外包装，噪音会淹没人 review。
3. **不轻易保存**；保存必须 consent。
4. consent 的前提是**结果先展示给人看过**。

由 1 推出：一个 façade 的价值 = 模型自己写不对 / 写不安全的那部分（输入校验、跨步不变量、状态一致性），**不是**"把几步打包省几行代码"。
由 2 推出：每一个新增 Report 模型都是输出噪音，必须证明它比"直接返回 DataArray + notebook 的 repr"更省。
由 3、4 推出：**任何在函数内部自动落盘的 API 都是绕开 consent 的旁路**。

---

## A 类：与标尺直接冲突，建议删除

### A1 `preprocess_batch` —— 计划里最大的一处冗余，且违反 intent 3/4

计划原文：*"按路径逐项加载、处理、保存、释放"*、`BatchPreprocessItem` 含 source path、`BatchProcessingReport`。

- 它把 load → preprocess → **save** 三件事封进一个黑箱，结果人在 notebook 里只看得到一份 report，**看不到任何一张图、任何一个 DataArray**。这直接违反"consent 的前提是结果先展示给人判断"：批量保存发生在人看到结果之前。
- 它同时违反 intent 1：逐项循环正是模型该在 notebook 里写出来的组合逻辑（`for p in paths: preprocess_cut(open_scan(p), ...)`），不是需要隐藏的实现。
- 成本不低：新增 2 个模型 + CPU budget 集成 + BrokenProcessPool / 并发发布 / 逐项失败 三类测试。

**建议**：不做批处理 façade。改为把"单次处理"做扎实（`preprocess_cut` / `preprocess_mapping`），批量由 notebook 里的循环 + 已有 `save` consent 链路承担。若确实要保留，必须改成**只处理不落盘**，返回 DataArray 列表。

### A2 `save_result` —— 与已存在的 consent 门重复

现状已经有完整的落盘 consent 链，不需要再包一层 façade：

- `security/code_scanner.py:592` `_FILE_WRITERS` / `_FILE_WRITE_METHOD_NAMES` → 任何 `.save` / `to_netcdf` / `to_csv` 都产出 `SAVE002` → `requires_explicit_consent`；
- `backend/notebook_unsafe.py:85-99` `_authorize()` 把 `requires_explicit_consent` 接到 `ConsentManager`；
- `mcp.require_consent` 总开关 + append-only notebook（人能看到上一条 cell 的输出，才决定要不要批准下一条）。

也就是说 "保存 → consent → 结果先展示" 这条链路**已经成立**（保存 cell 之前必然先有一个展示 cell，因为 notebook 是 append-only 的）。`save_result` 再加 `overwrite` 保护、`SaveReport(path/status/size/hash/warnings)`，是第二套策略 + 一份新的输出噪音（intent 2）。

**建议**：用 native `da.save(path)`，把"保存前必须先展示"写成 `prompts.yaml` 的一条规则 + 一条 scanner 提示，而不是新 façade。注意 skill 今天刚改成 `da.save(path)`（不是 `to_netcdf`），方向是一致的。

### A3 `mcp_list_resources` 的 6 个内联 matplotlib 模板 —— 计划完全没碰的最大冗余

`config/metadata_baseline.yaml:88-261` 有 7 个 resource（figure_conventions + fermi_surface / dispersion_single / dispersion_grid / waterfall / before_after / kz_map），每个都带**整段可运行模板**，且 `notebook_write_with_api_check` 强制"画图前必须先读"（`notebook_unsafe.py:189`）。

后果：模型被引导去**手写 pcolormesh**，而不是调用 `plot_batch` / `plot_validation_pair` / `show_mapping_slice`。这同时违反 intent 1（绕过黑箱）和 intent 2（手写模板会带出 Figure 对象、多余 display、`v = np.nanpercentile(...)` 之类的中间输出）。

计划一边给 `plot_validation_pair` 加 `shared_scale: Literal["auto", True, False]`、给 `plot_batch` 加严格 colorbar 语义，一边保留引导手写代码的模板库 —— **两套东西互相抵消**。

**建议**：6 个模板收敛成"façade 调用示例"（一行 `plot_batch(...)` / `plot_validation_pair(...)`），`mcp_list_resources` 只留 `figure_conventions` 样式约定。这样"画图前必须读"的 gate 仍值得保留，但它约束的是**样式**而不是代码。

### A4 `internal` 暴露层 + `register_l112_loader` 条目

计划把 `register_l112_loader`、`batch_execution_lock` 定为 `internal`，并规定"internal 永不进入模型索引"，还要为它写可见性测试。但同一个计划里又说 `open_scan` **内部自动完成 L112 loader 注册**。

即：这个 internal 层是专门为一个即将被自动化的函数建的。"不入索引"不需要一个暴露等级 —— 索引只从 `peaksMCP.overrides` 构建即可，其他模块天然不在索引里。**建议**：删除 internal 层与这两个 manifest 条目，函数保留在 Python 里当实现细节。

---

## B 类：重复实现，建议合并

### B1 改名兼容的四套机制 → 一套

计划里同时有：

- "保留一个发布周期的 v2 兼容解析器"；
- `legacy_ids` 可解析到新 canonical 但永不出现在搜索结果；
- "旧模块 discovery entry 被投影掉"；
- "CI 检查同一名称/实现/callable 不得生成两个 searchable entries"。

四套都在解决同一件事：改名后的旧入口。但这个仓库是自包含的，YAML **没有外部消费者**（`load_api_overrides(path=...)` 的默认路径就是仓库内那一份）。直接改成"索引只扫 `peaksMCP.overrides` 一个模块"就够了，其余三套都是纯维护成本。

### B2 转换的三个入口

`convert_experiment`（新 façade）+ `convert_pxt` + `convert_path`（都保留为 advanced）= 三入口做同一件事。而 advanced "仅在精确名称或 `include_advanced=True` 时返回"，模型实际只会拿到 façade。**建议**：只留 `convert_experiment`，另两个降级为不索引的内部函数；`default_output_dir` / `index_from_path` 同理。

### B3 打开的三个入口

`open_scan`（新 façade）+ `load_pxt`（advanced）+ native `load`。intent 1 明确"必要时使用 peaks 的原生函数"，而加载恰恰是 native `load` 已经做好的事。`open_scan` 唯一的真实增量是"L112 loader 自动注册"—— 这应该修在 `load_pxt` 内部（顺带解决 A4），不值得新增 façade。

### B4 六个 Report 模型 → 一两个

`GoldCalibration` / `SaveReport` / `ConversionReport` / `BatchProcessingReport` / `ProcessingReport` / `ExperimentSummary`（外加 `ScanKind` 七值枚举），再加 `ProcessingResult` 包 `ProcessingReport` 的双层结构。

按 intent 2，这些都是输出噪音。而且：
- `ProcessingReport.complete` / `.selection` 是为"禁止内部切片"这条规则服务的 —— 该规则用**输入维度校验 + 直接报错**就能实现，不需要在报告里留字段让人读；
- `SaveReport` 的 `hash` / `size` / `warnings` 在"不轻易保存"的前提下几乎不会被读。

**建议**：处理类 façade 直接返回 DataArray（校验失败就抛异常，异常本身就是最好的 review 信息）；报告类统一成一种极简结构（路径 / 状态 / 关键数值）。

### B5 分类规则的两处 gate

计划第 3 节要求"在 `pxt_utils.metadata` 内建立唯一的分类实现"，这是对的；但随后 `preprocess_cut` / `preprocess_mapping` 又各自做"只接受 2D / 3D"校验，再加一条 "idx 26 式分类冲突" gate。集中化之后 façade 只需查一次分类结果，维度校验和冲突判断是同一件事的两处实现。**建议**：façade 只做 `kind = classify(...)` 的一次断言。

### B6 CPU 预算参数化过度

`ResourceBudget` + `BatchExecutor` + `batch_execution_lock` 已经是一套完整机制（`batch/`）。计划又给 `convert_experiment`、`preprocess_batch` 各加一个 `cpu_limit_percent=60` 参数。CPU 预算是**机器级全局策略**，应留在 profile / `ResourceBudget` 里，不需要每个 API 暴露一个旋钮（多一个参数就多一份要 review 的输出）。

---

## C 类：元数据与 schema 过度设计

| 项 | 问题 | 建议 |
|---|---|---|
| `category`(7) + `kind`(9) | 两套正交分类，还要给 `peaks_search_api` 加两个过滤参数。约 25 个 API 用一套足够，`kind` 基本可从签名推出 | 保留一套（建议 `kind`） |
| `shadows_native` | 明确"本轮不允许任何实际条目使用" —— 空字段 | 删 |
| `tier` + `exposure` | `tier=override` 与 `exposure in {facade, advanced, internal}` 语义重叠，`project: true` 又是第三个同义标记 | 保留 `tier`（native/override）+ `project` 两值即可 |
| 三份文档来源 | manifest 的 `summary/inputs/returns/preconditions/side_effects/errors/example` + native 的 `docstring_note` + `signatures.py` 运行时抽真实 docstring | `get` 一次返回 6 段本身就违反 intent 2；压到 `summary` + `signature` + `example`，其余并入 example |
| 测试矩阵 | "每个 façade 覆盖成功/错误类型/空维度/全 NaN/缺 metadata/upstream exception" × 8 façade | 空维度 / 全 NaN / 缺 metadata 是**共享输入校验**，下沉后测一次，不必每 façade 重复 |

### 一个数量上的佐证

当前 `api_overrides.yaml` 有 18 个 project 条目。计划给出 **8 façade + 3 plotting + 15 advanced + 2 internal = 28 个条目**，而实际函数只有 17（18 减去删除的 `publication_grid`）+ 8 新 = 25 个。**条目数 > 函数数**，多出来的就是 `convert_pxt`/`convert_path`（被 `convert_experiment` 包裹）、`load_pxt`（被 `open_scan` 包裹）这类重复登记。

---

## D 类：计划没覆盖、但 intent 权重最高的两件事

计划的绝大部分篇幅花在 **manifest / schema / 搜索可见性** 上，而 intent 2、3、4 指向的两处改动它基本没写：

### D1 输出规范化应该合并进"一次执行"

现状：模型执行完 cell 后，要再单独调 `notebook_read_active_cell_output` 才能看到输出；`tools.py:131` 的 `_output_content()` 已经做了规范化（markdown→文本、图片只计数、interactive 只留标记），但它挂在**另一个工具**上。同时还有 `notebook_list_variables` / `notebook_read_variable` / `notebook_read_content` / `notebook_move_cursor` 四个读取工具，构成第二条获取数据视图的路径。

按 intent 2，理想形态是：**一次 `notebook_write_with_api_check` 返回 = 该 cell 的规范化输出**（函数返回什么就看到什么，多余的一律不进 context）。这一条改动对"减少 review 噪音"的贡献，大于计划里整个 schema 重构。**建议**：把 `_output_content` 接到 `write_with_api_check` 的返回值上，并评估 `notebook_list_variables` / `notebook_read_variable` 是否可以收敛掉。

### D2 "保存前必须先展示"应是一条显式门禁

链路其实已经具备（append-only notebook + SAVE002 consent + 人能看到上一条 cell），缺的只是一条**显式规则**：包含写盘调用时，要求上一条 cell 已经产出过可见输出（图或摘要）。这是 `prompts.yaml` 一行 + scanner 一条提示的工作量，不需要任何新 façade —— 而计划新增的 `save_result` / `preprocess_batch` 恰恰是绕开这条链路的旁路。

---

## 建议的最小集合（删掉冗余后）

**保留的 façade（4 个）**

1. `fit_gold_reference` —— 真实增量：委托 upstream `fit_gold` + 离群点策略 + JSON 安全 calibration（跨 cell 传递状态，模型自己写容易错）。
2. `preprocess_cut` —— 真实增量：EF correction + normal emission + `k_convert` 的顺序不变量 + 输入 2D 校验。
3. `preprocess_mapping` —— 真实增量：3D 完整 cube，禁止中心切片后谎报完成。
4. `inspect_experiment` —— 真实增量：把 `read_meta` 的记录表压成一份人能扫的摘要（gold 标志、维度、theta offset、冲突）。

**可作为 façade 或内部函数（`convert_experiment`）**：单文件/目录统一走一份 CPU budget 与 metadata 路径是有价值的，但要与 `convert_pxt`/`convert_path` 二选一，不要并存。

**绘图（3 个）**：`plot_batch` / `plot_validation_pair` / `show_mapping_slice` 保留，但必须配合 A3 删模板库，否则黑箱绘图永远用不起来。

**删除**：`open_scan`、`save_result`、`preprocess_batch`、`publication_grid`、`internal` 层、v2 兼容解析器、`legacy_ids`、canonical 投影 + ghost CI、`shadows_native`、`category` 或 `kind` 之一。

**新增但不是 façade**：输出规范化（D1）+ 保存前展示门禁（D2）。这两条才是 intent 里权重最高的部分。
