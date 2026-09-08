# peaksMCP 全面代码审查

- **日期**：2026-09-08
- **基线**：`main` @ `a58da6f`（工作区有 1 处改动 + 1 个未跟踪文件）
- **规模**：14,070 行 Python（52 个模块）+ TypeScript 前端扩展；git 跟踪 127 个文件 / 11.4 MB
- **门禁实测**：`ruff check peaksMCP tests tools` → All checks passed；`pytest -m 'not e2e'` → 378 passed, 1 failed
- **审查方法**：4 路并行子系统深读 + 关键结论逐条实测复现（扫描器绕过、token 泄漏、`os.link` 异常、依赖声明、工具清单、文档漂移）

> 关于那个 1 个失败测试：见 §7.1，**是审查环境沙箱的产物，不是项目 bug**，但暴露了一个真实的健壮性问题。

---

## 总体评价

这是一个**设计水准明显高于平均**的项目。分层清晰（MCP 服务在 kernel 内、supervisor 在 kernel 外），安全模型有明确的分层主张（AST 硬阻断 + consent + 审计），`append-only` 日志、原子写、CPU 预算这些"知道自己要防什么"的设计在同类 MCP 桥接项目里很少见。README 和 AGENTS.md 的质量也远好于一般个人项目。

问题在于：**几处最关键的安全声称，在实测中站不住**。AST 扫描器作为"always-on 硬阻断"，可被一行 `sys.modules[...]` 绕过；AGENTS.md 白纸黑字写的"绝不返回原始 Jupyter token"，在三个地方被违反。这两条一旦被依赖，就会形成虚假的安全感——这比没有安全设计更危险。

其余问题是典型的"工程债"：文档漂移、测试自洽循环、CI 门禁偏松、打包缺依赖。

---

## 1. 【P0】AST 代码扫描器可被一行绕过

**文件**：`peaksMCP/server/jupyter_peaks/security/code_scanner.py:634-655`

扫描器用 `name.split(".")[0]` 判定模块归属，而 `sys` 不在 `_BLOCK_MODULES`（:293）、`sys.modules` 也不在 `_REFLECTION_BASES`（:297）。实测结果（`scan_code(...).blocked`）：

| 载荷 | `blocked` | 结论 |
|---|---|---|
| `import os; os.system('id')` | `True` | 正常拦截 |
| `import subprocess; subprocess.run(['id'])` | `True` | 正常拦截 |
| `import os; getattr(os,'sys'+'tem')('id')` | `True` | 正常拦截（拼接也拦得住） |
| `import sys; sys.modules['subprocess'].run(['id'])` | **`False`** | **绕过** |
| `import sys; sys.modules['os'].popen('id')` | **`False`** | **绕过** |
| `import sys; sys.modules['shutil'].rmtree(...)` | **`False`** | **绕过** |
| `import sys; f=sys.modules['builtins'].open(p,'w'); f.write(x)` | **`False`** | **绕过（可写文件）** |
| `get_ipython().getoutput('id')` | **`False`** | **绕过** |

**为什么严重**：

1. 扫描器的强制执行点只有一处 —— `backend/notebook_unsafe.py:63-64` 的 `if scan and scan.blocked`。`blocked=False` 就直接放行，没有第二道防线。
2. 存在**端到端可利用链**：`sys.modules['builtins'].open(...).write(...)` 扫描通过 → 叶子名 `open`/`write` 命中 `GENERIC_METHODS`（`api_allowlists.py:52`）→ `notebook_unsafe.py:266-271` 的 API 校验判为 generic 放行。两层都过。
3. **读方向**同样成立：`sys.modules['builtins'].open('~/.peaksMCP/runfile.json').read()` 可取出 Jupyter token —— 与 §2 叠加。
4. `consent` 默认关闭（见 §3），所以多数部署下这是**唯一**的防线。

**IPython 扫描器同理**：`ipython_scanner.py:8-12` 只匹配 `!`、少数 magic 与 `get_ipython().system/run_*_magic`，`getoutput` 漏网。

**建议**：黑名单式的 `name.split(".")[0]` 无法覆盖属性/下标动态取值。至少做三件事 —— (a) 把 `sys.modules` / `sys` 纳入阻断与反射基；(b) 对 `Subscript` 取值的模块解析做保守处理（无法静态确定归属时按最高风险处理）；(c) 把 `open` 从 `GENERIC_METHODS` 移出，改为按参数模式（写模式）单独判定。长期看，这类"允许清单"模型需要收敛到真正的 allowlist 而非 blocklist。

---

## 2. 【P0】Jupyter token 明文外泄，与文档声明直接冲突

`AGENTS.md:26-28` 声明："Never return the raw Jupyter token from the dashboard status API." 实测：

- `app/api.py:559` — `snapshot_notebook` 在 JSON 响应里返回 `open_url = "...?token={supervisor.token}"`（原始 token）
- `app/api.py:588` — `open_notebook` 302 重定向同样携带 `?token=`（落入浏览器地址栏与历史记录）
- `app/runtime.py:166` — token 作为**命令行参数** `--ServerApp.token=...` 传入，同机任何用户 `ps aux` 可见
- `cli.py:353` — `command_dash` 把含 `?token=` 的 URL `print` 到 stdout 并交给 `webbrowser`

部分保护是到位的：`/api/status`（`runtime.py:673-683`）确实不含 token，`cli.py:30` 的 `_public_run_state` 与 `runtime.py:306` 的日志脱敏也做了。但覆盖面不完整，**声明与实际不符**。

**建议**：API 只返回一次性短期兑换凭证（重定向到 `/auth?code=...` 再 302 到带 token 的 URL），token 全程不出现在 JSON 响应体与命令行；Jupyter 改用 `JUPYTER_TOKEN` 环境变量或 `--ServerApp.token_file`。

---

## 3. 【P1】安全模型实际只有一档，且默认是关闭的

- `require_consent` 默认 `False`（`backend/base.py:60`、`jupyter_mcp_extension.py:29-31`、`app/profiles.py:31`）。声称的 safe/unsafe/dangerous 三档，在默认配置下**全部退化为"扫描器 + 审计"**。
- 且 `notebook_unsafe.py:85-87` 的判定使 **safe 与 unsafe 行为完全一致**，档位形同虚设。

叠加 §1：默认部署 = 一个可被绕过的扫描器 + 审计日志。建议在文档中明确写出"默认模式下的实际防护边界"，避免用户误以为 consent 在保护他们。

---

## 4. 【P1】`append-only` 保证依赖前端自律

MCP 侧确认扎实：15 个工具中确实没有 `delete_cell`，两个 mutation 工具只做末尾追加。

但前端仍**完整实现并落盘**了删除路径：
- `extensions/jupyterlab/src/index.ts:280-299` — `case 'delete_cell'` → `NotebookActions.deleteCells(notebook)` → 保存到磁盘
- `src/index.ts:94` — consent UI 保留 `删除` / `覆盖` 分支
- `src/index.ts:315` — `delete_cell` 在持久化操作列表里

也就是说，"notebook 是不可变日志"这一保证，靠的是**没有 MCP 工具暴露它**，而不是机制上不存在。任何拿到 comm 通道的调用方都能删。建议要么物理删除该分支，要么在后端 comm handler 里硬拒 `delete_cell`。

---

## 5. 【P1】`peaks` 未声明为依赖 —— `pip install peaksMCP` 得到的是不可用的包

`discovery/index.py:84,190,376` 与 `signatures.py:153` 都 `import peaks`，但 `pyproject.toml:11-31` 的 `dependencies` 里**没有** `peaks`。目前只是靠 `ci.yml:14` 手工 `pip install peaks-arpes@git+...` 才跑得通。

同时所有依赖只设下限（唯一有上限的是 `jupyterlab<5`），在安全敏感项目里偏松。

---

## 6. 【P1】运行时与数据管线的可靠性问题

### 6.1 CPU 节流会静默丢弃整批任务
`batch/executor.py:88-110`：单次 `wait_for_capacity(timeout=60)` 超时后，**把剩余全部项标为 `skipped` 并 break，不再重试**。`psutil.cpu_percent()` 测的是整机 CPU，若 Jupyter/浏览器已占用 55%，gate 一直不开 → 100 个文件可能只转 3 个，而 `ConversionReport.failed == 0`（`models.py:100`），只能靠 `output_exists=False` 发现。

另 `resource_budget.py:201-220` 存在死区：`resume=50` / `preemptive=58.5`，均值落在两者之间时既不 set 也不 clear gate，若此前被 clear 则永不恢复。

### 6.2 进程树清理只在超时分支生效
`app/runtime.py:225-238`：`killpg` 只在 `wait(8)` 超时的分支内执行。正常路径仅向 JupyterLab 组长发 SIGTERM，而 `start_new_session=True`（:180）产生的同组孙进程（kernel）依赖 JupyterLab 自行回收。文档"stop 拆掉整棵进程树"仅在超时路径成立。

### 6.3 runfile 可被异常路径抹掉
`observability/runfile.py:111-121` 的 `remove_runfile()` 无 PID 归属校验。`runtime.py:718-730` 中 `start()` 抛异常后 `finally: self.stop()` 仍会删除 runfile —— 会**抹掉存活中的另一个 host 的 runfile**，CLI 随即失联。

### 6.4 端口占用误判
`runtime.py:35-45` 的 `_port_owner` 遍历 `psutil.net_connections` 时未过滤 `status == LISTEN`（对比 `cli.py:123` 正确过滤了）。TIME_WAIT 的同端口 socket 会被误判为占用 → `restart_jupyter` 后紧接重启必然失败。

### 6.5 原子写不落盘、残留不清理
- **无 `fsync`**：`converter.py:380`、`models.py:65-66` rename 前都没 fsync（全仓只有 `runfile.py:57` 用了）。rename 保证原子性（读者看不到半截文件），但**不保证崩溃后不丢**。
- **`.part` 无清理**：`models.py:58-66` 无 `try/finally`，写一半失败（磁盘满）就永久留下 `.part`；全仓无任何清理逻辑。
- **硬链依赖**：`converter.py:298` 用 `os.link` 做 create-if-absent，且只捕获 `FileExistsError`。见 §7.1。
- 源文件保护本身**扎实**（`converter.py:275-283` 含 symlink/samefile 校验，强制 `.nc` 后缀），未发现源 PXT 被写/覆盖/删除的路径。唯一瑕疵：`converter.py:469-471` 会把 `experiment_metadata.json` 写进原始目录，属污染。

---

## 7. 测试与 CI

### 7.1 那个失败的测试（重要澄清）

`tests/unit/test_pxt.py::test_non_force_does_not_overwrite_concurrent_publisher` 报 `'failed' == 'skipped'`。

**这不是项目 bug**。根因是审查环境的 sitecustomize 沙箱把 `os.link` 的 EEXIST 包装成了 `PermissionError`，而 `converter.py:298` 只捕获 `FileExistsError`。用干净解释器实测：`os.link` 目标已存在 → `FileExistsError`（正确行为，测试应通过）。

但它暴露了一个**真实的健壮性问题**：`_publish_output` 只 catch `FileExistsError`。在不支持硬链的 FS、受限容器、或 macOS App Sandbox 这类环境下，硬链会以 `OSError`/`PermissionError` 失败 → 整批标记 `failed` 而非优雅降级。**建议扩展捕获范围或加 `os.link` 可用性探测 + 退化为 `O_EXCL` 创建**。

### 7.2 工具清单存在第 4 份副本，且测试自洽循环
工具名清单有 4 处：`core/tools.py`、`metadata_baseline.yaml`、`tests/unit/test_tools.py`、`transport/stdio_proxy.py:11-19`。**第 4 份没进 AGENTS.md:142-148 的同步清单**，而 `tests/unit/test_transport.py:23` 又用 `stdio_proxy.EXPECTED_TOOL_NAMES` 自己生成输入 —— 自洽循环，永远绿。加/删工具时 `mcp-ping` 会报 missing/unexpected，而测试发现不了。（当前 15 个名字与 tools.py 一致，暂未出错。）

### 7.3 半数测试断言偏弱
- `config/metadata.py:31` 对缺失项回退 `title=name`，故 `test_tools.py:49` / `test_extension_and_package.py:47` 的 `metadata["title"]` 恒真
- `test_extension_and_package.py:24-34,61-65` 用源码字符串字面量做断言（测实现细节，重构即假红）

### 7.4 e2e 在 CI 中执行 0 次
排除生效（`pyproject.toml:75` + `ci.yml:33`），但 41 条真浏览器/真 kernel 断言从未被门控；`test_e2e_live.py:149,259,372` 还会因 Chrome 不可启动**静默 skip**。CI 也无 `cache: pip`、无 `timeout-minutes`、无覆盖率门禁（`pytest-cov` 装了但没用 `--cov-fail-under`）。

### 7.5 ruff 配置偏松
`pyproject.toml:86-88` 仅启用 E/F/I/B/UP 且 `ignore=["E501"]`，使 `line-length=100` 形同虚设；安全敏感项目未启用 `S`/`TRY`/`PL`/`C4`/`RUF`；只 `check` 不 `format --check`。

---

## 8. 文档漂移

| 位置 | 问题 |
|---|---|
| `docs/ARCHITECTURE.md:39` | `peaksMCP launch` 子命令不存在（`cli.py:648-717` 无） |
| `docs/ARCHITECTURE.md:13,22,25` | 仍称 "Supervisor"，`AGENTS.md:73-74` 已改名 dashboard host |
| `core/tools.py:215` | docstring 称 "twelve read-only"，实际注册 13 个 |
| `CHANGELOG.md` | 自初始 commit（2026-08-27）未再更新；`f72b45d` 删 process_cut、`a69a6c4` 删 `notebook_delete_cell`、`966f1f3` 加 `mcp_list_resources` 均无记录 |
| `AGENTS.md:142-148` | 工具变更清单漏了 `transport/stdio_proxy.py`（见 §7.2） |

---

## 9. 数据与算法正确性

- **`discovery/signatures.py:119-128`** — `_bound` 用逗号切分参数，实测 `annotate(self, text='a,b', n=1)` 被改坏成 `annotate(text='a, b', n=1)`。含逗号的字符串默认值会被破坏。
- **`discovery/signatures.py:29-30`** — `_source_path` 只对 `peaks.` 前缀剥壳，`peaksMCP.*` 会拼出 `peaks/peaksMCP/...` → 源码提取恒失败，只能回落 introspect。
- **`discovery/index.py:320`** — 只遍历 `tree.body`：类方法、`if TYPE_CHECKING:` / `try:` 内定义、动态生成 API 全部漏掉；同名重载被 `_merge_duplicates:348-359` 折叠，只保留首个签名。
- **`plotting/validation.py:104,129`** — `values` 按 `data.dims` 原顺序取，却传给 `pcolormesh(x, e, values)`，隐含假定 dims 为 `(eV, other)`。同仓 `workflows/slice_view.py:146` 显式做了换位，两处不一致 → 数组转置时形状不匹配。
- **`pxt_utils/metadata.py:165`** — `records_out.sort(key=lambda r: r["index"])`，非整数 key 回退为 str（:131-133），混合 int/str 排序会 `TypeError`。
- **指纹缓存每次搜索全量 walk** — `tools.py:56 → base.py:28` 每次 `peaks_search_api` 都 `source_fingerprint()` 全量 `os.walk` + `stat`；`_SKIP_DIRS`（`index.py:21`）未排除 `node_modules`。缓存逻辑（mtime_ns+size）正确，但成本随调用次数线性放大。
- **`config/metadata.py:12`** — `lru_cache` 永不失效，且 `:72` 直接返回缓存内的**可变 dict**，调用方可污染。

---

## 10. 并发与状态

- **共享状态两把锁**：`state.cell_outputs` / `active_cell_output` 在 `backend/notebook.py:167-173` 无锁改写，comm 线程在 `active_cell_bridge.py:104-137` 用 bridge `_lock` 改写，而 `base.py:69` 的 `state.lock` 完全没用于此 → 竞态。
- **`mcp_server.py:132-133`** — `start()` 不等待绑定、不捕获线程内异常，端口占用时静默失败。
- **`notebook.py:216-222`** — `wait_for_kernel` 用 `time.sleep` 阻塞 worker 线程最长 30s。
- **生命周期无互斥** — 全仓仅 `batch/resource_budget.py:42` 用 `flock`；CLI/host 无锁，两个 `peaksMCP dash` 并发可产生双 host（`cli.py:307` 检查与 `:316` spawn 之间有窗口）。
- **`notebook.py:197`** — `move_cursor` 未判 `bridge is None`（与 :188 不一致）→ `AttributeError`。

**附带发现**：`jupyter.host` 无 loopback 守卫（dashboard 有 `runtime.py:370-375` 的 `allow_remote` 门禁，MCP 有 `mcp_server.py:123`）。profile 设为 `0.0.0.0` 即把带 token 的 JupyterLab 暴露到网络。默认 `app/defaults/default.yaml` 是 127.0.0.1，未发现硬编码放宽。

---

## 11. 做得好的地方（值得保留）

- **源文件保护扎实** — `converter.py:275-283` 的 symlink/samefile/后缀多重校验，未发现任何源 PXT 被修改的路径
- **`_publish_output` 的设计意图正确** — 硬链做 create-if-absent 是很聪明的原子原语，只是异常面收得太窄
- **AST 扫描器整体认真** — 725 行、有 alias 追踪、`getattr` 常量拼接也拦得住，基础是对的
- **仓库卫生干净** — `.gitignore` 完善；工作区 522MB 全是本地构建产物（node_modules 389M / .yarn 89M），**git 只跟踪 127 个文件 11.4MB**，无凭证泄漏（唯一 token 字样是测试假 token），wheel 149KB 不含 node_modules 与 tests
- **`/api/status` 确实不含 token**，前端 `app.js` 只轮询 `/api/status`、不把 token 落 localStorage
- **版本号 0.1.0 四处一致**，端口 8123/8765 与代码一致
- **ruff clean，378/379 测试通过**
- **`ext_install.py`、`api.py:454-471` 的 `tool_call` 白名单** 防护到位

---

## 建议的修复顺序

| 优先级 | 事项 | 位置 |
|---|---|---|
| **P0** | 堵 `sys.modules` 绕过 + 收紧 `open` 的 generic 放行 | `code_scanner.py:634-655`、`api_allowlists.py:52` |
| **P0** | token 不再出现在 API 响应与命令行 | `api.py:559,588`、`runtime.py:166`、`cli.py:353` |
| **P1** | 前端物理移除 `delete_cell` 分支 | `src/index.ts:280-299,315` |
| **P1** | `peaks` 加入 `dependencies` | `pyproject.toml:11-31` |
| **P1** | 硬链失败优雅降级 + `fsync` + `.part` 清理 | `converter.py:298`、`models.py:58-66` |
| **P1** | 节流超时改为重试而非丢弃整批；修死区 | `executor.py:88-110`、`resource_budget.py:201-220` |
| **P1** | runfile 加 PID 归属校验；`killpg` 移出超时分支 | `runfile.py:111-121`、`runtime.py:225-238` |
| **P2** | 工具清单单一来源（现在 4 份）；测试改用真实注册列表 | `stdio_proxy.py:11-19` + 测试 |
| **P2** | 文档/CHANGELOG 同步；`stdioproxy` 进 AGENTS 清单 | §8 |
| **P2** | ruff 启用 `S`/`TRY`/`PL`，CI 加缓存与覆盖率门禁 | `pyproject.toml:86-88`、`ci.yml` |

---

*本报告基于静态审查 + 关键路径实测。所有行号对应 `main` @ `a58da6f`。标记为"实测确认"的结论均已用可执行代码复现。*
