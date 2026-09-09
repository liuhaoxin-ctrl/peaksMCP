# 端到端 Agent 基准 · 预处理一批 ARPES 数据

这份基准回答一个问题：

> 换一个 agent（pi-agent、Claude Desktop、WorkBuddy、任意配好 MCP 的），
> 给它一句话目标，它能不能**自己**把一批原始 ARPES 数据预处理完？
> 跑完之后，我能不能立刻知道**该改系统的哪个地方**？

不是单元测试，也不是"让 LLM 给 LLM 打分"。整份基准只读 agent 跑完后
留在磁盘上的东西——审计日志、产物、notebook——**不读任何 agent 的对话记录**。

---

## 30 秒版

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python

# 0. 先证明评分器本身没坏（应该 100%）
$PY benchmark/run_case.py selftest

# 1. 准备一次运行（建目录、算答案键、渲染提示词）
$PY benchmark/run_case.py init --name run01

# 2. 把 runs/run01/prompt.md 里的 P1 段粘给 agent，让它跑完
#    （人只负责在 notebook 的保存卡上点同意）

# 3. 打分
$PY benchmark/run_case.py grade runs/run01
```

第 3 步会直接打印「该改哪里」，每条失败都带上子系统和修法。

---

## 前置条件

| 项 | 说明 |
|---|---|
| Python | 用 `peaks` 环境：`/opt/homebrew/Caskroom/miniforge/base/envs/peaks/bin/python`。需要 `xarray` + `pyyaml`（用于打开产物比对）。**不需要** import peaksMCP 本身 |
| peaksMCP 在跑 | `peaksMCP dash`（会拉起 Jupyter + in-kernel MCP server）。用 `peaksMCP status` 确认 |
| Notebook 打开 | 在 Jupyter 里打开 `runs/<id>/work.ipynb`。保存同意卡渲染在这里，没人看就没人批准 |
| agent 接好 MCP | agent 那边能看到 `search` / `get` / `inspect_notebook` / `run_cell` / `save_with_consent` |
| 原始数据 | 默认 `/Users/haoxin/Documents/实验数据/BP260623/data`（28 个 PXT + datasheet.csv），可用 `--data` 换 |

> **为什么必须有 notebook 开着**：`save_with_consent` 是 fail-closed 的——
> 没有人在保存卡上点同意就不写盘。这是设计，不是 bug。所以这一轮实验
> 人是"审批者"，不是"操作者"；agent 全程自己干。

---

## Step 0 · 先自检，再拿 agent 试

```bash
$PY benchmark/run_case.py selftest
```

它会**造一次应该拿满分的运行**（人工处理好的 `*_processed.nc` + 一份与真实
审计日志 schema 一致的黄金痕迹），然后跑完整评分：**24 条检查必须全过**
（含审计相关的 A1/A2/R3/R4/V3/O1/O2/O3/S3——不再跳过）。随后再跑 **6 组毒化
负向对照**：每次只注入一种缺陷（直写 / delete_cell / denied / API-block /
执行错误 / 绕过 get），断言恰好对应的一组检查翻红。

期望输出 `自检通过：黄金 24 项全过，毒化对照 6 组全部符合预期`。
临时目录默认清理，想看现场加 `--keep`。

**如果自检都过不了，说明是评分器坏了，不是 agent 的问题。** 这一步必须在
拿真 agent 试之前跑通，否则你分不清低分是谁的锅。

---

## Step 1 · init

```bash
$PY benchmark/run_case.py init --name run01
# 常用变体：
#   --limit 3        只取前 3 条 cut，快速冒烟（几十秒跑完）
#   --data /path/to/data
#   --copy           复制数据而不是建软链（跨平台/外置盘时用）
```

生成 `benchmark/runs/run01/`：

```
input          -> 原始数据（软链）
output/        空目录，agent 应该把产物放这里
work.ipynb     空 notebook，agent 的工作现场（kernelspec 已对齐受管内核）
answer_key.json  标准答案（见下）
env.json       本次运行的路径与审计日志位置
prompt.md      渲染好的提示词（含 P1/P2 两个变体，给人看）
prompt-p1.txt   P1 裸提示纯文本（绝对路径已填好，直接粘给 agent）
prompt-p2.txt   P2 装备提示纯文本
```

**答案键是独立算出来的**：`run_case.py` 自己解析 `datasheet.csv`，
**绝不调用 `inspect_experiment`** 来生成期望值——否则分类逻辑有 bug 时，
基准会自己给自己打满分。

init 会打印答案键摘要，先扫一眼：

```
答案键：gold=[20]  cut=14 条  theta_offset=1.5
```

> 本数据集里有一个**刻意的歧义**：index 19 的 Comment 写着 "Sweep，Au"，
> 但 `Data format` 是 sweep；真正的金参考是 index 20（"Au sweep"）。
> 答案键只看 `Data format`。agent 如果选了 19，`C3_gold_index_correct` 会红——
> 这是故意的，用来测它有没有真的按元数据分类。

---

## Step 2 · 把提示词给 agent

打开 `runs/run01/`，用 **`prompt-p1.txt`（P1 裸提示）或 `prompt-p2.txt`（P2
装备提示）** 二选一，把整段粘给 agent。完整版在 `prompt.md`，**一次只用一段**：

| 变体 | 内容 | 用来测 |
|---|---|---|
| **P1 裸提示** | 只有目标（把这批数据预处理完，产物放哪），不提任何工具 | 纯粹的**可发现性**：agent 能不能自己找到路 |
| **P2 装备提示** | 目标 + 工具面指引（有 search/get、有黑箱 adapter、有 notebook、保存要人批准） | **能力**：给了地图能不能走到终点 |

两个变体的判读方式：

- **P1 挂、P2 过** → 不是能力问题，是**可发现性**问题。该改 Contract / Access：
  `server_instructions` 没说清入口，`search` 排不出关键 API，manifest 别名不够。
- **两个都挂** → 是**能力/流程**问题。该看 Run / Show / Save：
  代码跑不通、看不到中间结果、或者根本没走到持久化网关。
- **两个都过** → 这一批数据的预处理可以交给 agent 了；换更刁的数据集继续加压。

**提示词刻意不教函数名、不教步骤。** 一旦你把 `load_data → inspect_experiment →
fit_gold → ...` 写进提示词，测的就不是系统，是你写的说明书。

### 给 agent 的时候要注意

- 别在提示词外面多嘴。agent 问"要不要我用 X 函数"，正确回答是"你自己决定"。
- 每个产物都会弹一张保存卡，**每来一张就点同意**。你不点，V3 会红，
  但那是人没批，不是 agent 没做——报告里会写成"没有任何成功的保存记录"。
- 跑完之前不要手动改 notebook。notebook 是 append-only 的，改了就污染证据。

---

## Step 3 · grade

```bash
$PY benchmark/run_case.py grade runs/run01
```

输出：

```
总分 27%  （run01）

  × Run              0%   失败 4
  × Save             8%   失败 3
  × Contract        33%   失败 3
    Show           100%   失败 0   跳过 1
    ...

该改哪里：
  · [Run] R1_all_targets_processed：该处理的都处理了（产物齐全）
      期望 14 个，实到 0 个
  · [Save] V2_no_direct_disk_write：没有绕过网关直接写盘
      出现直写模式：['to_netcdf', "open(...,'w')"]
  ...
```

同时落两个文件：

- **`report.md`** — 给人看。每条失败带"观察 + 证据 + **怎么改**"（fix 文案来自 `rubric.yaml`）。
- **`result.json`** — 给机器看。`compare` 和 CI 都吃这个。

### 分数怎么读

- `跳过 N`（`passed: null`）表示**证据不足，无法判定**，不计入分母。
  缺审计日志 ≠ 做得差，所以不能当 0 分算——否则分数没有意义。
- **别只看总分，看失败项。** 「什么都不做」的空跑也能拿 30% 左右的底分——
  因为它没直写、没删 cell、没残留 `.part`，这些检查天生是过的。
  真正咬人的是 `R1`（产物没出来）、`V1`（产物没在目录里）、`C1/C2`（没走黑箱），
  这几条一红就是真的没做成。
- 审计日志是全局共享的，`grade` 会默认只统计 `env.json` 里 `created_at`
  之后的事件。想手动框窗口用 `--since 2026-09-09T18:00:00`。
- 想指向另一次运行的 notebook / 产物 / 日志：`--notebook` / `--output` / `--audit`。

---

## Step 4 · 该改哪里

`report.md` 的「该改哪里」按权重从高到低排。24 条检查分到六个子系统：

| 子系统 | 检查 | 它在问 |
|---|---|---|
| **Contract** | C1–C5 | agent 用黑箱入口了吗？分类、theta 偏移、EF 是不是从契约拿的而不是猜的？ |
| **Access** | A1–A3 | 用到的原生 API 都先 `search`/`get` 过吗？有没有被"未验证 API"拦下？ |
| **Run** | R1–R4 | 该处理的都处理了吗？gold 只拟合一次吗？notebook 只追加吗？执行成功率？ |
| **Show** | S1–S3 | agent **看到**分类结果了吗？出验证图了吗？有多少 cell 是"盲跑"的？ |
| **Save** | V1–V4 | 产物在约定目录吗？有没有绕过网关直写？每个产物都有批准记录吗？ |
| **Observability** | O1–O3 | 事后能不能重建「一次执行 → 用了哪些 API → 产出什么」？ |
| **Result** | Q1–Q4 | 产物本身对不对——k 空间、EF 归零、theta 归零、与人工结果一致 |

改 `rubric.yaml` 里某条的 `fix` 文案，报告里的"怎么改"就跟着变，不用动 Python。

---

## Step 5 · 改完再跑一遍，对比

```bash
$PY benchmark/run_case.py init --name run02
# ... 粘 prompt，让 agent 跑 ...
$PY benchmark/run_case.py grade runs/run02

$PY benchmark/run_case.py compare runs/run01/result.json runs/run02/result.json
```

```
检查                            run01        -> run02
------------------------------------------------------------------------
V2_no_direct_disk_write         失败         -> 通过
V3_consent_trail_complete       失败         -> 通过
------------------------------------------------------------------------
总分 27% -> 55%（2 项变化）
```

**这就是闭环**：改一个地方 → 重跑同一份提示词 → 看那一条是不是真的翻绿。
如果改完分数没动，要么改错了地方，要么这条检查的判定逻辑本身有问题。

---

## 加一条检查

1. 在 `run_case.py` 的某个 `check_*` 里 `out.append(Result("X_new_check", passed, detail, evidence))`。
   `passed` 用 `None` 表示"证据不足，跳过"。
2. 在 `rubric.yaml` 加同名条目：`subsystem` / `weight` / `title` / `fix`。
3. 跑 `selftest`。**如果你的新检查在参考流水线上都过不了，说明它写错了**——
   参考流水线代表的是"人做得对的样子"。

---

## 已知坑

- **`input` 是软链。** 换机器或数据在外置盘时加 `--copy`。
- **产物没落在 `run*/output/`。** `grade` 会在 `run_dir` 下 `rglob("*_processed.nc")`，
  所以散落在别处也能找到，但 `V1` 会因为"不在约定目录"扣分——这是对的。
- **人工参考缺失。** `--reference` 指向的目录里没有对应文件时，`Q4` 会跳过，
  其他 Q 照常判。参考文件命名约定：`<stem>_processed.nc`。
- **工具名又改了。** 项目换过好几轮工具名（`peaks_search_api` → `search`、
  `notebook_write_with_api_check` → `run_cell`、`save_result` → `save_with_consent`）。
  `run_case.py` 顶部有 `SEARCH_TOOLS` / `GET_TOOLS` / `RUN_TOOLS` / `SAVE_TOOLS`
  四张别名表，按语义匹配而不是字面名。再改名时往里加即可。
- **`benchmark/runs/` 已 gitignore。** 里面是实验现场（软链 + 产物 + 报告），不进版本库。
