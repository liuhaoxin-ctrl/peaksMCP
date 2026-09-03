---
name: cut-preprocessing
description: 使用peaksMCP调用peaks包，在notebook中对cut 数据做预处理（费米面调平和置0 → 高对称点置0 → k空间转换）。当用户要求处理 cut 数据 / 预处理 / 调平 / 转 k 空间时使用。
---

# Cut 数据预处理

## 推荐入口:`process_cut`

对 2D sweep(`eV`, `theta_par`)做标准预处理(费米边调平/置0 → 高对称点置0 → k 空间)
优先调用 peaksMCP 提供的确定性工作流函数,不要手搓:

```python
from peaksMCP.workflows import process_cut

result = process_cut(
    da,                            # 2D DataArray (eV, theta_par)
    theta_offset=...,              # 高对称点角度偏移(度):来自 experiment_metadata.json
                                   # 的 agent note,或 askuserquestion 问用户,绝不臆造
    ef_correction=ef_from_gold,    # 传 gold fit 的 EF → apply 模式(普通 sweep 数据)
    # ef_correction=None → Au 数据模式:在本条数据上 fit_gold,返回 EF_quality 收敛报告
)
da_k = result["data"]              # (eV, kx),kx=0 即高对称点
```

要点:

- `theta_offset` 必须显式传入;不要依赖函数内部的自动查找(转换出的 .nc 上
  `experiment_metadata_json` 不含顶层 notes,自动路径取不到会**静默按 0° 处理**)。
- Au 金数据用 `ef_correction=None` 拟合(产出 `EF_quality` 收敛报告);普通 sweep
  传入金上拟合好的 `ef_correction`,避免每条都重拟合。
- 返回 dict 含 `data` / `EF_correction` / `EF_quality` / `theta_offset_deg` / `geometry`。
- 需要细粒度控制 EF/offset/fit 参数或分步展示时,才退回 peaks 公开 API 逐步实现
  (`fit_gold` → `metadata.set_EF_correction` → `metadata.set_normal_emission` →
  `k_convert`)。

## 获取数据

 **读取 metadata 获取 input 信息**：数据目录旁有实验记录表`experiment_metadata.json`，包含：
   - 每个实验（Index）的 input 文件（BP_XXXX）、类型（sweep/mapping/Au）
   - 偏振（S/P）、中心能量、Pass E. 等参数
   - 对每个数据的comments
   - 金数据（`Data format` 标注 Au）用于费米能拟合：转换后每条 record 带
     `is_gold_reference: true`（以及原文 `experiment.data_format`），直接用该
     字段挑选金记录，不要靠猜表头文字
   - 根据 datasheet 选出要处理的 cut（2D sweep）文件

## 工作流

- 将E_k-k数据的费米边调平，并费米能置0 → 将角度空间的高对称点置0 → 转换到k空间
- 先核对再调用：search_api 看作用域、get_api 看返回结构、inspect 看运行时类型证据（type/dims/shape/attrs...），凭证据写代码，不凭假设
- numpy 组合前先断言 shape，不同维度用 [:, None] / [None, :] 显式广播
- 优先使用peaks公开 API，里面有完整docstring，不复刻 peaks 内部函数 
-  进行金poly4拟合时，要先排除离群值再进行拟合

## 需要的参数

- `theta_par_offset_deg`（高对称点位置）：在experiment_metadata.json给agent的note中或者是用户给定（askuserquestion tool），无法从数据推导
- 费米能：金数据拟合（`fit_gold`，datasheet 标注 Au 的）> 问用户（askuserquestion tool）
- 偏振/能量等其他metadata只用于判断，不参与计算


## 产物约定

使用print和plot向用户说明**关键**信息。