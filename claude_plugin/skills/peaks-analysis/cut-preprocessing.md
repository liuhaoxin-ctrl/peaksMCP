---
name: cut-preprocessing
description: 使用peaks-agent tools调用peaks包，在notebook中对cut 数据做预处理（费米面调平和置0 → 高对称点置0 → k空间转换）。当用户要求处理 cut 数据 / 预处理 / 调平 / 转 k 空间时使用。数据目录旁的 datasheet.csv 里有需要的 input 文件信息和参数。
---

# Cut 数据预处理

## 何时使用

用户要求处理 cut（sweep / fix）数据（预处理/ 调平费米面 / k 空间转换）时。


## 获取数据

 **读取 datasheet 获取 input 信息**：数据目录旁有实验记录表`datasheet.csv`，包含：
   - 每个实验（Index）的 input 文件（BP_XXXX）、类型（sweep/mapping/Au）
   - 偏振（S/P）、中心能量、Pass E. 等参数
   - 对每个数据的comments
   - 根据 datasheet 选出要处理的 cut（2D sweep）文件，金数据（标注 Au）用于费米能拟合
   - 探测器极端角处信号弱，应当排除离群值

## 工作流

- 将E_k-k数据的费米边调平，并费米能置0 → 将角度空间的高对称点置0 → 转换到k空间
- 先核对再调用：search_api 看作用域、get_api 看返回结构、inspect 看运行时类型证据（type/dims/shape/attrs...），凭证据写代码，不凭假设
- numpy 组合前先断言 shape，不同维度用 [:, None] / [None, :] 显式广播
- 不复刻 peaks 内部函数，用公开 API + 属性验证（确需内部先 help() 查签名）


## 产物约定

只有最终结果`<stem>_pre.nc`写入该 notebook 的
`<stem>_output/` 目录；使用print和plot向用户说明关键信息。

## 关键参数（不猜测，来源按优先级）

- `theta_par_offset_deg`（高对称点位置）：datasheet.csv给agent的note中/ 用户给定（askuserquestion tool），无法从数据推导
- 费米能：金数据拟合（`fit_gold`，datasheet 标注 Au 的）> 问用户（askuserquestion tool）
- 偏振/能量等只用于判断，不参与计算
