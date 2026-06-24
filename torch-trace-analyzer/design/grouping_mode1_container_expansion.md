## 背景
现有两种 group：
- A-group（容器 group）：PyTorch Module 边界，如 self.resblocks
- B-group（callloc_group）：相邻相同 call location 的 op 归一组

模式 1 目标：游离在容器外、但其前后向仅连接容器子孙的 op，归入容器，而不是单独成 callloc_group。

## 术语
- C：目标容器
- S：C 的全部子孙节点集合
- T：候选游离节点集合（初始为所有游离节点，即不属于任何容器子孙的节点）
- 游离节点：不属于任何容器子孙的节点

## 归入条件
节点 v 可归入 C，当且仅当：
- v 的所有直接 in-edge 的 src ∈ S ∪ T
- v 的所有直接 out-edge 的 dst ∈ S ∪ T

## 算法：收缩法（O(n+e)）
1. T = 所有游离节点
2. 对 T 中每个 v：若存在 in-edge src ∉ S ∪ T，或 out-edge dst ∉ S ∪ T，则从 T 中删除 v
3. 重复步骤 2 直到 T 稳定（不动点）
4. 剩余 T 整体归入 C

复杂度说明：每个节点最多被删除一次，每条边最多被检查两次，总复杂度 O(n+e)。

## 边界约定
- out-edge 有任意一条出 S ∪ T 的节点不归入，不创建出口 port
- 模式 1 先于 B-group 处理，以容器为单位整体评估
- 多容器竞争（v 的入边来自 C1、C2 子孙）：归入两者的公共父容器
- 典型案例 `C→v→w→C`：v、w 都不被删除，整体归入 C

## 执行顺序
模式 1 容器扩展 → B-group（callloc_group）

## 待设计
- 多容器竞争时公共父容器的精确计算方式
- 模式 2（Sequential Chain）和模式 3（Parallel Siblings）