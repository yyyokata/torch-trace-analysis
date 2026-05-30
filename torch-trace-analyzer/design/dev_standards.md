# 项目编程与开发规范汇总

## 1. Git 工作方式与工作目录

### 1.1 工作目录标准结构
所有开发工作在 `workspace/develop/` 下进行，每个任务拥有独立的克隆副本，严禁在同一 clone 中并发操作：
- `develop/master/`: 只读基线。
- `develop/{task_name}/`: 具体任务目录（如 `timing/`, `ast/` 等）。
  - 每个任务目录下必须同时包含 `torch-trace-analyzer` 和 `storage` 两个 clone。
  - 脚本实际路径：`{task_dir}/torch-trace-analyzer/torch-trace-analyzer/scripts/analyze_trace.py`。

### 1.2 任务开始前强制检查清单（最高优先级）
在开始任何文件修改前，必须完成以下清单确认：
1. **独立 Clone**：确认工作目录是独立 clone（`git remote -v` 指向对应 remote）。
2. **分支正确**：确认已切换到正确的 `feat/*` 分支，不在 master 上开发。
3. **无文件重叠**：确认工作目录与其他并发任务操作的文件不重叠。
4. **违规后果**：跳过检查直接修改属于高危操作，必须立即回退并重新从独立 clone 开始。

---

## 2. 合入规范 (High Risk)

### 2.1 研发分支 (External GitHub)
合入 `master` 前必须完成以下四步（顺序不可颠倒）：
1. **Rebase master**：feat 分支先 rebase 到最新 master，解决所有冲突。
2. **全量测试**：运行 `python3 testset/test_dag_rules.py`，必须 **7/7 PASS**。
3. **提 MR**：测试通过后提交 GitHub MR，并将链接发给用户。
4. **用户合入**：由用户 review 后合入，严禁自行 push master。

### 2.2 测试分支 (Internal storage)
1. **Rebase 时机**：随研发分支同步 rebase。
2. **合入流程**：Rebase -> 全量测试（记录结果） -> 直接 `git push origin master` 合入。

---

## 3. 文件操作与 Workspace 清理

- **严禁并发修改**：多任务严禁在同一目录或修改同一文件。
- **严禁 `git add -f`**：不得绕过 `.gitignore` 提交 output/、*.tar.gz 等产物文件。
- **即时清理**：
  - 任务完成后删除所有 `*.bak`、`*.backup` 等临时备份文件。
  - feat 分支合入后，从内部仓库 `WorkSpace/` 目录删除对应的实验脚本（如 `analyze_trace_timing_fix.py`）。
  - **保留内容**：只保留当前生产脚本及配套文档。

---

## 4. 设计文档管理

- **Markdown 优先**：不再使用飞书文档。设计文档直接生成 `.md` 文件写入仓库的 `design/` 目录。
- **随代码同步**：文档随代码一起 commit 提交。
- **一文档一主题**：每个设计主题独立文件，避免内容堆叠。

---

## 5. 代码规范

### 5.1 AST 优先原则
- **禁止正则兜底**：所有源码解析（调用点定位、attr 识别、数据流追踪等）必须使用 AST walk 实现，不得使用正则作为 fallback。

### 5.2 模块化与行数限制
- **文件行数限制**：单个 `.py` 文件行数不得超过 **2000 行**。
- **逻辑抽离**：基础设施和辅助函数应抽离至 `utils.py` 或 `containers.py` 等独立模块。
- **显式操作**：禁止使用 `globals().update()` 做跨模块符号注入，必须使用显式 import。

### 5.3 复用优先
- 设计新函数前必须先检查现有实现能否复用或扩展，不得盲目新增。

### 5.4 Mock Runtime 规范
- **强制开启 FakeTensorMode**：`main_model_mock.py` 入口顶部必须硬编码 `os.environ["MOCK_RUN_DISABLE_FAKE_TENSOR"] = "0"`，不留 fallback 开关。

---

## 6. 测试规范

### 6.1 修复流程 (不得跳步)
1. **根因分析**：准确定位代码位置并说明原因。
2. **测例设计**：先输出测例方案（mock 数据 + 关键断言）给用户审核。
3. **用户确认**：等用户确认方案后再编写修复代码。
4. **修复与验证**：修复后必须跑通 `test_dag_rules.py` 7/7 PASS。
5. **测例落地**：在测试文件中补充回归测例，严禁仅靠人工验证替代代码测例。

### 6.2 节点数修改原则
- **节点数只能增加**：静态树展开新路径导致节点数增加是正确行为。
- **减少即 Regression**：如果新改动导致节点数比基线减少，视为 bug 需要修复。
- **变化必分析**：任何节点数变化前必须先分析清楚原因，不得未经分析直接回退。

### 6.3 覆盖率要求
- 必须遵守 `README.md` 中定义的准入阈值。
- 补测优先级：`timeline_analysis` 主链路 > 高复杂度辅助函数。

---

## 7. Trace 数据处理原则

- **后端核心化**：所有 Trace 数据的计算、分类、聚合或时长统计必须在 `analyze_trace.py` 中完成。
- **前端纯展示**：前端（`frontend_html.py` / JS）仅负责展示，不得引入任何 trace 数据处理逻辑。

---

## 8. 安全改动规则

- **备份基线**：任务开始前将目标文件备份为 `{filename}_{task_id}_BASELINE.bak` 作为回退点。
- **故障回退**：若改动导致测试失败（如 7/7 FAILED），必须立即从基线恢复，严禁在破损文件上叠加修改。
- **验证后交付**：每次基线验证通过后，必须将生成的 HTML 文件打包交付给用户。
