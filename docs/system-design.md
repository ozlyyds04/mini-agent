# 系统设计

> 面向读者的版本。实现细节以代码为准；本文说明**为什么这么拆**、**数据怎么流**、**取舍在哪**。

## 1. 分层

```
┌──────────────────────────────────────────────────────────────┐
│  cli.py           交互层：多窗口切换、流式展示、命令           │
│                   （只是一个事件订阅者，不含业务逻辑）          │
└───────────────▲──────────────────────────────────────────────┘
                │ EventBus（tool.call / tool.result / compression / memory.recall …）
┌───────────────┴──────────────────────────────────────────────┐
│  runtime/loop.py  编排层：四步循环、轮次控制、取消、异步边界   │
│  runtime/executor.py    工具执行：校验→超时→异常→ToolResult    │
│  runtime/async_tools.py 异步任务：提交/队列/step 边界注入      │
└───┬───────────┬──────────────┬──────────────┬────────────────┘
    │           │              │              │
┌───▼────┐ ┌────▼─────┐ ┌──────▼──────┐ ┌─────▼──────┐
│ llm/   │ │ tools/   │ │ context/    │ │ session/   │
│ 协议   │ │ 注册+执行│ │ 组装+压缩   │ │ 持久化     │
│ 归一化 │ │ Schema   │ │ budget      │ │ JSONL      │
└───┬────┘ └──────────┘ └──────┬──────┘ └─────┬──────┘
    │                          │              │
┌───▼─────────┐          ┌─────▼──────┐ ┌─────▼──────┐
│ memory/     │          │ trace/     │ │ artifacts/ │
│ 按需召回    │          │ span+JSONL │ │ 大结果归档 │
└─────────────┘          └────────────┘ └────────────┘
```

**依赖方向是单向的**：`loop` 依赖所有层，但没有一层反过来依赖 `loop`（工具通过 `ToolContext` 拿到会话，而不是 import runtime）。这样每一层都能单独测试——140 个用例里绝大多数都是单层测试。

## 2. 一次请求的数据流

```
用户输入
  │
  ├─① session.append(user message) + 落盘 JSONL
  │
  ├─② 记忆召回（仅当触发词命中）→ extra_system
  │
  └─③ loop:
       ├─ 异步结果 drain（step 边界注入）
       ├─ compressor.maybe_compress(session)      # 按水位 L0→L3
       ├─ builder.build(session, tools, extra_system)
       │     system = 角色 + 时间 + 工具清单 + 结构化状态(TODO) + pinned + 摘要 + 召回片段
       │     messages = 最近 K 轮（tool_call/tool_result 严格配对，超长工具结果截断）
       ├─ llm.chat(messages, tools)  ──► 归一化 LLMResponse
       │
       ├─ 无 tool_calls → append assistant → 返回
       └─ 有 tool_calls → executor.execute(每个 call) → append tool 结果 → 回到循环头
```

关键点：**每一轮都重新组装 context**，而不是维护一个不断变大的 messages 数组。压缩、召回、状态注入都依赖这个设计才成立。

## 3. 关键决策记录（ADR）

### ADR-1　不依赖任何 Agent 框架

- **背景**：题目明确要求从零实现主流程。
- **决策**：手写循环、手写 HTTP、手写 Schema 生成。
- **后果**：多了约 500 行基础代码；换来的是每一处行为都可解释、可测试，且 clone 即跑。

### ADR-2　零第三方依赖

- **备选**：`openai` SDK（代码更短）、`pydantic`（配置校验）、`tiktoken`（精确 token）。
- **决策**：全部不用。token 用字符分类估算（CJK≈1/字、ASCII≈1/4），配置用 dataclass。
- **权衡**：token 估算有 ±20% 误差，所以压缩水位留了 8% 安全余量（L3 在 92% 触发，不是 100%）。
- **理由**：面试官 clone 下来直接跑，比多 20 行优雅代码重要。

### ADR-3　工具 Schema 由类型注解生成

- **备选**：每个工具手写 JSON Schema。
- **决策**：`@tool` 装饰器用 `inspect.signature` + `get_type_hints` + docstring 生成。
- **后果**：支持 `Literal`（→ enum）、`Optional`、`list[X]`、默认值、参数描述；不支持嵌套 dataclass（当前 5 个工具不需要）。

### ADR-4　工具失败转为观测，而不是异常

- 未知工具 / 参数错 / 超时 / 内部异常 → `ToolResult(ok=False, error=…)`，回喂给模型并附修正建议。
- **理由**：Agent 的容错单元是"一轮对话"而不是"一次函数调用"。直接抛异常会让一次小失误废掉整轮。

### ADR-5　压缩在请求前做，且必须可降级

- 水位：60% / 80% / 92% 三档。
- 摘要调用失败 → `CompressionReport(ok=False)`，不抛异常；builder 的硬预算兜底。
- **理由**：压缩是为了"不要超窗"，如果压缩本身把请求搞挂就本末倒置了。

### ADR-6　记忆召回默认关闭式触发

- 只有出现指代/时间词才召回；召回结果放 system 且带"非当前指令"声明。
- **理由**：过度召回比不召回更糟——会让模型执行半个月前的旧指令。

### ADR-7　异步结果以 `role="user"` 注入

- 备选：复用 `role="tool"` + 原 `tool_call_id`。
- **决策**：不行。同一条 assistant 消息的 tool_call_id 已被"已受理"占用，重复使用违反协议（Anthropic 会直接报错，OpenAI 系行为未定义）。
- **后果**：注入消息带 `[异步工具完成]` 前缀，并在 system 中说明"这类通知是运行时插入的"。

### ADR-8　同一 session 串行

- 进程内 `RLock`（store 内部方法会嵌套加锁，必须可重入），run 用非阻塞获取，失败返回 busy。
- **未做**：跨进程文件锁。单机 CLI 场景不需要，分布式场景应交给外层调度（见架构题模块四）。

## 4. 并发模型

| 位置 | 模型 | 说明 |
|---|---|---|
| 主循环 | 单线程同步 | 确定性优先，便于复现 |
| 工具执行 | 线程池 + `future.result(timeout)` | 超时可控；工具共享 session 状态，默认串行 |
| 异步工具 | 独立线程池 + `queue.Queue` | 提交即返回，结果在 step 边界取走 |
| session | 每 id 一把 `RLock` | 同一会话串行；不同会话并行 |
| CLI | 请求跑在 worker 线程 | Ctrl-C 时主线程设置 CancelToken，不强杀 |

**明确不做**：同一 session 内并行执行多个工具。`parallel_tool_calls` 默认关闭——todo 这类工具会改共享状态，并行会让结果不可复现。

## 5. 可测试性设计

三条硬约束让 140 个测试能全离线跑：

1. **LLM 是可注入的**：`AgentRuntime(config, llm, …)` 里 `llm` 只要有 `chat()`；测试用 `FakeLLM`。
2. **时间与网络不在核心路径上**：核心循环不做 `time.sleep` / 不发请求；重试在 `LLMClient` 内部，测试用 mock 替换。
3. **存储可指向临时目录**：`Config.data_dir` 一个字段决定 sessions / artifacts / traces 的位置。

## 6. 已知局限

| 局限 | 影响 | 如果继续做 |
|---|---|---|
| token 是估算 | 压缩水位有偏差 | 接 tiktoken / 各厂商 tokenizer |
| 记忆是关键词打分 | 语义召回弱 | 换向量检索 + rerank，接口不变 |
| 单机单进程 | 不支持多实例共享 session | session 加文件锁 / 迁到 Redis + 事件表 |
| 工具并行默认关闭 | 多工具场景慢一点 | 无状态工具标记 `parallel_safe=True` 后再开 |
| 摘要是一次性合并 | 极长会话可能丢早期细节 | 分层摘要 + 关键实体表持久化 |
| 没有权限/审批层 | 工具可执行任何允许的操作 | 加 `ToolPolicy`：按工具名/参数白名单 + 人工确认（对应架构题模块五） |
