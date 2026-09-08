# VR 场景 Tool-Calling Agent 重构版

## 交付状态

本版基于用户提供的 agent-project-improved.rar，保留参考库、场景规划、图像生成、接缝修复、图像审核、重试、音乐与结果导出功能，将原来的固定主流程拆成模型可调用的工具，合并为一个 UI。

**已完成代码与离线测试；尚未完成真实模型、GPU、Suno、Chroma 及浏览器端到端验收。** 当前验证环境没有这些服务和数据，也无法下载 UI 依赖。不能据此声称更快、缓存命中率提高、图像质量提高或生产可用。

## 启动

推荐复制本目录到原项目旁的新目录，使用原本能运行 GPU 生成的 Python 环境，不覆盖唯一的原项目副本。

1. 将原有 `PictureData/`、`PictureBase/`、`models/` 放到本目录。RAR 没有包含这些资产，因此本交付也不包含。
2. 复制 `.env.example` 为 `.env`，填写原有服务凭据。Chroma 的 Embedding 凭据是 `CHROMA_OPENAI_API_KEY`，未填时使用 `OPENAI_API_KEY`。
3. 在原已能运行生成链路的环境内安装 `python -m pip install -r requirements-agent.txt`。新环境还需要 `requirements.txt` 中的完整原依赖及适配 GPU 的 PyTorch。完整原依赖组合未在本次环境重新验证。
4. **先切换工作目录到本项目目录**，运行 `python UnifiedApp.py`。
5. 浏览器打开 `http://127.0.0.1:8000/gui`。

旧命令 `python AgentAPP_Interactive.py`、`python AgentAPP_Passive.py` 现在都只是统一入口的兼容启动器，不再各自创建页面或占用两个端口。不要同时运行三个命令。

UI 顶部选择：

- **直接描述（Passive）**：输入一段需求，点击生成；保留原单次状态提取逻辑。
- **对话引导（Interactive）**：发送消息进行对话，信息足够后点击生成，也可点击“使用当前信息”；风险规则仍然优先。
- 切换模式或清空时，清理对话状态、输入、图片、音乐和结果，避免复用隐藏的旧状态。

生产图像仍依赖原模型和本地资产；不能只提供 API key 就直接运行完整系统。

## 架构和模型自主权

`UnifiedApp.py` 处理输入与页面，`PassiveState.py` 和 `DialogTherapistAgent.py` 生成原有结构化上下文。
`ToolAgent.py` 通过 LiteLLM `acompletion(tools=..., tool_choice="auto")` 请求模型决策，读取 `tool_calls`，验证参数，执行注册函数，再发送包含匹配 `tool_call_id` 的工具结果。
`AgentBackend.py` 持有会话领域对象并执行实际业务能力。

这里的“任意工具”指**任意已注册且当前满足前置条件的领域工具**，不包括任意 Python、Shell、HTTP URL 或文件系统访问。允许模型决定场景批次、生成和审核的交错顺序、失败场景修订顺序，以及是否查询状态。程序不会根据阶段硬编码替模型挑选下一工具。

| 工具 | 参数 | 前置条件与实际作用 |
|---|---|---|
| retrieve_context | 无 | 根据原结构化状态和原策略映射检索全局参考；会话内重复调用复用结果 |
| plan_session | 无 | 先检索；复用原计划生成、Schema 修复与逐场景参考绑定 |
| start_music | 无 | 先规划；启动两条音乐请求的后台线程，重复调用不重复提交 |
| generate_scenes | steps: 1—10 的唯一编号列表 | 仅未生成场景；调用原 SDXL/ControlNet/IP-Adapter 生成并修复接缝 |
| audit_scenes | steps | 仅已生成且待审核候选；计算指标并调用原多模态审核 |
| refine_scenes | steps | 仅审核失败且重试未耗尽；复用参考重检索、提示词修订、种子策略与重新生成 |
| inspect_session | 无 | 读取紧凑的权威状态 |
| finish_session | 无 | 所有场景必须通过、初始生成失败或达到重试上限；保存最佳候选、后处理、等待音乐并导出 |

工具 Schema 由 Pydantic 生成，拒绝额外字段、非整数/重复/越界步骤。程序拒绝未知工具、重复 call ID、越权步骤、提前结束、重复初始生成与超额重试。不会执行模型提供的文件路径。注册表定义的是可执行能力；没有伪造一个只写文档却未加载的“Skill”。

对话采集仍是用户驱动的前置阶段，风险检查是不可绕过的程序规则。这两部分没有为了增加 Agent 调用次数而强行改成自由调度；自主工具调用发生在内容生成执行阶段。

## 上下文工程与缓存

- **分离职责**：调度模型只看场景状态、重试次数、短审核反馈；完整用户状态、检索描述、计划及图像由对应专用模块使用。
- **句柄而非大对象**：工具参数是 scene step ID，图片不以 Base64 反复进入调度上下文。完整数据保存在后端和日志。
- **固定前缀**：系统提示词和工具 Schema 顺序稳定，动态状态放在后面。
- **有限历史**：保留最近四个完整 assistant/tool 交换组，并追加最新后端快照。按完整交换裁剪，避免遗留没有对应调用的 tool 消息。保留供应商返回的消息元数据。
- **执行与输出预算**：默认最多 64 次调度决策、96 次工具调用，每次输出 token 上限 1800。它们不是严格的总成本或 GPU 墙钟时间预算。
- **记录用量**：每次调度响应的 usage、耗时写入 `coordinator_usage`，供应商返回的 cached_tokens 等字段按原样保留；专用子模块调用用量尚未统一汇总。
- **缓存边界**：服务端可能复用相同前缀的 KV 状态，但托管 API 不让本代码直接管理 KV Cache。`AGENT_PROMPT_CACHE_KEY` 默认关闭，只有确认模型供应商和 LiteLLM 路由支持时才启用。没有设置跨用户的整份回答缓存，没有缓存随机生成图片。
- **上下文长度与前缀复用存在取舍**：裁剪历史降低输入长度，但会牺牲部分长历史前缀命中，不能同时保证最大命中率和最短上下文。

没有为本项目部署 vLLM/PagedAttention，也没有修改模型内部 KV 分配。不能在简历中写“自研 KV Cache”。

## 实际的执行优化

原 `generate_image` 虽声明为 async，内部主要是同步计算。本版在工作线程执行这些调用，让事件循环能调度音乐请求。音乐工具启动后，GPU 工作和音乐网络请求可以重叠；测试用同步事件验证了这一点，但实际节省时间尚未测量。

图像生成、接缝修复、指标模型与后处理仍通过同一文件锁串行使用共享 GPU 资源，模型初始化也在锁内。不同会话可以在非 GPU 阶段推进。取消等待锁或取消 GPU 请求时，等待工作线程正确退出再释放锁，防止后台线程仍在使用管线时另一个请求进入。

长音乐请求退出时会等待后台线程结束，而不是声称取消 asyncio Task 就停止 HTTP 请求。原 50 次轮询逻辑保留，补充轮询 HTTP timeout；这不是快速中断机制。

工具批量执行降低调度次数，但额外的调度 LLM 调用也会增加成本和延迟。固定任务的纯流程可能仍更快；本版的收益是可观察、受约束的模型工具决策，速度需要实测。

## 保持的功能与明确变化

保留：Embedding 模型、Chroma 查询与规则排序、计划验证与一次修复、SDXL/ControlNet/IP-Adapter 参数及生成逻辑、接缝修复、指标、审核提示词、每场景最多三次修订、最佳候选、风险拦截、音乐兜底、PhotoFinisher 和原主要输出字段。

变化：

1. 固定主循环替换为模型工具调用循环，允许按不同场景批次推进。随机结果和执行顺序不保证与原版本完全一致。
2. 不再默认把有失败的输出标记为 Success：缺图或仍有 FAIL 时是 `Partial`，新增 `missing_steps`、`planned_scene_count`。
3. `quality_metrics` 和审核仍对应最终 PhotoFinisher 之前的图像，新增 `metrics_stage` 明确标记，避免伪称后处理结果也已审核。
4. `GET /get-latest-session` 要求显式 `session_id`，不再跨用户返回全局最近一条；内存最多保留 32 个普通结果。磁盘日志不随内存淘汰删除。
5. API 的 `POST /generate-session` 使用 `mode` 选择输入方式，保留 `model_name`。统一 UI 的“模型设置”保留模型选择；仅接受 `LLM_MODELS` 配置的模型名称。旧选项保留不代表各供应商当前已实际连通。`AGENT_MODEL` 可单独指定调度模型，未设置时跟随 UI 选择。切换模型会清空旧对话和结果。
6. 对于未知工具、循环预算耗尽或服务故障，记录失败，不自动绕回旧固定流程掩盖 Agent 调度失败。

## API 示例

Passive 请求体：

```json
{"mode":"passive","description":"最近有些疲惫，希望看温暖、开阔的湖边自然环境。"}
```

Interactive 请求体使用同一路径，传 `mode="interactive"` 及原格式的 `dialog_state`，或直接使用统一 UI 完成对话。

结果位置：`static/results/{passive|interactive}/{session_id}/session.json`，全局追加记录仍为 `static/results/session_logs.jsonl`。每个完成或调度失败的记录包括工具名、实参、错误、耗时和决策用量；它是排查记录，不提供重启断点恢复。

本服务默认 localhost，未新增登录、租户权限或公开部署配置。不要将本地研究原型描述为生产服务。

## 验证

无需 GPU/API 的测试：

```text
python -m unittest discover -s tests -v
python -m compileall -q .
```

本次 19 项测试通过：原 9 项，加上工具参数限制、真实后端配合模拟模型的完整工具循环、前置条件、模型选择不同批次、重试预算/最佳候选、部分结果、风险前置、未知工具/坏 JSON/循环预算，以及两项取消与锁行为测试。

模拟的是模型响应、图片生成、审核和指标服务；实际执行的是 ToolAgent 协议处理与 AgentBackend 业务编排。因此不是只检查返回固定字符串，也不等同于真实模型和 UI 已经联通。

真实环境验收建议：

1. 打开同一 `/gui` 页面，切换两种模式，确认输入、旧对话和结果会清空。
2. 用原有效资产分别完成一次 Passive 和 Interactive 请求，检查十个场景、两条音乐与 Unity 配置。
3. 查看 session.json 的 tool_trace，确认工具选择来自模型响应；抽查失败场景确实经过 refine 后再次 audit。
4. 分别测试空输入、风险输入、缺参考库、图像失败和音乐失败。
5. 比较新旧版本多次运行的总时长、调度 token、实际缓存字段和审核通过情况；未完成比较前不声称优化百分比。

## 参考资料

- OpenAI Function Calling：https://developers.openai.com/api/docs/guides/function-calling
- OpenAI Prompt Caching：https://developers.openai.com/api/docs/guides/prompt-caching
- LiteLLM Tool Calling：https://docs.litellm.ai/docs/completion/function_call
- LiteLLM Prompt Caching：https://docs.litellm.ai/docs/completion/prompt_caching

参考的是工具调用协议、上下文安排及缓存边界；没有直接搬入模板的生产指标。
