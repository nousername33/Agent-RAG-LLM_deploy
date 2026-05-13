# Edge-LLM-RAG-Voice

全离线、模块化的智能座舱语音系统，基于 RK3576 NPU 实现端到端语音交互闭环：**ASR 语音识别 → Agent 调度 → RAG 知识检索 + LLM 推理 → TTS 语音合成**，端侧部署，全程离线。

## 技术栈

`Linux` `C++17` `Python` `RK3576 NPU` `Sherpa-ONNX ASR` `LangChain Agent` `RAG` `DeepSeek-R1 LoRA` `RKLLM w4a16 量化` `VITS TTS` `ZeroMQ` `CMake` `多线程`

## 系统架构

```
┌─────────────┐    ZMQ     ┌──────────────────────────────────┐    ZMQ     ┌─────────────┐
│   ASR 模块   │──PUSH:5555→│          Agent 模块 (Python)      │──REQ:7777→│   TTS 模块   │
│ Sherpa-ONNX │            │                                  │            │ VITS + ALSA │
│ 流式识别     │            │  IntentClassifier → ReAct Loop   │            │ 双缓冲队列   │
│ VAD 0.4s    │            │  ToolRegistry → Memory           │            │ 生产者-消费者 │
└─────────────┘            │           │           │          │            └─────────────┘
                           │     RAG Tool     LLM Tool        │
                           │       │              │            │
                           └───────┼──────────────┼────────────┘
                                   │              │
                                   ▼              ▼
                           ┌──────────┐   ┌──────────────┐
                           │ RAG 模块  │   │   LLM 模块    │
                           │ text2vec │   │ DeepSeek-R1   │
                           │ 768维向量 │   │ 1.5B w4a16   │
                           │ 余弦检索  │   │ RKLLM+NPU    │
                           └──────────┘   │ ZMQ PUSH:5559 │
                                          └──────────────┘
```

## 模块说明

### 1. Agent 模块 (`agent/`)

基于 LangChain 的 Tool-based Agent，统一调度 RAG 检索和 LLM 推理，是整个系统的调度中枢。

**核心流程：**

1. **输入归一化** — ASR 文本 + 对话历史 + Query 重写（指代消解）
2. **阶段1 轻量意图分类** — 基于关键词+句式规则，6 类意图粗粒度路由
3. **阶段2 手动 ReAct 循环** — Thought → Action → Action Input → Observation 推理链
4. **Memory 管理** — 三层记忆（短期上下文窗口 / Query 重写 / 结构化状态）

**意图分类器** (`agent/src/intent/intent_classifier.py`):

| 意图类型 | 处理路径 |
|---------|---------|
| GREETING | 直接回复模板，不调用 LLM/Tool |
| COMMAND | 直接回复确认 |
| FACTUAL | 优先 RAG 检索 (`retrieve_vehicle_knowledge` + `generate_with_context`) |
| COMPLEX_EXPLANATION | 混合 RAG + LLM |
| OPEN_GENERATION | 纯 LLM 生成 (`generate_answer`) |
| UNKNOWN | 全量工具，Agent 自行决策 |

**Tool 工具链** (`agent/src/tools/`):

| Tool | 功能 |
|------|------|
| `retrieve_vehicle_knowledge` | 向量检索车辆手册知识 |
| `generate_answer` | 纯 LLM 直接生成回答 |
| `generate_with_context` | RAG 检索 + LLM 融合生成 |

### 2. RAG 模块 (`automotive_edge_rag/`)

基于 text2vec-base-chinese Embedding 模型构建车载知识库向量检索系统。

**技术链路：**
- **数据处理**: 解析车辆使用手册 Markdown → 按章节切分文本片段 → SentenceTransformer 编码 → 768 维语义向量
- **向量存储**: `vehicle_embeddings.npy` + `vehicle_data.pkl` + `similarity_matrix.npy`
- **检索**: 查询编码 → 余弦相似度计算 → Top-K 结果返回（默认 threshold=0.5, top_k=5）
- **Python 实现**: `vehicle_vector_search.py` — 供 Agent 同进程调用
- **C++ 实现**: pybind11 嵌入 Python 解释器，供 C++ 主流程调用

**查询分类器** (`automotive_edge_rag/cpp/query_classifier.cpp`):

C++ 规则分类器，将查询分为 5 类：EMERGENCY（紧急故障）、FACTUAL（事实型）、COMPLEX（复杂解释型）、CREATIVE（开放生成型）、UNKNOWN。

**三种响应模式：**
- **纯 RAG** — 事实型/紧急查询，直接返回手册内容
- **纯 LLM** — 开放生成型，由 LLM 自由回答
- **RAG+LLM 混合** — 复杂查询，RAG 检索注入 `<rag>` 标签后交由 LLM 生成

### 3. LLM 模块 (`llm/`)

DeepSeek-R1-Distill-Qwen-1.5B 模型的端侧部署与推理。

**模型适配：**
- **LoRA 微调**: 基于 Llama Factory，构建座舱领域数据集进行指令微调 → 导出 LoRA adapter 权重 → `lora/merge_lora.py` 合并基座模型 → 完整 HF 模型
- **RK 量化部署**: 使用 RK 工具链对合并模型进行 **w4a16 量化** → 导出 `.rkllm` 文件
- **NPU 绑定**: NPU + 2 小核 (CPU0/CPU2) 绑定，约 8% 性能损失换取推理稳定性
- **推理参数**: top_k=1, top_p=0.95, temperature=0.8, max_new_tokens=100

**LoRA → RK 部署流水线** (`lora/`):
详见 [lora/README.md](lora/README.md)，关键步骤：训练数据准备 → Llama Factory 微调 → 权重合并 → 量化校准 → RKLLM 导出。

**伪流式输出** (`llm/rknn-llm/examples/.../llm_demo.cpp`):

RKLLM 回调中逐字符接收，遇中文标点（，。；！？）即分段发送，缩短首响应延迟。

**RAG 上下文注入:**

LLM 接收 `<rag>` 标签分割的 prompt（`user_query<rag>rag_context`），自动构建带 RAG 上下文的 chat template。

**通信协议:**
- ZMQ REP `:8899` 接收 Agent 发送的 prompt
- ZMQ PUSH `:5559` 将生成结果推送给 Agent

### 4. TTS 模块 (`tts/`)

VITS 架构语音合成，支持中文前端处理与实时音频播放。

**模型架构:**
- TextEncoder + StochasticDurationPredictor + DurationPredictor
- 多 Generator 支持：HiFi-GAN / Multi-Scale / IStft / Multi-Band
- 中文 TN 文本正则化 + 分词 + G2P 注音
- 英文 IPA 音标支持

**双缓冲队列** (`tts/tts_server/include/MessageQueue.h`):

```
文本队列 (mutex + condition_variable)  →  Synthesis 线程  →  音频队列 (mutex + condition_variable)  →  ALSA 播放线程
```

生产者-消费者模式消除播放卡顿，合成与播放并行执行。

**实时优化:**
- CPU 核心绑定 + 实时调度策略 (SCHED_FIFO)
- 16kHz 单声道 ALSA 输出
- 文本分段合成，首段优先播放

### 5. ZeroMQ 通信模块 (`zmq-comm-kit/`)

统一的模块间异步通信库，封装标准 ZMQ 通信模式。

**接口:**
- `ZmqServer` — ZMQ_REP 绑定端口，接收请求并回复
- `ZmqClient` — ZMQ_REQ 连接端口，发送请求并等待响应
- `ZmqInterface` — 基类，统一 socket 生命周期管理

**通信拓扑:**

| 方向 | 模式 | 端口 | 用途 |
|------|------|------|------|
| ASR → Agent | PUSH-PULL | :5555 | 语音识别文本推送 |
| Agent → LLM | REQ-REP | :8899 | 发送 prompt，接收 ACK |
| LLM → Agent | PUSH-PULL | :5559 | LLM 生成结果推送 |
| Agent → TTS | REQ-REP | :7777 | 发送回复文本，TTS 合成播放 |

### 6. Voice/ASR 模块 (`voice/`)

基于 Sherpa-ONNX 流式 ASR，使用 **Zipformer 中英双语小模型** 进行离线语音识别。
- VAD 静音阈值优化至 **0.4s** 加速端点判定
- 麦克风暂停机制抑制回声误触发
- 识别结果通过 ZMQ PUSH 发送至 Agent

## 端到端数据流

```
1. 麦克风采集音频
      ↓
2. Sherpa-ONNX 流式 ASR 识别 (VAD 0.4s)
      ↓ ZMQ PUSH :5555
3. Agent 接收文本 → InputNormalizer (Query 重写)
      ↓
4. Stage1 IntentClassifier (关键词规则, <1ms)
      ↓
   ├─ GREETING/COMMAND → 直接回复模板
   └─ 其他 → Stage2 ReAct Loop
              ├─ retrieve_vehicle_knowledge → RAG 向量检索 (余弦相似度)
              ├─ generate_answer → ZMQ REQ :8899 → LLM 推理 → ZMQ PUSH :5559
              └─ generate_with_context → RAG + LLM 融合
      ↓
5. Agent 生成最终回复
      ↓ ZMQ REQ :7777
6. TTS 文本合成 → 双缓冲队列 → ALSA 播放
```

## 性能优化

- **ASR**: VAD 静音阈值 0.4s，加速端点检测
- **LLM 伪流式**: 逐标点分段发送，首响应 ≤0.9s，分段时延下降 50%
- **TTS 双缓冲**: 合成与播放并行，消除卡顿
- **阻塞控制**: 麦克风暂停机制实现零等待，抑制回声误触发
- **NPU 调优** (`perf.sh`): 固定 NPU 1GHz / CPU 2.2GHz / GPU 950MHz / DDR 2.1GHz 频率，禁用 CPU idle 状态
- **NPU 核心绑定**: NPU + 2 小核 (CPU0/CPU2)，牺牲约 8% 性能换取稳定性
- **端到端闭环**: 语音输入 → RAG 检索 → LLM 思考 → 语音输出 ≤ 4s

## 项目结构

```
Edge_LLM_RAG_Voice/
├── agent/                          # Agent 调度模块 (Python)
│   ├── config.yaml                 # 全局配置
│   ├── requirements.txt
│   └── src/
│       ├── main.py                 # 入口：主循环 + ZMQ 通信初始化
│       ├── core/
│       │   └── request_orchestrator.py  # 请求编排器：两阶段决策 + ReAct 循环
│       ├── intent/
│       │   ├── intent_classifier.py     # 轻量意图分类（关键词规则）
│       │   └── query_rewriter.py        # Query 重写（指代消解）
│       ├── tools/
│       │   ├── tool_registry.py         # Tool 注册中心
│       │   ├── rag_search.py            # RAG 检索 Tool
│       │   ├── llm_generate.py          # 纯 LLM 生成 Tool
│       │   └── llm_rag_generate.py      # RAG+LLM 融合 Tool
│       ├── memory/
│       │   └── conversation_memory.py   # 三层对话记忆管理
│       ├── communication/
│       │   ├── zmq_bridge.py            # ZMQ PULL/REQ socket 封装
│       │   └── llm_client.py            # LLM ZMQ 通信客户端
│       └── llm/
│           └── zmq_llm.py               # LangChain BaseLLM 适配 RKNN LLM
│
├── automotive_edge_rag/            # RAG 检索模块
│   ├── models/                     # text2vec-base-chinese 模型文件
│   ├── python/
│   │   ├── vehicle_vector_search.py     # 向量检索核心
│   │   ├── vehicle_data_processor.py    # 手册数据预处理
│   │   └── run_demo.py
│   ├── cpp/
│   │   ├── edge_llm_rag_system.h/cpp    # C++ RAG 系统 (pybind11 嵌入 Python)
│   │   ├── query_classifier.h/cpp       # C++ 查询分类器
│   │   └── demo_main.cpp               # C++ 主流程 demo
│   └── python_cpp/
│       └── persistent_search_cli.cpp    # C++ 交互式检索 CLI
│
├── lora/                           # LoRA 微调 → 部署工具链
│   ├── README.md                   # LoRA 流程完整文档
│   ├── prepare_cockpit_data.py     # 座舱领域训练数据集准备
│   ├── merge_lora.py               # LoRA adapter 与基座模型合并
│   └── convert_pipeline.sh         # 端到端自动化流水线
│
├── llm/                            # LLM 推理模块
│   ├── models/                     # 模型存储目录
│   └── rknn-llm/
│       ├── examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/
│       │   ├── deploy/src/llm_demo.cpp  # RKLLM 部署主程序
│       │   └── export/                  # 模型导出 + 量化脚本
│       ├── rkllm-runtime/               # RKLLM 运行时库
│       └── scripts/                     # NPU/CPU 频率固化脚本
│
├── tts/                            # TTS 语音合成模块
│   ├── CMakeLists.txt
│   ├── include/                    # 公开头文件
│   ├── src/                        # VITS 模型实现 + 中文前端
│   ├── tts_server/
│   │   └── include/MessageQueue.h  # 双缓冲消息队列
│   ├── test/main.cpp               # TTS 服务主程序
│   └── models/                     # TTS 模型权重
│
├── voice/                          # ASR 语音识别模块
│   ├── models/                     # Sherpa-ONNX Zipformer 模型
│   └── sherpa-onnx/                # Sherpa-ONNX 运行时
│
├── zmq-comm-kit/                   # ZeroMQ 通信组件库
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── ZmqInterface.h          # 基类
│   │   ├── ZmqServer.h             # REP Server
│   │   └── ZmqClient.h             # REQ Client
│   └── src/
│       ├── ZmqInterface.cpp
│       ├── ZmqServer.cpp
│       └── ZmqClient.cpp
│
└── perf.sh                         # RK3576 性能调优脚本
```

## 构建与运行

### 依赖

- **C++**: CMake ≥3.12, libzmq, Python3-dev, pybind11, ALSA, PortAudio
- **Python**: langchain-core, sentence-transformers, numpy, scikit-learn, pyzmq, PyYAML
- **RK3576**: RKLLM runtime, Sherpa-ONNX runtime

### 编译 ZMQ 通信组件
```bash
cd zmq-comm-kit && mkdir -p build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### 编译 TTS 模块
```bash
cd tts && mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### 编译 RAG C++ 模块
```bash
cd automotive_edge_rag/cpp && mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### 编译 LLM 模块
参考 `llm/rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/Readme.md`

### 运行

1. **启动 LLM 服务**: `./llm_demo <model_path>`
2. **启动 TTS 服务**: `./tts_server`
3. **启动 Agent**: `python -m agent.src.main`
4. **启动 ASR**: `./sherpa-onnx-streaming-asr`
5. **性能调优**: `sh perf.sh`

## 许可证

本项目各部分遵循其原始组件的许可证条款。
- RKLLM SDK: Apache 2.0
- Sherpa-ONNX: Apache 2.0
- Eigen 3.4.0: MPL 2.0
