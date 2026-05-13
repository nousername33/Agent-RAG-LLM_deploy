# LoRA 微调 → RK 部署 完整流程

本目录包含 DeepSeek-R1-Distill-Qwen-1.5B 模型 **LoRA 领域微调 → 权重合并 → RKLLM 量化导出** 的完整工具链。

## 背景

本项目的 LLM 部分先使用 **Llama Factory** 低代码平台在座舱领域数据集上进行 LoRA 微调，再将微调后的模型通过 **RKLLM 工具链** 进行 w4a16 量化，最终部署到 RK3576 NPU 上运行。

Llama Factory 只输出训练过程，训练后导出的是 **LoRA adapter 权重文件** (约几 MB~几十 MB)，不是完整模型。必须将其与基座模型合并后，才能喂给 RKLLM 工具链进行量化导出。

## 完整流程

```
┌─────────────────┐
│ 座舱领域数据集    │  (车辆手册 + 座舱对话 + RAG增强QA)
│ cockpit_data.json│
└────────┬────────┘
         │  Llama Factory LoRA SFT
         ▼
┌─────────────────┐
│ LoRA Adapter     │  (adapter_model.safetensors + adapter_config.json)
│ ~几十 MB         │
└────────┬────────┘
         │ ① merge_lora.py         ← 本目录提供
         ▼
┌─────────────────┐
│ 合并后的完整模型  │  (标准 HuggingFace 格式)
│ ~3 GB (bf16)    │
└────────┬────────┘
         │ ② generate_data_quant.py  ← llm/rknn-llm/.../export/ 已有
         ▼
┌─────────────────┐
│ 量化校准数据      │  (data_quant.json)
└────────┬────────┘
         │ ③ export_rkllm.py        ← llm/rknn-llm/.../export/ 已有
         ▼
┌─────────────────────────────┐
│ RKLLM 部署模型               │
│ model_W4A16_RK3576.rkllm    │  ← 部署到 RK3576 NPU
│ ~500 MB (w4a16 量化)         │
└─────────────────────────────┘
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `prepare_cockpit_data.py` | 座舱领域训练数据集准备脚本，支持从车辆手册生成 QA 对、座舱对话模板、RAG 增强 QA |
| `merge_lora.py` | **核心脚本**：将 LoRA adapter 权重与基座模型合并，输出完整 HF 模型 |
| `convert_pipeline.sh` | 端到端自动化流水线，串联合并→校准→导出三步 |

## 使用步骤

### 步骤 1: 准备训练数据

```bash
# 从车辆手册和对话模板生成训练数据集
python lora/prepare_cockpit_data.py \
    --manual_file /path/to/vehicle_manual.txt \
    --output cockpit_train_data.json \
    --format alpaca \
    --include_rag_enhanced
```

输出 `cockpit_train_data.json` 可直接导入 Llama Factory。

### 步骤 2: Llama Factory LoRA 微调

在 Llama Factory Web UI 中配置:

| 参数 | 推荐值 |
|------|--------|
| Model | `DeepSeek-R1-Distill-Qwen-1.5B` |
| Finetuning Type | `lora` |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Target | `q_proj,v_proj,k_proj,o_proj` |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Template | `deepseek` |
| Cutoff Length | 512 |

训练完成后导出 LoRA adapter 目录 (包含 `adapter_model.safetensors` + `adapter_config.json`)。

### 步骤 3: 合并 LoRA 权重

```bash
python lora/merge_lora.py \
    --base_model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --lora_path /path/to/lora_adapter \
    --output_dir ./output/merged_model \
    --dtype bfloat16
```

### 步骤 4: 量化导出 RKLLM 模型

**方式一: 使用自动流水线**

```bash
bash lora/convert_pipeline.sh \
    /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    /path/to/lora_adapter \
    ./output
```

**方式二: 手动执行**

```bash
# 4a. 生成校准数据
python llm/rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export/generate_data_quant.py \
    -m ./output/merged_model \
    -o ./output/data_quant.json

# 4b. 修改 export_rkllm.py 中的 modelpath 指向 merged_model, dataset 指向 data_quant.json
# 4c. 导出 RKLLM
cd llm/rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export
python export_rkllm.py
```

## 关键参数说明

### merge_lora.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_model` | 必填 | 基座模型路径 (HuggingFace 格式) |
| `--lora_path` | 必填 | LoRA adapter 目录 (含 `adapter_config.json`) |
| `--output_dir` | `./merged_model` | 合并后模型输出目录 |
| `--dtype` | `bfloat16` | 保存精度，RK 量化建议 bfloat16 |
| `--device` | `cuda` | 合并时使用的设备 |

### RKLLM 量化参数 (在 export_rkllm.py 中配置)

| 参数 | 座舱场景推荐值 | 说明 |
|------|---------------|------|
| `quantized_dtype` | `W4A16` | w4a16 量化，减小模型 4 倍 |
| `quantized_algorithm` | `grq` | GRQ 算法，w4a16 场景效果更优 |
| `target_platform` | `RK3576` | 目标芯片平台 |
| `num_npu_core` | 2 | 使用 2 个 NPU 核心 |
| `optimization_level` | 1 | 优化等级 |

## 注意事项

1. **合并必须在有 GPU 的环境执行**，LoRA merge_and_unload 需要将模型加载到显存
2. **校准数据推荐使用领域相关数据**，可修改 `generate_data_quant.py` 中的 `input_text` 列表，替换为座舱领域的典型对话，以获得更好的量化精度
3. **合并后模型约 3GB (bf16)**，量化后 `.rkllm` 文件约 500MB
4. **PEFT 版本兼容性**: 确认 `peft` 版本与 Llama Factory 训练时使用的版本一致，避免 adapter 加载失败
