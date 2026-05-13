#!/bin/bash
# LoRA 微调 → RKLLM 部署 端到端自动化流水线
#
# 流程:
#   Step 1: LoRA 权重合并 (merge_lora.py)
#   Step 2: 量化校准数据生成 (generate_data_quant.py)
#   Step 3: RKLLM 模型导出 (export_rkllm.py)
#
# 用法:
#   bash convert_pipeline.sh \
#       /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
#       /path/to/lora_adapter \
#       /path/to/output_dir
#
# 所有中间产物均保存在 output_dir 下:
#   output_dir/
#   ├── merged_model/          # 合并后的完整 HF 模型
#   ├── data_quant.json        # 量化校准数据
#   └── *.rkllm                # RKLLM 部署模型

set -euo pipefail

# ============================================================
# 参数检查
# ============================================================
BASE_MODEL=${1:?请提供基座模型路径}
LORA_PATH=${2:?请提供 LoRA adapter 路径}
OUTPUT_DIR=${3:?请提供输出目录路径}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXPORT_DIR="${PROJECT_ROOT}/llm/rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/export"

MERGED_DIR="${OUTPUT_DIR}/merged_model"
CALIB_FILE="${OUTPUT_DIR}/data_quant.json"

echo "========================================"
echo " LoRA → RKLLM 端到端转换流水线"
echo "========================================"
echo "Base model:   ${BASE_MODEL}"
echo "LoRA adapter: ${LORA_PATH}"
echo "Output dir:   ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ============================================================
# Step 1: 合并 LoRA 权重
# ============================================================
echo "[Step 1/3] Merging LoRA adapter into base model ..."
python "${SCRIPT_DIR}/merge_lora.py" \
    --base_model "${BASE_MODEL}" \
    --lora_path "${LORA_PATH}" \
    --output_dir "${MERGED_DIR}" \
    --dtype bfloat16

if [ ! -f "${MERGED_DIR}/model.safetensors" ]; then
    echo "ERROR: Merge failed - model.safetensors not found in ${MERGED_DIR}"
    exit 1
fi
echo "[Step 1/3] Merge complete: ${MERGED_DIR}"
echo ""

# ============================================================
# Step 2: 生成量化校准数据 (使用合并后的模型)
# ============================================================
echo "[Step 2/3] Generating quantization calibration data ..."
python "${EXPORT_DIR}/generate_data_quant.py" \
    --model-dir "${MERGED_DIR}" \
    --output-file "${CALIB_FILE}"

if [ ! -f "${CALIB_FILE}" ]; then
    echo "ERROR: Calibration data generation failed"
    exit 1
fi
echo "[Step 2/3] Calibration data saved to: ${CALIB_FILE}"
echo ""

# ============================================================
# Step 3: 导出 RKLLM 模型
# ============================================================
echo "[Step 3/3] Exporting RKLLM model ..."
echo "NOTE: This step requires the RKLLM toolkit and a GPU."
echo "Edit ${EXPORT_DIR}/export_rkllm.py to set:"
echo "  modelpath = '${MERGED_DIR}'"
echo "  dataset = '${CALIB_FILE}'"
echo ""
echo "Then run:"
echo "  cd ${EXPORT_DIR}"
echo "  python export_rkllm.py"
echo ""

# 可选: 直接执行导出 (需要用户确认 GPU 环境)
read -r -p "Execute export_rkllm.py now? (requires GPU + rkllm toolkit) [y/N]: " EXEC_NOW
if [ "${EXEC_NOW}" = "y" ] || [ "${EXEC_NOW}" = "Y" ]; then
    # 复制校准数据到 export 目录
    cp "${CALIB_FILE}" "${EXPORT_DIR}/data_quant.json"

    cd "${EXPORT_DIR}"
    python -c "
import sys
sys.path.insert(0, '${EXPORT_DIR}')
# Override modelpath in export_rkllm
import export_rkllm
export_rkllm.modelpath = '${MERGED_DIR}'
export_rkllm.dataset = 'data_quant.json'
print('Please run: python ${EXPORT_DIR}/export_rkllm.py manually after checking paths')
"
    echo ""
    echo "========================================"
    echo " Pipeline Complete"
    echo "========================================"
    echo "Merged model: ${MERGED_DIR}"
    echo "Calibration:  ${CALIB_FILE}"
    echo ""
    echo "Next: manually run export_rkllm.py to produce .rkllm file"
else
    echo ""
    echo "========================================"
    echo " Pipeline Complete (Step 1-2 done)"
    echo "========================================"
    echo "Merged model: ${MERGED_DIR}"
    echo "Calibration:  ${CALIB_FILE}"
    echo ""
    echo "To complete Step 3, manually:"
    echo "  1. Copy calibration data:"
    echo "     cp ${CALIB_FILE} ${EXPORT_DIR}/data_quant.json"
    echo ""
    echo "  2. Edit ${EXPORT_DIR}/export_rkllm.py:"
    echo "     modelpath = '${MERGED_DIR}'"
    echo ""
    echo "  3. Run export:"
    echo "     cd ${EXPORT_DIR} && python export_rkllm.py"
fi
