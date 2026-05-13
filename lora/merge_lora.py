#!/usr/bin/env python3
"""LoRA 权重合并脚本。

将从 Llama Factory 等平台训练导出的 LoRA adapter 权重
与基座模型 DeepSeek-R1-Distill-Qwen-1.5B 合并，输出完整的
HuggingFace 格式模型目录，供 RKLLM 工具链进行量化和导出。

合并流程:
  1. 加载基座模型 (AutoModelForCausalLM + AutoTokenizer)
  2. 加载 LoRA adapter (PeftModel.from_pretrained)
  3. 合并权重并卸载 adapter (merge_and_unload)
  4. 保存合并后的完整模型

依赖:
  pip install transformers peft torch accelerate

用法:
  python merge_lora.py \
    --base_model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --lora_path /path/to/lora_adapter \
    --output_dir /path/to/merged_model \
    --push_to_hub False
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model", "-b",
        type=str,
        required=True,
        help="Path to base model (DeepSeek-R1-Distill-Qwen-1.5B)",
    )
    parser.add_argument(
        "--lora_path", "-l",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (contains adapter_model.safetensors + adapter_config.json)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./merged_model",
        help="Output directory for merged model (default: ./merged_model)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for loading and saving (default: bfloat16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for model loading (default: cuda)",
    )
    parser.add_argument(
        "--skip_save_tokenizer",
        action="store_true",
        help="Skip copying tokenizer files from base model",
    )
    return parser.parse_args()


def validate_paths(args):
    if not os.path.isdir(args.base_model):
        logger.error("Base model directory not found: %s", args.base_model)
        sys.exit(1)

    lora_config = os.path.join(args.lora_path, "adapter_config.json")
    if not os.path.isfile(lora_config):
        logger.error(
            "adapter_config.json not found in LoRA path: %s. "
            "Make sure this is a valid PEFT adapter directory.",
            args.lora_path,
        )
        sys.exit(1)

    logger.info("Base model: %s", args.base_model)
    logger.info("LoRA adapter: %s", args.lora_path)
    logger.info("Output directory: %s", args.output_dir)


def get_dtype(dtype_str: str):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def merge(args):
    dtype = get_dtype(args.dtype)

    # 1. Load base model
    logger.info("Loading base model from %s ...", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if args.device == "cuda" else None,
    )

    # 2. Load LoRA adapter
    from peft import PeftModel
    logger.info("Loading LoRA adapter from %s ...", args.lora_path)
    model = PeftModel.from_pretrained(model, args.lora_path)
    logger.info("LoRA adapter loaded successfully")

    # 3. Merge and unload
    logger.info("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()
    logger.info("Merge completed")

    # 4. Save merged model
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Saving merged model to %s ...", args.output_dir)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    logger.info("Model saved")

    # 5. Save tokenizer
    if not args.skip_save_tokenizer:
        logger.info("Copying tokenizer from base model ...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Tokenizer saved")

    logger.info("Merge complete. Merged model saved to: %s", args.output_dir)
    logger.info(
        "Next step: run convert_pipeline.sh or directly use export_rkllm.py "
        "with --model_dir=%s", args.output_dir
    )


def main():
    args = parse_args()
    validate_paths(args)
    merge(args)


if __name__ == "__main__":
    main()
