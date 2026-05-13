#!/usr/bin/env python3
"""座舱领域微调数据集准备脚本。

将车载手册、座舱对话等内容转换为 Llama Factory / HuggingFace
兼容的指令微调数据集格式。输出为 JSON/JSONL 文件，可直接用于
Llama Factory 的 supervised fine-tuning 流程。

Llama Factory 支持的格式 (alpaca/sharegpt):
  - alpaca:  [{"instruction": "...", "input": "...", "output": "..."}, ...]
  - sharegpt: [{"conversations": [{"role": "user", "content": "..."},
                                   {"role": "assistant", "content": "..."}]}, ...]

DeepSeek-R1 模型使用特定的 chat template:
  <｜User｜>用户输入<｜Assistant｜><think>\n推理过程\n</think>\n最终回答

用法:
  python prepare_cockpit_data.py \
    --manual_file /path/to/vehicle_manual.txt \
    --qa_file /path/to/qa_pairs.json \
    --output train_data.json \
    --format alpaca
"""

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 座舱领域 系统提示词模板
# ============================================================
SYSTEM_PROMPTS = [
    "你是一个车载智能助手，专门帮助用户解答车辆使用、保养和故障处理相关问题。",
    "你是智能座舱AI语音助手，为用户提供车辆操作指导、安全驾驶建议和道路信息。",
    "你是一位资深的汽车技术专家，擅长用简洁易懂的语言解释复杂的车辆知识。",
]

# ============================================================
# 1. 从车辆手册生成 QA 对
# ============================================================
TEMPLATE_QUESTIONS = [
    "如何{section}？",
    "{section}应该怎么做？",
    "请介绍一下{section}",
    "{section}是什么意思？",
    "我的车出现了{subsection}的问题，怎么办？",
    "怎样正确使用{section}？",
    "{section}的注意事项有哪些？",
    "什么情况下需要{section}？",
    "请详细说明{subsection}的操作步骤",
    "{section}的常见问题有哪些？",
]


def parse_markdown_manual(file_path: str) -> List[Dict[str, str]]:
    """解析车辆手册 Markdown 文件，提取章节结构。"""
    if not os.path.exists(file_path):
        logger.warning("Manual file not found: %s", file_path)
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = []
    lines = content.split("\n")
    current_section = None
    current_subsection = None
    current_content = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            if current_section and current_content:
                sections.append({
                    "section": current_section,
                    "subsection": current_subsection or current_section,
                    "content": "\n".join(current_content),
                })
            current_section = line[3:]
            current_subsection = None
            current_content = []
        elif line.startswith("### "):
            if current_subsection and current_content:
                sections.append({
                    "section": current_section,
                    "subsection": current_subsection,
                    "content": "\n".join(current_content),
                })
            current_subsection = line[4:]
            current_content = []
        else:
            current_content.append(line)

    if current_section and current_content:
        sections.append({
            "section": current_section,
            "subsection": current_subsection or current_section,
            "content": "\n".join(current_content),
        })

    logger.info("Parsed %d sections from manual", len(sections))
    return sections


def manual_to_qa_pairs(sections: List[Dict]) -> List[Dict]:
    """从章节生成 instruction-回答对。"""
    pairs = []
    for s in sections:
        template = random.choice(TEMPLATE_QUESTIONS)
        question = template.format(
            section=s.get("section", "操作方法"),
            subsection=s.get("subsection", "此功能"),
        )
        # 为 DeepSeek-R1 生成含思考过程的回答
        think = f"用户询问的是关于{s['section']}的问题，我需要从车辆手册知识中给出准确的解答。"
        pairs.append({
            "instruction": question,
            "input": "",
            "output": f"<think>\n{think}\n</think>\n{s['content']}",
            "system": random.choice(SYSTEM_PROMPTS),
        })
    logger.info("Generated %d QA pairs from manual", len(pairs))
    return pairs


# ============================================================
# 2. 座舱对话模板生成
# ============================================================
COCKPIT_SCENARIOS = [
    # (用户输入, R1思考过程, 助手回答)
    (
        "仪表盘上发动机灯亮了，这严重吗？",
        "用户询问发动机故障灯亮的严重程度。需要区分：如果是黄灯常亮可继续短途行驶但需尽快检查；红灯或闪烁必须立即停车。",
        "发动机故障灯亮起需要根据颜色判断：如果是黄色常亮，车辆可以继续短距离行驶，但请尽快到维修站检查。"
        "如果灯是红色或者闪烁，请立即安全停车并联系救援，切勿继续行驶。您方便看一下灯的颜色吗？",
    ),
    (
        "夏天开空调不制冷怎么办？",
        "用户反映空调不制冷。可能原因有：制冷剂不足、空调滤芯堵塞、压缩机故障、冷凝器脏污。按从简到繁排查。",
        "建议按以下顺序排查：1）检查空调滤芯是否超过1年未更换；2）确认A/C开关已打开，温度调至最低；"
        "3）检查出风口风量是否正常——风量小可能是滤芯堵塞，风量正常不冷可能是制冷剂不足。需要到店检查制冷系统。",
    ),
    (
        "这车机油多久换一次比较好？",
        "用户询问保养周期。需要给出常规建议和分类讨论：涡轮增压vs自然吸气、全合成vs半合成机油。",
        "一般建议每5000-10000公里或6个月更换一次机油，以先到为准。使用全合成机油可延长到10000公里。"
        "但涡轮增压车型建议5000-7500公里更换。请参考仪表盘保养提醒或车辆使用手册的保养计划。",
    ),
    (
        "自动启停功能怎么关闭？",
        "用户想关闭自动启停。这是常见的操作指令类问题，需要给出具体的按钮位置和说明。",
        "自动启停关闭按钮通常在中控台或方向盘左侧，标有'A-off'或'@'符号，按下即可关闭。"
        "注意：大部分车型每次启动后自动启停会默认开启，需要重新手动关闭。您可以在中控区找一下这个按钮。",
    ),
    (
        "高速公路开车突然爆胎该怎么办？",
        "紧急安全类问题。必须强调安全第一：握紧方向盘、不猛踩刹车、缓慢减速、靠边停车。",
        "这是紧急情况，请记住以下要点：1）紧握方向盘保持直线行驶，不要猛打方向；"
        "2）不要急刹车，松开油门让车辆自然减速；3）速度降到60km/h以下后轻踩刹车靠边停车；"
        "4）打开双闪，在车后150米放置三角警示牌，人员撤离到护栏外再换备胎或呼叫救援。安全第一！",
    ),
    (
        "你好，今天有什么能帮我的？",
        "用户发起了问候，这是友好的开场，我需要以热情专业的态度回应。",
        "你好！我是你的随车语音助手，关于车辆使用、保养维修、安全驾驶方面的问题，"
        "我都能帮你解答。有什么需要了解的吗？",
    ),
    (
        "导航说偏航了，怎么重新规划路线？",
        "用户遇到导航偏航问题。需要指导用户操作车载导航系统重新规划。",
        "别担心，车载导航会自动检测偏航并在3-5秒内重新计算路线。如果未自动规划，"
        "可以点屏幕上的'重新规划'或'重新计算'按钮，系统会基于当前新位置计算最佳路线。",
    ),
    (
        "晚上开车大灯不够亮是什么原因？",
        "用户反馈大灯亮度不足。可能原因：灯罩老化发黄、灯泡老化衰减、电压不足、大灯高度调节不当。",
        "大灯变暗常见原因：1）灯罩长期暴晒发黄老化——可做灯罩翻新抛光；"
        "2）卤素灯泡使用超2年亮度衰减——建议更换LED或新卤素灯泡；"
        "3）大灯高度调节旋钮位置过低——在方向盘左下方调高。先检查灯罩和灯泡，多数情况是灯泡老化。",
    ),
    (
        "下了高速之后刹车有焦味正常吗？",
        "用户询问刹车异味。如果是长下坡或频繁制动后出现短期焦味，一般是刹车片和刹车盘摩擦过热。但如果持续存在需要重视。",
        "如果是长下坡或频繁制动后短暂出现焦味，属于正常现象，刹车片与刹车盘摩擦过热导致。"
        "建议在下坡时使用低档位发动机制动，减少刹车负担。但如果焦味持续不退，或者刹车时出现抖动、异响，"
        "可能是刹车片磨损到极限或刹车卡钳卡滞，需要尽快检查。",
    ),
    (
        "车子停了一段时间电瓶没电了怎么办？",
        "用户遇到电瓶亏电。需要给出搭电启动的步骤和安全注意事项。",
        "电瓶亏电可以这样处理：1）如果有搭电线：找另一辆车，先接正极再接负极，启动后先拆负极再拆正极；"
        "2）如果有应急启动电源：直接夹上电瓶正负极即可启动；3）如果是手动挡：可以尝试推车启动。"
        "启动后至少行驶30分钟给电瓶充电。如果经常亏电，建议检查电瓶健康度或是否存在漏电。",
    ),
]


def generate_dialogue_samples() -> List[Dict]:
    """生成座舱对话训练样本。"""
    samples = []
    for query, think, answer in COCKPIT_SCENARIOS:
        system = random.choice(SYSTEM_PROMPTS)
        samples.append({
            "instruction": query,
            "input": "",
            "output": f"<think>\n{think}\n</think>\n{answer}",
            "system": system,
        })
    logger.info("Generated %d dialogue samples", len(samples))
    return samples


# ============================================================
# 3. RAG 增强型 QA 生成 (标注知识来源)
# ============================================================
def generate_rag_enhanced_qa(sections: List[Dict]) -> List[Dict]:
    """生成标注了知识来源的 QA 对，用于训练模型"基于证据回答"的能力。"""
    pairs = []
    for s in sections[:20]:  # 取前 20 个章节生成 RAG 增强样本
        think = (
            f"用户询问{s['section']}相关问题。我需要依据车辆手册中关于"
            f"{s['subsection'] or s['section']}的信息进行回答。如果手册中有明确说明就引用，"
            f"如果没有则如实告知用户。"
        )
        answer = f"根据车辆使用手册，{s['content'][:500]}"
        pairs.append({
            "instruction": f"查一下{s['subsection'] or s['section']}怎么操作",
            "input": "",
            "output": f"<think>\n{think}\n</think>\n{answer}",
            "system": random.choice(SYSTEM_PROMPTS),
        })
    logger.info("Generated %d RAG-enhanced QA pairs", len(pairs))
    return pairs


# ============================================================
# 4. 导出与格式转换
# ============================================================
def convert_to_sharegpt(samples: List[Dict]) -> List[Dict]:
    """将 alpaca 格式转换为 ShareGPT 格式。"""
    sharegpt_data = []
    for s in samples:
        conversations = []
        if s.get("system"):
            conversations.append({"role": "system", "content": s["system"]})
        user_content = s["instruction"]
        if s.get("input"):
            user_content += f"\n{s['input']}"
        conversations.append({"role": "user", "content": user_content})
        conversations.append({"role": "assistant", "content": s["output"]})
        sharegpt_data.append({"conversations": conversations})
    return sharegpt_data


def save_dataset(samples: List[Dict], output_path: str, fmt: str = "alpaca"):
    """保存数据集到文件。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "sharegpt":
        samples = convert_to_sharegpt(samples)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info("Dataset saved to %s (%d samples, format=%s)", output_path, len(samples), fmt)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cockpit domain fine-tuning dataset"
    )
    parser.add_argument(
        "--manual_file",
        type=str,
        default=None,
        help="Path to vehicle manual markdown file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="cockpit_train_data.json",
        help="Output path for the training dataset (default: cockpit_train_data.json)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt"],
        help="Output format: alpaca or sharegpt (default: alpaca)",
    )
    parser.add_argument(
        "--include_rag_enhanced",
        action="store_true",
        help="Include RAG-enhanced QA pairs",
    )
    parser.add_argument(
        "--augment_count",
        type=int,
        default=0,
        help="Number of augmented samples to generate via template variation (0 = skip)",
    )
    args = parser.parse_args()

    all_samples = []

    # 1. 座舱对话模板 (始终包含)
    dialogue_samples = generate_dialogue_samples()
    all_samples.extend(dialogue_samples)

    # 2. 车辆手册 → QA 对
    if args.manual_file and os.path.exists(args.manual_file):
        sections = parse_markdown_manual(args.manual_file)
        qa_pairs = manual_to_qa_pairs(sections)
        all_samples.extend(qa_pairs)

        if args.include_rag_enhanced:
            rag_qa = generate_rag_enhanced_qa(sections)
            all_samples.extend(rag_qa)
    else:
        logger.info(
            "No manual file provided or not found. "
            "Skipping manual-based QA generation. "
            "Use --manual_file /path/to/vehicle_manual.txt to include."
        )

    # 3. 数据增强
    if args.augment_count > 0:
        logger.info("Augmenting with %d template variations ...", args.augment_count)
        augmented = []
        for _ in range(args.augment_count):
            base = random.choice(all_samples)
            augmented.append(dict(base))
        all_samples.extend(augmented)

    # 4. 打乱并保存
    random.shuffle(all_samples)
    logger.info("Total training samples: %d", len(all_samples))

    save_dataset(all_samples, args.output, args.format)

    logger.info(
        "\n============================================\n"
        "Dataset ready for Llama Factory training.\n"
        "Usage in Llama Factory:\n"
        "  --dataset %s\n"
        "  --template deepseek\n"
        "  --finetuning_type lora\n"
        "============================================",
        os.path.abspath(args.output),
    )


if __name__ == "__main__":
    main()
