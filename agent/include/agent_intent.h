#ifndef AGENT_INTENT_H
#define AGENT_INTENT_H

#include <string>
#include <unordered_map>
#include <vector>

// 意图类型定义
enum IntentType {
    INTENT_GREETING,    // 问候（如：你好/嗨）
    INTENT_COMMAND,     // 简单指令（如：打开空调/播放音乐）
    INTENT_KNOWLEDGE,   // 知识库查询（如：车机系统怎么升级）
    INTENT_UNKNOWN      // 未知意图
};

// 意图识别类（轻量化，无模型依赖）
class AgentIntent {
public:
    AgentIntent();
    // 识别意图（输入ASR文本，输出意图类型）
    IntentType RecognizeIntent(const std::string& asr_text);

private:
    // 关键词匹配表（可配置，板端可修改）
    std::unordered_map<IntentType, std::vector<std::string>> intent_keywords_;
    // 文本预处理（去空格/小写）
    std::string PreprocessText(const std::string& text);
};

#endif // AGENT_INTENT_H