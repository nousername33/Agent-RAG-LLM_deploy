#include "agent_intent.h"
#include <algorithm>
#include <cctype>

AgentIntent::AgentIntent() {
    // 初始化关键词（可从配置文件加载，此处硬编码为最小入侵）
    intent_keywords_[INTENT_GREETING] = {"你好", "嗨", "哈喽", "早上好", "晚上好"};
    intent_keywords_[INTENT_COMMAND] = {"打开空调", "关闭空调", "播放音乐", "暂停音乐", "导航回家"};
    intent_keywords_[INTENT_KNOWLEDGE] = {"怎么", "如何", "升级", "故障", "设置", "教程"};
}

std::string AgentIntent::PreprocessText(const std::string& text) {
    std::string res;
    for (char c : text) {
        if (!isspace(c)) {
            res += tolower(c); // 小写化（适配中文无影响）
        }
    }
    return res;
}

IntentType AgentIntent::RecognizeIntent(const std::string& asr_text) {
    std::string text = PreprocessText(asr_text);
    
    // 匹配问候意图
    for (const std::string& keyword : intent_keywords_[INTENT_GREETING]) {
        if (text.find(keyword) != std::string::npos) {
            return INTENT_GREETING;
        }
    }
    
    // 匹配简单指令
    for (const std::string& keyword : intent_keywords_[INTENT_COMMAND]) {
        if (text.find(keyword) != std::string::npos) {
            return INTENT_COMMAND;
        }
    }
    
    // 匹配知识库查询
    for (const std::string& keyword : intent_keywords_[INTENT_KNOWLEDGE]) {
        if (text.find(keyword) != std::string::npos) {
            return INTENT_KNOWLEDGE;
        }
    }
    
    return INTENT_UNKNOWN;
}