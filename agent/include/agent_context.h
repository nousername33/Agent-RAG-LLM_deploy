#ifndef AGENT_CONTEXT_H
#define AGENT_CONTEXT_H

#include <string>
#include <deque>
#include <mutex>

// 上下文配置（适配板端资源）
#define MAX_CONTEXT_LEN 5  // 最多保存5轮上下文
#define CONTEXT_TIMEOUT 30 // 上下文超时时间（秒）

// 上下文条目
struct ContextItem {
    std::string user_text;  // 用户输入
    std::string agent_reply;// Agent回复
    long timestamp;         // 时间戳（秒）
};

// 上下文管理类（线程安全）
class AgentContext {
public:
    AgentContext();
    // 添加上下文
    void AddContext(const std::string& user_text, const std::string& agent_reply);
    // 获取最近上下文
    std::deque<ContextItem> GetRecentContext();
    // 清理超时上下文
    void CleanTimeoutContext();

private:
    std::deque<ContextItem> context_queue_;
    std::mutex mutex_;
    // 获取当前时间戳（秒）
    long GetCurrentTimestamp();
};

#endif // AGENT_CONTEXT_H