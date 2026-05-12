#include "agent_context.h"
#include <ctime>
#include <algorithm>

AgentContext::AgentContext() {
    context_queue_.clear();
}

long AgentContext::GetCurrentTimestamp() {
    return time(nullptr);
}

void AgentContext::AddContext(const std::string& user_text, const std::string& agent_reply) {
    std::lock_guard<std::mutex> lock(mutex_);
    CleanTimeoutContext();
    
    ContextItem item;
    item.user_text = user_text;
    item.agent_reply = agent_reply;
    item.timestamp = GetCurrentTimestamp();
    
    context_queue_.push_back(item);
    
    // 限制上下文长度
    if (context_queue_.size() > MAX_CONTEXT_LEN) {
        context_queue_.pop_front();
    }
}

std::deque<ContextItem> AgentContext::GetRecentContext() {
    std::lock_guard<std::mutex> lock(mutex_);
    CleanTimeoutContext();
    return context_queue_;
}

void AgentContext::CleanTimeoutContext() {
    long now = GetCurrentTimestamp();
    for (auto it = context_queue_.begin(); it != context_queue_.end();) {
        if (now - it->timestamp > CONTEXT_TIMEOUT) {
            it = context_queue_.erase(it);
        } else {
            ++it;
        }
    }
}