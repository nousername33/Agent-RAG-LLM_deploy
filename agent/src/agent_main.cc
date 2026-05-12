// #include "agent_intent.h"
// #include "agent_context.h"
// #include "agent_zmq_comm.h"
// #include <iostream>
// #include <string>

// int main() {
//     // 初始化模块（轻量化，板端启动耗时＜100ms）
//     AgentIntent intent_recognizer;
//     AgentContext context_manager;
//     AgentZmqComm zmq_comm;
    
//     std::cout << "Edge Agent started on RK3576 (offline mode)" << std::endl;
    
//     // 主循环（非阻塞，适配板端）
//     while (true) {
//         // 1. 接收ASR文本
//         std::string asr_text = zmq_comm.RecvFromASR();
//         if (asr_text.empty()) {
//             usleep(10000); // 10ms休眠，降低CPU占用
//             continue;
//         }
        
//         std::cout << "Received ASR text: " << asr_text << std::endl;
        
//         // 2. 识别意图
//         IntentType intent = intent_recognizer.RecognizeIntent(asr_text);
//         std::string agent_reply;
        
//         // 3. 分支逻辑处理
//         switch (intent) {
//             case INTENT_GREETING:
//                 agent_reply = "你好呀！有什么我能帮助你的吗？";
//                 break;
                
//             case INTENT_COMMAND:
//                 agent_reply = "已收到你的指令，正在为你执行！";
//                 break;
                
//             case INTENT_KNOWLEDGE:
//                 // 调用RAG+LLM（复用原有模块）
//                 zmq_comm.SendToRAG(asr_text);
//                 zmq_comm.SendToLLM(asr_text);
//                 // 接收RAG/LLM结果
//                 agent_reply = zmq_comm.RecvFromRAGLLM();
//                 if (agent_reply.empty()) {
//                     agent_reply = "抱歉，我暂时无法回答这个问题。";
//                 }
//                 break;
                
//             case INTENT_UNKNOWN:
//                 agent_reply = "我没太理解你的意思，可以再说一遍吗？";
//                 break;
                
//             default:
//                 agent_reply = "抱歉，我有点卡顿了，请重试！";
//                 break;
//         }
        
//         // 4. 保存上下文
//         context_manager.AddContext(asr_text, agent_reply);
        
//         // 5. 发送到TTS
//         zmq_comm.SendToTTS(agent_reply);
//         std::cout << "Agent reply: " << agent_reply << std::endl;
//     }
    
//     return 0;
// }


#include <iostream>
#include <string>
#include <zmq.hpp>

int main() {
    // 0MQ 上下文
    zmq::context_t ctx(1);

    // 1. 接收 ASR 文本（ASR -> Agent）
    zmq::socket_t pull_asr(ctx, ZMQ_PULL);
    pull_asr.bind("tcp://*:5555");

    // 2. 转发文本到 LLM（Agent -> LLM）
    zmq::socket_t push_llm(ctx, ZMQ_PUSH);
    push_llm.connect("tcp://127.0.0.1:8899");

    // 3. 接收 LLM 回复（LLM -> Agent）
    zmq::socket_t pull_llm(ctx, ZMQ_PULL);
    pull_llm.bind("tcp://*:5559");

    std::cout << "=====================================" << std::endl;
    std::cout << "✅ Agent 智能中间层 启动成功（本地PC测试版）" << std::endl;
    std::cout << "=====================================" << std::endl;

    while (true) {
        // -------- 1. 接收 ASR 结果 --------
        zmq::message_t asr_msg;
        pull_asr.recv(asr_msg);
        std::string user_text = std::string((char*)asr_msg.data(), asr_msg.size());
        std::cout << "\n[Agent] 🎤 用户语音转文字：" << user_text << std::endl;

        // -------- 2. 转发给 LLM --------
        zmq::message_t llm_input(user_text.size());
        memcpy(llm_input.data(), user_text.data(), user_text.size());
        push_llm.send(llm_input);
        std::cout << "[Agent] 🔄 已转发给 LLM 大模型" << std::endl;

        // -------- 3. 接收 LLM 输出 --------
        zmq::message_t llm_reply;
        pull_llm.recv(llm_reply);
        std::string llm_output = std::string((char*)llm_reply.data(), llm_reply.size());
        std::cout << "[Agent] 🤖 LLM 回复结果：" << llm_output << std::endl;
    }

    return 0;
}