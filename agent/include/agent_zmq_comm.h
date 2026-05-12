
#ifndef AGENT_ZMQ_COMM_H
#define AGENT_ZMQ_COMM_H

#include <zmq.hpp>
#include <string>

// ZeroMQ通信配置（复用原有项目的端口）
#define ZMQ_ASR_AGENT_PORT "5555"    // ASR→Agent
#define ZMQ_AGENT_RAG_PORT "5556"    // Agent→RAG
#define ZMQ_AGENT_TTS_PORT "5557"    // Agent→TTS
#define ZMQ_AGENT_LLM_PORT "5558"    // Agent→LLM

// Agent通信类（复用原有zmq-comm-kit）
class AgentZmqComm {
public:
    AgentZmqComm();
    ~AgentZmqComm();
    // 接收ASR文本
    std::string RecvFromASR();
    // 发送数据到RAG
    void SendToRAG(const std::string& data);
    // 发送数据到LLM
    void SendToLLM(const std::string& data);
    // 发送文本到TTS
    void SendToTTS(const std::string& text);
    // 接收RAG/LLM返回结果
    std::string RecvFromRAGLLM();

private:
    zmq::context_t context_;
    zmq::socket_t asr_socket_;    // 接收ASR
    zmq::socket_t rag_socket_;    // 发送到RAG
    zmq::socket_t llm_socket_;    // 发送到LLM
    zmq::socket_t tts_socket_;    // 发送到TTS
    zmq::socket_t result_socket_; // 接收RAG/LLM结果
};

#endif // AGENT_ZMQ_COMM_H