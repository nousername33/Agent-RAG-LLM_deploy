#include "agent_zmq_comm.h"
#include <iostream>

AgentZmqComm::AgentZmqComm() 
    : context_(1),
      asr_socket_(context_, ZMQ_PULL),
      rag_socket_(context_, ZMQ_PUSH),
      llm_socket_(context_, ZMQ_PUSH),
      tts_socket_(context_, ZMQ_PUSH),
      result_socket_(context_, ZMQ_PULL) {
    
    // 绑定/连接端口（非阻塞，适配板端）
    asr_socket_.bind("tcp://*:" ZMQ_ASR_AGENT_PORT);
    rag_socket_.connect("tcp://127.0.0.1:" ZMQ_AGENT_RAG_PORT);
    llm_socket_.connect("tcp://127.0.0.1:" ZMQ_AGENT_LLM_PORT);
    tts_socket_.connect("tcp://127.0.0.1:" ZMQ_AGENT_TTS_PORT);
    result_socket_.bind("tcp://*:5559"); // 接收RAG/LLM结果
    
    // 设置超时（避免阻塞）
    int timeout = 1000; // 1秒
    asr_socket_.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    result_socket_.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
}

AgentZmqComm::~AgentZmqComm() {
    asr_socket_.close();
    rag_socket_.close();
    llm_socket_.close();
    tts_socket_.close();
    result_socket_.close();
    context_.close();
}

std::string AgentZmqComm::RecvFromASR() {
    zmq::message_t msg;
    try {
        if (asr_socket_.recv(&msg, zmq::recv_flags::none)) {
            return std::string(static_cast<char*>(msg.data()), msg.size());
        }
    } catch (...) {
        return "";
    }
    return "";
}

void AgentZmqComm::SendToRAG(const std::string& data) {
    zmq::message_t msg(data.size());
    memcpy(msg.data(), data.c_str(), data.size());
    rag_socket_.send(msg, zmq::send_flags::none);
}

void AgentZmqComm::SendToLLM(const std::string& data) {
    zmq::message_t msg(data.size());
    memcpy(msg.data(), data.c_str(), data.size());
    llm_socket_.send(msg, zmq::send_flags::none);
}

void AgentZmqComm::SendToTTS(const std::string& text) {
    zmq::message_t msg(text.size());
    memcpy(msg.data(), text.c_str(), text.size());
    tts_socket_.send(msg, zmq::send_flags::none);
}

std::string AgentZmqComm::RecvFromRAGLLM() {
    zmq::message_t msg;
    try {
        if (result_socket_.recv(&msg, zmq::recv_flags::none)) {
            return std::string(static_cast<char*>(msg.data()), msg.size());
        }
    } catch (...) {
        return "";
    }
    return "";
}