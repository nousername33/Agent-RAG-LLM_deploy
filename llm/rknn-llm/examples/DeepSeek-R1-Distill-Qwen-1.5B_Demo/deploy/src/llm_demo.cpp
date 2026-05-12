// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <set>
#include "ZmqServer.h"
#include "ZmqClient.h"
#include <cwchar>
#include <locale>
#include <clocale>
#include <cstdlib>
#include <codecvt>
#include "../../../../../../zmq-comm-kit/include/zmq.hpp"

using namespace std;
LLMHandle llmHandle = nullptr;

zmq_component::ZmqServer server("tcp://*:8899");
zmq_component::ZmqClient client("tcp://localhost:7777");

std::wstring buffer_;
static const std::set<wchar_t> split_chars = {
    L'：',
    L'，',
    L'。',
    L'\n',
    L'；',
    L'！',
    L'？'};

bool is_valid_utf8_continuation(uint8_t c)
{
    return (c & 0xC0) == 0x80;
}

std::wstring extract_after_think(const std::wstring &input)
{
    // 定义要过滤的字符（宽字符版本）
    const std::wstring punct = L" \t\n\r*#@$%^&，。：、；！？【】（）“”‘’";

    // 过滤处理
    std::wstring filtered;
    for (wchar_t c : input)
    {
        if (punct.find(c) == std::wstring::npos)
        {
            filtered += c;
        }
    }

    return filtered;
}

std::wstring utf8_to_wstring(const std::string &str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}

std::string wstring_to_utf8(const std::wstring &str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(str);
}
void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        {
            cout << "程序即将退出" << endl;
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
    }
    exit(signal);
}

void send_response(const std::wstring &text)
{
    std::string response_str = wstring_to_utf8(extract_after_think(text));

    // [Agent接管TTS调度] LLM不再直连TTS，由Agent统一决定是否发送到TTS
    // auto response = client.request(response_str);
    // std::cout << "[tts -> llm] received : " << response << std::endl;

    // ===================== 发送结果给 Agent =====================
    zmq::context_t ctx(1);
    zmq::socket_t sock(ctx, ZMQ_PUSH);
    sock.connect("tcp://127.0.0.1:5559");
    zmq::message_t msg(response_str.c_str(), response_str.size());
    sock.send(msg, zmq::send_flags::none);
    // ============================================================

}

void callback(RKLLMResult *result, void *userdata, LLMCallState state)
{

    if (state == RKLLM_RUN_FINISH)
    {
        if (!buffer_.empty())
        {
            std::string response_str = wstring_to_utf8(extract_after_think(buffer_)) + " END";
            // [Agent接管TTS调度] LLM不再直连TTS
            // auto response = client.request(response_str);
            // std::cout << "[tts -> llm] received: " << response << std::endl;
            // ===================== 发送结果给 Agent =====================
            zmq::context_t ctx(1);
            zmq::socket_t sock(ctx, ZMQ_PUSH);
            sock.connect("tcp://127.0.0.1:5559");
            zmq::message_t msg(response_str.c_str(), response_str.size());
            sock.send(msg, zmq::send_flags::none);
            // ============================================================
            buffer_.clear();
        }
        else
        {
            // [Agent接管TTS调度] LLM不再直连TTS
            // auto response = client.request("END");
            // std::cout << "[tts -> llm] received: " << response << std::endl;
            // ===================== 发送结果给 Agent =====================
            zmq::context_t ctx(1);
            zmq::socket_t sock(ctx, ZMQ_PUSH);
            sock.connect("tcp://127.0.0.1:5559");
            zmq::message_t msg("END", 3);
            sock.send(msg, zmq::send_flags::none);
            // ============================================================
        }

        printf("\n");
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\\run error\n");
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        /* ================================================================================================================
        若使用GET_LAST_HIDDEN_LAYER功能,callback接口会回传内存指针:last_hidden_layer,token数量:num_tokens与隐藏层大小:embd_size
        通过这三个参数可以取得last_hidden_layer中的数据
        注:需要在当前callback中获取,若未及时获取,下一次callback会将该指针释放
        ===============================================================================================================*/
        if (result->last_hidden_layer.embd_size != 0 && result->last_hidden_layer.num_tokens != 0)
        {
            int data_size = result->last_hidden_layer.embd_size * result->last_hidden_layer.num_tokens * sizeof(float);
            printf("\ndata_size:%d", data_size);
            std::ofstream outFile("last_hidden_layer.bin", std::ios::binary);
            if (outFile.is_open())
            {
                outFile.write(reinterpret_cast<const char *>(result->last_hidden_layer.hidden_states), data_size);
                outFile.close();
                std::cout << "Data saved to output.bin successfully!" << std::endl;
            }
            else
            {
                std::cerr << "Failed to open the file for writing!" << std::endl;
            }
        }

        printf("%s", result->text);

        // std::wstring wide_text = utf8_to_wstring(result->text);
        // buffer_ += wide_text;

        // if (split_chars.count(wide_text))
        // {
        //     std::cout << "split_chars"<< wstring_to_utf8(wide_text)<<std::endl;
        //     if (!buffer_.empty())
        //     {
        //         auto response = client.request(wstring_to_utf8(buffer_));
        //         std::cout << "Client received: " << response << std::endl;

        //         buffer_.clear();
        //     }
        // }
        std::wstring wide_text = utf8_to_wstring(result->text);

        for (wchar_t c : wide_text)
        {
            buffer_ += c;

            if (split_chars.count(c))
            {
                if (!buffer_.empty())
                {
                    send_response(buffer_);
                    buffer_.clear();
                }
            }
        }
    }
}

void Init(const string &model_path)
{

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model_path.c_str();

    param.top_k = 1;
    param.top_p = 0.95;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;

    param.max_new_tokens = 100;
    param.max_context_len = 256;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;
    param.extend_param.embed_flash = 1;
    param.extend_param.enabled_cpus_num = 2;
    param.extend_param.enabled_cpus_mask = CPU0 | CPU2;

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0)
    {
        printf("rkllm init success\n");
    }
    else
    {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }
}

std::string build_prompt_with_rag(const std::string &rag_context)
{
    const std::string PLACEHOLDER = "{rag_context}";
    std::string prompt =
        "你是一款智能座舱AI助手：\n"
        "1. 使用口语化表达\n"
        "# 知识检索规范\n"
        "回答必须基于：\n"
        "{rag_context}"; // 预留占位符

    size_t pos = prompt.find(PLACEHOLDER);
    if (pos != std::string::npos)
    {
        prompt.replace(pos, PLACEHOLDER.length(), rag_context);
    }
    return prompt;
}

std::pair<std::string, std::string> split_rag_tag(const std::string &llm_query)
{
    const std::string TAG = "<rag>";
    size_t tag_pos = llm_query.find(TAG);

    if (tag_pos == std::string::npos)
    {
        return {llm_query, ""};
    }

    return {
        llm_query.substr(0, tag_pos),
        llm_query.substr(tag_pos + TAG.length())};
}
void receive_asr_data_and_process()
{
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;

    rkllm_infer_params.keep_history = 0;

    RKLLMInput rkllm_input;

    while (true)
    {
        std::string input_str;

        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        input_str = server.receive();
        std::cout << "[voice -> llm] received: " << input_str << std::endl;
        server.send("llm sucess reply !!!");

        auto [user_query, rag_context] = split_rag_tag(input_str);
        rkllm_input.prompt_input = (char *)user_query.c_str();
        if (!rag_context.empty()){
            std::string system_prompt = build_prompt_with_rag(rag_context);

            rkllm_set_chat_template(llmHandle, system_prompt.c_str(), "<｜User｜>", "<｜Assistant｜><think>\n</think>");
        }else{
            rkllm_set_chat_template(llmHandle, "", "<｜User｜>", "<｜Assistant｜><think>\n</think>");
        }
        

        // 若要使用普通推理功能,则配置rkllm_infer_mode为RKLLM_INFER_GENERATE或不配置参数
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    }
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "en_US.UTF-8");

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " model_path\n";
        return 1;
    }

    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");

    Init(argv[1]);

    receive_asr_data_and_process();

    rkllm_destroy(llmHandle);

    return 0;
}
