cmake_minimum_required(VERSION 3.10)
project(edge_agent)

# 编译选项（适配RK3576）
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=armv8-a -mtune=cortex-a55")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread -lzmq")

# 头文件路径（复用原有zmq-comm-kit）
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/zmq-comm-kit/include
)

# 源文件
file(GLOB AGENT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
)

# 生成可执行文件（板端运行）
# add_executable(edge_agent ${AGENT_SRC})
add_executable(agent src/agent_main.cpp)
target_link_libraries(agent zmq pthread)
# 链接ZeroMQ（复用原有库）
# target_link_libraries(edge_agent zmq)

# 安装到板端目录
# install(TARGETS edge_agent DESTINATION /usr/bin/edge_agent)