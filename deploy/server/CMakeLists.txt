cmake_minimum_required(VERSION 3.14)
project(vlx_seek_server LANGUAGES C CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 部署服务复用 cpp_ggml 的 C API（vlx_seek_capi.h）+ 唯一 ggml 源（llama.cpp）。
# 上游代码零改动：这里只把 cpp_ggml 作为子目录引入并链接 vlxvision/llama。
set(VLX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "仓库根目录")
set(LLAMA_DIR "${VLX_ROOT}/cpp_ggml/third_party/llama.cpp")

find_package(Threads REQUIRED)

add_subdirectory("${VLX_ROOT}/cpp_ggml" cpp_ggml-build EXCLUDE_FROM_ALL)

# cpp-httplib 在 llama.cpp 里只在 LLAMA_BUILD_COMMON 时构建（本构建没开），
# 因此直接把 vendor 里的单文件实现编进服务目标，避免引入外部依赖。
set(HTTPLIB_DIR "${LLAMA_DIR}/vendor/cpp-httplib")
set(NLOHMANN_DIR "${LLAMA_DIR}/vendor")

add_executable(vlx-seek-server
    vlx_seek_server.cpp
    "${HTTPLIB_DIR}/httplib.cpp"
)

target_include_directories(vlx-seek-server PRIVATE
    "${VLX_ROOT}/cpp_ggml/include"
    "${HTTPLIB_DIR}"
    "${NLOHMANN_DIR}"
)

# 图像 URL 支持 https：vendor 的 cpp-httplib 只有在定义
# CPPHTTPLIB_OPENSSL_SUPPORT 时才启用 TLS。
find_package(OpenSSL REQUIRED)
target_compile_definitions(vlx-seek-server PRIVATE
    CPPHTTPLIB_OPENSSL_SUPPORT=1
    CPPHTTPLIB_TCP_NODELAY=1
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    CPPHTTPLIB_THREAD_POOL_COUNT=16
)

target_link_libraries(vlx-seek-server PRIVATE vlxvision llama Threads::Threads
    OpenSSL::SSL OpenSSL::Crypto)

# vlxvision 里含融合 CUDA RoIAlign 目标文件，静态库不会传递 cudart 依赖
if(VLX_GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(vlx-seek-server PRIVATE CUDA::cudart)
endif()
