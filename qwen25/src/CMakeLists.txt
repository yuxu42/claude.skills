# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)
project(run_qwen25 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the composable_pipeline project
set(CP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../openvino.pipeline.mx")
set(CP_SRC "${CP_ROOT}/src/cpp/src")

# Find OpenVINO
if(DEFINED OpenVINO_DIR)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
else()
    # Try system install or build location
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

# Find nlohmann_json (header-only)
set(JSON_CANDIDATES
    "${CP_ROOT}/thirdparty/openvino/thirdparty/json/nlohmann_json/include"
    "${CP_ROOT}/thirdparty/openvino/thirdparty/json/nlohmann_json/single_include"
    "${CP_ROOT}/thirdparty/nlohmann_json/include"
)
set(JSON_INCLUDE "")
foreach(candidate ${JSON_CANDIDATES})
    if(EXISTS "${candidate}/nlohmann/json.hpp")
        set(JSON_INCLUDE "${candidate}")
        break()
    endif()
endforeach()
if(JSON_INCLUDE STREQUAL "")
    message(FATAL_ERROR "nlohmann_json not found in any candidate path")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE "${JSON_INCLUDE}")

# Composable pipeline include paths
set(CP_INCLUDE_DIRS
    "${CP_SRC}"
    "${CP_SRC}/modeling"
    "${CP_SRC}/utils"
    "${CP_ROOT}/src/cpp/src"
)

# Link against libcomposable_pipeline.so directly
set(CP_LIB "${CP_ROOT}/build/cp310-cp310-linux_x86_64/src/cpp/libcomposable_pipeline.so")
if(NOT EXISTS "${CP_LIB}")
    message(FATAL_ERROR "libcomposable_pipeline.so not found at ${CP_LIB}. Build composable_pipeline first.")
endif()

add_library(composable_pipeline_imported SHARED IMPORTED)
set_target_properties(composable_pipeline_imported PROPERTIES
    IMPORTED_LOCATION "${CP_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${CP_SRC}"
)

# Build the Qwen2.5 modeling library
add_library(modeling_qwen25 STATIC
    modeling_qwen25.cpp
)
target_include_directories(modeling_qwen25 PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${CP_INCLUDE_DIRS}
)
target_link_libraries(modeling_qwen25 PUBLIC
    composable_pipeline_imported
    openvino::runtime
)

# Build the runner executable
add_executable(run_qwen25 run_qwen25.cpp)
target_include_directories(run_qwen25 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${CP_INCLUDE_DIRS}
)
target_link_libraries(run_qwen25 PRIVATE
    modeling_qwen25
    composable_pipeline_imported
    openvino::runtime
)
target_link_libraries(run_qwen25 PRIVATE nlohmann_json)
