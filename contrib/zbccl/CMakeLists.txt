# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set(ZBCCL_BASE_DIR ${PROJECT_SOURCE_DIR}/contrib/zbccl/)
set(ZBCCL_SRC_DIR ${PROJECT_SOURCE_DIR}/contrib/zbccl/src)

# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ZBCCL_BASE_DIR}/python/zbccl/lib)

# host side files
FILE(GLOB OP_SRCS
    ${ZBCCL_SRC_DIR}/csrc/ccl/zbccl_op_allgather.cpp
)

# set the so name
set(OP_PLUGIN_NAME zbccl)

ascendc_library(zbccl_kernel STATIC
    ${ZBCCL_SRC_DIR}/csrc/ccl/device/zbccl_kernel_allgather.cpp
)

ascendc_include_directories(zbccl_kernel PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/aclnn
    ${ASCEND_INCLUDE_DIR}/aclnn/opdev
    ${ZBCCL_SRC_DIR}/include
)

# create shared library libzccl.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        zbccl_kernel
        torch_npu
        ascendcl
        tiling_api
        register
        platform
        ascendalog
        dl
        shmem
        mf_smem
        mf_hybm_core
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
        ${SHMEM_HOME_PATH}/shmem/lib
        ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${ZBCCL_SRC_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${ASCEND_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
)