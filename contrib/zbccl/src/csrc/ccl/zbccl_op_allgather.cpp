/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * ZBCCL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ZBCCL_OP_ALLGATHER_H
#define ZBCCL_OP_ALLGATHER_H

#include <memory>
#include "acl/acl.h"
#include "shmem_api.h"
#include "aclrtlaunch_allgather.h"
#include "zbccl.h"
#include "zbccl_op.h"
#include "../common/zbccl_functions.h"
#include "../common/zbccl_defines.h"

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t BIG_DATA_SIZE = 40 * 1024 * 1024;

namespace zbccl {

std::shared_ptr<AllGatherTilingData> get_tiling(int32_t block_dim, uint64_t elements, int team_id)
{
    int64_t pe_size = shmem_team_n_pes(team_id);
    auto tiling_data = std::make_shared<AllGatherTilingData>();
    tiling_data->input_num_per_core = elements / block_dim;
    tiling_data->input_last_num_core = elements - (block_dim - 1) * tiling_data->input_num_per_core;
    const uint32_t core_per_rank = block_dim / pe_size;
    tiling_data->output_core_per_rank = core_per_rank;
    tiling_data->output_num_per_core = elements / core_per_rank;
    tiling_data->output_last_num_core = elements - (core_per_rank - 1) * tiling_data->output_num_per_core;
    return tiling_data;
}

int32_t zbccl_all_gather(void *send_buff, void *recv_buff, size_t send_count, zbccl_datatype_t data_type,
                         size_t team_id, aclrtStream stream)
{
    int32_t block_dim = 0;
    if (send_count * getSizeFromTypeEnum(data_type) < BIG_DATA_SIZE) {
        block_dim = 8;
    }else {
        block_dim = 16;
    }
    int magic = 1024;

    void *tiling_device_ptr;
    aclrtMalloc(&tiling_device_ptr, sizeof(AllGatherTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    std::shared_ptr<AllGatherTilingData> tiling_host;
    aclrtMallocHost(reinterpret_cast<void**>(tiling_host.get()), sizeof(AllGatherTilingData));
    tiling_host = get_tiling(block_dim, send_count, team_id);

    aclrtMemcpy(tiling_device_ptr, sizeof(AllGatherTilingData), tiling_host.get(), sizeof(AllGatherTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    uint64_t ffts_addr = shmemx_get_ffts_config();
    size_t gva_size = block_dim * SYNC_FLAG_INTERVAL * sizeof(int) + GVA_BUFF_MAX_SIZE;
    void *gva = shmem_malloc(gva_size);
    aclrtMemset(gva, gva_size, 0, gva_size);
    int data_type_int = static_cast<int>(data_type);

    ACLRT_LAUNCH_KERNEL(allgather)(block_dim, stream, send_buff, recv_buff, gva, send_count, data_type_int, team_id, ffts_addr, magic, tiling_device_ptr);
    aclrtFreeHost(tiling_host.get());
    aclrtFree(tiling_device_ptr);
    shmem_free(gva);
    return 0;
}

int32_t zbccl_all_gather_zero_buffer(void *send_buff, void *recv_buff, size_t send_count, zbccl_datatype_t data_type,
                         size_t team_id, aclrtStream stream)
{
    int32_t block_dim = 0;
    if (send_count * getSizeFromTypeEnum(data_type) < BIG_DATA_SIZE) {
        block_dim = 8;
    }else {
        block_dim = 16;
    }
    int magic = 1024;

    void *tiling_device_ptr;
    aclrtMalloc(&tiling_device_ptr, sizeof(AllGatherTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    std::shared_ptr<AllGatherTilingData> tiling_host;
    aclrtMallocHost(reinterpret_cast<void**>(tiling_host.get()), sizeof(AllGatherTilingData));
    tiling_host = get_tiling(block_dim, send_count, team_id);

    aclrtMemcpy(tiling_device_ptr, sizeof(AllGatherTilingData), tiling_host.get(), sizeof(AllGatherTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    uint64_t ffts_addr = shmemx_get_ffts_config();
    size_t gva_size = block_dim * SYNC_FLAG_INTERVAL * sizeof(int) + GVA_BUFF_MAX_SIZE;
    void *gva = shmem_malloc(gva_size);
    aclrtMemset(gva, gva_size, 0, gva_size);
    int data_type_int = static_cast<int>(data_type);

    ACLRT_LAUNCH_KERNEL(allgather)(block_dim, stream, send_buff, recv_buff, gva, send_count, data_type_int, team_id, ffts_addr, magic, tiling_device_ptr);
    aclrtFreeHost(tiling_host.get());
    aclrtFree(tiling_device_ptr);
    shmem_free(gva);
    return 0;
}

}  // namespace zccl

extern "C" ZBCCL_API int32_t zbccl_all_gather(void *send_buff, void *recv_buff, size_t send_count, zbccl_datatype_t data_type,
                         size_t team_id, aclrtStream stream) 
{
    return zbccl::zbccl_all_gather(send_buff, recv_buff, send_count, data_type, team_id, stream);
}

#endif  // ZBCCL_OP_ALLGATHER_H
