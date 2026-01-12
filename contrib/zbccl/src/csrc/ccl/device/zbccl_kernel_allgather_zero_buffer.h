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
#ifndef ZBCCL_KERNEL_ALLGATHER_ZERO_BUFFER_H
#define ZBCCL_KERNEL_ALLGATHER_ZERO_BUFFER_H

#include "kernel_operator.h"
#include "shmem_api.h"
#include "hardware.h"
#include "mem.h"
#include "zbccl.h"
#include "../zbccl_op.h"

using namespace AscendC;

class AllGatherZeroBufferKernel
{
public:
    __aicore__ inline AllGatherZeroBufferKernel() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t elements, int32_t team_id, 
        uint64_t ffts_addr, GM_ADDR tiling_data_in)
    {
        auto *tiling_data = reinterpret_cast<__gm__ zbccl::AllGatherTilingData*>(tiling_data_in);
        this->input_num_per_core = tiling_data->input_num_per_core;
        this->output_num_per_core = tiling_data->output_num_per_core;
        this->output_core_per_rank = tiling_data->output_core_per_rank;
        this->input_last_num_core = tiling_data->input_last_num_core;
        this->output_last_num_core = tiling_data->output_last_num_core;
        shmemx_set_ffts_config(ffts_addr);
        AllGatherOrigin<T>(input, output, elements, team_id);
    }

    template<typename T>
    __aicore__ inline void AllGatherOrigin(GM_ADDR inputGM, GM_ADDR outputGM, uint64_t elements, int32_t team_id)
    {
        const int64_t aivNum = AscendC::GetBlockNum();
        const int64_t aivIndex = AscendC::GetBlockIdx();

        int64_t my_rank = shmem_team_my_pe(team_id);
        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)inputGM, elements);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)outputGM);

        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<T> tmp_buff = buf.GetBuffer<BufferType::ASCEND_UB, T>(64);

        // data move parameters
        const int64_t core_per_rank = this->output_core_per_rank;
        const int64_t core_rank_idx = aivIndex % core_per_rank;
        const int64_t x = aivIndex / core_per_rank;

        // [AllGather Step 2] symmetric mem -> local output.
        uint32_t num_per_core = this->output_num_per_core;
        uint32_t output_offset = x * elements + core_rank_idx * num_per_core;
        uint32_t input_offset = core_rank_idx * num_per_core;
        if (core_rank_idx == core_per_rank - 1) {
            num_per_core = this->output_last_num_core;
        }

        shmem_mte_get_mem_nbi(outputGT[output_offset], inputGT[input_offset], tmp_buff, num_per_core, x, EVENT_ID0);
    }

private:
    uint32_t input_num_per_core;
    uint32_t output_num_per_core;
    uint32_t output_core_per_rank;
    uint32_t input_last_num_core;
    uint32_t output_last_num_core;
};


#endif  // ZBCCL_KERNEL_ALLGATHER_H
