// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MINER_IMPL_GPU_H
#define DYNAMIC_MINER_IMPL_GPU_H

#if defined(HAVE_CONFIG_H)
#include "config/dynamic-config.h"
#endif

#ifdef ENABLE_GPU
#include "crypto/argon2gpu/common.h"
#include "miner/internal/miner-base.h"

#if HAVE_CUDA
#include "crypto/argon2gpu/cuda/cuda-exception.h"
#include "crypto/argon2gpu/cuda/processing-unit.h"
#else
#include "crypto/argon2gpu/opencl/opencl.h"
#include "crypto/argon2gpu/opencl/processing-unit.h"
#endif


class MinerContext;

namespace gpu
{
using Params = argon2gpu::Argon2Params;

#if HAVE_CUDA
using ProcessingUnit = argon2gpu::cuda::ProcessingUnit;
using Device = argon2gpu::cuda::Device;
using Context = argon2gpu::cuda::GlobalContext;
using ProgramContext = argon2gpu::cuda::ProgramContext;
#else
using ProcessingUnit = argon2gpu::opencl::ProcessingUnit;
using Device = argon2gpu::opencl::Device;
using Context = argon2gpu::opencl::GlobalContext;
using ProgramContext = argon2gpu::opencl::ProgramContext;
#endif
} // namespace gpu

class GPUMiner final : public MinerBase
{
public:
    GPUMiner(MinerContextRef ctx, std::size_t device_index);
    virtual ~GPUMiner() = default;

    static unsigned int TotalDevices()
    {
        gpu::Context global;
        return global.getAllDevices().size();
    };

    virtual const char* DeviceName() override { return "GPU"; };

protected:
    virtual int64_t TryMineBlock(CBlock& block) override;

private:
    gpu::Context _global;
    gpu::Params _params;
    gpu::Device _device;
    gpu::ProgramContext _context;
    std::size_t _batch_size_target;
    gpu::ProcessingUnit _processing_unit;
};

#endif // ENABLE_GPU
#endif // DYNAMIC_MINER_IMPL_GPU_H
