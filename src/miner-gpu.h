// Copyright (c) 2018 Łukassz Kurowski <crackcomm@gmail.com>

#ifndef DYNAMIC_ARGON2_GPU
#define DYNAMIC_ARGON2_GPU

#ifdef ENABLE_GPU
#include <cstddef>

#include "crypto/argon2gpu/common.h"

#if HAVE_CUDA
#include "crypto/argon2gpu/cuda/cuda-exception.h"
#include "crypto/argon2gpu/cuda/processing-unit.h"
#else
#include "crypto/argon2gpu/opencl/opencl.h"
#include "crypto/argon2gpu/opencl/processing-unit.h"
#endif

using Argon2GPUParams = argon2gpu::Argon2Params;

#if HAVE_CUDA
using Argon2GPU = argon2gpu::cuda::ProcessingUnit;
using Argon2GPUDevice = argon2gpu::cuda::Device;
using Argon2GPUContext = argon2gpu::cuda::GlobalContext;
using Argon2GPUProgramContext = argon2gpu::cuda::ProgramContext;
#else
using Argon2GPU = argon2gpu::opencl::ProcessingUnit;
using Argon2GPUDevice = argon2gpu::opencl::Device;
using Argon2GPUContext = argon2gpu::opencl::GlobalContext;
using Argon2GPUProgramContext = argon2gpu::opencl::ProgramContext;
#endif

static std::size_t GetGPUDeviceCount()
{
    static Argon2GPUContext global;
    return global.getAllDevices().size();
}

template <class Pu>
inline uint256 GetBlockHashGPU(const CBlockHeader* block, const Pu& pu)
{
    static unsigned char pblank[1];
    const auto pBegin = BEGIN(block->nVersion);
    const auto pEnd = END(block->nNonce);
    const void* input = (pBegin == pEnd ? pblank : static_cast<const void*>(&pBegin[0]));

    uint256 hashResult;
    pu->setInputAndSalt(0, input, INPUT_BYTES);
    pu->beginProcessing();
    pu->endProcessing();
    pu->getHash(0, (uint8_t*)&hashResult);
    return hashResult;
}
#endif // ENABLE_GPU

#endif // DYNAMIC_ARGON2_GPU
