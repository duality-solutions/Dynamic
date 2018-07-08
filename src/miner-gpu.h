// Copyright (c) 2018 Łukassz Kurowski <crackcomm@gmail.com>

#ifndef DYNAMIC_ARGON2_GPU
#define DYNAMIC_ARGON2_GPU

#ifdef ENABLE_GPU

#include <cstddef>

#include "crypto/argon2gpu/common.h"

// TODO: configure script
#define HAVE_CUDA 1

#ifdef HAVE_CUDA
#include "crypto/argon2gpu/cuda/cuda-exception.h"
#include "crypto/argon2gpu/cuda/processing-unit.h"
#else
#include "crypto/argon2gpu/opencl/opencl.h"
#include "crypto/argon2gpu/opencl/processing-unit.h"
#endif

using Argon2GPUParams = argon2gpu::Argon2Params;

#ifdef HAVE_CUDA
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

static Argon2GPU GetProcessingUnit(std::size_t nDeviceIndex, bool fGPU) {
    if (!fGPU) {
        Argon2GPU processingUnit(nullptr, nullptr, nullptr, 1, false, false);
        return processingUnit;
    }
    else {
        // Argon2GPU processingUnit = GetGPUProcessingUnit(nDeviceIndex);
        Argon2GPUContext global;
        auto& devices = global.getAllDevices();
        auto& device = devices[nDeviceIndex];
        Argon2GPUProgramContext context(&global, {device}, argon2gpu::ARGON2_D, argon2gpu::ARGON2_VERSION_10);
        Argon2GPUParams params((std::size_t)OUTPUT_BYTES, 2, 500, 8);
        Argon2GPU processingUnit(&context, &params, &device, 1, false, false);
        return processingUnit;
    }
}

template <class Pu>
inline uint256 GetBlockHashGPU(const CBlockHeader* block, const Pu& pu)
{
    static unsigned char pblank[1];
    const auto pBegin = BEGIN(block->nVersion);
    const auto pEnd = END(block->nNonce);
    const void* input = (pBegin == pEnd ? pblank : static_cast<const void*>(&pBegin[0]));

    uint256 hashResult;
    pu->setInputAndSalt(0, (const void*)input, INPUT_BYTES);
    pu->beginProcessing();
    pu->endProcessing();
    pu->getHash(0, (uint8_t*)&hashResult);
    return hashResult;
}
#else
static std::size_t GetGPUDeviceCount()
{
    return 0;
}
#endif // ENABLE_GPU

#endif // DYNAMIC_ARGON2_GPU
