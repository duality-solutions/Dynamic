/*
 * Copyright (C) 2015-2019  Ehsan Dalvand <dalvand.ehsan@gmail.com>, Łukasz Kurowski <crackcomm@gmail.com>, Ondrej Mosnacek <omosnacek@gmail.com>
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation: either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "crypto/argon2gpu/opencl/kernel-runner.h"

#include <stdexcept>

#define THREADS_PER_LANE 32

namespace argon2gpu
{
namespace opencl
{

KernelRunner::KernelRunner(const ProgramContext* programContext,
    const Argon2Params* params,
    const Device* device,
    std::uint32_t batchSize,
    bool bySegment,
    bool precompute)
    : programContext(programContext), params(params), batchSize(batchSize),
      bySegment(bySegment), precompute(precompute),
      memorySize(params->getMemorySize() * static_cast<std::size_t>(batchSize))
{
    auto context = programContext->getContext();
    std::uint32_t passes = params->getTimeCost();
    std::uint32_t lanes = params->getLanes();
    std::uint32_t segmentBlocks = params->getSegmentBlocks();

    queue = cl::CommandQueue(context, device->getCLDevice());

    memoryBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, memorySize);

    kernel = cl::Kernel(programContext->getProgram(), "argon2d_fill");
    kernel.setArg<cl::Buffer>(1, memoryBuffer);
    kernel.setArg<cl_uint>(2, passes);
    kernel.setArg<cl_uint>(3, lanes);
    kernel.setArg<cl_uint>(4, segmentBlocks);

    inputBuffer     = cl::Buffer(context, CL_MEM_WRITE_ONLY, 80);
    resultBuffer    = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t));

    kernelInit      = cl::Kernel(programContext->getProgram(), "argon2d_initialize");
    kernelInit.setArg<cl::Buffer>(0, memoryBuffer);
    kernelInit.setArg<cl::Buffer>(1, inputBuffer);

    kernelFinal     = cl::Kernel(programContext->getProgram(), "argon2d_finalize");
    kernelFinal.setArg<cl::Buffer>(0, memoryBuffer);
    kernelFinal.setArg<cl::Buffer>(1, resultBuffer);
}

void KernelRunner::run(std::uint32_t lanesPerBlock, std::uint32_t jobsPerBlock)
{
    std::uint32_t lanes = params->getLanes();
    std::size_t shmemSize = THREADS_PER_LANE * lanesPerBlock * jobsPerBlock * sizeof(cl_uint) * 2;

    cl::NDRange globalRange{THREADS_PER_LANE * lanes, batchSize};
    cl::NDRange localRange{THREADS_PER_LANE * lanesPerBlock, jobsPerBlock};

    kernel.setArg<cl::LocalSpaceArg>(0, {shmemSize});

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);
}

void KernelRunner::init(const void* input){
    std::uint32_t umax = std::numeric_limits<uint32_t>::max();
    queue.enqueueWriteBuffer(inputBuffer, true, 0, 80, input);
    queue.enqueueWriteBuffer(resultBuffer, true, 0, sizeof(cl_uint), static_cast<void*>(&umax));
}

void KernelRunner::fillFirstBlocks(const std::uint32_t startNonce)
{
    std::uint32_t lanes = params->getLanes();
    std::uint32_t jobsPerBlock = (batchSize<16) ? 1 : 16;
    cl::NDRange global{ lanes*2, batchSize };
    cl::NDRange local{ lanes*2, jobsPerBlock };

    kernelInit.setArg<cl_uint>(2, startNonce);

    queue.enqueueNDRangeKernel(kernelInit, cl::NullRange, global, local);

}

void KernelRunner::finalize(const std::uint32_t startNonce, const std::uint64_t target)
{
    std::uint32_t jobsPerBlock = (batchSize<8) ? 1 : 8;
    std::size_t smem = 129 * sizeof(cl_ulong) * jobsPerBlock + 18 * sizeof(cl_ulong) * jobsPerBlock;

    cl::NDRange global{ 4, batchSize };
    cl::NDRange local{ 4, jobsPerBlock };

    kernelFinal.setArg<cl::LocalSpaceArg>(2, cl::__local(smem));
    kernelFinal.setArg<cl_uint>(3, startNonce);
    kernelFinal.setArg<cl_ulong>(4, target);

    queue.enqueueNDRangeKernel(kernelFinal, cl::NullRange, global, local);

}


std::uint32_t KernelRunner::readResultNonce()
{
    queue.enqueueReadBuffer(resultBuffer, true, 0, sizeof(cl_uint), static_cast<void*>(&res_nonce));
    return res_nonce;
}

} // namespace opencl
} // namespace argon2gpu
