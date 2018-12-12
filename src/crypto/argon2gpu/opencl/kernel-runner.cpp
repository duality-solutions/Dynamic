/*
 * Copyright (C) 2015-2019 Łukasz Kurowski <crackcomm@gmail.com>, Ondrej Mosnacek <omosnacek@gmail.com>
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

#ifndef NDEBUG
#include <iostream>
#endif

#define THREADS_PER_LANE 32

namespace argon2gpu
{
namespace opencl
{
enum {
    ARGON2_REFS_PER_BLOCK = ARGON2_BLOCK_SIZE / (2 * sizeof(cl_uint)),
};

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

    queue = cl::CommandQueue(context, device->getCLDevice(),
        CL_QUEUE_PROFILING_ENABLE);

#ifndef NDEBUG
    std::cerr << "[INFO] Allocating " << memorySize << " bytes for memory..."
              << std::endl;
#endif

    memoryBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, memorySize);

//    Type type = programContext->getArgon2Type();
//    if ((type == ARGON2_I || type == ARGON2_ID) && precompute) {
//        std::uint32_t segments =
//            type == ARGON2_ID ? lanes * (ARGON2_SYNC_POINTS / 2) : passes * lanes * ARGON2_SYNC_POINTS;
//
//        std::size_t refsSize = segments * segmentBlocks * sizeof(cl_uint) * 2;
//
//#ifndef NDEBUG
//        std::cerr << "[INFO] Allocating " << refsSize << " bytes for refs..."
//                  << std::endl;
//#endif
//
//        refsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, refsSize);
//
//        precomputeRefs();
//    }
//
//    static const char* KERNEL_NAMES[2][2] = {
//        {
//            "argon2_kernel_oneshot",
//            "argon2_kernel_segment",
//        },
//        {
//            "argon2_kernel_oneshot_precompute",
//            "argon2_kernel_segment_precompute",
//        }};

//    kernel = cl::Kernel(programContext->getProgram(),
//        KERNEL_NAMES[precompute][bySegment]);

    kernel = cl::Kernel(programContext->getProgram(), "argon2d_fill");
    kernel.setArg<cl::Buffer>(1, memoryBuffer);
    kernel.setArg<cl_uint>(2, passes);
	kernel.setArg<cl_uint>(3, lanes);
	kernel.setArg<cl_uint>(4, segmentBlocks);

//    if (precompute) {
//        kernel.setArg<cl::Buffer>(2, refsBuffer);
//        kernel.setArg<cl_uint>(3, passes);
//        kernel.setArg<cl_uint>(4, lanes);
//        kernel.setArg<cl_uint>(5, segmentBlocks);
//    } else {
//        kernel.setArg<cl_uint>(2, passes);
//        kernel.setArg<cl_uint>(3, lanes);
//        kernel.setArg<cl_uint>(4, segmentBlocks);
//    }

    inputBuffer 	= cl::Buffer(context, CL_MEM_WRITE_ONLY, 80);
    resultBuffer 	= cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t));

    kernelInit 		= cl::Kernel(programContext->getProgram(), "argon2d_initialize");
    kernelInit.setArg<cl::Buffer>(0, memoryBuffer);
    kernelInit.setArg<cl::Buffer>(1, inputBuffer);

    kernelFinal 	= cl::Kernel(programContext->getProgram(), "argon2d_finalize");
    kernelFinal.setArg<cl::Buffer>(0, memoryBuffer);
    kernelFinal.setArg<cl::Buffer>(1, resultBuffer);
}

void KernelRunner::precomputeRefs()
{
    std::uint32_t passes = params->getTimeCost();
    std::uint32_t lanes = params->getLanes();
    std::uint32_t segmentBlocks = params->getSegmentBlocks();
    std::uint32_t segmentAddrBlocks =
        (segmentBlocks + ARGON2_REFS_PER_BLOCK - 1) / ARGON2_REFS_PER_BLOCK;
    std::uint32_t segments = programContext->getArgon2Type() == ARGON2_ID ? lanes * (ARGON2_SYNC_POINTS / 2) : passes * lanes * ARGON2_SYNC_POINTS;

    std::size_t shmemSize = THREADS_PER_LANE * sizeof(cl_uint) * 2;

    cl::Kernel kernel = cl::Kernel(programContext->getProgram(),
        "argon2_precompute_kernel");
    kernel.setArg<cl::LocalSpaceArg>(0, {shmemSize});
    kernel.setArg<cl::Buffer>(1, refsBuffer);
    kernel.setArg<cl_uint>(2, passes);
    kernel.setArg<cl_uint>(3, lanes);
    kernel.setArg<cl_uint>(4, segmentBlocks);

    cl::NDRange globalRange{THREADS_PER_LANE * segments * segmentAddrBlocks};
    cl::NDRange localRange{THREADS_PER_LANE};
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);
    queue.finish();
}

void* KernelRunner::mapInputMemory(std::uint32_t jobId)
{
    std::size_t memorySize = params->getMemorySize();
    std::size_t mappedSize = params->getLanes() * 2 * ARGON2_BLOCK_SIZE;
    return queue.enqueueMapBuffer(memoryBuffer, true, CL_MAP_WRITE,
        memorySize * jobId, mappedSize);
}

void KernelRunner::unmapInputMemory(void* memory)
{
    queue.enqueueUnmapMemObject(memoryBuffer, memory);
}

void* KernelRunner::mapOutputMemory(std::uint32_t jobId)
{
    std::size_t memorySize = params->getMemorySize();
    std::size_t mappedSize = static_cast<std::size_t>(params->getLanes()) * ARGON2_BLOCK_SIZE;
    std::size_t mappedOffset = memorySize * (jobId + 1) - mappedSize;
    return queue.enqueueMapBuffer(memoryBuffer, true, CL_MAP_READ,
        mappedOffset, mappedSize);
}

void KernelRunner::unmapOutputMemory(void* memory)
{
    queue.enqueueUnmapMemObject(memoryBuffer, memory);
}

void KernelRunner::run(std::uint32_t lanesPerBlock, std::uint32_t jobsPerBlock)
{
    std::uint32_t lanes = params->getLanes();
//    std::uint32_t passes = params->getTimeCost();

//    if (bySegment) {
//        if (lanesPerBlock > lanes || lanes % lanesPerBlock != 0) {
//            throw std::logic_error("Invalid lanesPerBlock!");
//        }
//    } else {
//        if (lanesPerBlock != lanes) {
//            throw std::logic_error("Invalid lanesPerBlock!");
//        }
//    }
//
//    if (jobsPerBlock > batchSize || batchSize % jobsPerBlock != 0) {
//        throw std::logic_error("Invalid jobsPerBlock!");
//    }

    cl::NDRange globalRange{THREADS_PER_LANE * lanes, batchSize};
    cl::NDRange localRange{THREADS_PER_LANE * lanesPerBlock, jobsPerBlock};

//    queue.enqueueMarker(&start);

    std::size_t shmemSize = THREADS_PER_LANE * lanesPerBlock * jobsPerBlock * sizeof(cl_uint) * 2;
    kernel.setArg<cl::LocalSpaceArg>(0, {shmemSize});
//    if (bySegment) {
//        for (std::uint32_t pass = 0; pass < passes; pass++) {
//            for (std::uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; slice++) {
//                kernel.setArg<cl_uint>(precompute ? 6 : 5, pass);
//                kernel.setArg<cl_uint>(precompute ? 7 : 6, slice);
//                queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//                    globalRange, localRange);
//            }
//        }
//    } else {
//        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
//            globalRange, localRange);
//    }

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);
//    queue.enqueueMarker(&end);
}

void KernelRunner::init(const void* input){
	//TODO(EhssanD) check the returned status
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
	cl::NDRange global{ 4, batchSize };
	cl::NDRange local{ 4, jobsPerBlock };
	std::size_t smem = 129 * sizeof(cl_ulong) * jobsPerBlock + 18 * sizeof(cl_ulong) * jobsPerBlock;

	kernelFinal.setArg<cl::LocalSpaceArg>(2, cl::__local(smem));
	kernelFinal.setArg<cl_uint>(3, startNonce);
	kernelFinal.setArg<cl_ulong>(4, target);

	queue.enqueueNDRangeKernel(kernelFinal, cl::NullRange, global, local);

}


uint32_t KernelRunner::readResultNonce()
{
	queue.enqueueReadBuffer(resultBuffer, true, 0, sizeof(cl_uint), static_cast<void*>(&res_nonce));
	return res_nonce;
}

float KernelRunner::finish()
{
    end.wait();

    cl_ulong nsStart = start.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong nsEnd = end.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    return (nsEnd - nsStart) / (1000.0F * 1000.0F);
}

} // namespace opencl
} // namespace argon2gpu
