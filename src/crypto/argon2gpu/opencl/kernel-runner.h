/*
 * Copyright (C) 2015-2018 Łukasz Kurowski <crackcomm@gmail.com>, Ondrej Mosnacek <omosnacek@gmail.com>
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

#ifndef ARGON2_OPENCL_KERNELRUNNER_H
#define ARGON2_OPENCL_KERNELRUNNER_H

#include "crypto/argon2gpu/common.h"
#include "crypto/argon2gpu/opencl/program-context.h"

namespace argon2gpu
{
namespace opencl
{
class KernelRunner
{
private:
    const ProgramContext* programContext;
    const Argon2Params* params;

    std::uint32_t batchSize;
    bool bySegment;
    bool precompute;

    cl::CommandQueue queue;
    cl::Kernel kernel;
    cl::Buffer memoryBuffer, refsBuffer;
    cl::Event start, end;

    std::size_t memorySize;

    void precomputeRefs();

public:
    std::uint32_t getMinLanesPerBlock() const
    {
        return bySegment ? 1 : params->getLanes();
    }
    std::uint32_t getMaxLanesPerBlock() const { return params->getLanes(); }

    std::uint32_t getMinJobsPerBlock() const { return 1; }
    std::uint32_t getMaxJobsPerBlock() const { return batchSize; }

    std::uint32_t getBatchSize() const { return batchSize; }

    KernelRunner(const ProgramContext* programContext,
        const Argon2Params* params,
        const Device* device,
        std::uint32_t batchSize,
        bool bySegment,
        bool precompute);

    void* mapInputMemory(std::uint32_t jobId);
    void unmapInputMemory(void* memory);

    void* mapOutputMemory(std::uint32_t jobId);
    void unmapOutputMemory(void* memory);

    void run(std::uint32_t lanesPerBlock, std::uint32_t jobsPerBlock);
    float finish();
};

} // namespace opencl
} // namespace argon2gpu

#endif // ARGON2_OPENCL_KERNELRUNNER_H
