/*
 * Copyright (C) 2017-2018 Łukasz Kurowski <crackcomm@gmail.com>
 * Copyright (C) 2015 Ondrej Mosnacek <omosnacek@gmail.com>
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

#ifndef ARGON2_CUDA_PROCESSINGUNIT_H
#define ARGON2_CUDA_PROCESSINGUNIT_H

#if HAVE_CUDA

#include <memory>

#include "crypto/argon2gpu/common.h"
#include "crypto/argon2gpu/cuda/program-context.h"
#include "crypto/argon2gpu/cuda/kernels.h"

namespace argon2gpu
{
namespace cuda
{
class ProcessingUnit
{
private:
    const ProgramContext* programContext;
    const Argon2Params* params;
    const Device* device;

    KernelRunner runner;
    std::uint32_t bestLanesPerBlock;
    std::uint32_t bestJobsPerBlock;

public:
    std::size_t getBatchSize() const { return runner.getBatchSize(); }

    ProcessingUnit(
        const ProgramContext* programContext,
        const Argon2Params* params,
        const Device* device,
        std::size_t batchSize,
        bool bySegment = true,
        bool precomputeRefs = false);

    void setInputAndSalt(std::size_t index, const void* input, const std::size_t inputSize);
    void getHash(std::size_t index, void* hash);

    void beginProcessing();
    void endProcessing();
};

} // namespace cuda
} // namespace argon2gpu

#else

#include <cstddef>

#include "crypto/argon2gpu/common.h"
#include "crypto/argon2gpu/cuda/program-context.h"

namespace argon2gpu
{
namespace cuda
{
class ProcessingUnit
{
public:
    std::size_t getBatchSize() const { return 0; }

    ProcessingUnit(
        const ProgramContext* programContext,
        const Argon2Params* params,
        const Device* device,
        std::size_t batchSize,
        bool bySegment = true,
        bool precomputeRefs = false)
    {
    }

    void setInputAndSalt(std::size_t index, const void* input, std::size_t inputSize) {}

    void getHash(std::size_t index, void* hash) {}

    void beginProcessing() {}
    void endProcessing() {}
};

} // namespace cuda
} // namespace argon2gpu

#endif /* HAVE_CUDA */

#endif // ARGON2_CUDA_PROCESSINGUNIT_H
