/*
 * Copyright (C) 2018-2019 Ehsan Dalvand <dalvand.ehsan@gmail.com>
 * Copyright (C) 2017-2019 Łukasz Kurowski <crackcomm@gmail.com>
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

#ifndef ARGON2_CUDA_KERNELS_H
#define ARGON2_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

/* workaround weird CMake/CUDA bug: */
#ifdef argon2
#undef argon2
#endif

namespace argon2gpu
{
namespace cuda
{

class KernelRunner
{
  private:
    std::uint32_t type, version;
    std::uint32_t passes, lanes, segmentBlocks;
    std::uint32_t batchSize;
    bool bySegment;
    bool precompute;
    int deviceIndex;

    cudaEvent_t start, end;
    cudaStream_t stream;
    void *memory;
    void *refs;

    std::uint32_t res_nonce;
    std::uint32_t *d_res_nonce;

    void runKernelOneshot(std::uint32_t lanesPerBlock,
                          std::uint32_t jobsPerBlock);

  public:
    std::uint32_t getMinLanesPerBlock() const { return bySegment ? 1 : lanes; }
    std::uint32_t getMaxLanesPerBlock() const { return lanes; }

    std::uint32_t getMinJobsPerBlock() const { return 1; }
    std::uint32_t getMaxJobsPerBlock() const { return batchSize; }

    std::uint32_t getBatchSize() const { return batchSize; }

    KernelRunner(std::uint32_t type, std::uint32_t version,
                 std::uint32_t passes, std::uint32_t lanes,
                 std::uint32_t segmentBlocks, std::uint32_t batchSize,
                 bool bySegment, bool precompute, int deviceIndex);
    ~KernelRunner();

    void run(std::uint32_t lanesPerBlock, std::uint32_t jobsPerBlock);
    void init(const void* input);
    void fillFirstBlocks(const std::uint32_t start_nonce);
    void finalize(const std::uint32_t startNonce, const std::uint64_t target);
    std::uint32_t readResultNonce();

};

} // namespace cuda
} // namespace argon2gpu

#endif // ARGON2_CUDA_KERNELS_H
