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


#include "crypto/argon2gpu/opencl/processing-unit.h"

namespace argon2gpu
{
namespace opencl
{

ProcessingUnit::ProcessingUnit(
    const ProgramContext* programContext,
    const Argon2Params* params,
    const Device* device,
    std::size_t batchSize,
    bool bySegment,
    bool precomputeRefs)
    : programContext(programContext), params(params), device(device),
      runner(programContext, params, device, batchSize, bySegment, precomputeRefs),
      bestLanesPerBlock(runner.getMinLanesPerBlock()),
      bestJobsPerBlock(runner.getMinJobsPerBlock()){}


std::uint32_t ProcessingUnit::scanNonces(
    const void* input, const std::uint32_t startNonce,
    const std::uint64_t target)
{
    runner.init(input);
    runner.fillFirstBlocks(startNonce);
    runner.run(bestLanesPerBlock, bestJobsPerBlock);
    runner.finalize(startNonce, target);
    return runner.readResultNonce();
}

} // namespace opencl
} // namespace argon2gpu
