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

#ifndef ARGON2_OPENCL_KERNELLOADER_H
#define ARGON2_OPENCL_KERNELLOADER_H

#include "crypto/argon2gpu/common.h"
#include "crypto/argon2gpu/opencl/opencl.h"

#include <string>

namespace argon2gpu
{
namespace opencl
{
namespace KernelLoader
{
cl::Program loadArgon2Program(
    const cl::Context& context,
    Type type,
    Version version,
    bool debug = false);
};

} // namespace opencl
} // namespace argon2gpu

#endif // ARGON2_OPENCL_KERNELLOADER_H
