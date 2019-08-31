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

#include "crypto/argon2gpu/opencl/program-context.h"
#include "crypto/argon2gpu/opencl/kernel-loader.h"

namespace argon2gpu
{
namespace opencl
{
ProgramContext::ProgramContext(
    const GlobalContext* globalContext,
    const std::vector<Device>& devices,
    Type type,
    Version version)
    : globalContext(globalContext), devices(), type(type), version(version)
{
    this->devices.reserve(devices.size());
    for (auto& device : devices) {
        this->devices.push_back(device.getCLDevice());
    }
    context = cl::Context(this->devices);
    program = KernelLoader::loadArgon2Program(context, type, version);
}

} // namespace opencl
} // namespace argon2gpu
