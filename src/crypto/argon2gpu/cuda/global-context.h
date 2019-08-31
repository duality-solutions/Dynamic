/*
 * Copyright (C) 2017-2019 Łukasz Kurowski <crackcomm@gmail.com>
 * Copyright (C) 2015, Ondrej Mosnacek <omosnacek@gmail.com>
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

#ifndef ARGON2_CUDA_GLOBALCONTEXT_H
#define ARGON2_CUDA_GLOBALCONTEXT_H

#include "crypto/argon2gpu/cuda/device.h"

#include <string>
#include <vector>

namespace argon2gpu
{
namespace cuda
{

class GlobalContext
{
  private:
    std::vector<Device> devices;

  public:
    const std::vector<Device> &getAllDevices() const { return devices; }

    GlobalContext();
};

} // namespace cuda
} // namespace argon2gpu

#endif // ARGON2_CUDA_GLOBALCONTEXT_H
