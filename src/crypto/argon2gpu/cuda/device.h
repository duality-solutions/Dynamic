/*
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

#ifndef ARGON2_CUDA_DEVICE_H
#define ARGON2_CUDA_DEVICE_H

#include <string>

namespace argon2gpu
{
namespace cuda
{

class Device
{
  private:
    int deviceIndex;

  public:
    std::string getName() const;
    std::string getInfo() const;
    std::size_t getTotalMemory() const;

    int getDeviceIndex() const { return deviceIndex; }

    /**
     * @brief Empty constructor.
     * NOTE: Calling methods other than the destructor on an instance initialized
     * with empty constructor results in undefined behavior.
     */
    Device() {}

    Device(int deviceIndex) : deviceIndex(deviceIndex)
    {
    }

    Device(const Device &) = default;
    Device(Device &&) = default;

    Device &operator=(const Device &) = default;
};

} // namespace cuda
} // namespace argon2gpu

#endif // ARGON2_CUDA_DEVICE_H
