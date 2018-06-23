/*
 * Copyright (C) 2017-2018 Łukasz Kurowski <crackcomm@gmail.com>
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

#ifndef ARGON2_CUDA_CUDAEXCEPTION_H
#define ARGON2_CUDA_CUDAEXCEPTION_H

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <exception>

namespace argon2gpu
{
namespace cuda
{

#if HAVE_CUDA

class CudaException : public std::exception
{
  private:
    cudaError_t res;

  public:
    CudaException(cudaError_t res) : res(res) {}

    const char *what() const noexcept override
    {
        return cudaGetErrorString(res);
    }

    static void check(cudaError_t res)
    {
        if (res != cudaSuccess)
        {
            throw CudaException(res);
        }
    }
};

#else

class CudaException : public std::exception
{
};

#endif

} // namespace cuda
} // namespace argon2gpu

#endif // ARGON2_CUDA_CUDAEXCEPTION_H
