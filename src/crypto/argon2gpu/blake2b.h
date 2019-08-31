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

#ifndef ARGON2_BLAKE2B_H
#define ARGON2_BLAKE2B_H

#include <cstdint>

#include <stddef.h>

namespace argon2gpu
{

class Blake2b
{
  public:
    enum
    {
        BLOCK_BYTES = 128,
        OUT_BYTES = 64,
    };

  private:
    std::uint64_t h[8];
    std::uint64_t t[2];
    std::uint8_t buf[BLOCK_BYTES];
    size_t bufLen;

    void compress(const void *block, std::uint64_t f0);
    void incrementCounter(std::uint64_t inc);

  public:
    Blake2b() : h(), t(), buf(), bufLen(0) {}

    void init(size_t outlen);
    void update(const void *in, size_t inLen);
    void final(void *out, size_t outLen);
};

} // namespace argon2gpu

#endif // ARGON2_BLAKE2B_H
