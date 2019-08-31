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

#ifndef ARGON2COMMON_H
#define ARGON2COMMON_H

#include <cstdint>

#include <stddef.h>

#include "crypto/argon2d/argon2.h"

namespace argon2gpu
{

enum
{
    ARGON2_BLOCK_SIZE = 1024,
    ARGON2_PREHASH_DIGEST_LENGTH = 64,
    ARGON2_PREHASH_SEED_LENGTH = 72
};

enum Type
{
    ARGON2_D = 0,
    ARGON2_I = 1,
    ARGON2_ID = 2
};

enum Version
{
    ARGON2_VERSION_10 = 0x10,
    ARGON2_VERSION_13 = 0x13
};

class Argon2Params
{
  private:
    std::uint32_t outLen;
    std::uint32_t t_cost, m_cost, lanes;

    std::uint32_t segmentBlocks;

    static void digestLong(void *out, size_t outLen,
                           const void *in, size_t inLen);

    void initialHash(void *out, const void *input, size_t inputLen,
                     Type type, Version version) const;

  public:
    std::uint32_t getOutputLength() const { return outLen; }

    std::uint32_t getTimeCost() const { return t_cost; }
    std::uint32_t getMemoryCost() const { return m_cost; }
    std::uint32_t getLanes() const { return lanes; }

    std::uint32_t getSegmentBlocks() const { return segmentBlocks; }
    std::uint32_t getLaneBlocks() const
    {
        return segmentBlocks * ARGON2_SYNC_POINTS;
    }
    std::uint32_t getMemoryBlocks() const { return getLaneBlocks() * lanes; }
    size_t getMemorySize() const
    {
        return static_cast<size_t>(getMemoryBlocks()) * ARGON2_BLOCK_SIZE;
    }

    Argon2Params(size_t outLen, size_t t_cost, size_t m_cost, size_t lanes);

    void fillFirstBlocks(void *memory, const void *pwd, size_t pwdLen,
                         Type type, Version version) const;

    void finalize(void *out, const void *memory) const;
};

} // namespace argon2gpu

#endif // ARGON2COMMON_H
