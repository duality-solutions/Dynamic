// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2015 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "primitives/block.h"

#include "crypto/common.h"

#include "hash.h"
#include "tinyformat.h"
#include "validation.h"
#include "utilstrencodings.h"
#include "crypto/randomx.h"

uint256 CBlockHeader::GetHash(bool hash_override) const
{
    if (chainActive.Height() > std::numeric_limits<uint64_t>::max() || hash_override) {
        uint256 k = epoch_cache->GetClosestEpoch(hashPrevBlock);
        return RXHashFunction(BEGIN(nVersion), END(nNonce), k.begin(), k.end());
    }
    return hash_Argon2d(BEGIN(nVersion), END(nNonce), 1);
}

std::string CBlock::ToString() const
{
    std::stringstream s;
    s << strprintf("CBlock(hash=%s, ver=%d, hashPrevBlock=%s, hashMerkleRoot=%s, nTime=%u, nBits=%08x, nNonce=%u, vtx=%u)\n",
        GetHash().ToString(),
        nVersion,
        hashPrevBlock.ToString(),
        hashMerkleRoot.ToString(),
        nTime, nBits, nNonce,
        vtx.size());
    for (unsigned int i = 0; i < vtx.size(); i++) {
        s << "  " << vtx[i]->ToString() << "\n";
    }
    return s.str();
}
