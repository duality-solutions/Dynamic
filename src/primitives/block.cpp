// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2015 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "primitives/block.h"

#include "crypto/common.h"

#include "chain.h"
#include "chainparams.h"
#include "hash.h"
#include "tinyformat.h"
#include "validation.h"
#include "utilstrencodings.h"
#include "proof/hash.cpp"

uint64_t GetHeight(const uint256& block_hash)
{
    CBlockIndex* pblockindex = mapBlockIndex[block_hash];
    assert(pblockindex);
    return pblockindex->nHeight;
}

bool ReadBlock(const uint64_t& nHeight, CBlock& block)
{
    CBlockIndex* pblockindex = chainActive[nHeight];

    if (!(pblockindex->nStatus & BLOCK_HAVE_DATA) && pblockindex->nTx > 0)
        return error("%s: invalid block index %d", __func__, nHeight);

    if (!ReadBlockFromDisk(block, pblockindex, Params().GetConsensus()))
        return error("%s: block not found, expected index: %x", __func__, nHeight);

    return true;
}

uint256 CBlockHeader::GetHash(bool hash_override) const
{
    if (chainActive.Height() > std::numeric_limits<uint64_t>::max() || hash_override) {
        uint64_t nHeight = GetHeight(hashPrevBlock);
        CBlock epoch_block;
        while (nHeight % 2048 != 0) {
            nHeight--;
        }
        assert(ReadBlock(nHeight, epoch_block));
        return RXHashFunction(BEGIN(nVersion), END(nNonce), epoch_block.GetHash().begin(), epoch_block.GetHash().end());
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
