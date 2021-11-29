// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.


#ifndef DYNAMIC_SWAP_SWAPDB_H
#define DYNAMIC_SWAP_SWAPDB_H

#include "swap/swapdata.h"
#include "dbwrapper.h"
#include "sync.h"

class CTxOut;

static CCriticalSection cs_swap;

class CSwapDB : public CDBWrapper {
public:
    CSwapDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "swaps", nCacheSize, fMemory, fWipe, obfuscate) {
    }
    bool AddSwap(const CSwapData& swap);
    bool ReadSwap(const std::vector<unsigned char>& vchSwap, CSwapData& swap);
    bool GetAllSwaps(std::vector<CSwapData>& vSwaps);
    bool ReadSwapTxId(const std::vector<unsigned char>& vchTxId, CSwapData& swap);
    bool EraseSwapTxId(const std::vector<unsigned char>& vchTxId);
};

bool AddSwap(const CSwapData& swap);
bool GetAllSwaps(std::vector<CSwapData>& vSwaps);
bool GetSwapTxId(const std::string& strTxId, CSwapData& swap);
bool SwapExists(const std::vector<unsigned char>& vchTxId, CSwapData& swap);
bool UndoAddSwap(const CSwapData& swap);
bool CheckSwapDB();
bool FlushSwapLevelDB();

extern CSwapDB *pSwapDB;

#endif // DYNAMIC_SWAP_SWAPDB_H