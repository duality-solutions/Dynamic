// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "swap/swapdb.h"

#include "bdap/utils.h"

#include <univalue.h>

#include <boost/thread.hpp>

CSwapDB *pSwapDB = NULL;

bool CSwapDB::AddSwap(const CSwapData& swap) 
{ 
    bool writeState = false;
    {
        LOCK(cs_swap);
        writeState = Write(make_pair(std::string("swap"), swap.vchTxId()), swap);
    }
    return writeState;
}

bool CSwapDB::ReadSwapTxId(const std::vector<unsigned char>& vchTxId, CSwapData& swap) 
{
    LOCK(cs_swap);
    return CDBWrapper::Read(make_pair(std::string("swap"), vchTxId), swap);
}

bool CSwapDB::EraseSwapTxId(const std::vector<unsigned char>& vchTxId)
{
    LOCK(cs_swap);
    CSwapData swap;
    if (ReadSwapTxId(vchTxId, swap))
        return CDBWrapper::Erase(make_pair(std::string("swap"), vchTxId));

    return false;
}

bool CSwapDB::GetAllSwaps(std::vector<CSwapData>& vSwaps)
{
    LOCK(cs_swap);
    std::pair<std::string, CharString> key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            bool fGetKey = pcursor->GetKey(key);
            if (fGetKey && key.first == "swap") {
                CSwapData swap;
                pcursor->GetValue(swap);
                vSwaps.push_back(swap);
            }
            pcursor->Next();
        }
        catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool AddSwap(const CSwapData& swap)
{
    CSwapData readSwap;
    if (!pSwapDB || pSwapDB->ReadSwapTxId(swap.vchTxId(), readSwap))
        return false;

    if (!pSwapDB->AddSwap(swap))
        return false;

    return true;
}

bool GetAllSwaps(std::vector<CSwapData>& vSwaps)
{
    if (!pSwapDB || !pSwapDB->GetAllSwaps(vSwaps))
        return false;
    return true;
}

bool GetSwapTxId(const std::string& strTxId, CSwapData& swap)
{
    if (!pSwapDB || !pSwapDB->ReadSwapTxId(vchFromString(strTxId), swap))
        return false;

    return !swap.IsNull();
}

bool SwapExists(const std::vector<unsigned char>& vchTxId, CSwapData& swap)
{
    if (!pSwapDB)
        return false;

    return pSwapDB->ReadSwapTxId(vchTxId, swap);
}

bool UndoAddSwap(const CSwapData& swap)
{
    if (!pSwapDB)
        return false;

    return pSwapDB->EraseSwapTxId(vchFromString(swap.TxId.ToString()));
}

bool CheckSwapDB()
{
    if (!pSwapDB)
        return false;

    return true;
}

bool FlushSwapLevelDB() 
{
    {
        LOCK(cs_swap);
        if (pSwapDB != NULL)
        {
            if (!pSwapDB->Flush()) {
                LogPrintf("Failed to flush Swap database!");
                return false;
            }
        }
    }
    return true;
}
