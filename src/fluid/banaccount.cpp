// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.


#include "fluid/banaccount.h"

#include "base58.h"
#include "bdap/utils.h"
#include "core_io.h"
#include "fluid/fluid.h"
#include "script/script.h"

#include <boost/thread.hpp>

CBanAccountDB* pBanAccountDB = NULL;

bool CheckBanAccountDB()
{
    if (!pBanAccountDB)
        return false;

    return true;
}

bool AddBanAccountEntry(const CBanAccount& entry)
{
    if (!CheckBanAccountDB())
        return false;

    return pBanAccountDB->AddBanAccountEntry(entry);
}

bool AddBanWalletEntry(const CBanAccount& entry)
{
    if (!CheckBanAccountDB())
        return false;

    return pBanAccountDB->AddBanWalletEntry(entry);
}

bool GetAllBanRecords(std::vector<CBanAccount>& entries)
{
    if (!CheckBanAccountDB())
        return false;

    return pBanAccountDB->GetAllBanRecords(entries);
}

bool BanRecordExists(const std::vector<unsigned char>& vchAccountAddress)
{
    if (!CheckBanAccountDB())
        return false;

    return pBanAccountDB->BanRecordExists(vchAccountAddress);
}

bool IsBannedWalletAddress(const CTransactionRef& pTx, CDynamicAddress& address)
{
    for (const CTxOut& out : pTx->vout) {
        CDynamicAddress outAddress = GetScriptAddress(out.scriptPubKey);
        if (pBanAccountDB->BanWalletRecordExists(vchFromString(outAddress.ToString()))) {
            address = outAddress;
            return true;
        }
    }
    return false;
}

CBanAccount::CBanAccount(const std::vector<unsigned char>& account, const int64_t& timestamp, 
                const std::vector<std::vector<unsigned char>>& addresses, const uint256& txid, const unsigned int& height)
        : vchAccountAddress(account), nTimeStamp(timestamp), vSovereignAddresses(addresses), txHash(txid), nHeight(height)
{
    nVersion = CBanAccount::CURRENT_VERSION;
    CTxDestination dest = DecodeDestination(stringFromVch(vchAccountAddress));
    if (IsValidDestination(dest)) {
        fIsWalletAddress = true;
    } else {
        fIsWalletAddress = false;
    }
}

CBanAccountDB::CBanAccountDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "banned-accounts", nCacheSize, fMemory, fWipe, obfuscate)
{
}

bool CBanAccountDB::AddBanAccountEntry(const CBanAccount& entry)
{
    LOCK(cs_ban_account);
    return Write(make_pair(std::string("account"), entry.vchAccountAddress), entry, true);
}

bool CBanAccountDB::AddBanWalletEntry(const CBanAccount& entry)
{
    LOCK(cs_ban_account);
    return Write(make_pair(std::string("wallet"), entry.vchAccountAddress), entry, true);
}

bool CBanAccountDB::GetAllBanRecords(std::vector<CBanAccount>& entries)
{
    LOCK(cs_ban_account);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CBanAccount entry;
        try {
            std::pair<std::string, std::vector<unsigned char>> key;
            if (pcursor->GetKey(key) && (key.first == "account" || key.first == "wallet")) {
                pcursor->GetValue(entry);
                if (!entry.IsNull()) {
                    entries.push_back(entry);
                }
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CBanAccountDB::GetAllBanWalletRecords(std::vector<CBanAccount>& entries)
{
    LOCK(cs_ban_account);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CBanAccount entry;
        try {
            std::pair<std::string, std::vector<unsigned char>> key;
            if (pcursor->GetKey(key) && key.first == "wallet") {
                pcursor->GetValue(entry);
                if (!entry.IsNull()) {
                    entries.push_back(entry);
                }
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CBanAccountDB::BanRecordExists(const std::vector<unsigned char>& vchAccountAddress)
{
    LOCK(cs_ban_account);
    CBanAccount entry;
    bool fFindAccountAddress = CDBWrapper::Read(make_pair(std::string("wallet"), vchAccountAddress), entry);
    if (!fFindAccountAddress)
        fFindAccountAddress = CDBWrapper::Read(make_pair(std::string("account"), vchAccountAddress), entry);

    return fFindAccountAddress;
}

bool CBanAccountDB::BanWalletRecordExists(const std::vector<unsigned char>& vchAccountAddress)
{
    LOCK(cs_ban_account);
    CBanAccount entry;
    return CDBWrapper::Read(make_pair(std::string("wallet"), vchAccountAddress), entry);
}