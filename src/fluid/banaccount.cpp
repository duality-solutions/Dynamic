// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.


#include "fluid/banaccount.h"

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

bool GetAllBanAccountRecords(std::vector<CBanAccount>& entries)
{
    if (!CheckBanAccountDB())
        return false;

    return pBanAccountDB->GetAllBanAccountRecords(entries);
}

CBanAccount::CBanAccount(const CScript& fluidScript, const std::vector<unsigned char>& fqdn, const int64_t& timestamp, 
                const std::vector<std::vector<unsigned char>>& addresses, const uint256& txid, const unsigned int& height)
        : vchFullObjectPath(fqdn), nTimeStamp(timestamp), vSovereignAddresses(addresses), txHash(txid), nHeight(height)
{
    nVersion = CBanAccount::CURRENT_VERSION;
    FluidScript = CharVectorFromString(ScriptToAsmStr(fluidScript));
}

CBanAccountDB::CBanAccountDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "banned-accounts", nCacheSize, fMemory, fWipe, obfuscate)
{
}

bool CBanAccountDB::AddBanAccountEntry(const CBanAccount& entry)
{
    LOCK(cs_ban_account);
    return Write(make_pair(std::string("account"), entry.vchFullObjectPath), entry, true);
}

bool CBanAccountDB::GetAllBanAccountRecords(std::vector<CBanAccount>& entries)
{
    LOCK(cs_ban_account);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CBanAccount entry;
        try {
            std::pair<std::string, std::vector<unsigned char>> key;
            if (pcursor->GetKey(key) && key.first == "account") {
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

bool CBanAccountDB::RecordExists(const std::vector<unsigned char>& vchFullObjectPath)
{
    LOCK(cs_ban_account);
    CBanAccount entry;
    return CDBWrapper::Read(make_pair(std::string("account"), vchFullObjectPath), entry);
}