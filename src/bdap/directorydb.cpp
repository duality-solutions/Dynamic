// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/directorydb.h"

#include "base58.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

CDirectoryDB *pDirectoryDB = NULL;

bool BuildDirectoryIndexerHistoryJson(const CDirectory& directory, UniValue& oName)
{
    return BuildBDAPJson(directory, oName);
}

std::string directoryFromOp(int op) 
{
    switch (op) {
        case OP_BDAP_NEW:
            return "bdap_new";
        case OP_BDAP_DELETE:
            return "bdap_delete";
        case OP_BDAP_ACTIVATE:
            return "bdap_activate";
        case OP_BDAP_MODIFY:
            return "bdap_update";
        case OP_BDAP_MODIFY_RDN:
            return "bdap_move";
        case OP_BDAP_EXECUTE_CODE:
            return "bdap_execute";
        case OP_BDAP_BIND:
            return "bdap_bind";
        case OP_BDAP_REVOKE:
            return "bdap_revoke";
        default:
            return "<unknown directory op>";
    }
}

bool GetDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory)
{
    if (!pDirectoryDB || !pDirectoryDB->ReadDirectory(vchObjectPath, directory)) {
        return false;
    }
    
    if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= directory.nExpireTime) {
        directory.SetNull();
        return false;
    }
    return true;
}

bool CDirectoryDB::AddDirectory(const CDirectory& directory, const int& op) 
{ 
    bool writeState = false;
    {
        LOCK(cs_bdap_directory);
        writeState = Write(make_pair(std::string("domain_component"), directory.GetFullObjectPath()), directory) 
                         && Write(make_pair(std::string("domain_wallet_address"), directory.WalletAddress), directory.GetFullObjectPath());
    }
    if (writeState)
        AddDirectoryIndex(directory, op);

    return writeState;
}

void CDirectoryDB::AddDirectoryIndex(const CDirectory& directory, const int& op) 
{
    UniValue oName(UniValue::VOBJ);
    if (BuildBDAPJson(directory, oName)) {
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_new");
        WriteDirectoryIndexHistory(directory, op);
    }
}

bool CDirectoryDB::ReadDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Read(make_pair(std::string("domain_component"), vchObjectPath), directory);
}

bool CDirectoryDB::ReadDirectoryAddress(const std::vector<unsigned char>& vchAddress, std::vector<unsigned char>& vchObjectPath) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Read(make_pair(std::string("domain_wallet_address"), vchAddress), vchObjectPath);
}

bool CDirectoryDB::EraseDirectory(const std::vector<unsigned char>& vchObjectPath) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Erase(make_pair(std::string("domain_component"), vchObjectPath));
}

bool CDirectoryDB::EraseDirectoryAddress(const std::vector<unsigned char>& vchAddress) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Erase(make_pair(std::string("domain_wallet_address"), vchAddress));
}

bool CDirectoryDB::DirectoryExists(const std::vector<unsigned char>& vchObjectPath)
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Exists(make_pair(std::string("domain_component"), vchObjectPath));
}

bool CDirectoryDB::DirectoryExistsAddress(const std::vector<unsigned char>& vchAddress) 
{
    LOCK(cs_bdap_directory);
    return CDBWrapper::Exists(make_pair(std::string("domain_wallet_address"), vchAddress));
}

bool CDirectoryDB::RemoveExpired(int& entriesRemoved)
{
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CDirectory directory;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "domain_component") {
                pcursor->GetValue(directory);
                  if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= directory.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDirectory(key.second);
                } 
                
            }
            else if (pcursor->GetKey(key) && key.first == "domain_wallet_address") {
                std::vector<unsigned char> value;
                CDirectory directory;
                pcursor->GetValue(value);
                if (GetDirectory(value, directory) && (unsigned int)chainActive.Tip()->GetMedianTimePast() >= directory.nExpireTime)
                {
                    entriesRemoved++;
                    EraseDirectoryAddress(directory.WalletAddress);
                }

            }
            pcursor->Next();
        } catch (std::exception &e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

void CDirectoryDB::WriteDirectoryIndexHistory(const CDirectory& directory, const int &op) 
{
    if (IsArgSet("-zmqpubbdaphistory")) {
        UniValue oName(UniValue::VOBJ);
        BuildDirectoryIndexerHistoryJson(directory, oName);
        oName.push_back(Pair("op", directoryFromOp(op)));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_history");
    }
}

void CDirectoryDB::WriteDirectoryIndex(const CDirectory& directory, const int &op) 
{
    if (IsArgSet("-zmqpubbdaprecord")) {
        UniValue oName(UniValue::VOBJ);
        oName.push_back(Pair("_id", stringFromVch(directory.OID)));
        CDynamicAddress address(EncodeBase58(directory.WalletAddress));
        oName.push_back(Pair("address", address.ToString()));
        oName.push_back(Pair("expires_on", directory.nExpireTime));
        oName.push_back(Pair("encryption_publickey", HexStr(directory.EncryptPublicKey)));
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "bdap_record");
    }
    WriteDirectoryIndexHistory(directory, op);
}

bool CDirectoryDB::UpdateDirectory(const std::vector<unsigned char>& vchObjectPath, CDirectory& directory)
{
    return true; //TODO (bdap): add update impl
}

bool CDirectoryDB::UpdateDirectoryAddress(const std::vector<unsigned char>& vchAddress, CDirectory& directory)
{
    return true; //TODO (bdap): add update impl
}
