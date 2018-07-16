// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/directorydb.h"

#include "validationinterface.h"

#include <univalue.h>

CDirectoryDB *pDirectoryDB = NULL;

bool CDirectoryDB::AddDirectory(const CDirectory& directory, const int& op) { 
    bool writeState = Write(make_pair(std::string("domain_component"), directory.DomainComponent), directory) 
                         && Write(make_pair(std::string("domain_wallet_address"), directory.WalletAddress), directory.DomainComponent);

    AddDirectoryIndex(directory, op);
    return writeState;
}

void CDirectoryDB::AddDirectoryIndex(const CDirectory& directory, const int& op) {
    UniValue oName(UniValue::VOBJ);
    if (BuildBDAPJson(directory, oName)) {
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "add.directory");
        //WriteDirectoryIndexHistory(directory, op);  //TODO: implement local leveldb storage.
    }
}