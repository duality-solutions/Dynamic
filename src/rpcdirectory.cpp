// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "directory.h"

#include "rpcprotocol.h"
#include "rpcserver.h"

#include <univalue.h>


UniValue directoryadd(const UniValue& params, bool fHelp) {
    if (fHelp || params.size() != 1) {
        throw std::runtime_error("directoryadd <directoryname>\nAdd directory to blockchain.\n");
    }

    std::vector<unsigned char> vchDirectoryName = vchFromValue(params[0]);
    CDirectory txDirectory;
    if (!pDirectoryDB || !pDirectoryDB->AddDirectory(txDirectory, OP_DIRECTORY_NEW))
        throw std::runtime_error("Failed to read from BDAP database");
	
    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("Failed to read from BDAP JSON object");
        
    return oName;
}

UniValue directorylist(const UniValue& params, bool fHelp) {
    if (fHelp || params.size() != 1) {
        throw std::runtime_error("directoryadd <directoryname>\nAdd directory to blockchain.\n");
    }

    std::vector<unsigned char> vchDirectoryName = vchFromValue(params[0]);
    CDirectory txDirectory;
    if (!pDirectoryDB || !pDirectoryDB->AddDirectory(txDirectory, OP_DIRECTORY_NEW))
        throw std::runtime_error("Failed to read from BDAP database");
	
    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("Failed to read from BDAP JSON object");
        
    return oName;
}

static const CRPCCommand commands[] =
{   //  category         name                        actor (function)           okSafeMode
#ifdef ENABLE_WALLET
    /* BDAP */
    { "bdap",            "directoryadd",             &directoryadd,                 true  },
    { "bdap",            "directorylist",            &directorylist,                true  },
#endif //ENABLE_WALLET
};

void RegisterDirectoryRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}