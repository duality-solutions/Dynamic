// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "directory.h"

#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"

#include <univalue.h>


UniValue addpublicname(const UniValue& params, bool fHelp) {
    if (fHelp || params.size() != 1) {
        throw std::runtime_error("addpublicname <publicname>\nAdd public name entry to blockchain directory.\n");
    }
    CharString vchObjectName = vchFromValue(params[0]);
    CharString vchPrefix(INTERNAL_DOMAIN_PREFIX.begin(), INTERNAL_DOMAIN_PREFIX.end());
    CharString domainName(DEFAULT_PUBLIC_DOMAIN.begin(), DEFAULT_PUBLIC_DOMAIN.end());

    CDynamicAddress newAddress();

    CDirectory txDirectory;
    txDirectory.DomainName = domainName;
    txDirectory.ObjectName = vchObjectName;

    // TODO: Add ability to pass in the wallet address and public key
    CKey privKey;
    privKey.MakeNewKey(true);
    CPubKey pubKey = privKey.GetPubKey();
    CharString vchPriKey(privKey.begin(), privKey.end());
    CharString vchPubKey(pubKey.begin(), pubKey.end());
    CDynamicAddress addressBDAP(pubKey.GetID());
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privKey, pubKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3501 - " + _("Error adding key to wallet"));

    std::string strAddress = addressBDAP.ToString();
    CharString walletAddress(strAddress.begin(), strAddress.end());
    txDirectory.WalletAddress = walletAddress;
    txDirectory.EncryptPublicKey = vchPubKey;
    txDirectory.EncryptPrivateKey = vchPriKey;
    txDirectory.nVersion = BDAP_TX_VERSION;
    if (!pDirectoryDB || !pDirectoryDB->AddDirectory(txDirectory, OP_BDAP_NEW))
        throw std::runtime_error("Failed to read from BDAP database");

    UniValue oName(UniValue::VOBJ);
    
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("Failed to read from BDAP JSON object");

    CharString data; //TODO put serialized CDirectory in data
    CScript scriptPubKey;
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << vchObjectName << OP_2DROP;
    
    CScript scriptData;
    scriptData << OP_RETURN << data;

    std::vector<CRecipient> vecSend;
    CRecipient recipient;
    CreateRecipient(scriptPubKey, recipient);
    CMutableTransaction tx;
    tx.nVersion = BDAP_TX_VERSION;
    tx.vin.clear();
    tx.vout.clear();

    for (auto& recp : vecSend) {
        tx.vout.push_back(CTxOut(recp.nAmount, recp.scriptPubKey));
    }
    
    // create BDAP OP_RETURN transaction
    return oName;
}

UniValue directorylist(const UniValue& params, bool fHelp) {
    if (fHelp || params.size() != 1) {
        throw std::runtime_error("directorylist <directoryname>\nAdd directory to blockchain.\n");
    }

    std::vector<unsigned char> vchDirectoryName = vchFromValue(params[0]);
    CDirectory txDirectory;
    if (!pDirectoryDB || !pDirectoryDB->AddDirectory(txDirectory, OP_BDAP_NEW))
        throw std::runtime_error("Failed to read from BDAP database");
	
    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("Failed to read from BDAP JSON object");
        
    return oName;
}

UniValue directoryupdate(const UniValue& params, bool fHelp) {
    if (fHelp || params.size() != 1) {
        throw std::runtime_error("directoryupdate <directoryname>\nAdd directory to blockchain.\n");
    }

    std::vector<unsigned char> vchDirectoryName = vchFromValue(params[0]);
    CDirectory txDirectory;
    if (!pDirectoryDB || !pDirectoryDB->AddDirectory(txDirectory, OP_BDAP_NEW))
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
    { "bdap",            "addpublicname",            &addpublicname,                true  },
    { "bdap",            "directorylist",            &directorylist,                true  },
    { "bdap",            "directoryupdate",          &directoryupdate,              true  },
#endif //ENABLE_WALLET
};

void RegisterDirectoryRPCCommands(CRPCTable &tableRPC)
{
    for (unsigned int vcidx = 0; vcidx < ARRAYLEN(commands); vcidx++)
        tableRPC.appendCommand(commands[vcidx].name, &commands[vcidx]);
}