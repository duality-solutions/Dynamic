// Copyright (c) 2018 Duality Blockchain Solutions Developers 
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "directory.h"

#include "rpcprotocol.h"
#include "rpcserver.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"

#include <univalue.h>

#include <boost/xpressive/xpressive_dynamic.hpp>

using namespace boost::xpressive;

extern void SendCustomTransaction(const CScript generatedScript, CWalletTx& wtxNew, CAmount nValue, bool fUseInstantSend=false, bool fIsBDAP=false);

UniValue addpublicname(const JSONRPCRequest& request) {
    if (request.params.size() != 2) {
        throw std::runtime_error("addpublicname <userid> <common name>\nAdd public name entry to blockchain directory.\n");
    }

    EnsureWalletIsUnlocked();

    // Adds a new name to channel zero.  OID = 0.0.block-height.tx-ordinal.0.0.0.0
    // Format object and domain names to lower case.
    std::string strObjectID = request.params[0].get_str();
    ToLowerCase(strObjectID);
    // Check if the object name is valid.
    sregex regexValidName = sregex::compile("^((?!-)[a-z0-9-]{2,64}(?<!-)\\.)+[a-z]{2,6}$");
    smatch sMatch;
    //if (!regex_search(strObjectID, sMatch, regexValidName))
    //    throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3500 - " + _("Invalid BDAP name.  Regular expression failed."));

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);
    CharString vchCommonName = vchFromValue(request.params[1]);

    CharString vchDomainComponent(DEFAULT_PUBLIC_DOMAIN.begin(), DEFAULT_PUBLIC_DOMAIN.end());
    CharString vchOrganizationalUnit(DEFAULT_PUBLIC_OU.begin(), DEFAULT_PUBLIC_OU.end());
    CharString vchOrganizationName (DEFAULT_ORGANIZATION_NAME.begin(), DEFAULT_ORGANIZATION_NAME.end());
    CharString vchOID (DEFAULT_OID_PREFIX.begin(), DEFAULT_OID_PREFIX.end());
    // Check if name already exists
    if (CheckIfNameExists(vchObjectID, vchOrganizationalUnit, vchDomainComponent))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3501 - " + _("This public name already exists"));

    CDirectory txDirectory;
    txDirectory.OID = vchOID;
    txDirectory.DomainComponent = vchDomainComponent;
    txDirectory.OrganizationalUnit = vchOrganizationalUnit;
    txDirectory.CommonName = vchCommonName;
    txDirectory.OrganizationName = vchOrganizationName;
    txDirectory.ObjectID = vchObjectID;
    txDirectory.fPublicObject = 1; //make entry public

    // TODO: Add ability to pass in the wallet address and public key
    CKey privWalletKey;
    privWalletKey.MakeNewKey(true);
    CPubKey pubWalletKey = privWalletKey.GetPubKey();
    CKeyID keyWalletID = pubWalletKey.GetID();
    
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privWalletKey, pubWalletKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3502 - " + _("Error adding receiving address key wo wallet for BDAP"));

    pwalletMain->SetAddressBook(keyWalletID, strObjectID, "receive");
    CDynamicAddress payWallet = CDynamicAddress(keyWalletID);
    std::string strWalletAddress = payWallet.ToString();
    CharString vchWalletAddress(strWalletAddress.begin(), strWalletAddress.end());
    txDirectory.WalletAddress = vchWalletAddress;

    CKey privEncryptKey;
    privEncryptKey.MakeNewKey(true);
    CPubKey pubEncryptKey = privEncryptKey.GetPubKey();
    std::string strPrivateEncryptKey = CDynamicSecret(privEncryptKey).ToString();
    CharString vchEncryptPriKey(strPrivateEncryptKey.begin(), strPrivateEncryptKey.end());
    CharString vchEncryptPubKey(pubEncryptKey.begin(), pubEncryptKey.end());
    if (pwalletMain && !pwalletMain->AddKeyPubKey(privEncryptKey, pubEncryptKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3503 - " + _("Error adding encrypt key to wallet for BDAP"));

    txDirectory.EncryptPublicKey = vchEncryptPubKey;
    txDirectory.EncryptPrivateKey = vchEncryptPriKey;

    CKey privSignKey;
    privSignKey.MakeNewKey(true);
    CPubKey pubSignKey = privSignKey.GetPubKey();
    CKeyID keySignID = pubSignKey.GetID();
    std::string strSignWalletAddress = CDynamicAddress(keySignID).ToString();
    CharString vchSignWalletAddress(strSignWalletAddress.begin(), strSignWalletAddress.end());

    if (pwalletMain && !pwalletMain->AddKeyPubKey(privSignKey, pubSignKey))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3504 - " + _("Error adding signature key to wallet for BDAP"));

    txDirectory.SignWalletAddress = vchSignWalletAddress;

    if (!pDirectoryDB || !pDirectoryDB->AddDirectory(txDirectory, OP_BDAP_NEW))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3505 - " + _("Failed to read from BDAP database"));

    CharString data;
    txDirectory.Serialize(data);
    
    // Create BDAP OP_RETURN Signature Scripts
    CScript scriptPubKey;
    std::vector<unsigned char> vchFullObjectPath = txDirectory.vchFullObjectPath();
    scriptPubKey << CScript::EncodeOP_N(OP_BDAP_NEW) << vchFullObjectPath << OP_2DROP;
    CScript scriptDestination;
    scriptDestination = GetScriptForDestination(payWallet.Get());
    scriptPubKey += scriptDestination;

    CScript scriptData;
    scriptData << OP_RETURN << data;
    scriptPubKey += scriptData;

    // Send the transaction
    float fYears = 1.0; //TODO use a variable for registration years.
    CWalletTx wtx;
    SendCustomTransaction(scriptPubKey, wtx, GetBDAPFee(scriptPubKey) * powf(3.1, fYears), true);
    txDirectory.txHash = wtx.GetHash();

    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("BDAP_ADD_PUBLIC_NAME_RPC_ERROR: ERRCODE: 3506 - " + _("Failed to read from BDAP JSON object"));

    return oName;
}

UniValue directorylist(const JSONRPCRequest& request) {
    if (request.params.size() != 1) {
        throw std::runtime_error("directorylist <directoryname>\nAdd directory to blockchain.\n");
    }

    std::vector<unsigned char> vchDirectoryName = vchFromValue(request.params[0]);
    CDirectory txDirectory;
    if (!pDirectoryDB || !pDirectoryDB->AddDirectory(txDirectory, OP_BDAP_NEW))
        throw std::runtime_error("Failed to read from BDAP database");
	
    UniValue oName(UniValue::VOBJ);
    if(!BuildBDAPJson(txDirectory, oName))
        throw std::runtime_error("Failed to read from BDAP JSON object");
        
    return oName;
}

UniValue directoryupdate(const JSONRPCRequest& request) {
    if (request.params.size() != 1) {
        throw std::runtime_error("directoryupdate <directoryname>\nAdd directory to blockchain.\n");
    }

    std::vector<unsigned char> vchDirectoryName = vchFromValue(request.params[0]);
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