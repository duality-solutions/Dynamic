// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "directory.h"

#include "fluid.h"
#include "policy/policy.h"
#include "rpcclient.h"
#include "rpcserver.h"
#include "validation.h"
#include "validationinterface.h"

#include <univalue.h>

CDirectoryDB *pDirectoryDB = NULL;

bool IsDirectoryTransaction(CScript txOut) {
    return (txOut.IsDirectoryScript(DIRECTORY_NEW_TX)
            || txOut.IsDirectoryScript(DIRECTORY_UPDATE_TX)
            || txOut.IsDirectoryScript(DIRECTORY_DELETE_TX)
            || txOut.IsDirectoryScript(DIRECTORY_ACTIVATE_TX)
           );
}

bool IsDirectoryDataOutput(const CTxOut& out) {
   txnouttype whichType;
    if (!IsStandard(out.scriptPubKey, whichType))
        return false;
    if (whichType == TX_NULL_DATA)
        return true;
   return false;
}

bool GetDirectoryTransaction(int nHeight, const uint256& hash, CTransaction& txOut, const Consensus::Params& consensusParams)
{
    if(nHeight < 0 || nHeight > chainActive.Height())
        return false;
    CBlockIndex *pindexSlow = NULL; 
    LOCK(cs_main);
    pindexSlow = chainActive[nHeight];
    if (pindexSlow) {
        CBlock block;
        if (ReadBlockFromDisk(block, pindexSlow, consensusParams)) {
            BOOST_FOREACH(const CTransaction &tx, block.vtx) {
                if (tx.GetHash() == hash) {
                    txOut = tx;
                    return true;
                }
            }
        }
    }
    return false;
}

std::string stringFromVch(const CharString& vch) {
    std::string res;
    std::vector<unsigned char>::const_iterator vi = vch.begin();
    while (vi != vch.end()) {
        res += (char) (*vi);
        vi++;
    }
    return res;
}

std::vector<unsigned char> vchFromValue(const UniValue& value) {
    std::string strName = value.get_str();
    unsigned char *strbeg = (unsigned char*) strName.c_str();
    return std::vector<unsigned char>(strbeg, strbeg + strName.size());
}

int GetDirectoryDataOutput(const CTransaction& tx) {
   for(unsigned int i = 0; i<tx.vout.size();i++) {
       if(IsDirectoryDataOutput(tx.vout[i]))
           return i;
    }
   return -1;
}

bool GetDirectoryData(const CScript& scriptPubKey, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash)
{
    CScript::const_iterator pc = scriptPubKey.begin();
    opcodetype opcode;
    if (!scriptPubKey.GetOp(pc, opcode))
        return false;
    if(opcode != OP_RETURN)
        return false;
    if (!scriptPubKey.GetOp(pc, opcode, vchData))
        return false;
    if (!scriptPubKey.GetOp(pc, opcode, vchHash))
        return false;
    return true;
}

bool GetDirectoryData(const CTransaction& tx, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash, int& nOut)
{
    nOut = GetDirectoryDataOutput(tx);
    if(nOut == -1)
       return false;

    const CScript &scriptPubKey = tx.vout[nOut].scriptPubKey;
    return GetDirectoryData(scriptPubKey, vchData, vchHash);
}

bool CDirectory::UnserializeFromTx(const CTransaction& tx) {
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetDirectoryData(tx, vchData, vchHash, nOut))
    {
        SetNull();
        return false;
    }
    if(!UnserializeFromData(vchData, vchHash))
    {
        return false;
    }
    return true;
}

void CDirectory::Serialize(std::vector<unsigned char>& vchData) {
    CDataStream dsBDAP(SER_NETWORK, PROTOCOL_VERSION);
    dsBDAP << *this;
    vchData = std::vector<unsigned char>(dsBDAP.begin(), dsBDAP.end());
}

bool CDirectory::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) {
    try {
        CDataStream dsBDAP(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsBDAP >> *this;

        std::vector<unsigned char> vchBDAPData;
        Serialize(vchBDAPData);
        const uint256 &calculatedHash = Hash(vchBDAPData.begin(), vchBDAPData.end());
        const std::vector<unsigned char> &vchRandBDAP = vchFromValue(calculatedHash.GetHex());
        if(vchRandBDAP != vchHash)
        {
            SetNull();
            return false;
        }
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

void CDirectoryDB::AddDirectoryIndex(const CDirectory& directory, const int& op) {
    UniValue oName(UniValue::VOBJ);
    std::string domainName = stringFromVch(directory.DomainName);
    oName.push_back(Pair("_id", domainName));
    //CDyamicAddress address(EncodeBase58(directory.vchAddress));
    oName.push_back(Pair("domain_name", domainName));
    GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "add.directory");
    //WriteDirectoryIndexHistory(directory, op);
}

bool BuildBDAPJson(const CDirectory& directory, UniValue& oName)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    oName.push_back(Pair("_id", stringFromVch(directory.OID)));
    oName.push_back(Pair("version", directory.nVersion));
    oName.push_back(Pair("domain_name", stringFromVch(directory.DomainName)));
    oName.push_back(Pair("object_name", stringFromVch(directory.ObjectName)));
    oName.push_back(Pair("object_type", directory.ObjectType));
    oName.push_back(Pair("address", EncodeBase58(directory.WalletAddress)));
    oName.push_back(Pair("public", (int)directory.fPublicObject));
    //loop SignWalletAddresses
    //oName.push_back(Pair("sign address", EncodeBase58(directory.SignWalletAddresses)));

    oName.push_back(Pair("encryption_publickey", HexStr(directory.EncryptPublicKey)));
    oName.push_back(Pair("encryption_privatekey", HexStr(directory.EncryptPrivateKey)));
    //oName.push_back(Pair("sigatures_required", directory.nSigaturesRequired));
    oName.push_back(Pair("ipfs_address", stringFromVch(directory.IPFSAddress)));
    oName.push_back(Pair("txid", directory.txHash.GetHex()));
    if ((unsigned int)chainActive.Height() >= directory.nHeight-1) {
        CBlockIndex *pindex = chainActive[directory.nHeight-1];
        if (pindex) {
            nTime = pindex->GetMedianTimePast();
        }
    }
    oName.push_back(Pair("time", nTime));
    //oName.push_back(Pair("height", directory.nHeight));
    expired_time = directory.nExpireTime;
    if(expired_time <= chainActive.Tip()->GetMedianTimePast())
    {
        expired = true;
    }
    oName.push_back(Pair("expires_on", expired_time));
    oName.push_back(Pair("expired", expired));
    
    oName.push_back(Pair("certificate", stringFromVch(directory.Certificate)));
    oName.push_back(Pair("private_data", stringFromVch(directory.PrivateData)));
    //oName.push_back(Pair("transaction_fee", directory.transactionFee);
    //oName.push_back(Pair("registration_fee", directory.registrationFeePerDay);
    // loop CheckpointHashes
    return true;
}

/*
SignWalletAddresses.clear();
transactionFee = 0;
registrationFeePerDay = 0;
CheckpointHashes.clear();

std::vector<std::pair<std::string, std::vector<unsigned char>>> CIdentityParameters::InitialiseAdminOwners()
{


}
*/