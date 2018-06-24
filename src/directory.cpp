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
#include "wallet/wallet.h"

#include <univalue.h>

CDirectoryDB *pDirectoryDB = NULL;

bool IsDirectoryTransaction(CScript txOut) {
    return (txOut.IsDirectoryScript(BDAP_NEW_TX)
            || txOut.IsDirectoryScript(BDAP_DELETE_TX)
            || txOut.IsDirectoryScript(BDAP_ACTIVATE_TX)
            || txOut.IsDirectoryScript(BDAP_MODIFY_TX)
            || txOut.IsDirectoryScript(BDAP_MODIFY_RDN_TX)
            || txOut.IsDirectoryScript(BDAP_EXECUTE_CODE_TX)
            || txOut.IsDirectoryScript(BDAP_BIND_TX)
            || txOut.IsDirectoryScript(BDAP_REVOKE_TX)
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

CDynamicAddress CDirectory::GetWalletAddress() const {
    return CDynamicAddress(stringFromVch(WalletAddress));
}

std::string CDirectory::GetFullObjectPath() const {
    return stringFromVch(ObjectID) + "@" + stringFromVch(OrganizationalUnit) + "." + stringFromVch(DomainComponent);
}

std::vector<unsigned char> CDirectory::vchFullObjectPath() const {
    std::string strFullObjectPath = GetFullObjectPath();
    std::vector<unsigned char> vchReturnValue(strFullObjectPath.begin(), strFullObjectPath.end());
    return vchReturnValue;
}

void CDirectoryDB::AddDirectoryIndex(const CDirectory& directory, const int& op) {
    UniValue oName(UniValue::VOBJ);
    if (BuildBDAPJson(directory, oName)) {
        GetMainSignals().NotifyBDAPUpdate(oName.write().c_str(), "add.directory");
        //WriteDirectoryIndexHistory(directory, op);  //TODO: implement local leveldb storage.
    }
}

bool BuildBDAPJson(const CDirectory& directory, UniValue& oName)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    oName.push_back(Pair("_id", stringFromVch(directory.OID)));
    oName.push_back(Pair("version", directory.nVersion));
    oName.push_back(Pair("domain_component", stringFromVch(directory.DomainComponent)));
    oName.push_back(Pair("common_name", stringFromVch(directory.CommonName)));
    oName.push_back(Pair("organizational_unit", stringFromVch(directory.OrganizationalUnit)));
    oName.push_back(Pair("organization_name", stringFromVch(directory.DomainComponent)));
    oName.push_back(Pair("object_id", stringFromVch(directory.ObjectID)));
    oName.push_back(Pair("object_full_path", stringFromVch(directory.vchFullObjectPath())));
    oName.push_back(Pair("object_type", directory.ObjectType));
    oName.push_back(Pair("wallet_address", stringFromVch(directory.WalletAddress)));
    oName.push_back(Pair("signature_address", stringFromVch(directory.SignWalletAddress)));
    oName.push_back(Pair("public", (int)directory.fPublicObject));
    oName.push_back(Pair("encryption_publickey", HexStr(directory.EncryptPublicKey)));
    oName.push_back(Pair("encryption_privatekey", stringFromVch(directory.EncryptPrivateKey)));
    oName.push_back(Pair("sigatures_required", (int)directory.nSigaturesRequired));
    oName.push_back(Pair("resource_pointer", stringFromVch(directory.ResourcePointer)));
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

void CreateRecipient(const CScript& scriptPubKey, CRecipient& recipient)
{
    CRecipient recp = {scriptPubKey, recipient.nAmount, false};
    recipient = recp;
    CTxOut txout(recipient.nAmount, scriptPubKey);
    size_t nSize = GetSerializeSize(txout, SER_DISK, 0) + 148u;
    recipient.nAmount = 3 * minRelayTxFee.GetFee(nSize);
}

void CreateFeeRecipient(CScript& scriptPubKey, const std::vector<unsigned char>& data, CRecipient& recipient)
{
    // add hash to data output (must match hash in inputs check with the tx scriptpubkey hash)
    uint256 hash = Hash(data.begin(), data.end());
    std::vector<unsigned char> vchHashRand = vchFromValue(hash.GetHex());
    scriptPubKey << vchHashRand;
    CRecipient recp = {scriptPubKey, 0, false};
    recipient = recp;
}

CAmount GetDataFee(const CScript& scriptPubKey)
{
    CAmount nFee = 0;
    CRecipient recp = {scriptPubKey, 0, false};
    CTxOut txout(0, scriptPubKey);
    size_t nSize = GetSerializeSize(txout, SER_DISK,0)+148u;
    nFee = CWallet::GetMinimumFee(nSize, nTxConfirmTarget, mempool);
    recp.nAmount = nFee;
    return recp.nAmount;
}

void ToLowerCase(CharString& vchValue) {
    std::string strValue;
    CharString::const_iterator vi = vchValue.begin();
    while (vi != vchValue.end()) 
    {
        strValue += std::tolower(*vi);
        vi++;
    }
    CharString vchNewValue(strValue.begin(), strValue.end());
    std::swap(vchValue, vchNewValue);
}

void ToLowerCase(std::string& strValue) {
    for(unsigned short loop=0;loop < strValue.size();loop++)
    {
        strValue[loop]=std::tolower(strValue[loop]);
    }
}

bool CheckIfNameExists(const CharString& vchObjectID, const CharString& vchOrganizationalUnit, const CharString& vchDomainComponent) {


    return false;
}

CAmount GetBDAPFee(const CScript& scriptPubKey)
{
    CAmount nFee = 0;
    CRecipient recp = {scriptPubKey, 0, false};
    CTxOut txout(0, scriptPubKey);
    size_t nSize = GetSerializeSize(txout, SER_DISK,0)+148u;
    nFee = CWallet::GetMinimumFee(nSize, nTxConfirmTarget, mempool);
    recp.nAmount = nFee;
    return recp.nAmount;
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