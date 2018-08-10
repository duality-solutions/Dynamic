// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"

#include "coins.h"
#include "fluid.h"
#include "policy/policy.h"
#include "rpcclient.h"
#include "rpcserver.h"
#include "validation.h"
#include "validationinterface.h"
#include "wallet/wallet.h"

#include <univalue.h>

#include <boost/algorithm/string/find.hpp>
#include <boost/xpressive/xpressive_dynamic.hpp>

using namespace boost::xpressive;

bool IsDomainEntryTransaction(const CScript& txOut)
{
    return (txOut.IsBDAPScript(BDAP_START)
            || txOut.IsBDAPScript(BDAP_NEW_TX)
            || txOut.IsBDAPScript(BDAP_DELETE_TX)
            || txOut.IsBDAPScript(BDAP_ACTIVATE_TX)
            || txOut.IsBDAPScript(BDAP_MODIFY_TX)
            || txOut.IsBDAPScript(BDAP_MODIFY_RDN_TX)
            || txOut.IsBDAPScript(BDAP_EXECUTE_CODE_TX)
            || txOut.IsBDAPScript(BDAP_BIND_TX)
            || txOut.IsBDAPScript(BDAP_REVOKE_TX)
           );
}

std::string DomainEntryFromOp(const int op) 
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
            return "<unknown bdap op>";
    }
}

bool IsDomainEntryDataOutput(const CTxOut& out) {
   txnouttype whichType;
    if (!IsStandard(out.scriptPubKey, whichType))
        return false;
    if (whichType == TX_NULL_DATA)
        return true;
   return false;
}

bool GetDomainEntryTransaction(int nHeight, const uint256& hash, CTransaction& txOut, const Consensus::Params& consensusParams)
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

std::vector<unsigned char> vchFromString(const std::string& str) 
{
    return std::vector<unsigned char>(str.begin(), str.end());
}

int GetDomainEntryDataOutput(const CTransaction& tx) {
   for(unsigned int i = 0; i<tx.vout.size();i++) {
       if(IsDomainEntryDataOutput(tx.vout[i]))
           return i;
    }
   return -1;
}

bool GetDomainEntryData(const CScript& scriptPubKey, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash)
{
    CScript::const_iterator pc = scriptPubKey.begin();
    opcodetype opcode;
    if (!scriptPubKey.GetOp(pc, opcode))
        return false;
    if(opcode != OP_RETURN)
        return false;
    if (!scriptPubKey.GetOp(pc, opcode, vchData))
        return false;

    uint256 hash;
    hash = Hash(vchData.begin(), vchData.end());
    vchHash = vchFromValue(hash.GetHex());

    return true;
}

bool GetDomainEntryData(const CTransaction& tx, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash, int& nOut)
{
    nOut = GetDomainEntryDataOutput(tx);
    if(nOut == -1)
       return false;

    const CScript &scriptPubKey = tx.vout[nOut].scriptPubKey;
    return GetDomainEntryData(scriptPubKey, vchData, vchHash);
}

bool CDomainEntry::UnserializeFromTx(const CTransaction& tx) {
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetDomainEntryData(tx, vchData, vchHash, nOut))
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

void CDomainEntry::Serialize(std::vector<unsigned char>& vchData) {
    CDataStream dsBDAP(SER_NETWORK, PROTOCOL_VERSION);
    dsBDAP << *this;
    vchData = std::vector<unsigned char>(dsBDAP.begin(), dsBDAP.end());
}

bool CDomainEntry::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) {
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

CDynamicAddress CDomainEntry::GetWalletAddress() const {
    return CDynamicAddress(stringFromVch(WalletAddress));
}

std::string CDomainEntry::GetFullObjectPath() const {
    return stringFromVch(ObjectID) + "@" + stringFromVch(OrganizationalUnit) + "." + stringFromVch(DomainComponent);
}

std::string CDomainEntry::GetObjectLocation() const {
    return stringFromVch(OrganizationalUnit) + "." + stringFromVch(DomainComponent);
}

std::vector<unsigned char> CDomainEntry::vchFullObjectPath() const {
    std::string strFullObjectPath = GetFullObjectPath();
    std::vector<unsigned char> vchReturnValue(strFullObjectPath.begin(), strFullObjectPath.end());
    return vchReturnValue;
}

std::vector<unsigned char> CDomainEntry::vchObjectLocation() const {
    std::string strObjectLocation = GetObjectLocation();
    std::vector<unsigned char> vchReturnValue(strObjectLocation.begin(), strObjectLocation.end());
    return vchReturnValue;
}

void CDomainEntry::AddCheckpoint(const uint32_t& height, const CharString& vchHash) 
{
    std::pair<uint32_t, CharString> pairNewCheckpoint;
    pairNewCheckpoint.first = height;
    pairNewCheckpoint.second = vchHash;
    CheckpointHashes.push_back(pairNewCheckpoint);
}

bool CDomainEntry::ValidateValues(std::string& errorMessage)
{
    smatch sMatch;
    std::string regExWithDot = "^((?!-)[a-z0-9-]{2," + std::to_string(MAX_OBJECT_NAME_LENGTH) + "}(?<!-)\\.)+[a-z]{2,6}$";
    std::string regExWithOutDot = "^((?!-)[a-z0-9-]{2," + std::to_string(MAX_OBJECT_NAME_LENGTH) + "}(?<!-))";

    // check domain name component
    std::string strDomainComponent = stringFromVch(DomainComponent);
    if (boost::find_first(strDomainComponent, "."))
    {
        sregex regexValidName = sregex::compile(regExWithDot);
        if (!regex_search(strDomainComponent, sMatch, regexValidName) || std::string(sMatch[0]) != strDomainComponent) {
            errorMessage = "Invalid BDAP domain name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }  
    }
    else
    {
        sregex regexValidName = sregex::compile(regExWithOutDot);
        if (!regex_search(strDomainComponent, sMatch, regexValidName) || std::string(sMatch[0]) != strDomainComponent) {
            errorMessage = "Invalid BDAP domain name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }
    }

    // check organizational unit component
    std::string strOrganizationalUnit = stringFromVch(OrganizationalUnit);
    if (boost::find_first(strOrganizationalUnit, "."))
    {
        sregex regexValidName = sregex::compile(regExWithDot);
        if (!regex_search(strOrganizationalUnit, sMatch, regexValidName) || std::string(sMatch[0]) != strOrganizationalUnit) {
            errorMessage = "Invalid BDAP organizational unit. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }  
    }
    else
    {
        sregex regexValidName = sregex::compile(regExWithOutDot);
        if (!regex_search(strOrganizationalUnit, sMatch, regexValidName) || std::string(sMatch[0]) != strOrganizationalUnit) {
            errorMessage = "Invalid BDAP organizational unit. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }
    }

    // check object name component
    std::string strObjectID = stringFromVch(ObjectID);
    if (boost::find_first(strObjectID, "."))
    {
        sregex regexValidName = sregex::compile(regExWithDot);
        if (!regex_search(strObjectID, sMatch, regexValidName) || std::string(sMatch[0]) != strObjectID) {
            errorMessage = "Invalid BDAP object name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }  
    }
    else
    {
        sregex regexValidName = sregex::compile(regExWithOutDot);
        if (!regex_search(strObjectID, sMatch, regexValidName) || std::string(sMatch[0]) != strObjectID) {
            errorMessage = "Invalid BDAP object name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }
    }

    // check object common name component
    std::string strCommonName = stringFromVch(CommonName);
    if (strCommonName.length() > MAX_COMMON_NAME_LENGTH) 
    {
        errorMessage = "Invalid BDAP common name. Can not have more than " + std::to_string(MAX_COMMON_NAME_LENGTH) + " characters.";
        return false;
    }

    // check object organization name component
    std::string strOrganizationName = stringFromVch(OrganizationName);
    if (strOrganizationName.length() > MAX_ORG_NAME_LENGTH) 
    {
        errorMessage = "Invalid BDAP organization name. Can not have more than " + std::to_string(MAX_ORG_NAME_LENGTH) + " characters.";
        return false;
    }

    // check resource pointer component
    std::string strResourcePointer = stringFromVch(ResourcePointer);
    if (strResourcePointer.length() > MAX_RESOURCE_POINTER_LENGTH) 
    {
        errorMessage = "Invalid BDAP resource pointer. Can not have more than " + std::to_string(MAX_RESOURCE_POINTER_LENGTH) + " characters.";
        return false;
    }

    // check private data component
    std::string strPrivateData = stringFromVch(PrivateData);
    if (strPrivateData.length() > MAX_PRIVATE_DATA_LENGTH) 
    {
        errorMessage = "Invalid BDAP private data. Can not have more than " + std::to_string(MAX_PRIVATE_DATA_LENGTH) + " characters.";
        return false;
    }

    // make sure the number of checkpoints does not exceed limit
    if (CheckpointHashes.size() > MAX_NUMBER_CHECKPOINTS) 
    {
        errorMessage = "Invalid BDAP checkpoint count. Can not have more than " + std::to_string(MAX_NUMBER_CHECKPOINTS) + " checkpoints for one entry.";
        return false;
    }
    // TODO: (bdap) check if EncryptPublicKey is valid
    // TODO: (bdap) check WalletAddress and SignWalletAddress
    return true;
}

bool BuildBDAPJson(const CDomainEntry& entry, UniValue& oName, bool fAbridged)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    if (!fAbridged) {
        oName.push_back(Pair("_id", stringFromVch(entry.OID)));
        oName.push_back(Pair("version", entry.nVersion));
        oName.push_back(Pair("domain_component", stringFromVch(entry.DomainComponent)));
        oName.push_back(Pair("common_name", stringFromVch(entry.CommonName)));
        oName.push_back(Pair("organizational_unit", stringFromVch(entry.OrganizationalUnit)));
        oName.push_back(Pair("organization_name", stringFromVch(entry.DomainComponent)));
        oName.push_back(Pair("object_id", stringFromVch(entry.ObjectID)));
        oName.push_back(Pair("object_full_path", stringFromVch(entry.vchFullObjectPath())));
        oName.push_back(Pair("object_type", entry.ObjectType));
        oName.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
        oName.push_back(Pair("signature_address", stringFromVch(entry.SignWalletAddress)));
        oName.push_back(Pair("public", (int)entry.fPublicObject));
        oName.push_back(Pair("encryption_publickey", HexStr(entry.EncryptPublicKey)));
        oName.push_back(Pair("sigatures_required", (int)entry.nSigaturesRequired));
        oName.push_back(Pair("resource_pointer", stringFromVch(entry.ResourcePointer)));
        oName.push_back(Pair("txid", entry.txHash.GetHex()));
        if ((unsigned int)chainActive.Height() >= entry.nHeight-1) {
            CBlockIndex *pindex = chainActive[entry.nHeight-1];
            if (pindex) {
                nTime = pindex->GetMedianTimePast();
            }
        }
        oName.push_back(Pair("time", nTime));
        //oName.push_back(Pair("height", entry.nHeight));
        expired_time = entry.nExpireTime;
        if(expired_time <= (unsigned int)chainActive.Tip()->GetMedianTimePast())
        {
            expired = true;
        }
        oName.push_back(Pair("expires_on", expired_time));
        oName.push_back(Pair("expired", expired));
        oName.push_back(Pair("private_data", stringFromVch(entry.PrivateData)));
        oName.push_back(Pair("transaction_fee", entry.transactionFee));
        oName.push_back(Pair("registration_fee", entry.registrationFeePerDay));
    }
    else {
        oName.push_back(Pair("common_name", stringFromVch(entry.CommonName)));
        oName.push_back(Pair("object_full_path", stringFromVch(entry.vchFullObjectPath())));
        oName.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
    }
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

bool DecodeDomainEntryTx(const CTransaction& tx, int& op, std::vector<std::vector<unsigned char> >& vvch) 
{
    bool found = false;

    for (unsigned int i = 0; i < tx.vout.size(); i++) {
        const CTxOut& out = tx.vout[i];
        vchCharString vvchRead;
        if (DecodeBDAPScript(out.scriptPubKey, op, vvchRead)) {
            found = true;
            vvch = vvchRead;
            break;
        }
    }
    if (!found)
        vvch.clear();

    return found;
}

bool FindDomainEntryInTx(const CCoinsViewCache &inputs, const CTransaction& tx, std::vector<std::vector<unsigned char> >& vvch)
{
    for (unsigned int i = 0; i < tx.vin.size(); i++) {
        const Coin& prevCoins = inputs.AccessCoin(tx.vin[i].prevout);
        if (prevCoins.IsSpent()) {
            continue;
        }
        // check unspent input for consensus before adding to a block
        int op;
        if (DecodeBDAPScript(prevCoins.out.scriptPubKey, op, vvch)) {
            return true;
        }
    }
    return false;
}

int GetDomainEntryOpType(const CScript& script)
{
    std::string ret;
    CScript::const_iterator it = script.begin();
    opcodetype op1 = OP_INVALIDOPCODE;
    opcodetype op2 = OP_INVALIDOPCODE;
    while (it != script.end()) {
        std::vector<unsigned char> vch;
        if (op1 == OP_INVALIDOPCODE)
        {
            if (script.GetOp2(it, op1, &vch)) 
            {
                if (op1 - OP_1NEGATE - 1 == OP_BDAP)
                {
                    continue;
                }
                else
                {
                    return 0;
                }
            }
        }
        else
        {
            if (script.GetOp2(it, op2, &vch)) 
            {
                if (op2 - OP_1NEGATE - 1  > OP_BDAP && op2 - OP_1NEGATE - 1 <= OP_BDAP_REVOKE)
                {
                    return (int)op2 - OP_1NEGATE - 1;
                }
                else
                {
                    return -1;
                }
            }
        }
    }
    return (int)op2;
}

std::string GetDomainEntryOpTypeString(const CScript& script)
{
    return DomainEntryFromOp(GetDomainEntryOpType(script));
}

bool GetDomainEntryOpScript(const CTransaction& tx, CScript& scriptDomainEntryOp, vchCharString& vvchOpParameters, int& op)
{
    for (unsigned int i = 0; i < tx.vout.size(); i++) 
    {
        const CTxOut& out = tx.vout[i];
        if (DecodeBDAPScript(out.scriptPubKey, op, vvchOpParameters)) 
        {
            scriptDomainEntryOp = out.scriptPubKey;
            return true;
        }
    }
    return false;
}