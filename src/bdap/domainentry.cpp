// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"

#include "chainparams.h"
#include "coins.h"
#include "policy/policy.h"
#include "rpcclient.h"
#include "rpcserver.h"
#include "txmempool.h"
#include "validation.h"
#include "validationinterface.h"
#include "wallet/wallet.h"

#include <univalue.h>

#include <boost/algorithm/string/find.hpp>
#include <boost/xpressive/xpressive_dynamic.hpp>

using namespace boost::xpressive;

namespace BDAP {
    std::string GetObjectTypeString(unsigned int nObjectType)
    {
        switch ((BDAP::ObjectType)nObjectType) {

            case BDAP::ObjectType::DEFAULT_TYPE:
                return "Default";
             case BDAP::ObjectType::USER_ACCOUNT:
                return "User Entry";
            case BDAP::ObjectType::GROUP:
                return "Group Entry";
            case BDAP::ObjectType::DEVICE_ACCOUNT:
                return "Device Entry";
            case BDAP::ObjectType::DOMAIN_ACCOUNT:
                return "Domain Entry";
            case BDAP::ObjectType::ORGANIZATIONAL_UNIT:
                return "OU Entry";
            case BDAP::ObjectType::CERTIFICATE:
                return "Certificate Entry";
            case BDAP::ObjectType::AUDIT:
                return "Audit Entry";
            case BDAP::ObjectType::CHANNEL:
                return "Channel Entry";
            case BDAP::ObjectType::CHECKPOINT:
                return "Channel Checkpoint Entry";
            case BDAP::ObjectType::BINDING_LINK:
                return "Binding Link Entry";
            case BDAP::ObjectType::IDENTITY:
                return "Identity Entry";
            case BDAP::ObjectType::IDENTITY_VERIFICATION:
                return "Identity Verification Entry";
            case BDAP::ObjectType::SMART_CONTRACT:
                return "Smart Contract Entry";
            default:
                return "Unknown";
        }
    }
    
    unsigned int GetObjectTypeInt(BDAP::ObjectType ObjectType)
    {
        return (unsigned int)ObjectType;
    }

    BDAP::ObjectType GetObjectTypeEnum(unsigned int nObjectType)
    {
        return (BDAP::ObjectType)nObjectType;
    }
}

std::string BDAPFromOp(const int op) 
{
    switch (op) {
        case OP_BDAP_NEW:
            return "bdap_new";
        case OP_BDAP_DELETE:
            return "bdap_delete";
        case OP_BDAP_REVOKE:
            return "bdap_revoke";
        case OP_BDAP_MODIFY:
            return "bdap_update";
        case OP_BDAP_MODIFY_RDN:
            return "bdap_move";
        case OP_BDAP_EXECUTE_CODE:
            return "bdap_execute";
        case OP_BDAP_BIND:
            return "bdap_link";
        case OP_BDAP_AUDIT:
            return "bdap_audit";
        case OP_BDAP_CERTIFICATE:
            return "bdap_certificate";
        case OP_BDAP_IDENTITY:
            return "bdap_identity";
        case OP_BDAP_ID_VERIFICATION:
            return "bdap_identity_verification";
        case OP_BDAP_CHANNEL:
            return "bdap_new_channel";
        case OP_BDAP_CHANNEL_CHECKPOINT:
            return "bdap_channel_checkpoint";
        default:
            return "<unknown bdap op>";
    }
}

bool IsBDAPDataOutput(const CTxOut& out) {
    txnouttype whichType;
    if (!IsStandard(out.scriptPubKey, whichType))
        return false;
    if (whichType == TX_NULL_DATA)
        return true;
   return false;
}

bool GetBDAPTransaction(int nHeight, const uint256& hash, CTransactionRef &txOut, const Consensus::Params& consensusParams)
{
    if(nHeight < 0 || nHeight > chainActive.Height())
        return false;

    CBlockIndex *pindexSlow = NULL;

    LOCK(cs_main);
    
    pindexSlow = chainActive[nHeight];
    
    if (pindexSlow) {
        CBlock block;
        if (ReadBlockFromDisk(block, pindexSlow, consensusParams)) {
                for (const auto& tx : block.vtx) {
                if (tx->GetHash() == hash) {
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

int GetBDAPDataOutput(const CTransactionRef& tx) {
   for(unsigned int i = 0; i < tx->vout.size();i++) {
       if(IsBDAPDataOutput(tx->vout[i]))
           return i;
    }
   return -1;
}

bool GetBDAPData(const CScript& scriptPubKey, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash)
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

bool GetBDAPData(const CTransactionRef& tx, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash, int& nOut)
{
    nOut = GetBDAPDataOutput(tx);
    if(nOut == -1)
       return false;

    const CScript &scriptPubKey = tx->vout[nOut].scriptPubKey;
    return GetBDAPData(scriptPubKey, vchData, vchHash);
}

bool GetBDAPData(const CTxOut& out, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash)
{
    return GetBDAPData(out.scriptPubKey, vchData, vchHash);
}

bool CDomainEntry::UnserializeFromTx(const CTransactionRef& tx) {
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetBDAPData(tx, vchData, vchHash, nOut))
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
    if (CommonName.size() > MAX_COMMON_NAME_LENGTH) 
    {
        errorMessage = "Invalid BDAP common name. Can not have more than " + std::to_string(MAX_COMMON_NAME_LENGTH) + " characters.";
        return false;
    }

    // check object organization name component
    if (OrganizationName.size() > MAX_ORG_NAME_LENGTH) 
    {
        errorMessage = "Invalid BDAP organization name. Can not have more than " + std::to_string(MAX_ORG_NAME_LENGTH) + " characters.";
        return false;
    }

    if (WalletAddress.size() > MAX_WALLET_ADDRESS_LENGTH) 
    {
        errorMessage = "Invalid BDAP wallet address. Can not have more than " + std::to_string(MAX_WALLET_ADDRESS_LENGTH) + " characters.";
        return false;
    }
    else {
        std::string strWalletAddress = stringFromVch(WalletAddress);
        CDynamicAddress entryAddress(strWalletAddress);
        if (!entryAddress.IsValid()) {
            errorMessage = "Invalid BDAP wallet address. Wallet address failed IsValid check.";
            return false;
        }
    }
    
    if (LinkAddress.size() > MAX_WALLET_ADDRESS_LENGTH) 
    {
        errorMessage = "Invalid BDAP link address. Can not have more than " + std::to_string(MAX_WALLET_ADDRESS_LENGTH) + " characters.";
        return false;
    }
    else {
        if (LinkAddress.size() > 0) {
            std::string strLinkAddress = stringFromVch(LinkAddress);
            CDynamicAddress entryLinkAddress(strLinkAddress);
            if (!entryLinkAddress.IsValid()) {
                errorMessage = "Invalid BDAP link address. Link wallet address failed IsValid check.";
                return false;
            }
        }
    }

    if (EncryptPublicKey.size() > MAX_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP encryption public key. Can not have more than " + std::to_string(MAX_KEY_LENGTH) + " characters.";
        return false;
    }
    else {
        if (EncryptPublicKey.size() > 0) {
            CPubKey entryEncryptPublicKey(EncryptPublicKey);
            if (!entryEncryptPublicKey.IsFullyValid()) {
                errorMessage = "Invalid BDAP encryption public key. Encryption public key failed IsFullyValid check.";
                return false;
            }
        }
    }

    return true;
}

/** Checks if BDAP transaction exists in the memory pool */
bool CDomainEntry::CheckIfExistsInMemPool(const CTxMemPool& pool, std::string& errorMessage)
{
    for (const CTxMemPoolEntry& e : pool.mapTx) {
        const CTransactionRef& tx = e.GetSharedTx();
        for (const CTxOut& txOut : tx->vout) {
            if (IsBDAPDataOutput(txOut)) {
                CDomainEntry domainEntry(tx);
                if (this->GetFullObjectPath() == domainEntry.GetFullObjectPath()) {
                    errorMessage = "CheckIfExistsInMemPool: A BDAP domain entry transaction for " + GetFullObjectPath() + " is already in the memory pool!";
                    return true;
                }
            }
        }
    }
    return false;
}

/** Checks if the domain entry transaction uses the entry's UTXO */
bool CDomainEntry::TxUsesPreviousUTXO(const CTransactionRef& tx)
{
    int nIn = GetBDAPOperationOutIndex(tx);
    COutPoint entryOutpoint = COutPoint(txHash, nIn);
    for (const CTxIn& txIn : tx->vin) {
        if (txIn.prevout == entryOutpoint)
            return true;
    }
    return false;
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
        oName.push_back(Pair("object_full_path", entry.GetFullObjectPath()));
        oName.push_back(Pair("object_type", entry.ObjectTypeString()));
        oName.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
        oName.push_back(Pair("public", (int)entry.fPublicObject));
        oName.push_back(Pair("encryption_publickey", HexStr(entry.EncryptPublicKey)));
        oName.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
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

bool DecodeBDAPTx(const CTransactionRef& tx, int& op, std::vector<std::vector<unsigned char> >& vvch) 
{
    bool found = false;

    for (unsigned int i = 0; i < tx->vout.size(); i++) {
        const CTxOut& out = tx->vout[i];
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

bool FindBDAPInTx(const CCoinsViewCache &inputs, const CTransaction& tx, std::vector<std::vector<unsigned char> >& vvch)
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

int GetBDAPOpType(const CScript& script)
{
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
                if (op2 - OP_1NEGATE - 1  > OP_BDAP && op2 - OP_1NEGATE - 1 <= OP_BDAP_CHANNEL_CHECKPOINT)
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

int GetBDAPOpType(const CTxOut& out)
{
    return GetBDAPOpType(out.scriptPubKey);
}

std::string GetBDAPOpTypeString(const CScript& script)
{
    return BDAPFromOp(GetBDAPOpType(script));
}

bool GetBDAPOpScript(const CTransactionRef& tx, CScript& scriptBDAPOp, vchCharString& vvchOpParameters, int& op)
{
    for (unsigned int i = 0; i < tx->vout.size(); i++) 
    {
        const CTxOut& out = tx->vout[i];
        if (DecodeBDAPScript(out.scriptPubKey, op, vvchOpParameters)) 
        {
            scriptBDAPOp = out.scriptPubKey;
            return true;
        }
    }
    return false;
}

bool GetBDAPOpScript(const CTransactionRef& tx, CScript& scriptBDAPOp)
{
    int op;
    vchCharString vvchOpParameters;
    return GetBDAPOpScript(tx, scriptBDAPOp, vvchOpParameters, op);
}

bool GetBDAPDataScript(const CTransaction& tx, CScript& scriptBDAPData)
{
    for (unsigned int i = 0; i < tx.vout.size(); i++) 
    {
        const CTxOut& out = tx.vout[i];
        if (out.scriptPubKey.IsUnspendable()) 
        {
            scriptBDAPData = out.scriptPubKey;
            return true;
        }
    }
    return false;
}

bool IsBDAPOperationOutput(const CTxOut& out)
{
    if (GetBDAPOpType(out.scriptPubKey) > 0)
        return true;
    return false;
}

int GetBDAPOpCodeFromOutput(const CTxOut& out)
{
    if (!IsBDAPOperationOutput(out)) {
        return 0;
    }

    return GetBDAPOpType(out.scriptPubKey);
}

std::string GetBDAPOpStringFromOutput(const CTxOut& out)
{
    return GetBDAPOpTypeString(out.scriptPubKey);
}

int GetBDAPOperationOutIndex(const CTransactionRef& tx) 
{
    for(unsigned int i = 0; i < tx->vout.size();i++) {
        if(IsBDAPOperationOutput(tx->vout[i]))
            return i;
    }
    return -1;
}

int GetBDAPOperationOutIndex(int nHeight, const uint256& txHash) 
{
    CTransactionRef tx;
    const Consensus::Params& consensusParams = Params().GetConsensus();
    if (!GetBDAPTransaction(nHeight, txHash, tx, consensusParams))
    {
        return -1;
    }
    return GetBDAPOperationOutIndex(tx);
}

bool GetDomainEntryFromRecipient(const std::vector<CRecipient>& vecSend, CDomainEntry& entry, std::string& strOpType) 
{
    for (const CRecipient& rec : vecSend) {
        CScript bdapScript = rec.scriptPubKey;
        if (bdapScript.IsUnspendable()) {
            std::vector<unsigned char> vchData;
            std::vector<unsigned char> vchHash;
            if (!GetBDAPData(bdapScript, vchData, vchHash)) 
            {
                return false;
            }
            entry.UnserializeFromData(vchData, vchHash);
        }
        else {
            strOpType = GetBDAPOpTypeString(bdapScript);
        }
    }
    if (!entry.IsNull() && strOpType.size() > 0) {
        return true;
    }
    return false;
}

CDynamicAddress GetScriptAddress(const CScript& pubScript)
{
    CTxDestination txDestination;
    ExtractDestination(pubScript, txDestination);
    CDynamicAddress entryAddress(txDestination);
    return entryAddress;
}