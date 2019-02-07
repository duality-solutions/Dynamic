// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/utils.h"

#include "chainparams.h"
#include "coins.h"
#include "core_io.h"
#include "policy/policy.h"
#include "serialize.h"
#include "uint256.h"
#include "validation.h"
#include "wallet/wallet.h"

#include <univalue.h>

namespace BDAP {
    std::string GetObjectTypeString(unsigned int nObjectType)
    {
        switch ((BDAP::ObjectType)nObjectType) {

            case BDAP::ObjectType::BDAP_DEFAULT_TYPE:
                return "Default";
             case BDAP::ObjectType::BDAP_USER:
                return "User Entry";
            case BDAP::ObjectType::BDAP_GROUP:
                return "Group Entry";
            case BDAP::ObjectType::BDAP_DEVICE:
                return "Device Entry";
            case BDAP::ObjectType::BDAP_DOMAIN:
                return "Domain Entry";
            case BDAP::ObjectType::BDAP_ORGANIZATIONAL_UNIT:
                return "OU Entry";
            case BDAP::ObjectType::BDAP_CERTIFICATE:
                return "Certificate Entry";
            case BDAP::ObjectType::BDAP_AUDIT:
                return "Audit Entry";
            case BDAP::ObjectType::BDAP_CHANNEL:
                return "Channel Entry";
            case BDAP::ObjectType::BDAP_CHECKPOINT:
                return "Channel Checkpoint Entry";
            case BDAP::ObjectType::BDAP_LINK_REQUEST:
                return "Link Request Entry";
            case BDAP::ObjectType::BDAP_LINK_ACCEPT:
                return "Link Accept Entry";
            case BDAP::ObjectType::BDAP_IDENTITY:
                return "Identity Entry";
            case BDAP::ObjectType::BDAP_IDENTITY_VERIFICATION:
                return "Identity Verification Entry";
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
        case OP_BDAP_ACCOUNT_ENTRY:
            return "bdap_account_entry";
        case OP_BDAP_LINK_REQUEST:
            return "bdap_link_request";
        case OP_BDAP_LINK_ACCEPT:
            return "bdap_link_accept";
        case OP_BDAP_AUDIT:
            return "bdap_audit";
        case OP_BDAP_CERTIFICATE:
            return "bdap_certificate";
        case OP_BDAP_IDENTITY:
            return "bdap_identity";
        case OP_BDAP_ID_VERIFICATION:
            return "bdap_identity_verification";
        case OP_BDAP_SIDECHAIN:
            return "bdap_sidechain";
        case OP_BDAP_SIDECHAIN_CHECKPOINT:
            return "bdap_sidechain_checkpoint";
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

bool DecodeBDAPTx(const CTransactionRef& tx, int& op1, int& op2, std::vector<std::vector<unsigned char> >& vvch) 
{
    bool found = false;

    for (unsigned int i = 0; i < tx->vout.size(); i++) {
        const CTxOut& out = tx->vout[i];
        vchCharString vvchRead;
        if (DecodeBDAPScript(out.scriptPubKey, op1, op2, vvchRead)) {
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
        int op1, op2;
        if (DecodeBDAPScript(prevCoins.out.scriptPubKey, op1, op2, vvch)) {
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
                if ((op1 - OP_1NEGATE - 1 == OP_BDAP_NEW) || 
                    (op1 - OP_1NEGATE - 1 == OP_BDAP_DELETE) || 
                    (op1 - OP_1NEGATE - 1 == OP_BDAP_REVOKE) || 
                    (op1 - OP_1NEGATE - 1 == OP_BDAP_MODIFY) ||
                    (op1 - OP_1NEGATE - 1 == OP_BDAP_MODIFY_RDN)
                )
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
                if (op2 - OP_1NEGATE - 1  > OP_BDAP_NEW && op2 - OP_1NEGATE - 1 <= OP_BDAP_SIDECHAIN_CHECKPOINT)
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


std::string GetBDAPOpTypeString(const int& op1, const int& op2)
{
    if (op1 == OP_BDAP_NEW && op2 == OP_BDAP_ACCOUNT_ENTRY) {
        return "bdap_new_account";
    }
    else if (op1 == OP_BDAP_DELETE && op2 == OP_BDAP_ACCOUNT_ENTRY) {
        return "bdap_delete_account";
    }
    else if (op1 == OP_BDAP_REVOKE && op2 == OP_BDAP_ACCOUNT_ENTRY) {
        return "bdap_revoke_account";
    }
    else if (op1 == OP_BDAP_MODIFY && op2 == OP_BDAP_ACCOUNT_ENTRY) {
        return "bdap_update_account";
    }
    else if (op1 == OP_BDAP_MODIFY_RDN && op2 == OP_BDAP_ACCOUNT_ENTRY) {
        return "bdap_move_account";
    }
    else if (op1 == OP_BDAP_NEW && op2 == OP_BDAP_LINK_REQUEST) {
        return "bdap_new_link_request";
    }
    else if (op1 == OP_BDAP_DELETE && op2 == OP_BDAP_LINK_REQUEST) {
        return "bdap_delete_link_request";
    }
    else if (op1 == OP_BDAP_MODIFY && op2 == OP_BDAP_LINK_REQUEST) {
        return "bdap_update_link_request";
    }
    else if (op1 == OP_BDAP_NEW && op2 == OP_BDAP_LINK_ACCEPT) {
        return "bdap_new_link_accept";
    }
    else if (op1 == OP_BDAP_DELETE && op2 == OP_BDAP_LINK_ACCEPT) {
        return "bdap_delete_link_accept";
    }
    else if (op1 == OP_BDAP_MODIFY && op2 == OP_BDAP_LINK_ACCEPT) {
        return "bdap_update_link_accept";
    }
    else {
        return "unknown";
    }
}

bool GetBDAPOpScript(const CTransactionRef& tx, CScript& scriptBDAPOp, vchCharString& vvchOpParameters, int& op1, int& op2)
{
    for (unsigned int i = 0; i < tx->vout.size(); i++) 
    {
        const CTxOut& out = tx->vout[i];
        if (DecodeBDAPScript(out.scriptPubKey, op1, op2, vvchOpParameters)) 
        {
            scriptBDAPOp = out.scriptPubKey;
            return true;
        }
    }
    return false;
}

bool GetBDAPOpScript(const CTransactionRef& tx, CScript& scriptBDAPOp)
{
    int op1, op2;
    vchCharString vvchOpParameters;
    return GetBDAPOpScript(tx, scriptBDAPOp, vvchOpParameters, op1, op2);
}

bool GetBDAPDataScript(const CTransaction& tx, CScript& scriptBDAPData)
{
    CTransactionRef ptx = MakeTransactionRef(tx);
    return GetBDAPDataScript(ptx, scriptBDAPData);
}

bool GetBDAPDataScript(const CTransactionRef& ptx, CScript& scriptBDAPData)
{
    for (const CTxOut& out : ptx->vout) 
    {
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

CDynamicAddress GetScriptAddress(const CScript& pubScript)
{
    CTxDestination txDestination;
    ExtractDestination(pubScript, txDestination);
    CDynamicAddress entryAddress(txDestination);
    return entryAddress;
}

bool ExtractOpTypeValue(const CScript& script, std::string& strOpType, std::vector<unsigned char>& vchValue)
{
    LogPrint("bdap", "%s -- Script = %s\n", __func__, ScriptToAsmStr(script));
    opcodetype opcode;
    std::string strPrefix;
    uint8_t i = 1;
    CScript::const_iterator itScript = script.begin();
    while (itScript < script.end()) {
        std::vector<unsigned char> vch;
        if (!script.GetOp(itScript, opcode, vch))
            return false;
        if (!(0 <= opcode && opcode <= OP_PUSHDATA4)) {
            strPrefix += GetOpName(opcode);
            if (i == 1)
                strPrefix += " ";
        }
        else {
            vchValue = vch;
        }
        if (i > 2)
            break;
        i++;
    }
    if (strPrefix.empty() || !(strPrefix.size() >= 3)) {
        LogPrintf("%s -- Error, incorrect prefix length. Script = %s\n", __func__, ScriptToAsmStr(script));
        return false;
    }
    else if (strPrefix == ("1 6")) {
        strOpType = "bdap_new_account";
    }
    else if (strPrefix == "2 6") {
        strOpType = "bdap_delete_account";
    }
    else if (strPrefix == "3 6") {
        strOpType = "bdap_revoke_account";
    }
    else if (strPrefix == "4 6") {
        strOpType = "bdap_update_account";
    }
    else if (strPrefix == "1 7") {
        strOpType = "bdap_new_link_request";
    }
    else if (strPrefix == "2 7") {
        strOpType = "bdap_delete_link_request";
    }
    else if (strPrefix == "4 7") {
        strOpType = "bdap_update_link_request";
    }
    else if (strPrefix == "1 8") {
        strOpType = "bdap_new_link_accept";
    }
    else if (strPrefix == "2 8") {
        strOpType = "bdap_delete_link_accept";
    }
    else if (strPrefix == "4 8") {
        strOpType = "bdap_update_link_accept";
    }
    else {
        return false;
    }
    LogPrintf("%s -- strOpType = %s, strPrefix = %s, vchValue = %s\n", __func__, strOpType, strPrefix, stringFromVch(vchValue));
    return true;
}

bool GetScriptOpTypeValue(const std::vector<CRecipient>& vecSend, CScript& bdapOpScript, std::string& strOpType, std::vector<unsigned char>& vchValue)
{
    LogPrint("bdap", "%s -- vecSend size = %u \n", __func__, vecSend.size());
    for (const CRecipient& rec : vecSend) {
        CScript script = rec.scriptPubKey;
        if (!script.IsUnspendable()) {
            if (ExtractOpTypeValue(script, strOpType, vchValue)) {
                bdapOpScript = script;
                break;
            }
        }
    }
    if (strOpType.size() > 0) {
        return true;
    }
    return false;
}

bool GetTransactionOpTypeValue(const CTransaction& tx, CScript& bdapOpScript, std::string& strOpType, std::vector<unsigned char>& vchValue)
{
    for (const CTxOut& out : tx.vout)
    {
        if (!out.scriptPubKey.IsUnspendable()) 
        {
            if (ExtractOpTypeValue(out.scriptPubKey, strOpType, vchValue)) {
                bdapOpScript = out.scriptPubKey;
                break;
            }
        }
    }
    if (strOpType.size() > 0) {
        return true;
    }
    return false;
}

// The version number for link data is always the first position.
int GetLinkVersionFromData(const std::vector<unsigned char>& vchData)
{
    if (!(vchData.size() > 0))
        return -1;

    return (int)vchData[0];
}

bool GetPreviousTxRefById(const uint256& prevTxId, CTransactionRef& prevTx)
{
    prevTx = MakeTransactionRef();
    uint256 hashBlock;
    if (!GetTransaction(prevTxId, prevTx, Params().GetConsensus(), hashBlock, true))
        return false;

    return true;
}