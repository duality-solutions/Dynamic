// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2015 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "script.h"

#include "assets/assets.h"
#include "standard.h"
#include "streams.h"
#include "tinyformat.h"
#include "utilstrencodings.h"
#include "version.h"

const char* GetOpName(opcodetype opcode)
{
    switch (opcode) {
    // push value
    case OP_0:
        return "0";
    case OP_PUSHDATA1:
        return "OP_PUSHDATA1";
    case OP_PUSHDATA2:
        return "OP_PUSHDATA2";
    case OP_PUSHDATA4:
        return "OP_PUSHDATA4";
    case OP_1NEGATE:
        return "-1";
    case OP_RESERVED:
        return "OP_RESERVED";
    case OP_1:
        return "1";
    case OP_2:
        return "2";
    case OP_3:
        return "3";
    case OP_4:
        return "4";
    case OP_5:
        return "5";
    case OP_6:
        return "6";
    case OP_7:
        return "7";
    case OP_8:
        return "8";
    case OP_9:
        return "9";
    case OP_10:
        return "10";
    case OP_11:
        return "11";
    case OP_12:
        return "12";
    case OP_13:
        return "13";
    case OP_14:
        return "14";
    case OP_15:
        return "15";
    case OP_16:
        return "16";

    // control
    case OP_NOP:
        return "OP_NOP";
    case OP_VER:
        return "OP_VER";
    case OP_IF:
        return "OP_IF";
    case OP_NOTIF:
        return "OP_NOTIF";
    case OP_VERIF:
        return "OP_VERIF";
    case OP_VERNOTIF:
        return "OP_VERNOTIF";
    case OP_ELSE:
        return "OP_ELSE";
    case OP_ENDIF:
        return "OP_ENDIF";
    case OP_VERIFY:
        return "OP_VERIFY";
    case OP_RETURN:
        return "OP_RETURN";

    // stack ops
    case OP_TOALTSTACK:
        return "OP_TOALTSTACK";
    case OP_FROMALTSTACK:
        return "OP_FROMALTSTACK";
    case OP_2DROP:
        return "OP_2DROP";
    case OP_2DUP:
        return "OP_2DUP";
    case OP_3DUP:
        return "OP_3DUP";
    case OP_2OVER:
        return "OP_2OVER";
    case OP_2ROT:
        return "OP_2ROT";
    case OP_2SWAP:
        return "OP_2SWAP";
    case OP_IFDUP:
        return "OP_IFDUP";
    case OP_DEPTH:
        return "OP_DEPTH";
    case OP_DROP:
        return "OP_DROP";
    case OP_DUP:
        return "OP_DUP";
    case OP_NIP:
        return "OP_NIP";
    case OP_OVER:
        return "OP_OVER";
    case OP_PICK:
        return "OP_PICK";
    case OP_ROLL:
        return "OP_ROLL";
    case OP_ROT:
        return "OP_ROT";
    case OP_SWAP:
        return "OP_SWAP";
    case OP_TUCK:
        return "OP_TUCK";

    // splice ops
    case OP_CAT:
        return "OP_CAT";
    case OP_SUBSTR:
        return "OP_SUBSTR";
    case OP_LEFT:
        return "OP_LEFT";
    case OP_RIGHT:
        return "OP_RIGHT";
    case OP_SIZE:
        return "OP_SIZE";

    // bit logic
    case OP_INVERT:
        return "OP_INVERT";
    case OP_AND:
        return "OP_AND";
    case OP_OR:
        return "OP_OR";
    case OP_XOR:
        return "OP_XOR";
    case OP_EQUAL:
        return "OP_EQUAL";
    case OP_EQUALVERIFY:
        return "OP_EQUALVERIFY";
    case OP_RESERVED1:
        return "OP_RESERVED1";
    case OP_RESERVED2:
        return "OP_RESERVED2";

    // numeric
    case OP_1ADD:
        return "OP_1ADD";
    case OP_1SUB:
        return "OP_1SUB";
    case OP_2MUL:
        return "OP_2MUL";
    case OP_2DIV:
        return "OP_2DIV";
    case OP_NEGATE:
        return "OP_NEGATE";
    case OP_ABS:
        return "OP_ABS";
    case OP_NOT:
        return "OP_NOT";
    case OP_0NOTEQUAL:
        return "OP_0NOTEQUAL";
    case OP_ADD:
        return "OP_ADD";
    case OP_SUB:
        return "OP_SUB";
    case OP_MUL:
        return "OP_MUL";
    case OP_DIV:
        return "OP_DIV";
    case OP_MOD:
        return "OP_MOD";
    case OP_LSHIFT:
        return "OP_LSHIFT";
    case OP_RSHIFT:
        return "OP_RSHIFT";
    case OP_BOOLAND:
        return "OP_BOOLAND";
    case OP_BOOLOR:
        return "OP_BOOLOR";
    case OP_NUMEQUAL:
        return "OP_NUMEQUAL";
    case OP_NUMEQUALVERIFY:
        return "OP_NUMEQUALVERIFY";
    case OP_NUMNOTEQUAL:
        return "OP_NUMNOTEQUAL";
    case OP_LESSTHAN:
        return "OP_LESSTHAN";
    case OP_GREATERTHAN:
        return "OP_GREATERTHAN";
    case OP_LESSTHANOREQUAL:
        return "OP_LESSTHANOREQUAL";
    case OP_GREATERTHANOREQUAL:
        return "OP_GREATERTHANOREQUAL";
    case OP_MIN:
        return "OP_MIN";
    case OP_MAX:
        return "OP_MAX";
    case OP_WITHIN:
        return "OP_WITHIN";

    // crypto
    case OP_RIPEMD160:
        return "OP_RIPEMD160";
    case OP_SHA1:
        return "OP_SHA1";
    case OP_SHA256:
        return "OP_SHA256";
    case OP_HASH160:
        return "OP_HASH160";
    case OP_HASH256:
        return "OP_HASH256";
    case OP_CODESEPARATOR:
        return "OP_CODESEPARATOR";
    case OP_CHECKSIG:
        return "OP_CHECKSIG";
    case OP_CHECKSIGVERIFY:
        return "OP_CHECKSIGVERIFY";
    case OP_CHECKMULTISIG:
        return "OP_CHECKMULTISIG";
    case OP_CHECKMULTISIGVERIFY:
        return "OP_CHECKMULTISIGVERIFY";

    // expanson
    case OP_NOP1:
        return "OP_NOP1";
    case OP_CHECKLOCKTIMEVERIFY:
        return "OP_CHECKLOCKTIMEVERIFY";
    case OP_CHECKSEQUENCEVERIFY:
        return "OP_CHECKSEQUENCEVERIFY";
    case OP_NOP4:
        return "OP_NOP4";
    case OP_NOP5:
        return "OP_NOP5";
    case OP_NOP6:
        return "OP_NOP6";
    case OP_NOP7:
        return "OP_NOP7";
    case OP_NOP8:
        return "OP_NOP8";
    case OP_NOP9:
        return "OP_NOP9";
    case OP_NOP10:
        return "OP_NOP10";

    // fluid
    case OP_MINT:
        return "OP_MINT";
    case OP_REWARD_DYNODE:
        return "OP_REWARD_DYNODE";
    case OP_REWARD_MINING:
        return "OP_REWARD_MINING";
    case OP_REWARD_STAKE:
        return "OP_REWARD_STAKE";
    case OP_SWAP_SOVEREIGN_ADDRESS:
        return "OP_SWAP_SOVEREIGN_ADDRESS";
    case OP_UPDATE_FEES:
        return "OP_UPDATE_FEES";
    case OP_FREEZE_ADDRESS:
        return "OP_FREEZE_ADDRESS";
    case OP_RELEASE_ADDRESS:
        return "OP_RELEASE_ADDRESS";

    // BDAP, directory access, user identity and certificate system
    case OP_BDAP_NEW:
        return "OP_BDAP_NEW";
    case OP_BDAP_DELETE:
        return "OP_BDAP_DELETE";
    case OP_BDAP_REVOKE:
        return "OP_BDAP_REVOKE";
    case OP_BDAP_MODIFY:
        return "OP_BDAP_MODIFY";
    case OP_BDAP_MOVE:
        return "OP_BDAP_MOVE";
    case OP_BDAP_ACCOUNT_ENTRY:
        return "OP_BDAP_ACCOUNT_ENTRY";
    case OP_BDAP_LINK_REQUEST:
        return "OP_BDAP_LINK_REQUEST";
    case OP_BDAP_LINK_ACCEPT:
        return "OP_BDAP_LINK_ACCEPT";
    case OP_BDAP_AUDIT:
        return "OP_BDAP_AUDIT";
    case OP_BDAP_CERTIFICATE:
        return "OP_BDAP_CERTIFICATE";
    case OP_BDAP_IDENTITY:
        return "OP_BDAP_IDENTITY";
    case OP_BDAP_ID_VERIFICATION:
        return "OP_BDAP_ID_VERIFICATION";
    case OP_BDAP_SIDECHAIN:
        return "OP_BDAP_SIDECHAIN";
    case OP_BDAP_SIDECHAIN_CHECKPOINT:
        return "OP_BDAP_SIDECHAIN_CHECKPOINT";
    case OP_BDAP_ASSET:
        return "OP_BDAP_ASSET";

    case OP_INVALIDOPCODE:
        return "OP_INVALIDOPCODE";

        // Note:
        //  The template matching params OP_SMALLINTEGER/etc are defined in opcodetype enum
        //  as kind of implementation hack, they are *NOT* real opcodes.  If found in real
        //  Script, just let the default: case deal with them.

    default:
        return "OP_UNKNOWN";
    }
}

// TODO (bdap): move functions below to seperate code file
bool IsBDAPOp(int op)
{
    return op == OP_BDAP_NEW || op == OP_BDAP_DELETE || op == OP_BDAP_EXPIRE || op == OP_BDAP_REVOKE || op == OP_BDAP_MODIFY || op == OP_BDAP_MOVE || 
    op == OP_BDAP_ACCOUNT_ENTRY || op == OP_BDAP_LINK_REQUEST || op == OP_BDAP_LINK_ACCEPT || op == OP_BDAP_AUDIT || op == OP_BDAP_CERTIFICATE || 
    op == OP_BDAP_IDENTITY || op == OP_BDAP_ID_VERIFICATION || op == OP_BDAP_SIDECHAIN || op == OP_BDAP_SIDECHAIN_CHECKPOINT || op == OP_BDAP_ASSET;
}

bool DecodeBDAPScript(const CScript& script, int& op1, int& op2, std::vector<std::vector<unsigned char> >& vvch, CScript::const_iterator& pc)
{
    opcodetype opcode;
    vvch.clear();
    if (!script.GetOp(pc, opcode))
        return false;
    if (opcode < OP_1 || opcode > OP_16)
        return false;

    op1 = CScript::DecodeOP_N(opcode);
    if (op1 != OP_BDAP_NEW && op1 != OP_BDAP_DELETE && op1 != OP_BDAP_EXPIRE && op1 != OP_BDAP_REVOKE && op1 != OP_BDAP_MODIFY && op1 != OP_BDAP_MOVE)
        return false;
    if (!script.GetOp(pc, opcode))
        return false;
    if (opcode < OP_1 || opcode > OP_16)
        return false;
    op2 = CScript::DecodeOP_N(opcode);
    if (!IsBDAPOp(op2))
        return false;
    bool found = false;
    for (;;) {
        std::vector<unsigned char> vch;
        if (!script.GetOp(pc, opcode, vch))
            return false;
        if (opcode == OP_DROP || opcode == OP_2DROP) {
            found = true;
            break;
        }
        if (!(opcode >= 0 && opcode <= OP_PUSHDATA4))
            return false;
        vvch.push_back(vch);
    }
    // move the pc to after any DROP or NOP
    while (opcode == OP_DROP || opcode == OP_2DROP) {
        if (!script.GetOp(pc, opcode))
            break;
    }

    pc--;
    return found;
}

bool DecodeBDAPScript(const CScript& script, int& op1, int& op2, std::vector<std::vector<unsigned char> >& vvch)
{
    CScript::const_iterator pc = script.begin();
    return DecodeBDAPScript(script, op1, op2, vvch, pc);
}

bool RemoveBDAPScript(const CScript& scriptIn, CScript& scriptOut)
{
    int op1, op2;
    std::vector<std::vector<unsigned char> > vvch;
    CScript::const_iterator pc = scriptIn.begin();

    if (!DecodeBDAPScript(scriptIn, op1, op2, vvch, pc))
        return false;
    scriptOut = CScript(pc, scriptIn.end());
    return true;
}
// TODO (bdap): move the above functions to seperate code file

unsigned int CScript::GetSigOpCount(bool fAccurate) const
{
    unsigned int n = 0;
    const_iterator pc = begin();
    opcodetype lastOpcode = OP_INVALIDOPCODE;
    while (pc < end()) {
        opcodetype opcode;
        if (!GetOp(pc, opcode))
            break;
        if (opcode == OP_CHECKSIG || opcode == OP_CHECKSIGVERIFY)
            n++;
        else if (opcode == OP_CHECKMULTISIG || opcode == OP_CHECKMULTISIGVERIFY) {
            if (fAccurate && lastOpcode >= OP_1 && lastOpcode <= OP_16)
                n += DecodeOP_N(lastOpcode);
            else
                n += MAX_PUBKEYS_PER_MULTISIG;
        }
        lastOpcode = opcode;
    }
    return n;
}

unsigned int CScript::GetSigOpCount(const CScript& scriptSig) const
{
    if (!IsPayToScriptHash())
        return GetSigOpCount(true);

    // This is a pay-to-script-hash scriptPubKey;
    // get the last item that the scriptSig
    // pushes onto the stack:
    const_iterator pc = scriptSig.begin();
    std::vector<unsigned char> data;
    while (pc < scriptSig.end()) {
        opcodetype opcode;
        if (!scriptSig.GetOp(pc, opcode, data))
            return 0;
        if (opcode > OP_16)
            return 0;
    }

    /// ... and return its opcount:
    CScript subscript(data.begin(), data.end());
    return subscript.GetSigOpCount(true);
}

bool CScript::IsPayToPublicKeyHash() const
{
    // Remove BDAP portion of the script
    CScript scriptPubKey;
    CScript scriptPubKeyOut;
    if (RemoveBDAPScript(*this, scriptPubKeyOut)) {
        scriptPubKey = scriptPubKeyOut;
    } else {
        scriptPubKey = *this;
    }

    // Extra-fast test for pay-to-pubkey-hash CScripts:
    return (this->size() == 25 &&
            (*this)[0] == OP_DUP &&
            (*this)[1] == OP_HASH160 &&
            (*this)[2] == 0x14 &&
            (*this)[23] == OP_EQUALVERIFY &&
            (*this)[24] == OP_CHECKSIG);
}

bool CScript::IsPayToScriptHash() const
{
    // Remove BDAP portion of the script
    CScript scriptPubKey;
    CScript scriptPubKeyOut;
    if (RemoveBDAPScript(*this, scriptPubKeyOut)) {
        scriptPubKey = scriptPubKeyOut;
    } else {
        scriptPubKey = *this;
    }
    // Extra-fast test for pay-to-script-hash CScripts:
    return (this->size() == 23 &&
            (*this)[0] == OP_HASH160 &&
            (*this)[1] == 0x14 &&
            (*this)[22] == OP_EQUAL);
}

/** ASSET START */
bool CScript::IsAssetScript() const
{
    int nType = 0;
    bool isOwner = false;
    int start = 0;
    return IsAssetScript(nType, isOwner, start);
}

bool CScript::IsAssetScript(int& nType, bool& isOwner) const
{
    int start = 0;
    return IsAssetScript(nType, isOwner, start);
}

bool CScript::IsAssetScript(int& nType, bool& fIsOwner, int& nStartingIndex) const
{
    if (this->size() > 31) {
        if ((*this)[25] == OP_DYN_ASSET) { // OP_DYN_ASSET is always in the 25 index of the script if it exists
            int index = -1;
            if ((*this)[27] == DYN_D) { // Check to see if DYN starts at 27 ( this->size() < 105)
                if ((*this)[28] == DYN_Y)
                    if ((*this)[29] == DYN_N)
                        index = 30;
            } else {
                if ((*this)[28] == DYN_D) // Check to see if DYN starts at 28 ( this->size() >= 105)
                    if ((*this)[29] == DYN_Y)
                        if ((*this)[30] == DYN_N)
                            index = 31;
            }

            if (index > 0) {
                nStartingIndex = index + 1; // Set the index where the asset data begins. Use to serialize the asset data into asset objects
                if ((*this)[index] == DYN_T) { // Transfer first anticipating more transfers than other assets operations
                    nType = TX_TRANSFER_ASSET;
                    return true;
                } else if ((*this)[index] == DYN_Q && this->size() > 39) {
                    nType = TX_NEW_ASSET;
                    fIsOwner = false;
                    return true;
                } else if ((*this)[index] == DYN_O) {
                    nType = TX_NEW_ASSET;
                    fIsOwner = true;
                    return true;
                } else if ((*this)[index] == DYN_D) {
                    nType = TX_REISSUE_ASSET;
                    return true;
                }
            }
        }
    }
    return false;
}

bool CScript::IsNewAsset() const
{

    int nType = 0;
    bool fIsOwner = false;
    if (IsAssetScript(nType, fIsOwner))
        return !fIsOwner && nType == TX_NEW_ASSET;

    return false;
}

bool CScript::IsOwnerAsset() const
{
    int nType = 0;
    bool fIsOwner = false;
    if (IsAssetScript(nType, fIsOwner))
        return fIsOwner && nType == TX_NEW_ASSET;

    return false;
}

bool CScript::IsReissueAsset() const
{
    int nType = 0;
    bool fIsOwner = false;
    if (IsAssetScript(nType, fIsOwner))
        return nType == TX_REISSUE_ASSET;

    return false;
}

bool CScript::IsTransferAsset() const
{
    int nType = 0;
    bool fIsOwner = false;
    if (IsAssetScript(nType, fIsOwner))
        return nType == TX_TRANSFER_ASSET;

    return false;
}

bool CScript::IsNullAsset() const
{
    return IsNullAssetTxDataScript() || IsNullGlobalRestrictionAssetTxDataScript() || IsNullAssetVerifierTxDataScript();
}

bool CScript::IsNullAssetTxDataScript() const
{
    return (this->size() > 23 &&
            (*this)[0] == OP_DYN_ASSET &&
            (*this)[1] == 0x14);
}

bool CScript::IsNullGlobalRestrictionAssetTxDataScript() const
{
    // 1 OP_DYN_ASSET followed by two OP_RESERVED + atleast 4 characters for the restricted name $ABC
    return (this->size() > 6 &&
            (*this)[0] == OP_DYN_ASSET &&
            (*this)[1] == OP_RESERVED &&
            (*this)[2] == OP_RESERVED);
}


bool CScript::IsNullAssetVerifierTxDataScript() const
{
    // 1 OP_DYN_ASSET followed by one OP_RESERVED
    return (this->size() > 3 &&
            (*this)[0] == OP_DYN_ASSET &&
            (*this)[1] == OP_RESERVED &&
            (*this)[2] != OP_RESERVED);
}
/** ASSET END */

bool CScript::IsPayToPublicKey() const
{
    // Remove BDAP portion of the script
    CScript scriptPubKey;
    CScript scriptPubKeyOut;
    if (RemoveBDAPScript(*this, scriptPubKeyOut)) {
        scriptPubKey = scriptPubKeyOut;
    } else {
        scriptPubKey = *this;
    }
    // Test for pay-to-pubkey CScript with both
    // compressed or uncompressed pubkey
    if (this->size() == 35) {
        return ((*this)[1] == 0x02 || (*this)[1] == 0x03) &&
               (*this)[34] == OP_CHECKSIG;
    }
    if (this->size() == 67) {
        return (*this)[1] == 0x04 &&
               (*this)[66] == OP_CHECKSIG;
    }
    return false;
}

bool CScript::IsPushOnly(const_iterator pc) const
{
    while (pc < end()) {
        opcodetype opcode;
        if (!GetOp(pc, opcode))
            return false;
        // Note that IsPushOnly() *does* consider OP_RESERVED to be a
        // push-type opcode, however execution of OP_RESERVED fails, so
        // it's not relevant to P2SH/BIP62 as the scriptSig would fail prior to
        // the P2SH special validation code being executed.
        if (opcode > OP_16)
            return false;
    }
    return true;
}

bool CScript::IsPushOnly() const
{
    return this->IsPushOnly(begin());
}

bool CScript::HasValidOps() const
{
    CScript::const_iterator it = begin();
    while (it < end()) {
        opcodetype opcode;
        std::vector<unsigned char> item;
        if (!GetOp(it, opcode, item) || opcode > MAX_OPCODE || item.size() > MAX_SCRIPT_ELEMENT_SIZE) {
            return false;
        }
    }
    return true;
}

bool CScript::IsUnspendable() const
{
    CAmount nAmount;
    return (size() > 0 && *begin() == OP_RETURN) || (size() > 0 && *begin() == OP_DYN_ASSET) || (size() > MAX_SCRIPT_SIZE) || (GetAssetAmountFromScript(*this, nAmount) && nAmount == 0);
}

/* ASSET START */
//!--------------------------------------------------------------------------------------------------------------------------!//
//! These are needed because script.h and script.cpp do not have access to asset.h and asset.cpp functions. This is
//! because the make file compiles them at different times. The script files are compiled with other
//! consensus files, and asset files are compiled with core files.

//! Used to check if an asset script contains zero assets. Is so, it should be unspendable
bool GetAssetAmountFromScript(const CScript& script, CAmount& nAmount)
{
    // Placeholder strings that will get set if you successfully get the transfer or asset from the script
    std::string address = "";
    std::string assetName = "";

    int nType = 0;
    bool fIsOwner = false;
    if (!script.IsAssetScript(nType, fIsOwner)) {
        return false;
    }

    txnouttype type = txnouttype(nType);

    // Get the New Asset or Transfer Asset from the scriptPubKey
    if (type == TX_NEW_ASSET && !fIsOwner) {
        if (AmountFromNewAssetScript(script, nAmount)) {
            return true;
        }
    } else if (type == TX_TRANSFER_ASSET) {
        if (AmountFromTransferScript(script, nAmount)) {
            return true;
        }
    } else if (type == TX_NEW_ASSET && fIsOwner) {
            nAmount = OWNER_ASSET_AMOUNT;
            return true;
    } else if (type == TX_REISSUE_ASSET) {
        if (AmountFromReissueScript(script, nAmount)) {
            return true;
        }
    }

    return false;
}

bool ScriptNewAsset(const CScript& scriptPubKey, int& nStartingIndex)
{
    int nType = 0;
    bool fIsOwner =false;
    if (scriptPubKey.IsAssetScript(nType, fIsOwner, nStartingIndex)) {
        return nType == TX_NEW_ASSET && !fIsOwner;
    }

    return false;
}

bool ScriptTransferAsset(const CScript& scriptPubKey, int& nStartingIndex)
{
    int nType = 0;
    bool fIsOwner =false;
    if (scriptPubKey.IsAssetScript(nType, fIsOwner, nStartingIndex)) {
        return nType == TX_TRANSFER_ASSET;
    }

    return false;
}

bool ScriptReissueAsset(const CScript& scriptPubKey, int& nStartingIndex)
{
    int nType = 0;
    bool fIsOwner =false;
    if (scriptPubKey.IsAssetScript(nType, fIsOwner, nStartingIndex)) {
        return nType == TX_REISSUE_ASSET;
    }

    return false;
}

bool AmountFromNewAssetScript(const CScript& scriptPubKey, CAmount& nAmount)
{
    int nStartingIndex = 0;
    if (!ScriptNewAsset(scriptPubKey, nStartingIndex))
        return false;

    std::vector<unsigned char> vchNewAsset;
    vchNewAsset.insert(vchNewAsset.end(), scriptPubKey.begin() + nStartingIndex, scriptPubKey.end());
    CDataStream ssAsset(vchNewAsset, SER_NETWORK, PROTOCOL_VERSION);

    CNewAsset assetNew;
    try {
        ssAsset >> assetNew;
    } catch(std::exception& e) {
        std::cout << "Failed to get the asset from the stream: " << e.what() << std::endl;
        return false;
    }

    nAmount = assetNew.nAmount;
    return true;
}

bool AmountFromTransferScript(const CScript& scriptPubKey, CAmount& nAmount)
{
    int nStartingIndex = 0;
    if (!ScriptTransferAsset(scriptPubKey, nStartingIndex))
        return false;

    std::vector<unsigned char> vchAsset;
    vchAsset.insert(vchAsset.end(), scriptPubKey.begin() + nStartingIndex, scriptPubKey.end());
    CDataStream ssAsset(vchAsset, SER_NETWORK, PROTOCOL_VERSION);

    CAssetTransfer asset;
    try {
        ssAsset >> asset;
    } catch(std::exception& e) {
        std::cout << "Failed to get the asset from the stream: " << e.what() << std::endl;
        return false;
    }

    nAmount = asset.nAmount;
    return true;
}

bool AmountFromReissueScript(const CScript& scriptPubKey, CAmount& nAmount)
{
    int nStartingIndex = 0;
    if (!ScriptReissueAsset(scriptPubKey, nStartingIndex))
        return false;

    std::vector<unsigned char> vchNewAsset;
    vchNewAsset.insert(vchNewAsset.end(), scriptPubKey.begin() + nStartingIndex, scriptPubKey.end());
    CDataStream ssAsset(vchNewAsset, SER_NETWORK, PROTOCOL_VERSION);

    CReissueAsset asset;
    try {
        ssAsset >> asset;
    } catch(std::exception& e) {
        std::cout << "Failed to get the asset from the stream: " << e.what() << std::endl;
        return false;
    }

    nAmount = asset.nAmount;
    return true;
}
//!--------------------------------------------------------------------------------------------------------------------------!//
/* ASSET END */