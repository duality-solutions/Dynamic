// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2015 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "script.h"

#include "tinyformat.h"
#include "utilstrencodings.h"

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
    case OP_BDAP_MODIFY_RDN:
        return "OP_BDAP_MODIFY_RDN";
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
    return op == OP_BDAP_NEW || op == OP_BDAP_DELETE || op == OP_BDAP_REVOKE || op == OP_BDAP_MODIFY || op == OP_BDAP_MODIFY_RDN || 
    op == OP_BDAP_ACCOUNT_ENTRY || op == OP_BDAP_LINK_REQUEST || op == OP_BDAP_LINK_ACCEPT || op == OP_BDAP_AUDIT || op == OP_BDAP_CERTIFICATE || 
    op == OP_BDAP_IDENTITY || op == OP_BDAP_ID_VERIFICATION || op == OP_BDAP_SIDECHAIN || op == OP_BDAP_SIDECHAIN_CHECKPOINT;
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
    if (op1 != OP_BDAP_NEW && op1 != OP_BDAP_DELETE && op1 != OP_BDAP_REVOKE && op1 != OP_BDAP_MODIFY && op1 != OP_BDAP_MODIFY_RDN)
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