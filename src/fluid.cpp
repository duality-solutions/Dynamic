// Copyright (c) 2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "fluid.h"

#include "chain.h"
#include "core_io.h"
#include "keepass.h"
#include "net.h"
#include "netbase.h"
#include "timedata.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"
#include "utiltime.h"
#include "validation.h"

#include "wallet/wallet.h"
#include "wallet/walletdb.h"

Fluid fluid;

#ifdef ENABLE_WALLET
extern CWallet* pwalletMain;
#endif //ENABLE_WALLET

bool IsTransactionFluid(CScript txOut) {
    return (txOut.IsProtocolInstruction(MINT_TX)
            || txOut.IsProtocolInstruction(DYNODE_MODFIY_TX)
            || txOut.IsProtocolInstruction(MINING_MODIFY_TX)
           );
}

/** Does client instance own address for engaging in processes - required for RPC (PS: NEEDS wallet) */
bool Fluid::InitiateFluidVerify(CDynamicAddress dynamicAddress) {
#ifdef ENABLE_WALLET
    LOCK2(cs_main, pwalletMain ? &pwalletMain->cs_wallet : NULL);
    CDynamicAddress address(dynamicAddress);

    if (address.IsValid()) {
        CTxDestination dest = address.Get();
        CScript scriptPubKey = GetScriptForDestination(dest);
        isminetype mine = pwalletMain ? IsMine(*pwalletMain, dest) : ISMINE_NO;

        return ((mine & ISMINE_SPENDABLE) ? true : false);
    }

    return false;
#else
    // Wallet cannot be accessed, cannot continue ahead!
    return false;
#endif //ENABLE_WALLET
}

/** Checks if any given address is a current master key (invoked by RPC) */
bool Fluid::IsGivenKeyMaster(CDynamicAddress inputKey) {
    std::vector<std::string> fluidManagers;
    GetLastBlockIndex(chainActive.Tip());
    CBlockIndex* pindex = chainActive.Tip();

    if (pindex != NULL)
        fluidManagers = pindex->fluidParams.fluidManagers;
    else
        fluidManagers = InitialiseAddresses();

    for (const std::string& address : fluidManagers) {
        CDynamicAddress attemptKey;
        attemptKey.SetString(address);
        if (inputKey.IsValid() && attemptKey.IsValid() && inputKey == attemptKey)
            return true;
    }

    return false;
}

/** Checks fluid transactoin operation script amount for invalid values. */
bool Fluid::CheckFluidOperationScript(const CScript& fluidScriptPubKey, const int64_t timeStamp, std::string& errorMessage, bool fSkipTimeStampCheck) {
    std::string strFluidOpScript = ScriptToAsmStr(fluidScriptPubKey);
    std::string verificationWithoutOpCode = GetRidOfScriptStatement(strFluidOpScript);
    std::string strOperationCode = GetRidOfScriptStatement(strFluidOpScript, 0);
    if (!fSkipTimeStampCheck) {
        if (!ExtractCheckTimestamp(strFluidOpScript, timeStamp)) {
            errorMessage = "CheckFluidOperationScript fluid timestamp is too old.";
            return false;
        }
    }
    if (IsHex(verificationWithoutOpCode)) {
        std::string strAmount;
        std::string strUnHexedFluidOpScript = HexToString(verificationWithoutOpCode);
        std::vector<std::string> vecSplitScript;
        SeperateString(strUnHexedFluidOpScript, vecSplitScript, "$");
        if (vecSplitScript.size() > 1) {
            strAmount = vecSplitScript[0];
            CAmount fluidAmount;
            if (ParseFixedPoint(strAmount, 8, &fluidAmount)) {
                if (fluidAmount < 0) {
                    errorMessage = "CheckFluidOperationScript fluid amount is less than zero: " + strAmount;
                    return false;
                }
                else if (strOperationCode == "OP_MINT" && (fluidAmount > FLUID_MAX_FOR_MINT)) {
                    errorMessage = "CheckFluidOperationScript fluid OP_MINT amount exceeds maximum: " + strAmount;
                    return false;
                }
                else if (strOperationCode == "OP_REWARD_MINING" && (fluidAmount > FLUID_MAX_REWARD_FOR_MINING)) {
                    errorMessage = "CheckFluidOperationScript fluid OP_REWARD_MINING amount exceeds maximum: " + strAmount;
                    return false;
                }
                else if (strOperationCode == "OP_REWARD_DYNODE" && (fluidAmount > FLUID_MAX_REWARD_FOR_DYNODE)) {
                    errorMessage = "CheckFluidOperationScript fluid OP_REWARD_DYNODE amount exceeds maximum: " + strAmount;
                    return false;
                }
            }
        }
        else {
            errorMessage = "CheckFluidOperationScript fluid token invalid. " + strUnHexedFluidOpScript;
            return false;
        }
    }
    else {
        errorMessage = "CheckFluidOperationScript fluid token is not hex. " + verificationWithoutOpCode;
        return false;
    }

    return true;
}

/** Checks whether fluid transaction is in the memory pool already */
bool Fluid::CheckIfExistsInMemPool(const CTxMemPool& pool, const CScript& fluidScriptPubKey, std::string& errorMessage) {

    for (const CTxMemPoolEntry& e : pool.mapTx) {
        const CTransaction& tx = e.GetTx();
        for (const CTxOut& txOut : tx.vout) {
            if (IsTransactionFluid(txOut.scriptPubKey)) {
                std::string strNewFluidScript = ScriptToAsmStr(fluidScriptPubKey);
                std::string strMemPoolFluidScript = ScriptToAsmStr(txOut.scriptPubKey);
                std::string strNewTxWithoutOpCode = GetRidOfScriptStatement(strNewFluidScript);
                std::string strMemPoolTxWithoutOpCode = GetRidOfScriptStatement(strMemPoolFluidScript);
                if (strNewTxWithoutOpCode == strMemPoolTxWithoutOpCode) {
                    errorMessage = "CheckIfExistsInMemPool: fluid transaction is already in the memory pool!";
                    LogPrintf("CheckIfExistsInMemPool: fluid transaction, %s is already in the memory pool! %s\n", tx.GetHash().ToString(), strNewTxWithoutOpCode);
                    return true;
                }
            }
        }
    }

    return false;
}

/** Checks whether as to parties have actually signed it - please use this with ones with the OP_CODE */
bool Fluid::CheckIfQuorumExists(const std::string consentToken, std::string &message, bool individual) {
    std::vector<std::string> fluidManagers;
    std::pair<CDynamicAddress, bool> keyOne;
    std::pair<CDynamicAddress, bool> keyTwo;
    std::pair<CDynamicAddress, bool> keyThree;
    keyOne.second = false, keyTwo.second = false;
    keyThree.second = false;

    GetLastBlockIndex(chainActive.Tip());
    CBlockIndex* pindex = chainActive.Tip();

    if (pindex != NULL)
        fluidManagers = pindex->fluidParams.fluidManagers;
    else
        fluidManagers = InitialiseAddresses();

    for (const std::string& address : fluidManagers) {
        CDynamicAddress attemptKey, xKey(address);

        if (!xKey.IsValid())
            return false;

        if (GenericVerifyInstruction(consentToken, attemptKey, message, 1) && xKey == attemptKey) {
            keyOne = std::make_pair(attemptKey.ToString(), true);
        }

        if (GenericVerifyInstruction(consentToken, attemptKey, message, 2) && xKey == attemptKey) {
            keyTwo = std::make_pair(attemptKey.ToString(), true);
        }

        if (GenericVerifyInstruction(consentToken, attemptKey, message, 3) && xKey == attemptKey) {
            keyThree = std::make_pair(attemptKey.ToString(), true);
        }
    }

    bool fValid = (keyOne.first.ToString() != keyTwo.first.ToString() && keyTwo.first.ToString() != keyThree.first.ToString()
                   && keyOne.first.ToString() != keyThree.first.ToString());

    LogPrint("fluid", "CheckIfQuorumExists(): Addresses validating this consent token are: %s, %s and %s\n", keyOne.first.ToString(), keyTwo.first.ToString(), keyThree.first.ToString());

    if (individual)
        return (keyOne.second || keyTwo.second || keyThree.second);
    else if (fValid)
        return (keyOne.second && keyTwo.second && keyThree.second);

    return false;
}


/** Checks whether as to parties have actually signed it - please use this with ones **without** the OP_CODE */
bool Fluid::CheckNonScriptQuorum(const std::string consentToken, std::string &message, bool individual) {
    std::string result = "12345 " + consentToken;
    return CheckIfQuorumExists(result, message, individual);
}

/** Because some things in life are meant to be intimate, like socks in a drawer */
bool Fluid::SignIntimateMessage(CDynamicAddress address, std::string unsignedMessage, std::string &stitchedMessage, bool stitch) {
#ifdef ENABLE_WALLET
    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << unsignedMessage;

    CDynamicAddress addr(address);

    CKeyID keyID;
    if (!addr.GetKeyID(keyID))
        return false;

    CKey key;
    if (!pwalletMain->GetKey(keyID, key))
        return false;

    std::vector<unsigned char> vchSig;
    if (!key.SignCompact(ss.GetHash(), vchSig))
        return false;
    else if(stitch)
        stitchedMessage = StitchString(unsignedMessage, EncodeBase64(&vchSig[0], vchSig.size()), false);
    else
        stitchedMessage = EncodeBase64(&vchSig[0], vchSig.size());

    return true;
#else
    return false;
#endif //ENABLE_WALLET
}

/** It will perform basic message signing functions */
bool Fluid::GenericSignMessage(const std::string message, std::string &signedString, CDynamicAddress signer) {
    if(!SignIntimateMessage(signer, message, signedString, true))
        return false;
    else
        ConvertToHex(signedString);

    return true;
}

/** It will append a signature of the new information */
bool Fluid::GenericConsentMessage(std::string message, std::string &signedString, CDynamicAddress signer) {
    std::string token, digest;

    if (!IsHex(message))
        return false;

    if(!CheckNonScriptQuorum(message, token, true))
        return false;

    if(token == "")
        return false;

    ConvertToString(message);

    if(!SignIntimateMessage(signer, token, digest, false))
        return false;

    signedString = StitchString(message, digest, false);

    ConvertToHex(signedString);

    return true;
}

/** Extract timestamp from a Fluid Transaction */
bool Fluid::ExtractCheckTimestamp(const std::string consentToken, const int64_t timeStamp) {
    std::string consentTokenNoScript = GetRidOfScriptStatement(consentToken);
    std::string dehexString = HexToString(consentTokenNoScript);
    std::vector<std::string> strs, ptrs;
    SeperateString(dehexString, strs, false);
    SeperateString(strs.at(0), ptrs, true);

    if(1 >= (int)strs.size())
        return false;

    std::string ls = ptrs.at(1);
    
    ScrubString(ls, true);
    int64_t tokenTimeStamp;
    ParseInt64(ls, &tokenTimeStamp);

    if (timeStamp > tokenTimeStamp + fluid.MAX_FLUID_TIME_DISTORT)
        return false;

    return true;
}

bool Fluid::ProcessFluidToken(const std::string consentToken, std::vector<std::string> &ptrs, int strVecNo) {
    std::string consentTokenNoScript = GetRidOfScriptStatement(consentToken);

    std::string message;
    if (!CheckNonScriptQuorum(consentTokenNoScript, message))
        return false;

    std::string dehexString = HexToString(consentTokenNoScript);

    std::vector<std::string> strs;
    SeperateString(dehexString, strs, false);
    SeperateString(strs.at(0), ptrs, true);

    if(strVecNo >= (int)strs.size())
        return false;

    return true;
}

/** It gets a number from the ASM of an OP_CODE without signature verification */
bool Fluid::GenericParseNumber(const std::string consentToken, const int64_t timeStamp, CAmount &coinAmount, bool txCheckPurpose) {
    std::vector<std::string> ptrs;

    if (!ProcessFluidToken(consentToken, ptrs, 1))
        return false;

    std::string lr = ptrs.at(0);
    ScrubString(lr, true);
    std::string ls = ptrs.at(1);
    ScrubString(ls, true);
    int64_t tokenTimeStamp;
    ParseInt64(ls, &tokenTimeStamp);

    if (timeStamp > tokenTimeStamp + fluid.MAX_FLUID_TIME_DISTORT && !txCheckPurpose)
        return false;

    ParseFixedPoint(lr, 8, &coinAmount);

    return true;
}

/** Individually checks the validity of an instruction */
bool Fluid::GenericVerifyInstruction(const std::string consentToken, CDynamicAddress& signer, std::string &messageTokenKey, int whereToLook)
{
    std::string consentTokenNoScript = GetRidOfScriptStatement(consentToken);
    messageTokenKey = "";
    std::vector<std::string> strs;

    ConvertToString(consentTokenNoScript);
    SeperateString(consentTokenNoScript, strs, false);

    messageTokenKey = strs.at(0);

    /* Don't even bother looking there there aren't enough digest keys or we are checking in the wrong place */
    if(whereToLook >= (int)strs.size() || whereToLook == 0)
        return false;

    std::string digestSignature = strs.at(whereToLook);

    bool fInvalid = false;
    std::vector<unsigned char> vchSig = DecodeBase64(digestSignature.c_str(), &fInvalid);

    if (fInvalid) {
        LogPrintf("GenericVerifyInstruction(): Digest Signature Found Invalid, Signature: %s \n", digestSignature);
        return false;
    }

    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << messageTokenKey;

    CPubKey pubkey;

    if (!pubkey.RecoverCompact(ss.GetHash(), vchSig)) {
        LogPrintf("GenericVerifyInstruction(): Public Key Recovery Failed! Hash: %s\n", ss.GetHash().ToString());
        return false;
    }

    signer = CDynamicAddress(pubkey.GetID());

    return true;
}

bool Fluid::ParseMintKey(const int64_t nTime, CDynamicAddress &destination, CAmount &coinAmount, std::string uniqueIdentifier, bool txCheckPurpose) {
    std::vector<std::string> ptrs;

    if (!ProcessFluidToken(uniqueIdentifier, ptrs, 1))
        return false;

    if(2 >= (int)ptrs.size())
        return false;

    std::string lr = ptrs.at(0);
    ScrubString(lr, true);
    std::string ls = ptrs.at(1);
    ScrubString(ls, true);
    int64_t tokenTimeStamp;
    ParseInt64(ls, &tokenTimeStamp);
    
    if (nTime > tokenTimeStamp + fluid.MAX_FLUID_TIME_DISTORT && !txCheckPurpose)
        return false;

    ParseFixedPoint(lr, 8, &coinAmount);

    std::string recipientAddress = ptrs.at(2);
    destination.SetString(recipientAddress);

    if(!destination.IsValid())
        return false;

    LogPrintf("ParseMintKey(): Token Data -- Address %s | Coins to be minted: %s | Time: %s\n", ptrs.at(2), coinAmount / COIN, ls);

    return true;
}

bool Fluid::GetMintingInstructions(const CBlockIndex* pblockindex, CDynamicAddress &toMintAddress, CAmount& mintAmount) {
    CBlock block;

    if (pblockindex != nullptr) {
        if(!ReadBlockFromDisk(block, pblockindex, Params().GetConsensus())) {
            LogPrintf("Unable to read from disk! Highly unlikely but has occured, may be bug or damaged blockchain copy!\n");
            return false;
        }
    } else {
        return false;
    }

    for (const CTransaction& tx : block.vtx) {
        for (const CTxOut& txout : tx.vout) {
            if (txout.scriptPubKey.IsProtocolInstruction(MINT_TX)) {
                std::string message;
                if (CheckIfQuorumExists(ScriptToAsmStr(txout.scriptPubKey), message))
                    return ParseMintKey(block.nTime, toMintAddress, mintAmount, ScriptToAsmStr(txout.scriptPubKey));
            }
        }
    }

    return false;
}

bool Fluid::GetProofOverrideRequest(const CBlockIndex* pblockindex, CAmount &coinAmount) {

    CBlock block;
    if (pblockindex != nullptr) {
        if(!ReadBlockFromDisk(block, pblockindex, Params().GetConsensus())) {
            LogPrintf("Unable to read from disk! Highly unlikely but has occured, may be bug or damaged blockchain copy!\n");
            return false;
        }
    }  else {
        return false;
    }

    for (const CTransaction& tx : block.vtx) {
        for (const CTxOut& txout : tx.vout) {
            if (txout.scriptPubKey.IsProtocolInstruction(MINING_MODIFY_TX)) {
                std::string message;
                if (CheckIfQuorumExists(ScriptToAsmStr(txout.scriptPubKey), message))
                    return GenericParseNumber(ScriptToAsmStr(txout.scriptPubKey), block.nTime, coinAmount);
            }
        }
    }
    return false;
}

bool Fluid::GetDynodeOverrideRequest(const CBlockIndex* pblockindex, CAmount &coinAmount) {
    CBlock block;

    if (pblockindex != nullptr) {
        if(!ReadBlockFromDisk(block, pblockindex, Params().GetConsensus())) {
            LogPrintf("Unable to read from disk! Highly unlikely but has occured, may be bug or damaged blockchain copy!\n");
            return false;
        }
    } else {
        return false;
    }

    for (const CTransaction& tx : block.vtx) {
        for (const CTxOut& txout : tx.vout) {
            if (txout.scriptPubKey.IsProtocolInstruction(DYNODE_MODFIY_TX)) {
                std::string message;
                if (CheckIfQuorumExists(ScriptToAsmStr(txout.scriptPubKey), message))
                    return GenericParseNumber(ScriptToAsmStr(txout.scriptPubKey), block.nTime, coinAmount);
            }
        }
    }
    return false;
}

void Fluid::AddFluidTransactionsToRecord(const CBlockIndex* pblockindex, std::vector<std::string>& transactionRecord) {
    CBlock block;
    std::string message;

    if (pblockindex != nullptr) {
        if(!ReadBlockFromDisk(block, pblockindex, Params().GetConsensus())) {
            LogPrintf("Unable to read from disk! Highly unlikely but has occured, may be bug or damaged blockchain copy!\n");
            return;
        }
    } else {
        return;
    }

    for (const CTransaction& tx : block.vtx) {
        for (const CTxOut& txout : tx.vout) {
            if (IsTransactionFluid(txout.scriptPubKey)) {
                if (!InsertTransactionToRecord(txout.scriptPubKey, transactionRecord)) {
                    LogPrintf("AddFluidTransactionsToRecord(): Script Database Entry: %s , FAILED!\n", ScriptToAsmStr(txout.scriptPubKey));
                }
            }
        }
    }
}

/* Check if transaction exists in record */
bool Fluid::CheckTransactionInRecord(CScript fluidInstruction, CBlockIndex* pindex) {
    std::string verificationString;
    CFluidEntry fluidIndex;
    if (chainActive.Height() <= fluid.FLUID_ACTIVATE_HEIGHT)
        return false;
    else if (pindex == nullptr) {
        GetLastBlockIndex(chainActive.Tip());
        fluidIndex = chainActive.Tip()->fluidParams;
    } else
        fluidIndex = pindex->fluidParams;

    std::vector<std::string> transactionRecord = fluidIndex.fluidHistory;

    if (IsTransactionFluid(fluidInstruction)) {
        verificationString = ScriptToAsmStr(fluidInstruction);
        std::string verificationWithoutOpCode = GetRidOfScriptStatement(verificationString);
        std::string message;
        if (CheckIfQuorumExists(verificationString, message)) {
            for (const std::string& existingRecord : transactionRecord)
            {
                std::string existingWithoutOpCode = GetRidOfScriptStatement(existingRecord);
                LogPrint("fluid", "CheckTransactionInRecord(): operation code removed. existingRecord  = %s verificationString = %s\n", existingWithoutOpCode, verificationWithoutOpCode);
                if (existingWithoutOpCode == verificationWithoutOpCode) {
                    LogPrintf("CheckTransactionInRecord(): Attempt to repeat Fluid Transaction: %s\n", existingRecord);
                    return true;
                }
            }
        }
    }

    return false;
}

/* Insertion of transaction script to record */
bool Fluid::InsertTransactionToRecord(CScript fluidInstruction, std::vector<std::string>& transactionRecord) {
    std::string verificationString;

    if (IsTransactionFluid(fluidInstruction)) {
        verificationString = ScriptToAsmStr(fluidInstruction);

        std::string message;
        if (CheckIfQuorumExists(verificationString, message)) {
            for (const std::string& existingRecord : transactionRecord)
            {
                if (existingRecord == verificationString) {
                    return false;
                }
            }
            transactionRecord.push_back(verificationString);
            return true;
        }
    }

    return false;
}

CAmount GetPoWBlockPayment(const int& nHeight)
{
    if (chainActive.Height() == 0) {
        CAmount nSubsidy = INITIAL_SUPERBLOCK_PAYMENT;
        LogPrint("superblock creation", "GetPoWBlockPayment() : create=%s nSubsidy=%d\n", FormatMoney(nSubsidy), nSubsidy);
        return nSubsidy;
    }
    else if (chainActive.Height() >= 1 && chainActive.Height() <= Params().GetConsensus().nRewardsStart) {
        LogPrint("zero-reward block creation", "GetPoWBlockPayment() : create=%s nSubsidy=%d\n", FormatMoney(BLOCKCHAIN_INIT_REWARD), BLOCKCHAIN_INIT_REWARD);
        return BLOCKCHAIN_INIT_REWARD; // Burn transaction fees
    }
    else if (chainActive.Height() > Params().GetConsensus().nRewardsStart) {
        LogPrint("creation", "GetPoWBlockPayment() : create=%s PoW Reward=%d\n", FormatMoney(PHASE_1_POW_REWARD), PHASE_1_POW_REWARD);
        return PHASE_1_POW_REWARD; // 1 DYN  and burn transaction fees
    }
    else
        return BLOCKCHAIN_INIT_REWARD; // Burn transaction fees
}

CAmount GetDynodePayment(bool fDynode)
{   
    if (fDynode && chainActive.Height() > Params().GetConsensus().nDynodePaymentsStartBlock && chainActive.Height() < Params().GetConsensus().nUpdateDiffAlgoHeight) {
        LogPrint("creation", "GetDynodePayment() : create=%s DN Payment=%d\n", FormatMoney(PHASE_1_DYNODE_PAYMENT), PHASE_1_DYNODE_PAYMENT);
        return PHASE_1_DYNODE_PAYMENT; // 0.382 DYN
    }
    else if (fDynode && chainActive.Height() > Params().GetConsensus().nDynodePaymentsStartBlock && chainActive.Height() >= Params().GetConsensus().nUpdateDiffAlgoHeight) {
        LogPrint("creation", "GetDynodePayment() : create=%s DN Payment=%d\n", FormatMoney(PHASE_2_DYNODE_PAYMENT), PHASE_2_DYNODE_PAYMENT);
        return PHASE_2_DYNODE_PAYMENT; // 1.618 DYN
    }
    else if ((fDynode && !fDynode) && chainActive.Height() <= Params().GetConsensus().nDynodePaymentsStartBlock) {
        LogPrint("creation", "GetDynodePayment() : create=%s DN Payment=%d\n", FormatMoney(BLOCKCHAIN_INIT_REWARD), BLOCKCHAIN_INIT_REWARD);
        return BLOCKCHAIN_INIT_REWARD;
    }
    else
        return BLOCKCHAIN_INIT_REWARD;
}

/** Passover code that will act as a switch to check if override did occur for Proof of Work Rewards **/
CAmount getBlockSubsidyWithOverride(const int& nHeight, CAmount lastOverrideCommand) {
    if (lastOverrideCommand != 0) {
        return lastOverrideCommand;
    } else {
        return GetPoWBlockPayment(nHeight);
    }
}

/** Passover code that will act as a switch to check if override did occur for Dynode Rewards **/
CAmount getDynodeSubsidyWithOverride(CAmount lastOverrideCommand, bool fDynode) {
    if (lastOverrideCommand != 0) {
        return lastOverrideCommand;
    } else {
        return GetDynodePayment(fDynode);
    }
}

bool Fluid::ValidationProcesses(CValidationState &state, CScript txOut, CAmount txValue) {
    std::string message;
    CAmount mintAmount;
    CDynamicAddress toMintAddress;

    if (IsTransactionFluid(txOut)) {
        if (!CheckIfQuorumExists(ScriptToAsmStr(txOut), message)) {
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-fluid-auth-failure");
        }

        if (txOut.IsProtocolInstruction(MINT_TX) &&
                !ParseMintKey(0, toMintAddress, mintAmount, ScriptToAsmStr(txOut), true)) {
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-fluid-mint-auth-failure");
        }

        if ((txOut.IsProtocolInstruction(DYNODE_MODFIY_TX) ||
                txOut.IsProtocolInstruction(MINING_MODIFY_TX)) &&
                !GenericParseNumber(ScriptToAsmStr(txOut), 0, mintAmount, true)) {
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-fluid-modify-parse-failure");
        }
    }

    return true;
}

bool Fluid::ProvisionalCheckTransaction(const CTransaction &transaction) {
    for (const CTxOut& txout : transaction.vout) {
        CScript txOut = txout.scriptPubKey;

        if (IsTransactionFluid(txOut) && CheckTransactionInRecord(txOut)) {
            LogPrintf("ProvisionalCheckTransaction(): Fluid Transaction %s has already been executed!\n", transaction.GetHash().ToString());
            return false;
        }
    }

    return true;
}

bool Fluid::CheckTransactionToBlock(const CTransaction &transaction, const CBlockHeader& blockHeader) {
    uint256 hash = blockHeader.GetHash();
    if (mapBlockIndex.count(hash) == 0)
        return true;

    CBlockIndex* pblockindex = mapBlockIndex[hash];

    for (const CTxOut& txout : transaction.vout) {
        CScript txOut = txout.scriptPubKey;

        if (IsTransactionFluid(txOut) && CheckTransactionInRecord(txOut, pblockindex)) {
            LogPrintf("CheckTransactionToBlock(): Fluid Transaction %s has already been executed!\n", transaction.GetHash().ToString());
            return false;
        }
    }

    return true;
}