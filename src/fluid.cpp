// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers

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

CFluid fluid;

#ifdef ENABLE_WALLET
extern CWallet* pwalletMain;
#endif //ENABLE_WALLET

bool IsTransactionFluid(CScript txOut) {
    return (txOut.IsProtocolInstruction(MINT_TX)
            || txOut.IsProtocolInstruction(DYNODE_MODFIY_TX)
            || txOut.IsProtocolInstruction(MINING_MODIFY_TX)
           );
}

/** Initialise sovereign identities that are able to run fluid commands */
std::vector<std::pair<std::string, CDynamicAddress>> CFluidParameters::InitialiseSovereignIdentities() {
    std::vector<std::pair<std::string, CDynamicAddress>> x;
    if (Params().NetworkIDString() == CBaseChainParams::MAIN) {
        x.push_back(std::make_pair("CEO",       CDynamicAddress("D9avNWVBmaUNevMNnkcLMrQpze8M2mKURu")));
        x.push_back(std::make_pair("CTO",       CDynamicAddress("DRoyjRoxP4qfeAiiZHX1dmSkbUJiBSXBt7")));
        x.push_back(std::make_pair("CFO",       CDynamicAddress("DHkD6oBQ5PtCiKo4wX8CRWrG61Vy5hEu4t")));
        x.push_back(std::make_pair("COO",       CDynamicAddress("DKyqamefa7YdbqrP5pdTfNVVuq1gerNhMH")));
        x.push_back(std::make_pair("CDOO",      CDynamicAddress("DUDE1zFKK4fezCgcxdGbFh4yHJMcg8qpoP")));
    }
    else if (Params().NetworkIDString() == CBaseChainParams::TESTNET) {
        x.push_back(std::make_pair("Test01",    CDynamicAddress("DSCex4e189aULrig3nLd42gVf7AbjTwnP5"))); //importprivatekey QVKXuZ2hSo2cT9BhkN3CApLuZYVsuzNvidJRt1ucyniHheZ2Pfq5
        x.push_back(std::make_pair("Test02",    CDynamicAddress("DMAh37n3RUdDxox3uiWAnc1zEPp5yFbHiL"))); //importprivatekey QU4VGDcVoej7nDZiyaSgoL7foG8xKiaVyk5odHnJdtyv4tYkmBw1 
        x.push_back(std::make_pair("Test03",    CDynamicAddress("DN4KvqtXyygooPV3oha72TyBB5nqBbkxwj"))); //importprivatekey QWjTe6sCFVtKBsXfrYDyrHzn7eBeJktsQnWzfiANkMd9PhVM4Qnp
        x.push_back(std::make_pair("Test04",    CDynamicAddress("DHVmS621KBBZJTJSxGDdLxoU7LCmpexWDa"))); //importprivatekey QScWuazWgWDTj8cXXz1YFKJW7mNJHJgMFY2FB6hkNyh3SJDUhPZt
        x.push_back(std::make_pair("Test05",    CDynamicAddress("DCZXDSRB3cJdCCUSerE4pvSfGQoXUivUxo"))); //importprivatekey QUt4pEDanRPzos3meoiNGUG9g7RctCtiwLoPjhDKfNPK99oLuzcU
    }
    else if (Params().NetworkIDString() == CBaseChainParams::REGTEST) {
        x.push_back(std::make_pair("RegTest01", CDynamicAddress("DSCex4e189aULrig3nLd42gVf7AbjTwnP5"))); //importprivatekey QVKXuZ2hSo2cT9BhkN3CApLuZYVsuzNvidJRt1ucyniHheZ2Pfq5
        x.push_back(std::make_pair("RegTest02", CDynamicAddress("DMAh37n3RUdDxox3uiWAnc1zEPp5yFbHiL"))); //importprivatekey QU4VGDcVoej7nDZiyaSgoL7foG8xKiaVyk5odHnJdtyv4tYkmBw1 
        x.push_back(std::make_pair("RegTest03", CDynamicAddress("DN4KvqtXyygooPV3oha72TyBB5nqBbkxwj"))); //importprivatekey QWjTe6sCFVtKBsXfrYDyrHzn7eBeJktsQnWzfiANkMd9PhVM4Qnp
        x.push_back(std::make_pair("RegTest04", CDynamicAddress("DHVmS621KBBZJTJSxGDdLxoU7LCmpexWDa"))); //importprivatekey QScWuazWgWDTj8cXXz1YFKJW7mNJHJgMFY2FB6hkNyh3SJDUhPZt
        x.push_back(std::make_pair("RegTest05", CDynamicAddress("DCZXDSRB3cJdCCUSerE4pvSfGQoXUivUxo"))); //importprivatekey QUt4pEDanRPzos3meoiNGUG9g7RctCtiwLoPjhDKfNPK99oLuzcU 
    }
    return x;
}

/** Checks if any given address is a current master key (invoked by RPC) */
bool CFluid::IsGivenKeyMaster(CDynamicAddress inputKey) {
    if (!inputKey.IsValid()) {
        return false;
    }

    std::vector<std::string> fluidSovereigns;
    GetLastBlockIndex(chainActive.Tip());
    CBlockIndex* pindex = chainActive.Tip();

    if (pindex != NULL)
        fluidSovereigns = pindex->fluidParams.fluidSovereigns;
    else
        fluidSovereigns = InitialiseAddresses();

    for (const std::string& strAddress : fluidSovereigns) {
        CDynamicAddress attemptKey(strAddress);
        if (attemptKey.IsValid() && inputKey == attemptKey)
            return true;
    }

    return false;
}

std::vector<std::string> CFluidParameters::InitialiseAddresses() {
    std::vector<std::string> initialSovereignAddresses;
    std::vector<std::pair<std::string, CDynamicAddress>> fluidIdentities = InitialiseSovereignIdentities();
    for (const std::pair<std::string, CDynamicAddress>& sovereignId : fluidIdentities) {
        initialSovereignAddresses.push_back(sovereignId.second.ToString());
    }
    return initialSovereignAddresses;
}

/** Checks fluid transactoin operation script amount for invalid values. */
bool CFluid::CheckFluidOperationScript(const CScript& fluidScriptPubKey, const int64_t timeStamp, std::string& errorMessage, bool fSkipTimeStampCheck) {
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
        SeparateString(strUnHexedFluidOpScript, vecSplitScript, "$");
        if (vecSplitScript.size() > 1) {
            strAmount = vecSplitScript[0];
            CAmount fluidAmount;
            if (ParseFixedPoint(strAmount, 8, &fluidAmount)) {
                if ((strOperationCode == "OP_REWARD_MINING" || strOperationCode == "OP_REWARD_DYNODE") && fluidAmount < 0) {
                    errorMessage = "CheckFluidOperationScript fluid reward amount is less than zero: " + strAmount;
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
bool CFluid::CheckIfExistsInMemPool(const CTxMemPool& pool, const CScript& fluidScriptPubKey, std::string& errorMessage) {

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
bool CFluid::CheckIfQuorumExists(const std::string consentToken, std::string &message, bool individual) {
    std::vector<std::string> fluidSovereigns;
    std::pair<CDynamicAddress, bool> keyOne;
    std::pair<CDynamicAddress, bool> keyTwo;
    std::pair<CDynamicAddress, bool> keyThree;
    keyOne.second = false, keyTwo.second = false;
    keyThree.second = false;

    GetLastBlockIndex(chainActive.Tip());
    CBlockIndex* pindex = chainActive.Tip();

    if (pindex != NULL)
        fluidSovereigns = pindex->fluidParams.fluidSovereigns;
    else
        fluidSovereigns = InitialiseAddresses();

    for (const std::string& address : fluidSovereigns) {
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
bool CFluid::CheckNonScriptQuorum(const std::string consentToken, std::string &message, bool individual) {
    std::string result = "12345 " + consentToken;
    return CheckIfQuorumExists(result, message, individual);
}

/** It will append a signature of the new information */
bool CFluid::GenericConsentMessage(std::string message, std::string &signedString, CDynamicAddress signer) {
    std::string token, digest;

    if (!IsHex(message))
        return false;

    if(!CheckNonScriptQuorum(message, token, true))
        return false;

    if(token == "")
        return false;

    ConvertToString(message);

    if(!SignTokenMessage(signer, token, digest, false))
        return false;

    signedString = StitchString(message, digest, false);

    ConvertToHex(signedString);

    return true;
}

/** Extract timestamp from a Fluid Transaction */
bool CFluid::ExtractCheckTimestamp(const std::string consentToken, const int64_t timeStamp) {
    std::string consentTokenNoScript = GetRidOfScriptStatement(consentToken);
    std::string dehexString = HexToString(consentTokenNoScript);
    std::vector<std::string> strs, ptrs;
    SeparateString(dehexString, strs, false);
    SeparateString(strs.at(0), ptrs, true);

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

bool CFluid::ProcessFluidToken(const std::string consentToken, std::vector<std::string>& ptrs, int strVecNo) {
    std::string consentTokenNoScript = GetRidOfScriptStatement(consentToken);

    std::string message;
    if (!CheckNonScriptQuorum(consentTokenNoScript, message))
        return false;

    std::string dehexString = HexToString(consentTokenNoScript);

    std::vector<std::string> strs;
    SeparateString(dehexString, strs, false);
    SeparateString(strs.at(0), ptrs, true);

    if(strVecNo >= (int)strs.size())
        return false;

    return true;
}

/** It gets a number from the ASM of an OP_CODE without signature verification */
bool CFluid::GenericParseNumber(const std::string consentToken, const int64_t timeStamp, CAmount& coinAmount, bool txCheckPurpose) {
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

CDynamicAddress CFluid::GetAddressFromDigestSignature(const std::string digestSignature, const std::string messageTokenKey) {
    bool fInvalid = false;
    std::vector<unsigned char> vchSig = DecodeBase64(digestSignature.c_str(), &fInvalid);
    
    if (fInvalid) {
        LogPrintf("GenericVerifyInstruction(): Digest Signature Found Invalid, Signature: %s \n", digestSignature);
        return nullptr;
    }

    CHashWriter ss(SER_GETHASH, 0);
    ss << strMessageMagic;
    ss << messageTokenKey;

    CPubKey pubkey;

    if (!pubkey.RecoverCompact(ss.GetHash(), vchSig)) {
        LogPrintf("GenericVerifyInstruction(): Public Key Recovery Failed! Hash: %s\n", ss.GetHash().ToString());
        return nullptr;
    }
    CDynamicAddress newAddress;
    newAddress.Set(pubkey.GetID());
    return newAddress;
}

/** Individually checks the validity of an instruction */
bool CFluid::GenericVerifyInstruction(const std::string consentToken, CDynamicAddress& signer, std::string& messageTokenKey, int whereToLook)
{
    std::string consentTokenNoScript = GetRidOfScriptStatement(consentToken);
    messageTokenKey = "";
    std::vector<std::string> strs;

    ConvertToString(consentTokenNoScript);
    SeparateString(consentTokenNoScript, strs, false);

    messageTokenKey = strs.at(0);

    /* Don't even bother looking there there aren't enough digest keys or we are checking in the wrong place */
    if(whereToLook >= (int)strs.size() || whereToLook == 0)
        return false;

    std::string digestSignature = strs.at(whereToLook);

    signer = GetAddressFromDigestSignature(digestSignature, messageTokenKey);

    return true;
}

bool CFluid::ParseMintKey(const int64_t nTime, CDynamicAddress& destination, CAmount& coinAmount, std::string uniqueIdentifier, bool txCheckPurpose) {
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

static bool GetFluidBlock(const CBlockIndex* pblockindex, CBlock& block)
{
    if (pblockindex != nullptr) {
        // Check for invalid block position and file. 
        const CDiskBlockPos pos = pblockindex->GetBlockPos();
        if (pos.nFile > -1 && pos.nPos > 0) {
            if(!ReadBlockFromDisk(block, pblockindex, Params().GetConsensus())) {
                LogPrintf("Unable to read from disk! Highly unlikely but has occured, may be bug or damaged blockchain copy!\n");
                return false;
            }
        }
        else {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool CFluid::GetMintingInstructions(const CBlockIndex* pblockindex, CDynamicAddress& toMintAddress, CAmount& mintAmount) {

    CBlock block;
    if (GetFluidBlock(pblockindex, block)) {
        for (const CTransaction& tx : block.vtx) {
            for (const CTxOut& txout : tx.vout) {
                if (txout.scriptPubKey.IsProtocolInstruction(MINT_TX)) {
                    std::string message;
                    if (CheckIfQuorumExists(ScriptToAsmStr(txout.scriptPubKey), message))
                        return ParseMintKey(block.nTime, toMintAddress, mintAmount, ScriptToAsmStr(txout.scriptPubKey));
                }
            }
        }
    }
    return false;
}

bool CFluid::GetProofOverrideRequest(const CBlockIndex* pblockindex, CAmount& coinAmount) {

    CBlock block;
    if (GetFluidBlock(pblockindex, block)) {
        for (const CTransaction& tx : block.vtx) {
            for (const CTxOut& txout : tx.vout) {
                if (txout.scriptPubKey.IsProtocolInstruction(MINING_MODIFY_TX)) {
                    std::string message;
                    if (CheckIfQuorumExists(ScriptToAsmStr(txout.scriptPubKey), message))
                        return GenericParseNumber(ScriptToAsmStr(txout.scriptPubKey), block.nTime, coinAmount);
                }
            }
        }
    }
    return false;
}

bool CFluid::GetDynodeOverrideRequest(const CBlockIndex* pblockindex, CAmount& coinAmount) {

    CBlock block;
    if (GetFluidBlock(pblockindex, block)) {
        for (const CTransaction& tx : block.vtx) {
            for (const CTxOut& txout : tx.vout) {
                if (txout.scriptPubKey.IsProtocolInstruction(DYNODE_MODFIY_TX)) {
                    std::string message;
                    if (CheckIfQuorumExists(ScriptToAsmStr(txout.scriptPubKey), message))
                        return GenericParseNumber(ScriptToAsmStr(txout.scriptPubKey), block.nTime, coinAmount);
                }
            }
        }
    }
    return false;
}

void CFluid::AddFluidTransactionsToRecord(const CBlockIndex* pblockindex, std::vector<std::string>& transactionRecord) {
    
    CBlock block;
    if (GetFluidBlock(pblockindex, block)) {
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
}

/* Check if transaction exists in record */
bool CFluid::CheckTransactionInRecord(CScript fluidInstruction, CBlockIndex* pindex) {
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
bool CFluid::InsertTransactionToRecord(CScript fluidInstruction, std::vector<std::string>& transactionRecord) {
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

bool CFluid::ValidationProcesses(CValidationState& state, CScript txOut, CAmount txValue) {
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

bool CFluid::ProvisionalCheckTransaction(const CTransaction& transaction) {
    for (const CTxOut& txout : transaction.vout) {
        CScript txOut = txout.scriptPubKey;

        if (IsTransactionFluid(txOut) && CheckTransactionInRecord(txOut)) {
            LogPrintf("ProvisionalCheckTransaction(): Fluid Transaction %s has already been executed!\n", transaction.GetHash().ToString());
            return false;
        }
    }

    return true;
}

bool CFluid::CheckTransactionToBlock(const CTransaction& transaction, const CBlockHeader& blockHeader) {
    uint256 hash = blockHeader.GetHash();
    return CheckTransactionToBlock(transaction, hash);
}

bool CFluid::CheckTransactionToBlock(const CTransaction& transaction, const uint256 hash) {
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