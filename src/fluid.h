// Copyright (c) 2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef FLUID_PROTOCOL_H
#define FLUID_PROTOCOL_H

#include "base58.h"
#include "amount.h"
#include "chain.h"
#include "script/script.h"
#include "consensus/validation.h"
#include "utilstrencodings.h"
#include "dbwrapper.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

class CBlock;
class CTxMemPool;
struct CBlockTemplate;

/** Configuration Framework */
class CFluidParameters {
public:
    static const int FLUID_ACTIVATE_HEIGHT = 10;
    static const int64_t MAX_FLUID_TIME_DISTORT = 60 * 60; // Maximum time distort = 1 hour.
    static const CAmount FLUID_TRANSACTION_COST = 100000 * COIN; // Cost to send a fluid transaction
    static const CAmount FLUID_MAX_REWARD_FOR_DYNODE = 1000 * COIN; // Max dynode block reward using fluid OP_REWARD_DYNODE
    static const CAmount FLUID_MAX_REWARD_FOR_MINING = 1000 * COIN; // Max mining block reward using fluid OP_REWARD_MINING
    static const CAmount FLUID_MAX_FOR_MINT = 1000000000 * COIN; // Max minting amount per fluid transaction


    //TODO (Amir): Change address and remove private keys below.
    std::vector<std::string> InitialiseAddresses() {
        std::vector<std::string> x;
        x.push_back("DSZcL75kYD6hWWoU1xecuZ9gMVB8fqccP6"); // CEO
        x.push_back("DPTotn3YNajtyAvAx9KQrKa98JwuKZKceD"); // CTO
        x.push_back("DJW7o975ycmJWBzzXQSP2nLSnECyGLTozM"); // CFO
        x.push_back("DAhJS4uWw2dfr2h3mofoi9ffBrhgqkrHbD"); // COO
        x.push_back("DJsGhpDtA9z4yquAFXiy48sgtQQcYhhq1K"); // CDOO
        return x;
    }

    std::vector<std::string> InitialiseIdentities() {
        std::vector<std::string> x;
        x.push_back("ChickenFishAndChips");
        x.push_back("AvanaTheMermaid");
        x.push_back("PomphretFish");
        x.push_back("LudicrousLunch");
        x.push_back("PromiscuousPanda");
        return x;
    }
};

/** Fluid Asset Management Framework */
class Fluid : public CFluidParameters, public HexFunctions {
public:
    void AddFluidTransactionsToRecord(const CBlockIndex* pblockindex, std::vector<std::string>& transactionRecord);
    void ReplaceFluidSovereigns(const CBlockHeader& blockHeader, std::vector<std::string>& fluidSovereigns);

    bool IsGivenKeyMaster(CDynamicAddress inputKey);
    bool CheckFluidOperationScript(const CScript& fluidScriptPubKey, const int64_t timeStamp, std::string& errorMessage, bool fSkipTimeStampCheck = false);
    bool CheckIfExistsInMemPool(const CTxMemPool& pool, const CScript& fluidScriptPubKey, std::string& errorMessage);
    bool CheckIfQuorumExists(const std::string consentToken, std::string &message, bool individual = false);
    bool GenericConsentMessage(std::string message, std::string &signedString, CDynamicAddress signer);
    bool CheckNonScriptQuorum(const std::string consentToken, std::string &message, bool individual = false);
    bool InitiateFluidVerify(CDynamicAddress dynamicAddress);
    bool SignIntimateMessage(CDynamicAddress address, std::string unsignedMessage, std::string &stitchedMessage, bool stitch = true);
    bool GenericSignMessage(const std::string message, std::string &signedString, CDynamicAddress signer);
    bool GenericParseNumber(const std::string consentToken, const int64_t timeStamp, CAmount &howMuch, bool txCheckPurpose=false);
    bool GenericVerifyInstruction(const std::string consentToken, CDynamicAddress &signer, std::string &messageTokenKey, int whereToLook=1);
    bool ParseMintKey(const int64_t nTime, CDynamicAddress &destination, CAmount &coinAmount, std::string uniqueIdentifier, bool txCheckPurpose=false);
    bool GetMintingInstructions(const CBlockIndex* pblockindex, CDynamicAddress &toMintAddress, CAmount& mintAmount);
    bool GetProofOverrideRequest(const CBlockIndex* pblockindex, CAmount &howMuch);
    bool GetDynodeOverrideRequest(const CBlockIndex* pblockindex, CAmount &howMuch);
    bool InsertTransactionToRecord(CScript fluidInstruction, std::vector<std::string>& transactionRecord);
    bool CheckTransactionInRecord(CScript fluidInstruction, CBlockIndex* pindex = NULL);
    bool ValidationProcesses(CValidationState& state, CScript txOut, CAmount txValue);
    bool ExtractCheckTimestamp(const std::string consentToken, const int64_t timeStamp);
    bool ProvisionalCheckTransaction(const CTransaction &transaction);
    bool CheckTransactionToBlock(const CTransaction &transaction, const CBlockHeader& blockHeader);
    bool ProcessFluidToken(const std::string consentToken, std::vector<std::string> &ptrs, int strVecNo);
};

/** Standard Reward Payment Determination Functions */
CAmount GetPoWBlockPayment(const int& nHeight);
CAmount GetDynodePayment(bool fDynode = true);

/** Override Logic Switch for Reward Payment Determination Functions */
CAmount getBlockSubsidyWithOverride(const int& nHeight, CAmount lastOverrideCommand);
CAmount getDynodeSubsidyWithOverride(CAmount lastOverrideCommand, bool fDynode = true);

void BuildFluidInformationIndex(CBlockIndex* pindex, CAmount &nExpectedBlockValue, bool fDynodePaid);
bool IsTransactionFluid(CScript txOut);

extern Fluid fluid;

#endif // FLUID_PROTOCOL_H