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
class CBlockTemplate;

/** Configuration Framework */
class CFluidParameters {
public:
    static const int FLUID_ACTIVATE_HEIGHT = 10;
    static const int MAX_FLUID_TIME_DISTORT = 8 * 60;
    static const int FEE_REDIRECT_HEIGHT = 100; //TODO (Amir): Change this to 250000
    static const CAmount MAX_FLUID_MINER_REWARD = 3; // Maximum DYN miner block reward amount 
    static const CAmount MAX_FLUID_DYNODE_REWARD = 5; // Maximum DYN Dynode block reward amount 
    static const CAmount MAX_FLUID_DYNODE_RECIPIENT_COUNT = 5; // Maximum Dynode recipient count per block

    //TODO (Amir): Change address and remove private keys below.
    std::string FEE_REDIRECT_ADDRESS = "DGkRapsj7yQSvYaf46pHYUPwJrakpMd3Sh"; // importprivkey 5iw3QNmomgDceCTEZCLtEaeqSsyJXvKf28fc9rCi1iymYzokt4L

    std::vector<std::string> InitialiseAddresses() {
        std::vector<std::string> x;
        x.push_back("DSZcL75kYD6hWWoU1xecuZ9gMVB8fqccP6"); // importprivkey 5hUhjdy2JgVgSrye7nB8SWkSTTMjW7rRDjkH1HcPh9q6DexXPdh
        x.push_back("DPTotn3YNajtyAvAx9KQrKa98JwuKZKceD"); // importprivkey 5iUn32hTbpQhH11rFrtkSZXxAjLcF81znxTsnQLhdFiG42cdKr8
        x.push_back("DJW7o975ycmJWBzzXQSP2nLSnECyGLTozM"); // importprivkey 5ihNmnWmvLRSDcwA4aFUXNjXakNTcDjpUGTXYuedwB2Xzp58CPo
        x.push_back("DAhJS4uWw2dfr2h3mofoi9ffBrhgqkrHbD"); // importprivkey 5hvA5xQEtH1CUzhxTCBrjgiQhPwrmenMGPL23DruYoduASR4wWm
        x.push_back("DJsGhpDtA9z4yquAFXiy48sgtQQcYhhq1K"); // importprivkey 5i2rPucptq3ojq5gccpa8FkdXuF1jiBiG4fvHB2bGUMQng1Y2iM
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
    void ReplaceFluidMasters(const CBlockHeader& blockHeader, std::vector<std::string>& fluidManagers);

    bool IsGivenKeyMaster(CDynamicAddress inputKey);
    bool CheckIfQuorumExists(std::string token, std::string &message, bool individual = false);
    bool GenericConsentMessage(std::string message, std::string &signedString, CDynamicAddress signer);
    bool CheckNonScriptQuorum(std::string token, std::string &message, bool individual = false);
    bool InitiateFluidVerify(CDynamicAddress dynamicAddress);
    bool SignIntimateMessage(CDynamicAddress address, std::string unsignedMessage, std::string &stitchedMessage, bool stitch = true);
    bool GenericSignMessage(std::string message, std::string &signedString, CDynamicAddress signer);
    bool GenericParseNumber(std::string scriptString, int64_t timeStamp, CAmount &howMuch, bool txCheckPurpose=false);
    bool GenericVerifyInstruction(std::string uniqueIdentifier, CDynamicAddress &signer, std::string &messageTokenKey, int whereToLook=1);
    bool ParseMintKey(int64_t nTime, CDynamicAddress &destination, CAmount &coinAmount, std::string uniqueIdentifier, bool txCheckPurpose=false);
    bool GetMintingInstructions(const CBlockIndex* pblockindex, CDynamicAddress &toMintAddress, CAmount& mintAmount);
    bool GetProofOverrideRequest(const CBlockIndex* pblockindex, CAmount &howMuch);
    bool GetDynodeOverrideRequest(const CBlockIndex* pblockindex, CAmount &howMuch);
    bool InsertTransactionToRecord(CScript fluidInstruction, std::vector<std::string>& transactionRecord);
    bool CheckTransactionInRecord(CScript fluidInstruction, CBlockIndex* pindex = NULL);
    bool ValidationProcesses(CValidationState& state, CScript txOut, CAmount txValue);
    bool ExtractCheckTimestamp(std::string scriptString, int64_t timeStamp);
    bool ProvisionalCheckTransaction(const CTransaction &transaction);
    bool CheckTransactionToBlock(const CTransaction &transaction, const CBlockHeader& blockHeader);
    bool ProcessFluidToken(std::string &scriptString, std::vector<std::string> &ptrs, int strVecNo);
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

