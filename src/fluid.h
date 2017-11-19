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
class CParameters {
public:
    static const int FLUID_ACTIVATE_HEIGHT = 10;
    static const int MAX_FLUID_TIME_DISTORT = 8 * 60;
    static const int FEE_REDIRECT_HEIGHT = 10;

    std::string FEE_REDIRECT_ADDRESS = "D8bMJRb7QaLAATMbMy6218gUoRXitipWzB"; // importprivkey QQMokyLh3HLNqV1XuTSnw9U5mq12qFSqaythSFoYGm1erSYSCSCi

    std::vector<std::string> InitialiseAddresses() {
        std::vector<std::string> x;
        x.push_back("D875Zfawgu8C2UeEzAEeUjy3xYPh1SVkdb"); // importprivkey QRk4PoYbvhyi1D1o8jMNcV6yLYKMa5cX1zRciSninmFQhX6vuAYg
        x.push_back("DBeJZkf8wjdkkxHQ2m2by9zPjdGHeMqy54"); // importprivkey QVHkqDiaZLgy1KZAkfyTNpVQtLEdg6hMBgi2SZ3khUZpVM5H64Aw
        x.push_back("DTbo2v1mCHnGA84JvitSdkGbRMrWVSm6u4"); // importprivkey QUyWN3JKSLZGLnKjYr6ZqY4hmRyzSQnBysknCv8pa2W31JrwsyC4
        x.push_back("DKBrQhZjVT6yeo16S4JsecFqtSQV8pojxR"); // importprivkey QRpJmBAZmu6hhBdsP5F2f45zZTs1us2SJofRrug7sJFAoogB2ytV
        x.push_back("DPish8VPe9iNrkb4dgXpx52z6XWjNQBXyC"); // importprivkey QTT7jTY72xmo2LHRHLqwUPYoDs7f9Je5hCrkRs1J7r9rE5NczbSd
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
class Fluid : public CParameters, public HexFunctions {
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

