// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers

#ifndef FLUID_PROTOCOL_H
#define FLUID_PROTOCOL_H

#include "amount.h"
#include "chain.h"
#include "consensus/validation.h"
#include "key_io.h"
#include "operations.h"
#include "script/script.h"
#include "utilstrencodings.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

class CBlock;
class CTxMemPool;
struct CBlockTemplate;
class CTransaction;

/** Configuration Framework */
class CFluidParameters
{
public:
    static const int FLUID_ACTIVATE_HEIGHT = 10;
    static const int64_t MAX_FLUID_TIME_DISTORT = 60 * 60;          // Maximum time distort = 1 hour.
    static const CAmount FLUID_TRANSACTION_COST = 100000 * COIN;    // Cost to send a fluid transaction
    static const CAmount FLUID_MAX_REWARD_FOR_DYNODE = 1000 * COIN; // Max dynode block reward using fluid OP_REWARD_DYNODE
    static const CAmount FLUID_MAX_REWARD_FOR_MINING = 1000 * COIN; // Max mining block reward using fluid OP_REWARD_MINING
    static const CAmount FLUID_MAX_FOR_MINT = 1000000000 * COIN;    // Max minting amount per fluid transaction

    std::vector<std::pair<std::string, CDynamicAddress> > InitialiseSovereignIdentities();

    std::vector<std::string> InitialiseAddresses();
    std::vector<std::vector<unsigned char> > InitialiseAddressCharVector();
};

std::vector<std::string> InitialiseAddresses();

/** Fluid Asset Management Framework */
class CFluid : public CFluidParameters, public COperations
{
public:
    void AddFluidTransactionsToRecord(const CBlockIndex* pblockindex, std::vector<std::string>& transactionRecord);
    void ReplaceFluidSovereigns(const CBlockHeader& blockHeader, std::vector<std::string>& fluidSovereigns);

    bool IsGivenKeyMaster(CDynamicAddress inputKey);

    bool CheckFluidOperationScript(const CScript& fluidScriptPubKey, const int64_t timeStamp, std::string& errorMessage, bool fSkipTimeStampCheck = false);
    bool CheckIfExistsInMemPool(const CTxMemPool& pool, const CScript& fluidScriptPubKey, std::string& errorMessage);
    bool CheckIfQuorumExists(const std::string consentToken, std::string& message, bool individual = false);
    bool CheckNonScriptQuorum(const std::string consentToken, std::string& message, bool individual = false);
    bool CheckTransactionInRecord(CScript fluidInstruction, CBlockIndex* pindex = NULL);

    bool GenericConsentMessage(std::string message, std::string& signedString, CDynamicAddress signer);
    bool GenericParseNumber(const std::string consentToken, const int64_t timeStamp, CAmount& howMuch, bool txCheckPurpose = false);
    bool GenericVerifyInstruction(const std::string consentToken, CDynamicAddress& signer, std::string& messageTokenKey, int whereToLook = 1);

    bool ExtractCheckTimestamp(const std::string consentToken, const int64_t timeStamp);
    bool ParseMintKey(const int64_t nTime, CDynamicAddress& destination, CAmount& coinAmount, std::string uniqueIdentifier, bool txCheckPurpose = false);
    bool ProcessFluidToken(const std::string consentToken, std::vector<std::string>& ptrs, int strVecNo);

    bool GetMintingInstructions(const CBlockIndex* pblockindex, CDynamicAddress& toMintAddress, CAmount& mintAmount);
    bool GetProofOverrideRequest(const CBlockIndex* pblockindex, CAmount& howMuch);
    bool GetDynodeOverrideRequest(const CBlockIndex* pblockindex, CAmount& howMuch);

    bool ValidationProcesses(CValidationState& state, CScript txOut, CAmount txValue);

    bool CheckTransactionToBlock(const CTransaction& transaction, const CBlockHeader& blockHeader);
    bool CheckTransactionToBlock(const CTransaction& transaction, const uint256 hash);

    bool ProvisionalCheckTransaction(const CTransaction& transaction);
    bool InsertTransactionToRecord(CScript fluidInstruction, std::vector<std::string>& transactionRecord);
    CDynamicAddress GetAddressFromDigestSignature(const std::string digestSignature, const std::string messageTokenKey);
};

/** Standard Reward Payment Determination Functions */
CAmount GetStandardPoWBlockPayment(const int nHeight);
CAmount GetStandardDynodePayment(const int nHeight);

void BuildFluidInformationIndex(CBlockIndex* pindex, CAmount& nExpectedBlockValue, bool fDynodePaid);
bool IsTransactionFluid(const CScript& txOut);
bool IsTransactionFluid(const CTransaction& tx, CScript& fluidScript);
int GetFluidOpCode(const CScript& fluidScript);

std::vector<unsigned char> CharVectorFromString(const std::string& str);
std::string StringFromCharVector(const std::vector<unsigned char>& vch);

extern CFluid fluid;

#endif // FLUID_PROTOCOL_H
