// Copyright (c) 2016-2021 Duality Blockchain Solutions Developers

#ifndef FLUID_PROTOCOL_H
#define FLUID_PROTOCOL_H

#include "amount.h"
#include "base58.h"
#include "chain.h"
#include "consensus/validation.h"
#include "operations.h"
#include "script/script.h"
#include "utilstrencodings.h"
#include "bdap/utils.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

class CBlock;
class CDomainEntry;
class CTxMemPool;
struct CBlockTemplate;
class CTransaction;

static const int FLUID_ACTIVATE_HEIGHT = 10;
static const int64_t MAX_FLUID_TIME_DISTORT = 60 * 60;          // Maximum time distort = 1 hour.
static const CAmount FLUID_TRANSACTION_COST = 100000 * COIN;    // Cost to send a fluid transaction
static const CAmount FLUID_MAX_REWARD_FOR_DYNODE = 1000 * COIN; // Max dynode block reward using fluid OP_REWARD_DYNODE
static const CAmount FLUID_MAX_REWARD_FOR_MINING = 1000 * COIN; // Max mining block reward using fluid OP_REWARD_MINING
static const CAmount FLUID_MAX_FOR_MINT = 1000000000 * COIN;    // Max minting amount per fluid transaction

/** Fluid Asset Management Framework */
class CFluid
{
public:
    void ReplaceFluidSovereigns(const CBlockHeader& blockHeader, std::vector<std::string>& fluidSovereigns);

    bool CheckFluidOperationScript(const CScript& fluidScriptPubKey, const int64_t& timeStamp, const bool fSkipTimeStampCheck = false);
    bool CheckIfExistsInMemPool(const CTxMemPool& pool, const CScript& fluidScriptPubKey);
    bool CheckIfQuorumExists(const std::string& consentToken, std::string& message, const bool individual = false);
    bool CheckNonScriptQuorum(const std::string& consentToken, std::string& message, const bool individual = false);
    bool CheckTransactionInRecord(const CScript& fluidInstruction, CBlockIndex* pindex = NULL);

    bool GenericConsentMessage(const std::string& message, std::string& signedString, const CDynamicAddress& signer);
    bool GenericParseNumber(const std::string consentToken, const int64_t timeStamp, CAmount& howMuch, bool txCheckPurpose = false);
    bool GenericVerifyInstruction(const std::string& consentToken, CDynamicAddress& signer, std::string& messageTokenKey, const int& whereToLook = 1);

    bool ExtractCheckTimestamp(const std::string& strOpCode, const std::string& consentToken, const int64_t& timeStamp);
    bool ParseMintKey(const int64_t& nTime, CDynamicAddress& destination, CAmount& coinAmount, const std::string& uniqueIdentifier, const bool txCheckPurpose = false);
    bool ProcessFluidToken(const std::string& consentToken, std::vector<std::string>& ptrs, const int& strVecNo);

    bool GetMintingInstructions(const CBlockIndex* pblockindex, CDynamicAddress& toMintAddress, CAmount& mintAmount);
    bool ValidationProcesses(CValidationState& state, const CScript& txOut, const CAmount& txValue);

    bool CheckTransactionToBlock(const CTransaction& transaction, const CBlockHeader& blockHeader);
    bool CheckTransactionToBlock(const CTransaction& transaction, const uint256 hash);

    bool ProvisionalCheckTransaction(const CTransaction& transaction);
    CDynamicAddress GetAddressFromDigestSignature(const std::string& digestSignature, const std::string& messageTokenKey);
    bool CheckAccountBanScript(const CScript& fluidScript, const uint256& txHashId, const unsigned int& nHeight, std::vector<CDomainEntry>& vBanAccounts);
    bool ExtractTimestampWithAddresses(const std::string& strOpCode, const CScript& fluidScript, int64_t& nTimeStamp, std::vector<std::vector<unsigned char>>& vSovereignAddresses);

};

class DSFluidObject
{
    typedef std::vector<unsigned char> byte_vec;
    static constexpr int CURRENT_VERSION = 1;

    friend class CFluidDynode;
    friend class CFluidMining;
    friend class CFluidMint;
    friend class CFluidSovereign;

private:
    int version;

    uint256  tx_hash;
    uint32_t tx_height;
    byte_vec tx_script;

    uint64_t obj_reward;
    uint64_t obj_time;

protected:
    DSFluidObject Null()
    {
        DSFluidObject obj;
        version = DSFluidObject::CURRENT_VERSION;
        tx_hash.SetNull();
        tx_script.clear();
        tx_height = 0;
        obj_reward = -1;
        obj_time = 0;
        obj_sigs.clear();
        obj_address.clear();

        return obj;
    }

public:
    byte_vec obj_address;
    std::set<byte_vec> obj_sigs;

    DSFluidObject()
    {
        *this = Null();
    }

    DSFluidObject(DSFluidObject&& obj)
    {
        std::swap(version, obj.version);
        std::swap(tx_hash, obj.tx_hash);
        std::swap(tx_script, obj.tx_script);
        std::swap(obj_reward, obj.obj_reward);
        std::swap(obj_time, obj.obj_time);
        std::swap(obj_sigs, obj.obj_sigs);
        std::swap(obj_address, obj.obj_address);
    }

    DSFluidObject& operator=(DSFluidObject&& obj)
    {
        std::swap(version, obj.version);
        std::swap(tx_hash, obj.tx_hash);
        std::swap(tx_script, obj.tx_script);
        std::swap(obj_reward, obj.obj_reward);
        std::swap(obj_time, obj.obj_time);
        std::swap(obj_sigs, obj.obj_sigs);
        std::swap(obj_address, obj.obj_address);
        return *this;
    }

    DSFluidObject(const DSFluidObject& obj) = default;
    DSFluidObject& operator=(const DSFluidObject& obj) = default;

    bool operator>(const DSFluidObject& obj) const
    {
        return obj_time > obj.obj_time;
    }

    bool operator<(const DSFluidObject& obj) const
    {
        return obj_time < obj.obj_time;
    }

    bool operator==(const DSFluidObject& obj) const
    {
        return tx_script == obj.tx_script;
    }

    bool operator!=(const DSFluidObject& obj) const
    {
        return !((*this) == obj);
    }

    void Initialise(byte_vec _vch, CAmount _amt, int64_t _t)
    {
        return InitialiseScriptRewardTime(_vch, _amt, _t);
    }

    void InitialiseScriptRewardTime(byte_vec _vch, CAmount _amt, int64_t _t)
    {
        tx_script = _vch;
        obj_reward = _amt;
        obj_time = _t;
    }

    void InitialiseHeightHash(CAmount _ht, uint256 _hash)
    {
        tx_height = _ht;
        tx_hash = _hash;
    }

    void SetAddress(byte_vec _vch)
    {
        obj_address = _vch;
    }

    uint64_t GetTime() const { return obj_time; }
    uint64_t GetHeight() const { return tx_height; }
    uint64_t GetReward() const { return obj_reward; }
    uint256 GetTransactionHash() const { return tx_hash; }
    uint32_t GetTransactionHeight() const { return tx_height; }
    byte_vec GetTransactionScript() const { return tx_script; }

    bool IsNull() { return *this == Null(); }
    void SetNull() const { DSFluidObject(); }
};

/** Standard Reward Payment Determination Functions */
CAmount GetStandardPoWBlockPayment(const int& nHeight);
CAmount GetStandardDynodePayment(const int& nHeight);

void BuildFluidInformationIndex(CBlockIndex* pindex, CAmount& nExpectedBlockValue, bool fDynodePaid);
bool IsTransactionFluid(const CTransaction& tx, CScript& fluidScript);
int GetFluidOpCode(const CScript& fluidScript);

std::vector<unsigned char> CharVectorFromString(const std::string& str);
std::string StringFromCharVector(const std::vector<unsigned char>& vch);
std::vector<unsigned char> FluidScriptToCharVector(const CScript& fluidScript);

template <typename T1>
bool ParseScript(const CScript& scriptPubKey, T1& object);

extern CFluid fluid;

#endif // FLUID_PROTOCOL_H
