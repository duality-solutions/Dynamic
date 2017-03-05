// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DYNODE_H
#define DYNAMIC_DYNODE_H

#include "key.h"
#include "main.h"
#include "net.h"
#include "timedata.h"

#include "spork.h"

class CDynode;
class CDynodeBroadcast;
class CDynodePing;

static const int DYNODE_CHECK_SECONDS                = 5;
static const int DYNODE_MIN_DNB_SECONDS              = 5 * 60;
static const int DYNODE_MIN_DNP_SECONDS              = 10 * 60;
static const int DYNODE_EXPIRATION_SECONDS           = 65 * 60;
static const int DYNODE_WATCHDOG_MAX_SECONDS         = 120 * 60;
static const int DYNODE_NEW_START_REQUIRED_SECONDS   = 180 * 60;

static const int DYNODE_POSE_BAN_MAX_SCORE           = 5;

//
// The Dynode Ping Class : Contains a different serialize method for sending pings from Dynodes throughout the network
//
class CDynodePing
{
public:
    CTxIn vin;
    uint256 blockHash;
    int64_t sigTime; //dnb message times
    std::vector<unsigned char> vchSig;
    //removed stop

    CDynodePing() :
        vin(),
        blockHash(),
        sigTime(0),
        vchSig()
        {}

    CDynodePing(CTxIn& vinNew);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vin);
        READWRITE(blockHash);
        READWRITE(sigTime);
        READWRITE(vchSig);
    }

    void swap(CDynodePing& first, CDynodePing& second) // nothrow
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two classes,
        // the two classes are effectively swapped
        swap(first.vin, second.vin);
        swap(first.blockHash, second.blockHash);
        swap(first.sigTime, second.sigTime);
        swap(first.vchSig, second.vchSig);
    }

    uint256 GetHash() const
    {
        CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
        ss << vin;
        ss << sigTime;
        return ss.GetHash();
    }

    bool IsExpired() { return GetTime() - sigTime > DYNODE_NEW_START_REQUIRED_SECONDS; }
    bool Sign(CKey& keyDynode, CPubKey& pubKeyDynode);
    bool CheckSignature(CPubKey& pubKeyDynode, int &nDos);
    bool SimpleCheck(int& nDos);
    bool CheckAndUpdate(CDynode* pdn, bool fFromNewBroadcast, int& nDos);
    void Relay();

    CDynodePing& operator=(CDynodePing from)
    {
        swap(*this, from);
        return *this;
    }
    friend bool operator==(const CDynodePing& a, const CDynodePing& b)
    {
        return a.vin == b.vin && a.blockHash == b.blockHash;
    }
    friend bool operator!=(const CDynodePing& a, const CDynodePing& b)
    {
        return !(a == b);
    }

};

struct dynode_info_t 
{
    dynode_info_t()
        : vin(),
          addr(),
          pubKeyCollateralAddress(),
          pubKeyDynode(),
          sigTime(0),
          nLastSsq(0),
          nTimeLastChecked(0),
          nTimeLastPaid(0),
          nTimeLastWatchdogVote(0),
          nTimeLastPing(0),
          nActiveState(0),
          nProtocolVersion(0),
          fInfoValid(false)
        {}

    CTxIn vin;
    CService addr;
    CPubKey pubKeyCollateralAddress;
    CPubKey pubKeyDynode;
    int64_t sigTime; //dnb message time
    int64_t nLastSsq; //the psq count from the last psq broadcast of this node
    int64_t nTimeLastChecked;
    int64_t nTimeLastPaid;
    int64_t nTimeLastWatchdogVote;
    int64_t nTimeLastPing;
    int nActiveState;
    int nProtocolVersion;
    bool fInfoValid;
};

//
// The Dynode Class. For managing the Privatesend process. It contains the input of the 1000DYN, signature to prove
// it's the one who own that ip address and code for calculating the payment election.
//
class CDynode
{
private:
    // critical section to protect the inner data structures
    mutable CCriticalSection cs;

public:
    enum state {
        DYNODE_PRE_ENABLED,
        DYNODE_ENABLED,
        DYNODE_EXPIRED,
        DYNODE_OUTPOINT_SPENT,
        DYNODE_UPDATE_REQUIRED,
        DYNODE_WATCHDOG_EXPIRED,
        DYNODE_NEW_START_REQUIRED,
        DYNODE_POSE_BAN
    };

    CTxIn vin;
    CService addr;
    CPubKey pubKeyCollateralAddress;
    CPubKey pubKeyDynode;
    CDynodePing lastPing;
    std::vector<unsigned char> vchSig;
    int64_t sigTime; //dnb message time
    int64_t nLastSsq; //the psq count from the last psq broadcast of this node
    int64_t nTimeLastChecked;
    int64_t nTimeLastPaid;
    int64_t nTimeLastWatchdogVote;
    int nActiveState;
    int nCacheCollateralBlock;
    int nBlockLastPaid;
    int nProtocolVersion;
    int nPoSeBanScore;
    int nPoSeBanHeight;
    bool fAllowMixingTx;
    bool fUnitTest;

    // KEEP TRACK OF GOVERNANCE ITEMS EACH DYNODE HAS VOTE UPON FOR RECALCULATION
    std::map<uint256, int> mapGovernanceObjectsVotedOn;

    CDynode();
    CDynode(const CDynode& other);
    CDynode(const CDynodeBroadcast& dnb);
    CDynode(CService addrNew, CTxIn vinNew, CPubKey pubKeyCollateralAddressNew, CPubKey pubKeyDynodeNew, int nProtocolVersionIn);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        LOCK(cs);
        READWRITE(vin);
        READWRITE(addr);
        READWRITE(pubKeyCollateralAddress);
        READWRITE(pubKeyDynode);
        READWRITE(lastPing);
        READWRITE(vchSig);
        READWRITE(sigTime);
        READWRITE(nLastSsq);
        READWRITE(nTimeLastChecked);
        READWRITE(nTimeLastPaid);
        READWRITE(nTimeLastWatchdogVote);
        READWRITE(nActiveState);
        READWRITE(nCacheCollateralBlock);
        READWRITE(nBlockLastPaid);
        READWRITE(nProtocolVersion);
        READWRITE(nPoSeBanScore);
        READWRITE(nPoSeBanHeight);
        READWRITE(fAllowMixingTx);
        READWRITE(fUnitTest);
        READWRITE(mapGovernanceObjectsVotedOn);
    }

    void swap(CDynode& first, CDynode& second) // nothrow
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two classes,
        // the two classes are effectively swapped
        swap(first.vin, second.vin);
        swap(first.addr, second.addr);
        swap(first.pubKeyCollateralAddress, second.pubKeyCollateralAddress);
        swap(first.pubKeyDynode, second.pubKeyDynode);
        swap(first.lastPing, second.lastPing);
        swap(first.vchSig, second.vchSig);
        swap(first.sigTime, second.sigTime);
        swap(first.nLastSsq, second.nLastSsq);
        swap(first.nTimeLastChecked, second.nTimeLastChecked);
        swap(first.nTimeLastPaid, second.nTimeLastPaid);
        swap(first.nTimeLastWatchdogVote, second.nTimeLastWatchdogVote);
        swap(first.nActiveState, second.nActiveState);
        swap(first.nCacheCollateralBlock, second.nCacheCollateralBlock);
        swap(first.nBlockLastPaid, second.nBlockLastPaid);
        swap(first.nProtocolVersion, second.nProtocolVersion);
        swap(first.nPoSeBanScore, second.nPoSeBanScore);
        swap(first.nPoSeBanHeight, second.nPoSeBanHeight);
        swap(first.fAllowMixingTx, second.fAllowMixingTx);
        swap(first.fUnitTest, second.fUnitTest);
        swap(first.mapGovernanceObjectsVotedOn, second.mapGovernanceObjectsVotedOn);
    }

    // CALCULATE A RANK AGAINST OF GIVEN BLOCK
    arith_uint256 CalculateScore(const uint256& blockHash);

    bool UpdateFromNewBroadcast(CDynodeBroadcast& dnb);

    void Check(bool fForce = false);

    bool IsBroadcastedWithin(int nSeconds) { return GetAdjustedTime() - sigTime < nSeconds; }

    bool IsPingedWithin(int nSeconds, int64_t nTimeToCheckAt = -1)
    {
        if(lastPing == CDynodePing()) return false;

        if(nTimeToCheckAt == -1) {
            nTimeToCheckAt = GetAdjustedTime();
        }
        return nTimeToCheckAt - lastPing.sigTime < nSeconds;
    }

    bool IsEnabled() { return nActiveState == DYNODE_ENABLED; }
    bool IsPreEnabled() { return nActiveState == DYNODE_PRE_ENABLED; }
    bool IsPoSeBanned() { return nActiveState == DYNODE_POSE_BAN; }
    // NOTE: this one relies on nPoSeBanScore, not on nActiveState as everything else here
    bool IsPoSeVerified() { return nPoSeBanScore <= -DYNODE_POSE_BAN_MAX_SCORE; }
    bool IsExpired() { return nActiveState == DYNODE_EXPIRED; }
    bool IsOutpointSpent() { return nActiveState == DYNODE_OUTPOINT_SPENT; }
    bool IsUpdateRequired() { return nActiveState == DYNODE_UPDATE_REQUIRED; }
    bool IsWatchdogExpired() { return nActiveState == DYNODE_WATCHDOG_EXPIRED; }
    bool IsNewStartRequired() { return nActiveState == DYNODE_NEW_START_REQUIRED; }

    static bool IsValidStateForAutoStart(int nActiveStateIn)
    {
        return  nActiveStateIn == DYNODE_ENABLED ||
                nActiveStateIn == DYNODE_PRE_ENABLED ||
                nActiveStateIn == DYNODE_EXPIRED ||
                nActiveStateIn == DYNODE_WATCHDOG_EXPIRED;
    }

   bool IsValidForPayment()
    {
        if(nActiveState == DYNODE_ENABLED) {
            return true;
        }
        if(!sporkManager.IsSporkActive(SPORK_14_REQUIRE_SENTINEL_FLAG) &&
           (nActiveState == DYNODE_WATCHDOG_EXPIRED)) {
            return true;
        }

        return false;
    }

    bool IsValidNetAddr();
    static bool IsValidNetAddr(CService addrIn);

    void IncreasePoSeBanScore() { if(nPoSeBanScore < DYNODE_POSE_BAN_MAX_SCORE) nPoSeBanScore++; }
    void DecreasePoSeBanScore() { if(nPoSeBanScore > -DYNODE_POSE_BAN_MAX_SCORE) nPoSeBanScore--; }

    dynode_info_t GetInfo();

    static std::string StateToString(int nStateIn);
    std::string GetStateString() const;
    std::string GetStatus() const;

    int GetCollateralAge();

    int GetLastPaidTime() { return nTimeLastPaid; }
    int GetLastPaidBlock() { return nBlockLastPaid; }
    void UpdateLastPaid(const CBlockIndex *pindex, int nMaxBlocksToScanBack);

    // KEEP TRACK OF EACH GOVERNANCE ITEM INCASE THIS NODE GOES OFFLINE, SO WE CAN RECALC THEIR STATUS
    void AddGovernanceVote(uint256 nGovernanceObjectHash);
    // RECALCULATE CACHED STATUS FLAGS FOR ALL AFFECTED OBJECTS
    void FlagGovernanceItemsAsDirty();

    void RemoveGovernanceObject(uint256 nGovernanceObjectHash);

    void UpdateWatchdogVoteTime();

    CDynode& operator=(CDynode from)
    {
        swap(*this, from);
        return *this;
    }
    friend bool operator==(const CDynode& a, const CDynode& b)
    {
        return a.vin == b.vin;
    }
    friend bool operator!=(const CDynode& a, const CDynode& b)
    {
        return !(a.vin == b.vin);
    }

};


//
// The Dynode Broadcast Class : Contains a different serialize method for sending Dynodes through the network
//

class CDynodeBroadcast : public CDynode
{
public:

    bool fRecovery;
    CDynodeBroadcast() : CDynode(), fRecovery(false) {}
    CDynodeBroadcast(const CDynode& dn) : CDynode(dn), fRecovery(false) {}
    CDynodeBroadcast(CService addrNew, CTxIn vinNew, CPubKey pubKeyCollateralAddressNew, CPubKey pubKeyDynodeNew, int nProtocolVersionIn) :
        CDynode(addrNew, vinNew, pubKeyCollateralAddressNew, pubKeyDynodeNew, nProtocolVersionIn), fRecovery(false) {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vin);
        READWRITE(addr);
        READWRITE(pubKeyCollateralAddress);
        READWRITE(pubKeyDynode);
        READWRITE(vchSig);
        READWRITE(sigTime);
        READWRITE(nProtocolVersion);
        READWRITE(lastPing);
    }

    uint256 GetHash() const
    {
        CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
            ss << vin;
            ss << pubKeyCollateralAddress;
            ss << sigTime;
        return ss.GetHash();
    }

    /// Create Dynode broadcast, needs to be relayed manually after that
    static bool Create(CTxIn vin, CService service, CKey keyCollateralAddressNew, CPubKey pubKeyCollateralAddressNew, CKey keyDynodeNew, CPubKey pubKeyDynodeNew, std::string &strErrorRet, CDynodeBroadcast &dnbRet);
    static bool Create(std::string strService, std::string strKey, std::string strTxHash, std::string strOutputIndex, std::string& strErrorRet, CDynodeBroadcast &dnbRet, bool fOffline = false);

    bool SimpleCheck(int& nDos);
    bool Update(CDynode* pdn, int& nDos);
    bool CheckOutpoint(int& nDos);

    bool Sign(CKey& keyCollateralAddress);
    bool CheckSignature(int& nDos);
    void Relay();
};

class CDynodeVerification
{
public:
    CTxIn vin1;
    CTxIn vin2;
    CService addr;
    int nonce;
    int nBlockHeight;
    std::vector<unsigned char> vchSig1;
    std::vector<unsigned char> vchSig2;

    CDynodeVerification() :
        vin1(),
        vin2(),
        addr(),
        nonce(0),
        nBlockHeight(0),
        vchSig1(),
        vchSig2()
        {}

    CDynodeVerification(CService addr, int nonce, int nBlockHeight) :
        vin1(),
        vin2(),
        addr(addr),
        nonce(nonce),
        nBlockHeight(nBlockHeight),
        vchSig1(),
        vchSig2()
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vin1);
        READWRITE(vin2);
        READWRITE(addr);
        READWRITE(nonce);
        READWRITE(nBlockHeight);
        READWRITE(vchSig1);
        READWRITE(vchSig2);
    }

    uint256 GetHash() const
    {
        CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
        ss << vin1;
        ss << vin2;
        ss << addr;
        ss << nonce;
        ss << nBlockHeight;
        return ss.GetHash();
    }

    void Relay() const
    {
        CInv inv(MSG_DYNODE_VERIFY, GetHash());
        RelayInv(inv);
    }
};

#endif // DYNAMIC_DYNODE_H
