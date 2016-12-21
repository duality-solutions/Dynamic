// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_STORMNODE_H
#define DARKSILK_STORMNODE_H

#include "key.h"
#include "main.h"
#include "net.h"
#include "spork.h"
#include "timedata.h"

class CStormnode;
class CStormnodeBroadcast;
class CStormnodePing;

static const int STORMNODE_MIN_SNP_SECONDS         = 10 * 60;
static const int STORMNODE_MIN_SNB_SECONDS         =  5 * 60;
static const int STORMNODE_EXPIRATION_SECONDS      = 65 * 60;
static const int STORMNODE_REMOVAL_SECONDS         = 75 * 60;
static const int STORMNODE_CHECK_SECONDS           = 5;
static const int STORMNODE_WATCHDOG_MAX_SECONDS    = 2 * 60 * 60;

static const int STORMNODE_POSE_BAN_MAX_SCORE      = 5;
//
// The Stormnode Ping Class : Contains a different serialize method for sending pings from stormnodes throughout the network
//

class CStormnodePing
{
public:
    CTxIn vin;
    uint256 blockHash;
    int64_t sigTime; //snb message times
    std::vector<unsigned char> vchSig;
    //removed stop

    CStormnodePing() :
        vin(),
        blockHash(),
        sigTime(0),
        vchSig()
        {}

    CStormnodePing(CTxIn& vinNew);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vin);
        READWRITE(blockHash);
        READWRITE(sigTime);
        READWRITE(vchSig);
    }

    void swap(CStormnodePing& first, CStormnodePing& second) // nothrow
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

    bool Sign(CKey& keyStormnode, CPubKey& pubKeyStormnode);
    bool CheckSignature(CPubKey& pubKeyStormnode, int &nDos);
    bool SimpleCheck(int& nDos);
    bool CheckAndUpdate(int& nDos);
    void Relay();

    CStormnodePing& operator=(CStormnodePing from)
    {
        swap(*this, from);
        return *this;
    }
    friend bool operator==(const CStormnodePing& a, const CStormnodePing& b)
    {
        return a.vin == b.vin && a.blockHash == b.blockHash;
    }
    friend bool operator!=(const CStormnodePing& a, const CStormnodePing& b)
    {
        return !(a == b);
    }

};

struct stormnode_info_t {

    stormnode_info_t()
        : vin(),
          addr(),
          pubKeyCollateralAddress(),
          pubKeyStormnode(),
          sigTime(0),
          nLastSsq(0),
          nTimeLastChecked(0),
          nTimeLastPaid(0),
          nTimeLastWatchdogVote(0),
          nActiveState(0),
          nProtocolVersion(0),
          fInfoValid(false)
        {}

    CTxIn vin;
    CService addr;
    CPubKey pubKeyCollateralAddress;
    CPubKey pubKeyStormnode;
    int64_t sigTime; //snb message time
    int64_t nLastSsq; //the ssq count from the last ssq broadcast of this node
    int64_t nTimeLastChecked;
    int64_t nTimeLastPaid;
    int64_t nTimeLastWatchdogVote;
    int nActiveState;
    int nProtocolVersion;
    bool fInfoValid;
};

//
// The Stormnode Class. For managing the Sandstorm process. It contains the input of the 1000DSLK, signature to prove
// it's the one who own that ip address and code for calculating the payment election.
//
class CStormnode
{
private:
    // critical section to protect the inner data structures
    mutable CCriticalSection cs;

public:
    enum state {
        STORMNODE_PRE_ENABLED,
        STORMNODE_ENABLED,
        STORMNODE_EXPIRED,
        STORMNODE_OUTPOINT_SPENT,
        STORMNODE_REMOVE,
        STORMNODE_WATCHDOG_EXPIRED,
        STORMNODE_POSE_BAN
    };

    CTxIn vin;
    CService addr;
    CPubKey pubKeyCollateralAddress;
    CPubKey pubKeyStormnode;
    CStormnodePing lastPing;
    std::vector<unsigned char> vchSig;
    int64_t sigTime; //snb message time
    int64_t nLastSsq; //the ssq count from the last ssq broadcast of this node
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

    // KEEP TRACK OF GOVERNANCE ITEMS EACH STORMNODE HAS VOTE UPON FOR RECALCULATION
    std::map<uint256, int> mapGovernanceObjectsVotedOn;

    CStormnode();
    CStormnode(const CStormnode& other);
    CStormnode(const CStormnodeBroadcast& snb);
    CStormnode(CService addrNew, CTxIn vinNew, CPubKey pubKeyCollateralAddressNew, CPubKey pubKeyStormnodeNew, int nProtocolVersionIn);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        LOCK(cs);
        READWRITE(vin);
        READWRITE(addr);
        READWRITE(pubKeyCollateralAddress);
        READWRITE(pubKeyStormnode);
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

    void swap(CStormnode& first, CStormnode& second) // nothrow
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two classes,
        // the two classes are effectively swapped
        swap(first.vin, second.vin);
        swap(first.addr, second.addr);
        swap(first.pubKeyCollateralAddress, second.pubKeyCollateralAddress);
        swap(first.pubKeyStormnode, second.pubKeyStormnode);
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

    bool UpdateFromNewBroadcast(CStormnodeBroadcast& snb);

    void Check(bool fForce = false);

    bool IsBroadcastedWithin(int nSeconds) { return GetAdjustedTime() - sigTime < nSeconds; }

    bool IsPingedWithin(int nSeconds, int64_t nTimeToCheckAt = -1)
    {
        if(lastPing == CStormnodePing()) return false;

        if(nTimeToCheckAt == -1) {
            nTimeToCheckAt = GetAdjustedTime();
        }
        return nTimeToCheckAt - lastPing.sigTime < nSeconds;
    }

    bool IsEnabled() { return nActiveState == STORMNODE_ENABLED; }
    bool IsPreEnabled() { return nActiveState == STORMNODE_PRE_ENABLED; }
    bool IsPoSeBanned() { return nActiveState == STORMNODE_POSE_BAN; }
    bool IsPoSeVerified() { return nPoSeBanScore <= -STORMNODE_POSE_BAN_MAX_SCORE; }

    bool IsWatchdogExpired() { return nActiveState == STORMNODE_WATCHDOG_EXPIRED; }

   bool IsValidForPayment()
    {
        if(nActiveState == STORMNODE_ENABLED) {
            return true;
        }
        if(!sporkManager.IsSporkActive(SPORK_14_REQUIRE_SENTINEL_FLAG) &&
           (nActiveState == STORMNODE_WATCHDOG_EXPIRED)) {
            return true;
        }

        return false;
    }

    bool IsValidNetAddr();
    static bool IsValidNetAddr(CService addrIn);

    void IncreasePoSeBanScore() { if(nPoSeBanScore < STORMNODE_POSE_BAN_MAX_SCORE) nPoSeBanScore++; }
    void DecreasePoSeBanScore() { if(nPoSeBanScore > -STORMNODE_POSE_BAN_MAX_SCORE) nPoSeBanScore--; }

    stormnode_info_t GetInfo();

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

    CStormnode& operator=(CStormnode from)
    {
        swap(*this, from);
        return *this;
    }
    friend bool operator==(const CStormnode& a, const CStormnode& b)
    {
        return a.vin == b.vin;
    }
    friend bool operator!=(const CStormnode& a, const CStormnode& b)
    {
        return !(a.vin == b.vin);
    }

};


//
// The Stormnode Broadcast Class : Contains a different serialize method for sending stormnodes through the network
//

class CStormnodeBroadcast : public CStormnode
{
public:

    CStormnodeBroadcast() : CStormnode() {}
    CStormnodeBroadcast(const CStormnode& sn) : CStormnode(sn) {}
    CStormnodeBroadcast(CService addrNew, CTxIn vinNew, CPubKey pubKeyCollateralAddressNew, CPubKey pubKeyStormnodeNew, int nProtocolVersionIn) :
        CStormnode(addrNew, vinNew, pubKeyCollateralAddressNew, pubKeyStormnodeNew, nProtocolVersionIn) {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vin);
        READWRITE(addr);
        READWRITE(pubKeyCollateralAddress);
        READWRITE(pubKeyStormnode);
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

    /// Create Stormnode broadcast, needs to be relayed manually after that
    static bool Create(CTxIn vin, CService service, CKey keyCollateralAddressNew, CPubKey pubKeyCollateralAddressNew, CKey keyStormnodeNew, CPubKey pubKeyStormnodeNew, std::string &strErrorRet, CStormnodeBroadcast &snbRet);
    static bool Create(std::string strService, std::string strKey, std::string strTxHash, std::string strOutputIndex, std::string& strErrorRet, CStormnodeBroadcast &snbRet, bool fOffline = false);

    bool SimpleCheck(int& nDos);
    bool Update(CStormnode* psn, int& nDos);
    bool CheckOutpoint(int& nDos);

    bool Sign(CKey& keyCollateralAddress);
    bool CheckSignature(int& nDos);
    void Relay();
};

class CStormnodeVerification
{
public:
    CTxIn vin1;
    CTxIn vin2;
    CService addr;
    int nonce;
    int nBlockHeight;
    std::vector<unsigned char> vchSig1;
    std::vector<unsigned char> vchSig2;

    CStormnodeVerification() :
        vin1(),
        vin2(),
        addr(),
        nonce(0),
        nBlockHeight(0),
        vchSig1(),
        vchSig2()
        {}

    CStormnodeVerification(CService addr, int nonce, int nBlockHeight) :
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
        CInv inv(MSG_STORMNODE_VERIFY, GetHash());
        RelayInv(inv);
    }
};

#endif // DARKSILK_STORMNODE_H
