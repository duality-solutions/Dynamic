// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef PRIVATESEND_H
#define PRIVATESEND_H

#include "chain.h"
#include "chainparams.h"
#include "primitives/transaction.h"
#include "pubkey.h"
#include "sync.h"
#include "tinyformat.h"
#include "timedata.h"

class CPrivateSend;
class CConnman;

// timeouts
static const int PRIVATESEND_AUTO_TIMEOUT_MIN       = 5;
static const int PRIVATESEND_AUTO_TIMEOUT_MAX       = 15;
static const int PRIVATESEND_QUEUE_TIMEOUT          = 30;
static const int PRIVATESEND_SIGNING_TIMEOUT        = 15;

//! minimum peer version accepted by mixing pool
static const int MIN_PRIVATESEND_PEER_PROTO_VERSION = 70500;

static const CAmount PRIVATESEND_ENTRY_MAX_SIZE     = 9;

// pool responses
enum PoolMessage {
    ERR_ALREADY_HAVE,
    ERR_DENOM,
    ERR_ENTRIES_FULL,
    ERR_EXISTING_TX,
    ERR_FEES,
    ERR_INVALID_COLLATERAL,
    ERR_INVALID_INPUT,
    ERR_INVALID_SCRIPT,
    ERR_INVALID_TX,
    ERR_MAXIMUM,
    ERR_DN_LIST,
    ERR_MODE,
    ERR_NON_STANDARD_PUBKEY,
    ERR_NOT_A_DN, // not used
    ERR_QUEUE_FULL,
    ERR_RECENT,
    ERR_SESSION,
    ERR_MISSING_TX,
    ERR_VERSION,
    MSG_NOERR,
    MSG_SUCCESS,
    MSG_ENTRIES_ADDED,
    MSG_POOL_MIN = ERR_ALREADY_HAVE,
    MSG_POOL_MAX = MSG_ENTRIES_ADDED
};

// pool states
enum PoolState {
    POOL_STATE_IDLE,
    POOL_STATE_QUEUE,
    POOL_STATE_ACCEPTING_ENTRIES,
    POOL_STATE_SIGNING,
    POOL_STATE_ERROR,
    POOL_STATE_SUCCESS,
    POOL_STATE_MIN = POOL_STATE_IDLE,
    POOL_STATE_MAX = POOL_STATE_SUCCESS
};

// status update message constants
enum PoolStatusUpdate {
    STATUS_REJECTED,
    STATUS_ACCEPTED
};

/** Holds an mixing input
 */
class CTxPSIn : public CTxIn
{
public:
    // memory only
    CScript prevPubKey;
    bool fHasSig; // flag to indicate if signed
    int nSentTimes; //times we've sent this anonymously

    CTxPSIn(const CTxIn& txin, const CScript& script) :
        CTxIn(txin),
        prevPubKey(script),
        fHasSig(false),
        nSentTimes(0)
        {}

    CTxPSIn() :
        CTxIn(),
        prevPubKey(),
        fHasSig(false),
        nSentTimes(0)
        {}
};

// A clients transaction in the mixing pool
class CPrivateSendEntry
{
public:
    std::vector<CTxPSIn> vecTxPSIn;
    std::vector<CTxOut> vecTxOut;
    CTransaction txCollateral;
    // memory only
    CService addr;

    CPrivateSendEntry() :
        vecTxPSIn(std::vector<CTxPSIn>()),
        vecTxOut(std::vector<CTxOut>()),
        txCollateral(CTransaction()),
        addr(CService())
        {}

    CPrivateSendEntry(const std::vector<CTxPSIn>& vecTxPSIn, const std::vector<CTxOut>& vecTxOut, const CTransaction& txCollateral) :
        vecTxPSIn(vecTxPSIn),
        vecTxOut(vecTxOut),
        txCollateral(txCollateral),
        addr(CService())
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vecTxPSIn);
        READWRITE(txCollateral);
        READWRITE(vecTxOut);
    }

    bool AddScriptSig(const CTxIn& txin);
};


/**
 * A currently inprogress mixing merge and denomination information
 */
class CPrivatesendQueue
{
public:
    int nDenom;
    CTxIn vin;
    int64_t nTime;
    bool fReady; //ready for submit
    std::vector<unsigned char> vchSig;
    // memory only
    bool fTried;

    CPrivatesendQueue() :
        nDenom(0),
        vin(CTxIn()),
        nTime(0),
        fReady(false),
        vchSig(std::vector<unsigned char>()),
        fTried(false)
        {}

    CPrivatesendQueue(int nDenom, COutPoint outpoint, int64_t nTime, bool fReady) :
        nDenom(nDenom),
        vin(CTxIn(outpoint)),
        nTime(nTime),
        fReady(fReady),
        vchSig(std::vector<unsigned char>()),
        fTried(false)
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(nDenom);
        READWRITE(vin);
        READWRITE(nTime);
        READWRITE(fReady);
        READWRITE(vchSig);
    }

    /** Sign this mixing transaction
     *  \return true if all conditions are met:
     *     1) we have an active Dynode,
     *     2) we have a valid Dynode private key,
     *     3) we signed the message successfully, and
     *     4) we verified the message successfully
     */
    bool Sign();
    /// Check if we have a valid Dynode address
    bool CheckSignature(const CPubKey& pubKeyDynode);

    bool Relay(CConnman& connman);

    /// Is this queue expired?
    bool IsExpired() { return GetAdjustedTime() - nTime > PRIVATESEND_QUEUE_TIMEOUT; }

    std::string ToString()
    {
        return strprintf("nDenom=%d, nTime=%lld, fReady=%s, fTried=%s, dynode=%s",
                        nDenom, nTime, fReady ? "true" : "false", fTried ? "true" : "false", vin.prevout.ToStringShort());
    }

    friend bool operator==(const CPrivatesendQueue& a, const CPrivatesendQueue& b)
    {
        return a.nDenom == b.nDenom && a.vin.prevout == b.vin.prevout && a.nTime == b.nTime && a.fReady == b.fReady;
    }
};

/** Helper class to store mixing transaction (tx) information.
 */
class CPrivatesendBroadcastTx
{
private:
    // memory only
    // when corresponding tx is 0-confirmed or conflicted, nConfirmedHeight is -1
    int nConfirmedHeight;

public:
    CTransaction tx;
    CTxIn vin;
    std::vector<unsigned char> vchSig;
    int64_t sigTime;

    CPrivatesendBroadcastTx() :
        nConfirmedHeight(-1),
        tx(),
        vin(),
        vchSig(),
        sigTime(0)
        {}

    CPrivatesendBroadcastTx(CTransaction tx, COutPoint outpoint, int64_t sigTime) :
        nConfirmedHeight(-1),
        tx(tx),
        vin(CTxIn(outpoint)),
        vchSig(),
        sigTime(sigTime)
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(tx);
        READWRITE(vin);
        READWRITE(vchSig);
        READWRITE(sigTime);
    }

    friend bool operator==(const CPrivatesendBroadcastTx& a, const CPrivatesendBroadcastTx& b)
    {
        return a.tx == b.tx;
    }
    friend bool operator!=(const CPrivatesendBroadcastTx& a, const CPrivatesendBroadcastTx& b)
    {
        return !(a == b);
    }
    explicit operator bool() const
    {
        return *this != CPrivatesendBroadcastTx();
    }

    bool Sign();
    bool CheckSignature(const CPubKey& pubKeyDynode);

    void SetConfirmedHeight(int nConfirmedHeightIn) { nConfirmedHeight = nConfirmedHeightIn; }
    bool IsExpired(int nHeight);
};

// base class
class CPrivateSendBase
{
protected:
    mutable CCriticalSection cs_privatesend;

    // The current mixing sessions in progress on the network
    std::vector<CPrivatesendQueue> vecPrivatesendQueue;

    std::vector<CPrivateSendEntry> vecEntries; // Dynode/clients entries

    PoolState nState; // should be one of the POOL_STATE_XXX values
    int64_t nTimeLastSuccessfulStep; // the time when last successful mixing step was performed, in UTC milliseconds

    int nSessionID; // 0 if no mixing session is active

    CMutableTransaction finalMutableTransaction; // the finalized transaction ready for signing

    void SetNull();
    void CheckQueue();

public:
    int nSessionDenom; //Users must submit an denom matching this

    CPrivateSendBase() { SetNull(); }

    int GetQueueSize() const { return vecPrivatesendQueue.size(); }
    int GetState() const { return nState; }
    std::string GetStateString() const;

    int GetEntriesCount() const { return vecEntries.size(); }
};

// helper class
class CPrivateSend
{
private:
    // make constructor, destructor and copying not available
    CPrivateSend() {}
    ~CPrivateSend() {}
    CPrivateSend(CPrivateSend const&) = delete;
    CPrivateSend& operator= (CPrivateSend const&) = delete;

    static const CAmount COLLATERAL = 0.001 * COIN;

    // static members
    static std::vector<CAmount> vecStandardDenominations;
    static std::map<uint256, CPrivatesendBroadcastTx> mapPSTX;

    static CCriticalSection cs_mappstx;

    static void CheckPSTXes(int nHeight);

public:
    static void InitStandardDenominations();
    static std::vector<CAmount> GetStandardDenominations() { return vecStandardDenominations; }
    static CAmount GetSmallestDenomination() { return vecStandardDenominations.back(); }

    /// Get the denominations for a specific amount of Dynamic.
    static int GetDenominationsByAmounts(const std::vector<CAmount>& vecAmount);

    static bool IsDenominatedAmount(CAmount nInputAmount);

    /// Get the denominations for a list of outputs (returns a bitshifted integer)
    static int GetDenominations(const std::vector<CTxOut>& vecTxOut, bool fSingleRandomDenom = false);
    static std::string GetDenominationsToString(int nDenom);
    static bool GetDenominationsBits(int nDenom, std::vector<int> &vecBitsRet);

    static std::string GetMessageByID(PoolMessage nMessageID);

    /// Get the maximum number of transactions for the pool
    static int GetMaxPoolTransactions() { return Params().PoolMaxTransactions(); }

    static CAmount GetMaxPoolAmount() { return vecStandardDenominations.empty() ? 0 : PRIVATESEND_ENTRY_MAX_SIZE * vecStandardDenominations.front(); }

    /// If the collateral is valid given by a client
    static bool IsCollateralValid(const CTransaction& txCollateral);
    static CAmount GetCollateralAmount() { return COLLATERAL; }
    static CAmount GetMaxCollateralAmount() { return COLLATERAL*4; }

    static bool IsCollateralAmount(CAmount nInputAmount);

    static void AddPSTX(const CPrivatesendBroadcastTx& pstx);
    static CPrivatesendBroadcastTx GetPSTX(const uint256& hash);

    static void UpdatedBlockTip(const CBlockIndex *pindex);

    static void SyncTransaction(const CTransaction& tx, const CBlock* pblock);
};

void ThreadCheckPrivateSend(CConnman& connman);

#endif