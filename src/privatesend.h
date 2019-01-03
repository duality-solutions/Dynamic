// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
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
#include "timedata.h"
#include "tinyformat.h"

class CPrivateSend;
class CConnman;

// timeouts
static const int PRIVATESEND_AUTO_TIMEOUT_MIN = 5;
static const int PRIVATESEND_AUTO_TIMEOUT_MAX = 15;
static const int PRIVATESEND_QUEUE_TIMEOUT = 30;
static const int PRIVATESEND_SIGNING_TIMEOUT = 15;

//! minimum peer version accepted by mixing pool
static const int MIN_PRIVATESEND_PEER_PROTO_VERSION = 70900;

static const CAmount PRIVATESEND_ENTRY_MAX_SIZE = 9;

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

    CTxPSIn(const CTxIn& txin, const CScript& script) : CTxIn(txin),
                                                        prevPubKey(script),
                                                        fHasSig(false)
    {
    }

    CTxPSIn() : CTxIn(),
                prevPubKey(),
                fHasSig(false)
    {
    }
};

class CPrivateSendAccept
{
public:
    int nDenom;
    CMutableTransaction txCollateral;

    CPrivateSendAccept() : nDenom(0),
                           txCollateral(CMutableTransaction()){};

    CPrivateSendAccept(int nDenom, const CMutableTransaction& txCollateral) : nDenom(nDenom),
                                                                              txCollateral(txCollateral){};

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(nDenom);
        READWRITE(txCollateral);
    }

    friend bool operator==(const CPrivateSendAccept& a, const CPrivateSendAccept& b)
    {
        return a.nDenom == b.nDenom && a.txCollateral == b.txCollateral;
    }
};

// A clients transaction in the mixing pool
class CPrivateSendEntry
{
public:
    std::vector<CTxPSIn> vecTxPSIn;
    std::vector<CTxOut> vecTxOut;
    CTransactionRef txCollateral;
    // memory only
    CService addr;

    CPrivateSendEntry() : vecTxPSIn(std::vector<CTxPSIn>()),
                          vecTxOut(std::vector<CTxOut>()),
                          txCollateral(MakeTransactionRef()),
                          addr(CService())
    {
    }

    CPrivateSendEntry(const std::vector<CTxPSIn>& vecTxPSIn, const std::vector<CTxOut>& vecTxOut, const CTransaction& txCollateral) : vecTxPSIn(vecTxPSIn),
                                                                                                                                      vecTxOut(vecTxOut),
                                                                                                                                      txCollateral(MakeTransactionRef(txCollateral)),
                                                                                                                                      addr(CService())
    {
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(vecTxPSIn);
        READWRITE(txCollateral);
        READWRITE(vecTxOut);
    }

    bool AddScriptSig(const CTxIn& txin);
};


/**
 * A currently inprogress mixing merge and denomination information
 */
class CPrivateSendQueue
{
public:
    int nDenom;
    COutPoint dynodeOutpoint;
    int64_t nTime;
    bool fReady; //ready for submit
    std::vector<unsigned char> vchSig;
    // memory only
    bool fTried;

    CPrivateSendQueue() : nDenom(0),
                          dynodeOutpoint(COutPoint()),
                          nTime(0),
                          fReady(false),
                          vchSig(std::vector<unsigned char>()),
                          fTried(false)
    {
    }

    CPrivateSendQueue(int nDenom, COutPoint outpoint, int64_t nTime, bool fReady) : nDenom(nDenom),
                                                                                    dynodeOutpoint(outpoint),
                                                                                    nTime(nTime),
                                                                                    fReady(fReady),
                                                                                    vchSig(std::vector<unsigned char>()),
                                                                                    fTried(false)
    {
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(nDenom);
        int nVersion = s.GetVersion();
        if (nVersion == 70900 && (s.GetType() & SER_NETWORK)) {
            // converting from/to old format
            CTxIn txin{};
            if (ser_action.ForRead()) {
                READWRITE(txin);
                dynodeOutpoint = txin.prevout;
            } else {
                txin = CTxIn(dynodeOutpoint);
                READWRITE(txin);
            }
        } else {
            // using new format directly
            READWRITE(dynodeOutpoint);
        }
        READWRITE(nTime);
        READWRITE(fReady);
        if (!(s.GetType() & SER_GETHASH)) {
            READWRITE(vchSig);
        }
    }

    uint256 GetSignatureHash() const;
    /** Sign this mixing transaction
     *  \return true if all conditions are met:
     *     1) we have an active Dynode,
     *     2) we have a valid Dynode private key,
     *     3) we signed the message successfully, and
     *     4) we verified the message successfully
     */
    bool Sign();
    /// Check if we have a valid Dynode address
    bool CheckSignature(const CPubKey& pubKeyDynode) const;

    bool Relay(CConnman& connman);

    /// Is this queue expired?
    bool IsExpired() { return GetAdjustedTime() - nTime > PRIVATESEND_QUEUE_TIMEOUT; }

    std::string ToString() const
    {
        return strprintf("nDenom=%d, nTime=%lld, fReady=%s, fTried=%s, dynode=%s",
            nDenom, nTime, fReady ? "true" : "false", fTried ? "true" : "false", dynodeOutpoint.ToStringShort());
    }

    friend bool operator==(const CPrivateSendQueue& a, const CPrivateSendQueue& b)
    {
        return a.nDenom == b.nDenom && a.dynodeOutpoint == b.dynodeOutpoint && a.nTime == b.nTime && a.fReady == b.fReady;
    }
};

/** Helper class to store mixing transaction (tx) information.
 */
class CPrivateSendBroadcastTx
{
private:
    // memory only
    // when corresponding tx is 0-confirmed or conflicted, nConfirmedHeight is -1
    int nConfirmedHeight;

public:
    CTransactionRef tx;
    COutPoint dynodeOutpoint;
    std::vector<unsigned char> vchSig;
    int64_t sigTime;

    CPrivateSendBroadcastTx() : nConfirmedHeight(-1),
                                tx(MakeTransactionRef()),
                                dynodeOutpoint(),
                                vchSig(),
                                sigTime(0)
    {
    }

    CPrivateSendBroadcastTx(const CTransactionRef& _tx, COutPoint _outpoint, int64_t _sigTime) : nConfirmedHeight(-1),
                                                                                                 tx(_tx),
                                                                                                 dynodeOutpoint(_outpoint),
                                                                                                 vchSig(),
                                                                                                 sigTime(_sigTime)
    {
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(tx);
        int nVersion = s.GetVersion();
        if (nVersion == 70900 && (s.GetType() & SER_NETWORK)) {
            // converting from/to old format
            CTxIn txin{};
            if (ser_action.ForRead()) {
                READWRITE(txin);
                dynodeOutpoint = txin.prevout;
            } else {
                txin = CTxIn(dynodeOutpoint);
                READWRITE(txin);
            }
        } else {
            // using new format directly
            READWRITE(dynodeOutpoint);
        }
        if (!(s.GetType() & SER_GETHASH)) {
            READWRITE(vchSig);
        }
        READWRITE(sigTime);
    }

    friend bool operator==(const CPrivateSendBroadcastTx& a, const CPrivateSendBroadcastTx& b)
    {
        return *a.tx == *b.tx;
    }
    friend bool operator!=(const CPrivateSendBroadcastTx& a, const CPrivateSendBroadcastTx& b)
    {
        return !(a == b);
    }
    explicit operator bool() const
    {
        return *this != CPrivateSendBroadcastTx();
    }

    uint256 GetSignatureHash() const;

    bool Sign();
    bool CheckSignature(const CPubKey& pubKeyDynode) const;

    void SetConfirmedHeight(int nConfirmedHeightIn) { nConfirmedHeight = nConfirmedHeightIn; }
    bool IsExpired(int nHeight);
};

// base class
class CPrivateSendBaseSession
{
protected:
    mutable CCriticalSection cs_privatesend;

    std::vector<CPrivateSendEntry> vecEntries; // Dynode/clients entries

    PoolState nState;                // should be one of the POOL_STATE_XXX values
    int64_t nTimeLastSuccessfulStep; // the time when last successful mixing step was performed, in UTC milliseconds

    int nSessionID; // 0 if no mixing session is active

    CMutableTransaction finalMutableTransaction; // the finalized transaction ready for signing

    void SetNull();

public:
    int nSessionDenom; //Users must submit a denom matching this

    CPrivateSendBaseSession() : vecEntries(),
                                nState(POOL_STATE_IDLE),
                                nTimeLastSuccessfulStep(0),
                                nSessionID(0),
                                finalMutableTransaction(),
                                nSessionDenom(0)
    {
    }

    int GetState() const { return nState; }
    std::string GetStateString() const;

    int GetEntriesCount() const { return vecEntries.size(); }
};

// base class
class CPrivateSendBaseManager
{
protected:
    mutable CCriticalSection cs_vecqueue;
    // The current mixing sessions in progress on the network
    std::vector<CPrivateSendQueue> vecPrivateSendQueue;
    void SetNull();
    void CheckQueue();

public:
    CPrivateSendBaseManager() : vecPrivateSendQueue() {}
    int GetQueueSize() const { return vecPrivateSendQueue.size(); }
    bool GetQueueItemAndTry(CPrivateSendQueue& psqRet);
};

// helper class
class CPrivateSend
{
private:
    // make constructor, destructor and copying not available
    CPrivateSend() {}
    ~CPrivateSend() {}
    CPrivateSend(CPrivateSend const&) = delete;
    CPrivateSend& operator=(CPrivateSend const&) = delete;

    // static members
    static std::vector<CAmount> vecStandardDenominations;
    static std::map<uint256, CPrivateSendBroadcastTx> mapPSTX;

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
    static bool GetDenominationsBits(int nDenom, std::vector<int>& vecBitsRet);

    static std::string GetMessageByID(PoolMessage nMessageID);

    /// Get the maximum number of transactions for the pool
    static int GetMaxPoolTransactions() { return Params().PoolMaxTransactions(); }

    static CAmount GetMaxPoolAmount() { return vecStandardDenominations.empty() ? 0 : PRIVATESEND_ENTRY_MAX_SIZE * vecStandardDenominations.front(); }

    /// If the collateral is valid given by a client
    static bool IsCollateralValid(const CTransaction& txCollateral);
    static CAmount GetCollateralAmount() { return GetSmallestDenomination() / 10; }
    static CAmount GetMaxCollateralAmount() { return GetCollateralAmount() * 4; }

    static bool IsCollateralAmount(CAmount nInputAmount);

    static void AddPSTX(const CPrivateSendBroadcastTx& pstx);
    static CPrivateSendBroadcastTx GetPSTX(const uint256& hash);

    static void UpdatedBlockTip(const CBlockIndex* pindex);
    static void SyncTransaction(const CTransaction& tx, const CBlockIndex* pindex, int posInBlock);
};

#endif