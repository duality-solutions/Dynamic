// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2017-2019 The Particl Core developers
// Copyright (c) 2014 The ShadowCoin developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_WALLET_WALLET_H
#define DYNAMIC_WALLET_WALLET_H

#include "amount.h"
#include "assets/assettypes.h"
#include "base58.h"
#include "bdap/linkstorage.h"
#include "policy/feerate.h"
#include "rpc/wallet.h"
#include "script/sign.h"
#include "streams.h"
#include "tinyformat.h"
#include "ui_interface.h"
#include "util.h"
#include "utilstrencodings.h"
#include "validationinterface.h"
#include "wallet/crypter.h"
#include "wallet/wallet_ismine.h"
#include "wallet/walletdb.h"

#include "privatesend.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <set>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

extern CWallet* pwalletMain;

/**
 * Settings
 */
extern CFeeRate payTxFee;
extern unsigned int nTxConfirmTarget;
extern bool bSpendZeroConfChange;
extern bool fSendFreeTransactions;
extern bool fWalletUnlockMixStakeOnly;
extern bool fWalletRbf;

//Set the following 2 constants together
static const unsigned int DEFAULT_KEYPOOL_SIZE = 200;
static const int DEFAULT_RESCAN_THRESHOLD = 150;

//! -paytxfee default
static const CAmount DEFAULT_TRANSACTION_FEE = 0;
//! -fallbackfee default
static const CAmount DEFAULT_FALLBACK_FEE = 1000;
//! -m_discard_rate default
static const CAmount DEFAULT_DISCARD_FEE = 25000;
//! -mintxfee default
static const CAmount DEFAULT_TRANSACTION_MINFEE = 1000;
//! minimum recommended increment for BIP 125 replacement txs
static const CAmount WALLET_INCREMENTAL_RELAY_FEE = 5000;
//! target minimum change amount
static const CAmount MIN_CHANGE = CENT;
//! final minimum change amount after paying for fees
static const CAmount MIN_FINAL_CHANGE = MIN_CHANGE / 2;
//! Default for -spendzeroconfchange
static const bool DEFAULT_SPEND_ZEROCONF_CHANGE = true;
//! Default for -sendfreetransactions
static const bool DEFAULT_SEND_FREE_TRANSACTIONS = false;
//! Default for -walletunlockmixstakeonly
static const bool WALLET_UNLOCKED_FOR_MIXING_STAKING_ONLY = false;
//! Default for -walletrejectlongchains
static const bool DEFAULT_WALLET_REJECT_LONG_CHAINS = false;
//! -txconfirmtarget default
static const unsigned int DEFAULT_TX_CONFIRM_TARGET = 10;
//! Largest (in bytes) free transaction we're willing to create
static const unsigned int MAX_FREE_TRANSACTION_CREATE_SIZE = 1000;
static const bool DEFAULT_WALLET_RBF = false;
static const bool DEFAULT_WALLETBROADCAST = true;
static const bool DEFAULT_DISABLE_WALLET = false;

extern const char* DEFAULT_WALLET_DAT;
extern const char* DEFAULT_WALLET_DAT_MNEMONIC;

//! if set, all keys will be derived by using BIP32
static const bool DEFAULT_USE_HD_WALLET = true;

bool AutoBackupWallet(CWallet* wallet, std::string strWalletFile, std::string& strBackupWarning, std::string& strBackupError);

class CAccountingEntry;
class CBlockIndex;
class CCoinControl;
class COutput;
class CReserveKey;
class CScheduler;
class CScript;
class CStakeInput;
class CTxMemPool;
class CWalletTx;

/** (client) version numbers for particular wallet features */
enum WalletFeature {
    FEATURE_BASE = 10500, // the earliest version new wallets supports (only useful for getinfo's clientversion output)

    FEATURE_WALLETCRYPT = 40000, // wallet encryption
    FEATURE_COMPRPUBKEY = 60000, // compressed public keys

    FEATURE_HD = 204000, // Hierarchical key derivation after BIP32 (HD Wallet), BIP44 (multi-coin), BIP39 (mnemonic)
                         // which uses on-the-fly private key derivation

    FEATURE_LATEST = 61000 // HD is optional, use FEATURE_COMPRPUBKEY as latest version
};

enum AvailableCoinsType {
    ALL_COINS,
    ONLY_DENOMINATED,
    ONLY_NONDENOMINATED,
    ONLY_1000, // find dynode outputs including locked ones (use with caution)
    ONLY_PRIVATESEND_COLLATERAL,
    STAKABLE_COINS // UTXO's that are valid for staking 
};

struct CompactTallyItem {
    CTxDestination txdest;
    CAmount nAmount;
    std::vector<COutPoint> vecOutPoints;
    CompactTallyItem()
    {
        nAmount = 0;
    }
};

/** A key pool entry */
class CKeyPool
{
public:
    int64_t nTime;
    CPubKey vchPubKey;
    bool fInternal; // for change outputs

    CKeyPool();
    CKeyPool(const CPubKey& vchPubKeyIn, bool fInternalIn);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        int nVersion = s.GetVersion();
        if (!(s.GetType() & SER_GETHASH))
            READWRITE(nVersion);
        READWRITE(nTime);
        READWRITE(vchPubKey);
        if (ser_action.ForRead()) {
            try {
                READWRITE(fInternal);
            } catch (std::ios_base::failure&) {
                /* flag as external address if we can't read the internal boolean
                   (this will be the case for any wallet before the HD chain split version) */
                fInternal = false;
            }
        } else {
            READWRITE(fInternal);
        }
    }
};

/** An Ed key pool entry */
class CEdKeyPool
{
public:
    int64_t nTime;
    std::vector<unsigned char> edPubKey;
    bool fInternal; // for change outputs

    CEdKeyPool();
    CEdKeyPool(const std::vector<unsigned char>& edPubKeyIn, bool fInternalIn);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        int nVersion = s.GetVersion();
        if (!(s.GetType() & SER_GETHASH))
            READWRITE(nVersion);
        READWRITE(nTime);
        READWRITE(edPubKey);
        if (ser_action.ForRead()) {
            try {
                READWRITE(fInternal);
            } catch (std::ios_base::failure&) {
                /* flag as external address if we can't read the internal boolean
                   (this will be the case for any wallet before the HD chain split version) */
                fInternal = false;
            }
        } else {
            READWRITE(fInternal);
        }
    }
};

/** Address book data */
class CAddressBookData
{
public:
    std::string name;
    std::string purpose;

    CAddressBookData()
    {
        purpose = "unknown";
    }

    typedef std::map<std::string, std::string> StringMap;
    StringMap destdata;
};

struct CRecipient {
    CScript scriptPubKey;
    CAmount nAmount;
    bool fSubtractFeeFromAmount;
};

typedef std::map<std::string, std::string> mapValue_t;


static inline void ReadOrderPos(int64_t& nOrderPos, mapValue_t& mapValue)
{
    if (!mapValue.count("n")) {
        nOrderPos = -1; // TODO: calculate elsewhere
        return;
    }
    nOrderPos = atoi64(mapValue["n"].c_str());
}


static inline void WriteOrderPos(const int64_t& nOrderPos, mapValue_t& mapValue)
{
    if (nOrderPos == -1)
        return;
    mapValue["n"] = i64tostr(nOrderPos);
}

struct COutputEntry {
    CTxDestination destination;
    CAmount amount;
    int vout;
};

/** ASSET START */
struct CAssetOutputEntry
{
    txnouttype type;
    std::string assetName;
    CTxDestination destination;
    CAmount nAmount;
    std::string message;
    int64_t expireTime;
    int vout;
};
/** ASSET END */

/** A transaction with a merkle branch linking it to the block chain. */
class CMerkleTx
{
private:
    /** Constant used in hashBlock to indicate tx has been abandoned */
    static const uint256 ABANDON_HASH;

public:
    CTransactionRef tx;
    uint256 hashBlock;

    /* An nIndex == -1 means that hashBlock (in nonzero) refers to the earliest
     * block in the chain we know this or any in-wallet dependency conflicts
     * with. Older clients interpret nIndex == -1 as unconfirmed for backward
     * compatibility.
     */
    int nIndex;

    CMerkleTx()
    {
        SetTx(MakeTransactionRef());
        Init();
    }

    CMerkleTx(CTransactionRef arg)
    {
        SetTx(std::move(arg));
        Init();
    }

    /** Helper conversion operator to allow passing CMerkleTx where CTransaction is expected.
     *  TODO: adapt callers and remove this operator. */
    operator const CTransaction&() const { return *tx; }

    void Init()
    {
        hashBlock = uint256();
        nIndex = -1;
    }

    void SetTx(CTransactionRef arg)
    {
        tx = std::move(arg);
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        std::vector<uint256> vMerkleBranch; // For compatibility with older versions.
        READWRITE(tx);
        READWRITE(hashBlock);
        READWRITE(vMerkleBranch);
        READWRITE(nIndex);
    }

    void SetMerkleBranch(const CBlockIndex* pIndex, int posInBlock);

    /**
     * Return depth of transaction in blockchain:
     * <0  : conflicts with a transaction this deep in the blockchain
     *  0  : in memory pool, waiting to be included in a block
     * >=1 : this many blocks deep in the main chain
     */
    int GetDepthInMainChain(const CBlockIndex*& pindexRet) const;
    int GetDepthInMainChain() const
    {
        const CBlockIndex* pindexRet;
        return GetDepthInMainChain(pindexRet);
    }
    bool IsInMainChain() const
    {
        const CBlockIndex* pindexRet;
        return GetDepthInMainChain(pindexRet) > 0;
    }
    bool IsLockedByInstantSend() const;
    int GetBlocksToMaturity() const;
    /** Pass this transaction to the mempool. Fails if absolute fee exceeds absurd fee. */
    bool AcceptToMemoryPool(const CAmount& nAbsurdFee, CValidationState& state);
    bool hashUnset() const { return (hashBlock.IsNull() || hashBlock == ABANDON_HASH); }
    bool isAbandoned() const { return (hashBlock == ABANDON_HASH); }
    void setAbandoned() { hashBlock = ABANDON_HASH; }

    const uint256& GetHash() const { return tx->GetHash(); }
    bool IsCoinBase() const { return tx->IsCoinBase(); }
    bool IsCoinStake() const { return tx->IsCoinStake(); }
};

/** 
 * A transaction with a bunch of additional info that only the owner cares about.
 * It includes any unrecorded transactions needed to link it back to the block chain.
 */
class CWalletTx : public CMerkleTx
{
private:
    const CWallet* pwallet;

public:
    mapValue_t mapValue;
    std::vector<std::pair<std::string, std::string> > vOrderForm;
    unsigned int fTimeReceivedIsTxTime;
    unsigned int nTimeReceived; //!< time received by this node
    unsigned int nTimeSmart;
    /**
     * From me flag is set to 1 for transactions that were created by the wallet
     * on this bitcoin node, and set to 0 for transactions that were created
     * externally and came in through the network or sendrawtransaction RPC.
     */
    char fFromMe;
    std::string strFromAccount;
    int64_t nOrderPos; //!< position in ordered transaction list

    // memory only
    mutable bool fFromMeCached;
    mutable bool fDebitCached;
    mutable bool fCreditCached;
    mutable bool fImmatureCreditCached;
    mutable bool fAvailableCreditCached;
    mutable bool fAnonymizedCreditCached;
    mutable bool fDenomUnconfCreditCached;
    mutable bool fDenomConfCreditCached;
    mutable bool fWatchDebitCached;
    mutable bool fWatchCreditCached;
    mutable bool fImmatureWatchCreditCached;
    mutable bool fAvailableWatchCreditCached;
    mutable bool fChangeCached;
    mutable bool fFromMeCachedValue;

    mutable CAmount nDebitCached;
    mutable CAmount nCreditCached;
    mutable CAmount nImmatureCreditCached;
    mutable CAmount nAvailableCreditCached;
    mutable CAmount nAnonymizedCreditCached;
    mutable CAmount nDenomUnconfCreditCached;
    mutable CAmount nDenomConfCreditCached;
    mutable CAmount nWatchDebitCached;
    mutable CAmount nWatchCreditCached;
    mutable CAmount nImmatureWatchCreditCached;
    mutable CAmount nAvailableWatchCreditCached;
    mutable CAmount nChangeCached;

    CWalletTx()
    {
        Init(nullptr);
    }

    CWalletTx(const CWallet* pwalletIn, CTransactionRef arg) : CMerkleTx(std::move(arg))
    {
        Init(pwalletIn);
    }

    void Init(const CWallet* pwalletIn)
    {
        pwallet = pwalletIn;
        mapValue.clear();
        vOrderForm.clear();
        fTimeReceivedIsTxTime = false;
        nTimeReceived = 0;
        nTimeSmart = 0;
        fFromMe = false;
        strFromAccount.clear();
        fFromMeCached = false;
        fDebitCached = false;
        fCreditCached = false;
        fImmatureCreditCached = false;
        fAvailableCreditCached = false;
        fAnonymizedCreditCached = false;
        fDenomUnconfCreditCached = false;
        fDenomConfCreditCached = false;
        fWatchDebitCached = false;
        fWatchCreditCached = false;
        fImmatureWatchCreditCached = false;
        fAvailableWatchCreditCached = false;
        fChangeCached = false;
        nDebitCached = 0;
        nCreditCached = 0;
        nImmatureCreditCached = 0;
        nAvailableCreditCached = 0;
        nAnonymizedCreditCached = 0;
        nDenomUnconfCreditCached = 0;
        nDenomConfCreditCached = 0;
        nWatchDebitCached = 0;
        nWatchCreditCached = 0;
        nAvailableWatchCreditCached = 0;
        nImmatureWatchCreditCached = 0;
        nChangeCached = 0;
        nOrderPos = -1;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        if (ser_action.ForRead())
            Init(nullptr);
        char fSpent = false;

        if (!ser_action.ForRead()) {
            mapValue["fromaccount"] = strFromAccount;

            WriteOrderPos(nOrderPos, mapValue);

            if (nTimeSmart)
                mapValue["timesmart"] = strprintf("%u", nTimeSmart);
        }

        READWRITE(*(CMerkleTx*)this);
        std::vector<CMerkleTx> vUnused; //! Used to be vtxPrev
        READWRITE(vUnused);
        READWRITE(mapValue);
        READWRITE(vOrderForm);
        READWRITE(fTimeReceivedIsTxTime);
        READWRITE(nTimeReceived);
        READWRITE(fFromMe);
        READWRITE(fSpent);

        if (ser_action.ForRead()) {
            strFromAccount = mapValue["fromaccount"];

            ReadOrderPos(nOrderPos, mapValue);

            nTimeSmart = mapValue.count("timesmart") ? (unsigned int)atoi64(mapValue["timesmart"]) : 0;
        }

        mapValue.erase("fromaccount");
        mapValue.erase("version");
        mapValue.erase("spent");
        mapValue.erase("n");
        mapValue.erase("timesmart");
    }

    //! make sure balances are recalculated
    void MarkDirty()
    {
        fCreditCached = false;
        fAvailableCreditCached = false;
        fAnonymizedCreditCached = false;
        fDenomUnconfCreditCached = false;
        fDenomConfCreditCached = false;
        fWatchDebitCached = false;
        fWatchCreditCached = false;
        fAvailableWatchCreditCached = false;
        fImmatureWatchCreditCached = false;
        fDebitCached = false;
        fChangeCached = false;
    }

    void BindWallet(CWallet* pwalletIn)
    {
        pwallet = pwalletIn;
        MarkDirty();
    }

    //! filter decides which addresses will count towards the debit
    CAmount GetDebit(const isminefilter& filter) const;
    CAmount GetCredit(const isminefilter& filter) const;
    CAmount GetImmatureCredit(bool fUseCache = true) const;
    CAmount GetAvailableCredit(bool fUseCache = true) const;
    CAmount GetImmatureWatchOnlyCredit(const bool& fUseCache = true) const;
    CAmount GetAvailableWatchOnlyCredit(const bool& fUseCache = true) const;
    CAmount GetChange() const;

    CAmount GetAnonymizedCredit(bool fUseCache = true) const;
    CAmount GetDenominatedCredit(bool unconfirmed, bool fUseCache = true) const;

    void GetAmounts(std::list<COutputEntry>& listReceived,
                    std::list<COutputEntry>& listSent, CAmount& nFee, std::string& strSentAccount, const isminefilter& filter) const;

    void GetAmounts(std::list<COutputEntry>& listReceived,
                    std::list<COutputEntry>& listSent, CAmount& nFee, std::string& strSentAccount, const isminefilter& filter, std::list<CAssetOutputEntry>& assetsReceived, std::list<CAssetOutputEntry>& assetsSent) const;

    void GetAccountAmounts(const std::string& strAccount, CAmount& nReceived, CAmount& nSent, CAmount& nFee, const isminefilter& filter) const;

    bool IsRelevantToMe(const isminefilter& filter) const
    {
        return (GetDebit(filter) > 0);
    }

    bool IsFromMe(const isminefilter& filter) const;

    // True if only scriptSigs are different
    bool IsEquivalentTo(const CWalletTx& tx) const;

    bool InMempool() const;
    bool IsTrusted() const;

    int64_t GetTxTime() const;
    int GetRequestCount() const;

    bool RelayWalletTransaction(CConnman* connman, const std::string& strCommand = "tx");

    std::set<uint256> GetConflicts() const;
    bool IsBDAP() const;
};

/* ASSET START */
class CInputCoin {
public:
    CInputCoin() {}
    CInputCoin(const CWalletTx* wtx, unsigned int nOutIndex) : walletTx(wtx), nOut(nOutIndex)
    {
        if (!walletTx)
            throw std::invalid_argument("walletTx should not be null");
        if (nOutIndex >= walletTx->tx->vout.size())
            throw std::out_of_range("The output index is out of range");

        outpoint = COutPoint(walletTx->GetHash(), nOutIndex);
        txout = walletTx->tx->vout[nOutIndex];
    }

    const CWalletTx* walletTx;
    unsigned int nOut;
    COutPoint outpoint;
    CTxOut txout;

    bool operator<(const CInputCoin& rhs) const {
        return outpoint < rhs.outpoint;
    }

    bool operator!=(const CInputCoin& rhs) const {
        return outpoint != rhs.outpoint;
    }

    bool operator==(const CInputCoin& rhs) const {
        return outpoint == rhs.outpoint;
    }
};
/* ASSET END */

class COutput
{
public:
    const CWalletTx* tx;
    int i;
    int nDepth;
    bool fSpendable;
    bool fSolvable;

    /**
     * Whether this output is considered safe to spend. Unconfirmed transactions
     * from outside keys and unconfirmed replacement transactions are considered
     * unsafe and will not be used to fund new spending transactions.
     */
    bool fSafe;

    COutput(const CWalletTx* txIn, int iIn, int nDepthIn, bool fSpendableIn, bool fSolvableIn, bool fSafeIn)
    {
        tx = txIn;
        i = iIn;
        nDepth = nDepthIn;
        fSpendable = fSpendableIn;
        fSolvable = fSolvableIn;
        fSafe = fSafeIn;
    }

    //Used with PrivateSend. Will return largest nondenom, then denominations, then very small inputs
    int Priority() const;

    std::string ToString() const;
};


/** Private key that includes an expiration date in case it never gets used. */
class CWalletKey
{
public:
    CPrivKey vchPrivKey;
    int64_t nTimeCreated;
    int64_t nTimeExpires;
    std::string strComment;
    //! todo: add something to note what created it (user, getnewaddress, change)
    //!   maybe should have a map<string, string> property map

    CWalletKey(int64_t nExpires = 0);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        int nVersion = s.GetVersion();
        if (!(s.GetType() & SER_GETHASH))
            READWRITE(nVersion);
        READWRITE(vchPrivKey);
        READWRITE(nTimeCreated);
        READWRITE(nTimeExpires);
        READWRITE(LIMITED_STRING(strComment, 65536));
    }
};

/** 
 * Internal transfers.
 * Database key is acentry<account><counter>.
 */
class CAccountingEntry
{
public:
    std::string strAccount;
    CAmount nCreditDebit;
    int64_t nTime;
    std::string strOtherAccount;
    std::string strComment;
    mapValue_t mapValue;
    int64_t nOrderPos; //! position in ordered transaction list
    uint64_t nEntryNo;

    CAccountingEntry()
    {
        SetNull();
    }

    void SetNull()
    {
        nCreditDebit = 0;
        nTime = 0;
        strAccount.clear();
        strOtherAccount.clear();
        strComment.clear();
        nOrderPos = -1;
        nEntryNo = 0;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        int nVersion = s.GetVersion();
        if (!(s.GetType() & SER_GETHASH))
            READWRITE(nVersion);
        //! Note: strAccount is serialized as part of the key, not here.
        READWRITE(nCreditDebit);
        READWRITE(nTime);
        READWRITE(LIMITED_STRING(strOtherAccount, 65536));

        if (!ser_action.ForRead()) {
            WriteOrderPos(nOrderPos, mapValue);

            if (!(mapValue.empty() && _ssExtra.empty())) {
                CDataStream ss(s.GetType(), s.GetVersion());
                ss.insert(ss.begin(), '\0');
                ss << mapValue;
                ss.insert(ss.end(), _ssExtra.begin(), _ssExtra.end());
                strComment.append(ss.str());
            }
        }

        READWRITE(LIMITED_STRING(strComment, 65536));

        size_t nSepPos = strComment.find("\0", 0, 1);
        if (ser_action.ForRead()) {
            mapValue.clear();
            if (std::string::npos != nSepPos) {
                CDataStream ss(std::vector<char>(strComment.begin() + nSepPos + 1, strComment.end()), s.GetType(), s.GetVersion());
                ss >> mapValue;
                _ssExtra = std::vector<char>(ss.begin(), ss.end());
            }
            ReadOrderPos(nOrderPos, mapValue);
        }
        if (std::string::npos != nSepPos)
            strComment.erase(nSepPos);

        mapValue.erase("n");
    }

private:
    std::vector<char> _ssExtra;
};

class CStealthKeyQueueData
{
// Used to get secret for keys created by stealth transaction with wallet locked
public:
    CStealthKeyQueueData() {}

    CStealthKeyQueueData(const CPubKey& pkEphem_, const CPubKey& pkScan_, const CPubKey& pkSpend_, const CKey& SharedKey_)
    {
        pkEphem = pkEphem_;
        pkScan = pkScan_;
        pkSpend = pkSpend_;
        SharedKey = SharedKey_;
    };

    CPubKey pkEphem;
    CPubKey pkScan;
    CPubKey pkSpend;
    CKey SharedKey;

    inline CStealthKeyQueueData operator=(const CStealthKeyQueueData& b) {
        pkEphem = b.pkEphem;
        pkScan = b.pkScan;
        pkSpend = b.pkSpend;
        SharedKey = b.SharedKey;
        return *this;
    }

    ADD_SERIALIZE_METHODS;
    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(pkEphem);
        READWRITE(pkScan);
        READWRITE(pkSpend);
        if (ser_action.ForRead()) {
            std::vector<unsigned char> vchSharedKey;
            READWRITE(vchSharedKey);
            SharedKey.Set(vchSharedKey.begin(), vchSharedKey.end(), true);
        } else {
            std::vector<unsigned char> vchSharedKey(SharedKey.begin(), SharedKey.end());
            READWRITE(vchSharedKey);
        }
    };
};

class CStealthAddressRaw
{
// Used to keep a list of stealth addresses used by this wallet by storing the raw format.
public:
    CStealthAddressRaw() {};

    CStealthAddressRaw(std::vector<uint8_t>& addrRaw_) : addrRaw(addrRaw_) {};
    std::vector<uint8_t> addrRaw;

    ADD_SERIALIZE_METHODS;
    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(addrRaw);
    };
};

/** 
 * A CWallet is an extension of a keystore, which also maintains a set of transactions and balances,
 * and provides the ability to create new transactions.
 */
class CWallet : public CCryptoKeyStore, public CValidationInterface
{
private:
    static std::atomic<bool> fFlushThreadRunning;

    /**
     * Select a set of coins such that nValueRet >= nTargetValue and at least
     * all coins from coinControl are selected; Never select unconfirmed coins
     * if they are not ours
     */
    bool SelectCoins(const std::vector<COutput>& vAvailableCoins, const CAmount& nTargetValue, std::set<CInputCoin>& setCoinsRet, CAmount& nValueRet, const CCoinControl* coinControl = nullptr, AvailableCoinsType nCoinType=ALL_COINS, bool fUseInstantSend = true) const;
/** ASSET START */
    bool SelectAssets(const std::map<std::string, std::vector<COutput> >& mapAvailableAssets, const std::map<std::string, CAmount>& mapAssetTargetValue, std::set<CInputCoin>& setCoinsRet, std::map<std::string, CAmount>& nValueRet) const;
/** ASSET END */

    CWalletDB* pwalletdbEncryption;

    //! the current wallet version: clients below this version are not able to load the wallet
    int nWalletVersion;

    //! the maximum wallet format version: memory-only variable that specifies to what version this wallet may be upgraded
    int nWalletMaxVersion;

    int64_t nNextResend;
    int64_t nLastResend;
    bool fBroadcastTransactions;

    bool fNeedToUpdateKeyPools = false;
    bool fNeedToUpdateLinks = false;
    bool fNeedToUpgradeWallet = false;
    
    mutable bool fAnonymizableTallyCached;
    mutable std::vector<CompactTallyItem> vecAnonymizableTallyCached;
    mutable bool fAnonymizableTallyCachedNonDenom;
    mutable std::vector<CompactTallyItem> vecAnonymizableTallyCachedNonDenom;

    /**
     * Used to keep track of spent outpoints, and
     * detect and report conflicts (double-spends or
     * mutated transactions where the mutant gets mined).
     */
    typedef std::multimap<COutPoint, uint256> TxSpends;
    TxSpends mapTxSpends;
    void AddToSpends(const COutPoint& outpoint, const uint256& wtxid);
    void AddToSpends(const uint256& wtxid);

    std::set<COutPoint> setWalletUTXO;

    /* Mark a transaction (and its in-wallet descendants) as conflicting with a particular block. */
    void MarkConflicted(const uint256& hashBlock, const uint256& hashTx);

    void SyncMetaData(std::pair<TxSpends::iterator, TxSpends::iterator>);

    /* Used by TransactionAddedToMemorypool/BlockConnected/Disconnected.
     * Should be called with pindexBlock and posInBlock if this is for a transaction that is included in a block. */
    void SyncTransaction(const CTransaction& tx, const CBlockIndex *pindex = nullptr, int posInBlock = 0);

    /* HD derive new child key (on internal or external chain) */
    void DeriveNewChildKey(const CKeyMetadata& metadata, CKey& secretRet, uint32_t nAccountIndex, bool fInternal /*= false*/);

    void DeriveEd25519ChildKey(const CKey& seed, CKeyEd25519& secretEdRet);

    /* HD derive new child stealth key from  */
    bool DeriveChildStealthKey(const CKey& key);

    void ReserveEdKeyForTransactions(const std::vector<unsigned char>& pubKeyToReserve);   

    bool ReserveKeyForTransactions(const CPubKey& pubKeyToReserve);   

    std::array<char, 32> ConvertSecureVector32ToArray(const std::vector<unsigned char, secure_allocator<unsigned char> >& vIn);

    std::set<int64_t> setInternalKeyPool;
    std::set<int64_t> setExternalKeyPool;

    std::set<int64_t> setInternalEdKeyPool;
    std::set<int64_t> setExternalEdKeyPool;

    unsigned int DynamicKeyPoolSize = (int64_t)0;

    std::vector<std::vector<unsigned char>> reservedEd25519PubKeys;

    int64_t nTimeFirstKey;

    /**
     * Private version of AddWatchOnly method which does not accept a
     * timestamp, and which will reset the wallet's nTimeFirstKey value to 1 if
     * the watch key did not previously have a timestamp associated with it.
     * Because this is an inherited virtual method, it is accessible despite
     * being marked private, but it is marked private anyway to encourage use
     * of the other AddWatchOnly which accepts a timestamp and sets
     * nTimeFirstKey more intelligently for more efficient rescans.
     */
    bool AddWatchOnly(const CScript& dest) override;

public:
    /*
     * Main wallet lock.
     * This lock protects all the fields added by CWallet
     *   except for:
     *      fFileBacked (immutable after instantiation)
     *      strWalletFile (immutable after instantiation)
     */
    mutable CCriticalSection cs_wallet;

    const std::string strWalletFile;

    bool fFileBacked;

    bool fNeedToRescanTransactions = false;
    CBlockIndex* rescan_index = nullptr;
    int ReserveKeyCount = 0;
    bool SaveRescanIndex = false;
    //unsigned int DynamicKeyPoolSize = (int64_t)0;

    bool WalletNeedsUpgrading()
    {
        return fNeedToUpgradeWallet;
    }

    void SetUpdateKeyPoolsAndLinks()
    {
        fNeedToUpdateKeyPools = true;
        fNeedToUpdateLinks = true;        
    }

    void LoadKeyPool(int nIndex, const CKeyPool& keypool)
    {
        if (keypool.fInternal) {
            setInternalKeyPool.insert(nIndex);
        } else {
            setExternalKeyPool.insert(nIndex);
        }

        // If no metadata exists yet, create a default with the pool key's
        // creation time. Note that this may be overwritten by actually
        // stored metadata for that key later, which is fine.
        CKeyID keyid = keypool.vchPubKey.GetID();
        if (mapKeyMetadata.count(keyid) == 0)
            mapKeyMetadata[keyid] = CKeyMetadata(keypool.nTime);
    }

    void LoadEdKeyPool(int nIndex, const CEdKeyPool& edkeypool)
    {
        if (edkeypool.fInternal) {
            setInternalEdKeyPool.insert(nIndex);
        } else {
            setExternalEdKeyPool.insert(nIndex);
        }

        /*
        // If no metadata exists yet, create a default with the pool key's
        // creation time. Note that this may be overwritten by actually
        // stored metadata for that key later, which is fine.
        CKeyID keyid = keypool.vchPubKey.GetID();
        if (mapKeyMetadata.count(keyid) == 0)
            mapKeyMetadata[keyid] = CKeyMetadata(keypool.nTime);
        */    
    }

    // Map from Key ID (for regular keys) or Script ID (for watch-only keys) to
    // key metadata.
    std::map<CTxDestination, CKeyMetadata> mapKeyMetadata;

    typedef std::map<unsigned int, CMasterKey> MasterKeyMap;
    MasterKeyMap mapMasterKeys;
    unsigned int nMasterKeyMaxID;

    // Stake Settings
    unsigned int nHashDrift;
    unsigned int nHashInterval;
    uint64_t nStakeSplitThreshold;
    int nStakeSetUpdateTime;

    //MultiSend
    std::vector<std::pair<std::string, int> > vMultiSend;
    bool fMultiSendStake;
    bool fMultiSendDynodeReward;
    bool fMultiSendNotify;
    std::string strMultiSendChangeAddress;
    int nLastMultiSendHeight;
    std::vector<std::string> vDisabledAddresses;

    //Auto Combine Inputs
    bool fCombineDust;
    CAmount nAutoCombineThreshold;

    CWallet()
    {
        SetNull();
    }

    CWallet(const std::string& strWalletFileIn) : strWalletFile(strWalletFileIn)
    {
        SetNull();

        fFileBacked = true;
    }

    ~CWallet()
    {
        delete pwalletdbEncryption;
        pwalletdbEncryption = nullptr;
    }

    void SetNull()
    {
        nWalletVersion = FEATURE_BASE;
        nWalletMaxVersion = FEATURE_BASE;
        fFileBacked = false;
        nMasterKeyMaxID = 0;
        pwalletdbEncryption = nullptr;
        nOrderPosNext = 0;
        nNextResend = 0;
        nLastResend = 0;
        nTimeFirstKey = 0;
        fBroadcastTransactions = false;
        fAnonymizableTallyCached = false;
        fAnonymizableTallyCachedNonDenom = false;
        vecAnonymizableTallyCached.clear();
        vecAnonymizableTallyCachedNonDenom.clear();
        nFoundStealth = 0;
        // Stake Settings
        nHashDrift = 45;
        nStakeSplitThreshold = 2000;
        nHashInterval = 22;
        nStakeSetUpdateTime = 300; // 5 minutes
        //MultiSend
        vMultiSend.clear();
        fMultiSendStake = false;
        fMultiSendDynodeReward = false;
        fMultiSendNotify = false;
        strMultiSendChangeAddress = "";
        nLastMultiSendHeight = 0;
        vDisabledAddresses.clear();
        //Auto Combine Dust
        fCombineDust = false;
        nAutoCombineThreshold = 0;
    }

    bool isMultiSendEnabled()
    {
        return fMultiSendDynodeReward || fMultiSendStake;
    }

    void setMultiSendDisabled()
    {
        fMultiSendDynodeReward = false;
        fMultiSendStake = false;
    }

    std::map<uint256, CWalletTx> mapWallet;
    std::list<CAccountingEntry> laccentries;

    typedef std::pair<CWalletTx*, CAccountingEntry*> TxPair;
    typedef std::multimap<int64_t, TxPair> TxItems;
    TxItems wtxOrdered;

    int64_t nOrderPosNext;
    std::map<uint256, int> mapRequestCount;

    std::map<CTxDestination, CAddressBookData> mapAddressBook;

    CPubKey vchDefaultKey;

    std::set<COutPoint> setLockedCoins;

    int64_t nKeysLeftSinceAutoBackup;

    std::map<CKeyID, CHDPubKey> mapHdPubKeys; //<! memory map of HD extended pubkeys
    std::map<CKeyID, CStealthAddress> mapStealthAddresses; //<! memory map of stealth addresses
    mutable CCriticalSection cs_mapStealthAddresses;
    std::vector<std::pair<CKeyID, CStealthKeyQueueData>> vStealthKeyQueue;
    mutable CCriticalSection cs_vStealthKeyQueue;
    uint32_t nFoundStealth; // for reporting, zero before use

    const CWalletTx* GetWalletTx(const uint256& hash) const;

    //! check whether we are allowed to upgrade (or already support) to the named feature
    bool CanSupportFeature(enum WalletFeature wf)
    {
        AssertLockHeld(cs_wallet);
        return nWalletMaxVersion >= wf;
    }

    std::map<CDynamicAddress, std::vector<COutput> > AvailableCoinsByAddress(bool fConfirmed = true, CAmount maxCoinValue = 0);

/**ASSETS START */
    /**
     * populate vCoins with vector of available COutputs, and populates vAssetCoins in fWithAssets is set to true.
     */
    void AvailableCoinsAll(std::vector<COutput>& vCoins, std::map<std::string, std::vector<COutput> >& mapAssetCoins,
                            bool fGetDYN = true, bool fOnlyAssets = false,
                            bool fOnlySafe = true, const CCoinControl* coinControl = nullptr,
                            const CAmount& nMinimumAmount = 1, const CAmount& nMaximumAmount = MAX_MONEY,
                            const CAmount& nMinimumSumAmount = MAX_MONEY, const uint64_t& nMaximumCount = 0,
                            const int& nMinDepth = 0, const int& nMaxDepth = 9999999, bool fOnlyConfirmed = true,
                            bool fIncludeZeroValue = false, AvailableCoinsType nCoinType = ALL_COINS, 
                            bool fUseInstantSend = false, bool fUseBDAP = false) const;

    /**
     * Helper function that calls AvailableCoinsAll, used for transfering assets
     */
    void AvailableAssets(std::map<std::string, std::vector<COutput> > &mapAssetCoins, bool fOnlySafe = true,
                         const CCoinControl* coinControl = nullptr, const CAmount &nMinimumAmount = 1,
                         const CAmount &nMaximumAmount = MAX_MONEY, const CAmount &nMinimumSumAmount = MAX_MONEY,
                         const uint64_t &nMaximumCount = 0, const int &nMinDepth = 0, const int &nMaxDepth = 9999999, 
                         bool fOnlyConfirmed = true, bool fIncludeZeroValue = false, AvailableCoinsType nCoinType = ALL_COINS, 
                         bool fUseInstantSend = false, bool fUseBDAP = false) const;
    /**
     * Helper function that calls AvailableCoinsAll, used to receive all coins, Assets and DYN
     */
    void AvailableCoinsWithAssets(std::vector<COutput> &vCoins, std::map<std::string, std::vector<COutput> > &mapAssetCoins,
                                  bool fOnlySafe = true, const CCoinControl* coinControl = nullptr, const CAmount &nMinimumAmount = 1,
                                  const CAmount &nMaximumAmount = MAX_MONEY, const CAmount &nMinimumSumAmount = MAX_MONEY,
                                  const uint64_t &nMaximumCount = 0, const int &nMinDepth = 0, const int &nMaxDepth = 9999999, 
                                  bool fOnlyConfirmed = true, bool fIncludeZeroValue = false, AvailableCoinsType nCoinType = ALL_COINS, 
                                  bool fUseInstantSend = false, bool fUseBDAP = false) const;
    /**
     * populate vCoins with vector of available COutputs.
     */
    void AvailableCoins(std::vector<COutput>& vCoins, bool fOnlySafe=true, const CCoinControl *coinControl = nullptr,
                        const CAmount& nMinimumAmount = 1, const CAmount& nMaximumAmount = MAX_MONEY,
                        const CAmount& nMinimumSumAmount = MAX_MONEY, const uint64_t& nMaximumCount = 0,
                        const int& nMinDepth = 0, const int& nMaxDepth = 9999999, bool fOnlyConfirmed = true,
                        bool fIncludeZeroValue = false, AvailableCoinsType nCoinType = ALL_COINS, 
                        bool fUseInstantSend = false, bool fUseBDAP = false) const;
/** ASSETS END */

    /**
     * populate vCoins with vector of available BDAP credits.
     */
    void AvailableBDAPCredits(std::vector<std::pair<CTxOut, COutPoint>>& vCredits, bool fOnlyConfirmed = true) const;
    /**
     * populate vCoins with vector of available COutputs for the specified address.
     */
    void GetBDAPCoins(std::vector<COutput>& vCoins, const CScript& prevScriptPubKey) const;
    /**
     * return BDAP Credits in Dynamic
     */
    CAmount GetBDAPDynamicAmount() const;

/** ASSET START */
    /**
     * Return list of available coins and locked coins grouped by non-change output address.
     */
    std::map<CTxDestination, std::vector<COutput>> ListCoins() const;

    /**
     * Return list of available assets and locked assets grouped by non-change output address.
     */
    std::map<CTxDestination, std::vector<COutput>> ListAssets() const;


    /**
     * Find non-change parent output.
     */
    const CTxOut& FindNonChangeParentOutput(const CTransaction& tx, int output) const;
/** ASSET END */

    /**
     * Shuffle and select coins until nTargetValue is reached while avoiding
     * small change; This method is stochastic for some inputs and upon
     * completion the coin set and corresponding actual target value is
     * assembled
     */
    bool SelectCoinsMinConf(const CAmount& nTargetValue, int nConfMine, int nConfTheirs, uint64_t nMaxAncestors, std::vector<COutput> vCoins, std::set<CInputCoin>& setCoinsRet, CAmount& nValueRet, AvailableCoinsType nCoinType=ALL_COINS, bool fUseInstantSend = false) const;

/** ASSET START */
    bool SelectAssetsMinConf(const CAmount& nTargetValue, const int nConfMine, const int nConfTheirs, const uint64_t nMaxAncestors, const std::string& strAssetName, std::vector<COutput> vCoins, std::set<CInputCoin>& setCoinsRet, CAmount& nValueRet) const;
/** ASSET END */

    // Coin selection
    bool SelectPSInOutPairsByDenominations(int nDenom, CAmount nValueMin, CAmount nValueMax, std::vector< std::pair<CTxPSIn, CTxOut> >& vecPSInOutPairsRet);
    bool GetCollateralTxPSIn(CTxPSIn& txpsinRet, CAmount& nValueRet) const;
    bool SelectPrivateCoins(CAmount nValueMin, CAmount nValueMax, std::vector<CTxIn>& vecTxInRet, CAmount& nValueRet, int nPrivateSendRoundsMin, int nPrivateSendRoundsMax) const;

    bool SelectCoinsGroupedByAddresses(std::vector<CompactTallyItem>& vecTallyRet, bool fSkipDenominated = true, bool fAnonymizable = true, bool fSkipUnconfirmed = true, int nMaxOupointsPerAddress = -1) const;

    /// Get 1000DYN output and keys which can be used for the Dynode
    bool GetDynodeOutpointAndKeys(COutPoint& outpointRet, CPubKey& pubKeyRet, CKey& keyRet, std::string strTxHash = "", std::string strOutputIndex = "");
    /// Extract txin information and keys from output
    bool GetOutpointAndKeysFromOutput(const COutput& out, COutPoint& outpointRet, CPubKey& pubKeyRet, CKey& keyRet);

    bool HasCollateralInputs(bool fOnlyConfirmed = true) const;
    int CountInputsWithAmount(CAmount nInputAmount);

    // get the PrivateSend chain depth for a given input
    int GetRealOutpointPrivateSendRounds(const COutPoint& outpoint, int nRounds = 0) const;
    // respect current settings
    int GetOutpointPrivateSendRounds(const COutPoint& outpoint) const;

    bool IsDenominated(const COutPoint& outpoint) const;

    bool IsSpent(const uint256& hash, unsigned int n) const;

    bool IsLockedCoin(uint256 hash, unsigned int n) const;
    void LockCoin(const COutPoint& output);
    void UnlockCoin(const COutPoint& output);
    void UnlockAllCoins();
    void ListLockedCoins(std::vector<COutPoint>& vOutpts) const;

    /**
     * keystore implementation
     * Generate a new key
     */
    CPubKey GenerateNewKey(uint32_t nAccountIndex, bool fInternal /*= false*/);
    void GenerateEdandStealthKey(CKey& keyIn);
    std::vector<unsigned char> GenerateNewEdKey(uint32_t nAccountIndex, bool fInternal, const CKey& seedIn = CKey());
    //! HaveDHTKey implementation that also checks the mapHdPubKeys
    bool HaveDHTKey(const CKeyID &address) const override;
    //! HaveKey implementation that also checks the mapHdPubKeys
    bool HaveKey(const CKeyID& address) const override;
    //! GetPubKey implementation that also checks the mapHdPubKeys
    bool GetPubKey(const CKeyID& address, CPubKey& vchPubKeyOut) const override;
    //! GetKey implementation that can derive a HD private key on the fly
    bool GetKey(const CKeyID& address, CKey& keyOut) const override;
    //! Gets Ed25519 private key from the wallet(database)
    bool GetDHTKey(const CKeyID& address, CKeyEd25519& keyOut) const override;
    //! Adds a HDPubKey into the wallet(database)
    bool AddHDPubKey(const CExtPubKey& extPubKey, bool fInternal);
    //! loads a HDPubKey into the wallets memory
    bool LoadHDPubKey(const CHDPubKey& hdPubKey);
    //! Adds a key to the store, and saves it to disk.
    bool AddKeyPubKey(const CKey& key, const CPubKey& pubkey) override;
    //! Adds an ed25519 keypair and saves it to disk.
    bool AddDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& pubkey) override;
    //! Adds a key to the store, without saving it to disk (used by LoadWallet)
    bool LoadKey(const CKey& key, const CPubKey& pubkey) { return CCryptoKeyStore::AddKeyPubKey(key, pubkey); }
    //! Adds a key to the store, without saving it to disk (used by LoadWallet)
    bool LoadDHTKey(const CKeyEd25519& key, const std::vector<unsigned char>& pubkey) { return CCryptoKeyStore::AddDHTKey(key, pubkey); }
    //! Load metadata (used by LoadWallet)
    bool LoadKeyMetadata(const CTxDestination& pubKey, const CKeyMetadata& metadata);

    bool LoadMinVersion(int nVersion)
    {
        AssertLockHeld(cs_wallet);
        nWalletVersion = nVersion;
        nWalletMaxVersion = std::max(nWalletMaxVersion, nVersion);
        return true;
    }
    void UpdateTimeFirstKey(int64_t nCreateTime);

    //! Adds an encrypted key to the store, and saves it to disk.
    bool AddCryptedKey(const CPubKey& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret) override;
    bool AddCryptedDHTKey(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret) override;
    //! Adds an encrypted key to the store, without saving it to disk (used by LoadWallet)
    bool LoadCryptedKey(const CPubKey& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret);
    bool LoadCryptedDHTKey(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchCryptedSecret);
    bool AddCScript(const CScript& redeemScript) override;
    bool LoadCScript(const CScript& redeemScript);

    //! Adds a destination data tuple to the store, and saves it to disk
    bool AddDestData(const CTxDestination& dest, const std::string& key, const std::string& value);
    //! Erases a destination data tuple in the store and on disk
    bool EraseDestData(const CTxDestination& dest, const std::string& key);
    //! Adds a destination data tuple to the store, without saving it to disk
    bool LoadDestData(const CTxDestination& dest, const std::string& key, const std::string& value);
    //! Look up a destination data tuple in the store, return true if found false otherwise
    bool GetDestData(const CTxDestination& dest, const std::string& key, std::string* value) const;

    //! Adds a watch-only address to the store, and saves it to disk.
    bool AddWatchOnly(const CScript& dest, int64_t nCreateTime);

    bool RemoveWatchOnly(const CScript& dest) override;
    //! Adds a watch-only address to the store, without saving it to disk (used by LoadWallet)
    bool LoadWatchOnly(const CScript& dest);

    bool Unlock(const SecureString& strWalletPassphrase, bool fForMixingOnly = false);
    bool ChangeWalletPassphrase(const SecureString& strOldWalletPassphrase, const SecureString& strNewWalletPassphrase);
    bool EncryptWallet(const SecureString& strWalletPassphrase);

    void GetKeyBirthTimes(std::map<CTxDestination, int64_t>& mapKeyBirth) const;

    /** 
     * Increment the next transaction order id
     * @return next transaction order id
     */
    int64_t IncOrderPosNext(CWalletDB* pwalletdb = nullptr);
    DBErrors ReorderTransactions();
    bool AccountMove(std::string strFrom, std::string strTo, CAmount nAmount, std::string strComment = "");
    bool GetAccountPubkey(CPubKey& pubKey, std::string strAccount, bool bForceNew = false);

    void MarkDirty();
    bool AddToWallet(const CWalletTx& wtxIn, bool fFlushOnClose = true);
    bool LoadToWallet(const CWalletTx& wtxIn);
    void TransactionAddedToMempool(const CTransactionRef& tx) override;
    void BlockConnected(const std::shared_ptr<const CBlock>& pblock, const CBlockIndex *pindex, const std::vector<CTransactionRef>& vtxConflicted) override;
    void BlockDisconnected(const std::shared_ptr<const CBlock>& pblock) override;
    bool AddToWalletIfInvolvingMe(const CTransaction& tx, const CBlockIndex* pIndex, int posInBlock, bool fUpdate);
    CBlockIndex* ScanForWalletTransactions(CBlockIndex* pindexStart, bool fUpdate = false);
    void ReacceptWalletTransactions();
    void ResendWalletTransactions(int64_t nBestBlockTime, CConnman* connman) override;
    std::vector<uint256> ResendWalletTransactionsBefore(int64_t nTime, CConnman* connman);
    CAmount GetBalance() const;
    CAmount GetTotal() const;
    CAmount GetStake() const;
    CAmount GetUnconfirmedBalance() const;
    CAmount GetImmatureBalance() const;
    CAmount GetWatchOnlyBalance() const;
    CAmount GetWatchOnlyStake() const;
    CAmount GetUnconfirmedWatchOnlyBalance() const;
    CAmount GetImmatureWatchOnlyBalance() const;

    CAmount GetAnonymizableBalance(bool fSkipDenominated = false, bool fSkipUnconfirmed = true) const;
    CAmount GetAnonymizedBalance() const;
    float GetAverageAnonymizedRounds() const;
    CAmount GetNormalizedAnonymizedBalance() const;
    CAmount GetNeedsToBeAnonymizedBalance(CAmount nMinBalance = 0) const;
    CAmount GetDenominatedBalance(bool unconfirmed = false) const;

    bool GetBudgetSystemCollateralTX(CTransactionRef& tx, uint256 hash, CAmount amount, bool fUseInstantSend);
    bool GetBudgetSystemCollateralTX(CWalletTx& tx, uint256 hash, CAmount amount, bool fUseInstantSend);

    /**
     * Insert additional inputs into the transaction by
     * calling CreateTransaction();
     */
    bool SignTransaction(CMutableTransaction& tx);
    bool FundTransaction(CMutableTransaction& tx, CAmount& nFeeRet, int& nChangePosInOut, std::string& strFailReason, bool includeWatching, bool lockUnspents, const std::set<int>& setSubtractFeeFromOutputs, CCoinControl coinControl);


    /**
     * Create a new transaction paying the recipients with a set of coins
     * selected by SelectCoins(); Also create the change output, when needed
     */
    /* ASSET START*/
    
    bool CreateTransactionWithAssets(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                                   std::string& strFailReason, const CCoinControl& coinControl, const std::vector<CNewAsset> assets, const CTxDestination destination, const AssetType& assetType, bool sign = true, AvailableCoinsType nCoinType = ALL_COINS, bool fUseInstantSend = false, bool fIsBDAP = false);

    bool CreateTransactionWithTransferAsset(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                                                     std::string& strFailReason, const CCoinControl& coinControl, bool sign = true, AvailableCoinsType nCoinType = ALL_COINS, bool fUseInstantSend = false, bool fIsBDAP = false);

    bool CreateTransactionWithReissueAsset(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                                                    std::string& strFailReason, const CCoinControl& coinControl, const CReissueAsset& reissueAsset, const CTxDestination destination, bool sign = true, AvailableCoinsType nCoinType = ALL_COINS, bool fUseInstantSend = false, bool fIsBDAP = false);

    bool CreateTransaction(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                           std::string& strFailReason, const CCoinControl& coinControl, bool sign = true, AvailableCoinsType nCoinType = ALL_COINS, bool fUseInstantSend = false, bool fIsBDAP = false);
    
    /*
     * Create a new transaction paying the recipients with a set of coins
     * selected by SelectCoins(); Also create the change output, when needed
     * @note passing nChangePosInOut as -1 will result in setting a random position
    */
    
    bool CreateTransactionAll(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet, int& nChangePosInOut,
                           std::string& strFailReason, const CCoinControl& coinControl, bool fNewAsset, const CNewAsset& asset, const CTxDestination dest, bool fTransferAsset, bool fReissueAsset, const CReissueAsset& reissueAsset, const AssetType& assetType, bool sign = true, AvailableCoinsType nCoinType = ALL_COINS, bool fUseInstantSend = false, bool fIsBDAP = false);

    bool CreateTransactionAll(const std::vector<CRecipient>& vecSend, CWalletTx& wtxNew, CReserveKey& reservekey, CAmount& nFeeRet,
                              int& nChangePosInOut, std::string& strFailReason, const CCoinControl& coinControl, bool fNewAsset, const std::vector<CNewAsset> assets, const CTxDestination destination, bool fTransferAsset, bool fReissueAsset, const CReissueAsset& reissueAsset, const AssetType& assetType, bool sign, AvailableCoinsType nCoinType, bool fUseInstantSend, bool fIsBDAP);

    bool CreateNewChangeAddress(CReserveKey& reservekey, CKeyID& keyID, std::string& strFailReason);

    /* ASSET END */

    bool CommitTransaction(CWalletTx& wtxNew, CReserveKey& reservekey, CConnman* connman, CValidationState& state, const std::string& strCommand = "tx");

    bool CreateCollateralTransaction(CMutableTransaction& txCollateral, std::string& strReason);
    bool ConvertList(std::vector<CTxIn> vecTxIn, std::vector<CAmount>& vecAmounts);

    void ListAccountCreditDebit(const std::string& strAccount, std::list<CAccountingEntry>& entries);
    bool AddAccountingEntry(const CAccountingEntry&);
    bool AddAccountingEntry(const CAccountingEntry&, CWalletDB* pwalletdb);
    template <typename ContainerType>
    bool DummySignTx(CMutableTransaction &txNew, const ContainerType &coins) const;

    static CFeeRate minTxFee;
    static CFeeRate fallbackFee;
    static CFeeRate m_discard_rate;

    bool NewKeyPool();
    bool NewEdKeyPool();
    size_t KeypoolCountExternalKeys();
    size_t KeypoolCountInternalKeys();
    size_t EdKeypoolCountExternalKeys();
    size_t EdKeypoolCountInternalKeys();
    bool SyncEdKeyPool(); 
    bool TopUpKeyPoolCombo(unsigned int kpSize = 0, bool fIncreaseSize = false);
    void ReserveKeysFromKeyPools(int64_t& nIndex, CKeyPool& keypool, CEdKeyPool& edkeypool, bool fInternal);
    void KeepKey(int64_t nIndex);
    void ReturnKey(int64_t nIndex, bool fInternal);
    bool GetKeysFromPool(CPubKey& pubkeyWallet, std::vector<unsigned char>& vchEd25519PubKey, bool fInternal);
    bool GetKeysFromPool(CPubKey& pubkeyWallet, std::vector<unsigned char>& vchEd25519PubKey, CStealthAddress& sxAddr, bool fInternal);
    int64_t GetOldestKeyPoolTime();
    void GetAllReserveKeys(std::set<CKeyID>& setAddress) const;
    void UpdateKeyPoolsFromTransactions(const std::string& strOpType, const std::vector<std::vector<unsigned char>>& vvchOpParameters);

    std::set<std::set<CTxDestination> > GetAddressGroupings();
    std::map<CTxDestination, CAmount> GetAddressBalances();

    CAmount GetAccountBalance(const std::string& strAccount, int nMinDepth, const isminefilter& filter, bool fAddLocked);
    CAmount GetAccountBalance(CWalletDB& walletdb, const std::string& strAccount, int nMinDepth, const isminefilter& filter, bool fAddLocked);
    std::set<CTxDestination> GetAccountAddresses(const std::string& strAccount) const;

    isminetype IsMine(const CTxIn& txin) const;
    /**
     * Returns amount of debit if the input matches the
     * filter, otherwise returns 0
     */
    CAmount GetDebit(const CTxIn& txin, const isminefilter& filter) const;
    CAmount GetDebit(const CTxIn& txin, const isminefilter& filter, CAssetOutputEntry& assetData) const;
    isminetype IsMine(const CTxOut& txout) const;
    CAmount GetCredit(const CTxOut& txout, const isminefilter& filter) const;
    bool IsChange(const CTxOut& txout) const;
    CAmount GetChange(const CTxOut& txout) const;
    bool IsMine(const CTransaction& tx) const;
    void AutoLockDynodeCollaterals();

    bool IsRelevantToMe(const CTransaction& tx) const;
    bool IsFromMe(const CTransaction& tx, const isminefilter& filter) const;
    CAmount GetDebit(const CTransaction& tx, const isminefilter& filter) const;
    /** Returns whether all of the inputs match the filter */
    bool IsAllFromMe(const CTransaction& tx, const isminefilter& filter) const;
    CAmount GetCredit(const CTransaction& tx, const isminefilter& filter) const;
    CAmount GetChange(const CTransaction& tx) const;
    void SetBestChain(const CBlockLocator& loc) override;

    DBErrors LoadWallet(bool& fFirstRunRet, const bool fImportMnemonic = false);
    DBErrors ZapSelectTx(std::vector<uint256>& vHashIn, std::vector<uint256>& vHashOut);
    DBErrors ZapWalletTx(std::vector<CWalletTx>& vWtx);

    bool SetAddressBook(const CTxDestination& address, const std::string& strName, const std::string& purpose);

    bool DelAddressBook(const CTxDestination& address);

    bool UpdatedTransaction(const uint256& hashTx) override;

    void Inventory(const uint256& hash) override
    {
        {
            LOCK(cs_wallet);
            std::map<uint256, int>::iterator mi = mapRequestCount.find(hash);
            if (mi != mapRequestCount.end())
                (*mi).second++;
        }
    }

    void GetScriptForMining(std::shared_ptr<CReserveScript>& script) override;
    void ResetRequestCount(const uint256& hash) override
    {
        LOCK(cs_wallet);
        mapRequestCount[hash] = 0;
    };

    unsigned int GetKeyPoolSize()
    {
        AssertLockHeld(cs_wallet); // set{Ex,In}ternalKeyPool
        return setInternalKeyPool.size() + setExternalKeyPool.size();
    }

    bool SetDefaultKey(const CPubKey& vchPubKey);

    //! signify that a particular wallet feature is now used. this may change nWalletVersion and nWalletMaxVersion if those are lower
    bool SetMinVersion(enum WalletFeature, CWalletDB* pwalletdbIn = nullptr, bool fExplicit = false);

    //! change which version we're allowed to upgrade to (note that this does not immediately imply upgrading to that format)
    bool SetMaxVersion(int nVersion);

    //! get the current wallet format (the oldest client version guaranteed to understand this wallet)
    int GetVersion()
    {
        LOCK(cs_wallet);
        return nWalletVersion;
    }

    //! Get wallet transactions that conflict with given transaction (spend same outputs)
    std::set<uint256> GetConflicts(const uint256& txid) const;

    //! Check if a given transaction has any of its outputs spent by another transaction in the wallet
    bool HasWalletSpend(const uint256& txid) const;

    //! Flush wallet (bitdb flush)
    void Flush(bool shutdown = false);

    void UpdateMyRestrictedAssets(std::string& address, std::string& asset_name,
                                  int type, uint32_t date);
    
    //! Verify the wallet database and perform salvage if required
    static bool Verify();

    /** 
     * Address book entry changed.
     * @note called with lock cs_wallet held.
     */
    boost::signals2::signal<void(CWallet* wallet, const CTxDestination& address, const std::string& label, bool isMine, const std::string& purpose, ChangeType status)> NotifyAddressBookChanged;

    /** 
     * Wallet transaction added, removed or updated.
     * @note called with lock cs_wallet held.
     */
    boost::signals2::signal<void(CWallet* wallet, const uint256& hashTx, ChangeType status)> NotifyTransactionChanged;

    /**
     * Wallets Restricted Asset Address data added or updated.
     * @note called with lock cs_wallet held.
     */
    boost::signals2::signal<void (CWallet *wallet, std::string& address, std::string& asset_name,
                                  int type, uint32_t date)> NotifyMyRestrictedAssetsChanged;
    
    /** Show progress e.g. for rescan */
    boost::signals2::signal<void(const std::string& title, int nProgress)> ShowProgress;

    /** Watch-only address added */
    boost::signals2::signal<void(bool fHaveWatchOnly)> NotifyWatchonlyChanged;

    /** Inquire whether this wallet broadcasts transactions. */
    bool GetBroadcastTransactions() const { return fBroadcastTransactions; }
    /** Set whether this wallet broadcasts transactions. */
    void SetBroadcastTransactions(bool broadcast) { fBroadcastTransactions = broadcast; }

    /* Mark a transaction (and it in-wallet descendants) as abandoned so its inputs may be respent. */
    bool AbandonTransaction(const uint256& hashTx);

    /** Mark a transaction as replaced by another transaction (e.g., BIP 125). */
    bool MarkReplaced(const uint256& originalHash, const uint256& newHash);

    /* Initializes the wallet, returns a new CWallet instance or a null pointer in case of an error */
    static CWallet* CreateWalletFromFile(const std::string walletFile, const bool fImportMnemonic = false);
    static bool InitLoadWallet();

    /**
     * Wallet post-init setup
     * Gives the wallet a chance to register repetitive tasks and complete post-init tasks
     */
    void postInitProcess(boost::thread_group& threadGroup);

    /* Initialize AutoBackup functionality */
    static bool InitAutoBackup();

    bool BackupWallet(const std::string& strDest);

    /**
     * HD Wallet Functions
     */

    /* Returns true if HD is enabled */
    bool IsHDEnabled();
    /* Generates a new HD chain */
    void GenerateNewHDChain();
    /* Set the HD chain model (chain child index counters) */
    bool SetHDChain(const CHDChain& chain, bool memonly);
    bool SetCryptedHDChain(const CHDChain& chain, bool memonly);
    bool GetDecryptedHDChain(CHDChain& hdChainRet);
    // Support mnemonic sub-chains
    void DeriveNewChildKeyBIP44BychainChildKey(CExtKey& chainChildKey, CKey& secret, bool internal, uint32_t* nInternalChainCounter, uint32_t* nExternalChainCounter);
    // Returns local BDAP DHT Public keys
    bool GetDHTPubKeys(std::vector<std::vector<unsigned char>>& vvchDHTPubKeys) const override;

    bool WriteLinkMessageInfo(const uint256& subjectID, const std::vector<unsigned char>& vchPubKey);
    bool EraseLinkMessageInfo(const uint256& subjectID);

    // Stealth Address Support
    bool GetStealthAddress(const CKeyID& keyid, CStealthAddress& sxAddr) const;
    bool ProcessStealthQueue();
    bool ProcessStealthOutput(const CTxDestination& address, std::vector<uint8_t>& vchEphemPK, uint32_t prefix, bool fHavePrefix, CKey& sShared);
    int CheckForStealthTxOut(const CTxOut* pTxOut, const CTxOut* pTxData);
    bool HasBDAPLinkTx(const CTransaction& tx, CScript& bdapOpScript);
    bool ScanForStealthOwnedOutputs(const CTransaction& tx);
    bool AddStealthAddress(const CStealthAddress& sxAddr);
    bool AddStealthToMap(const std::pair<CKeyID, CStealthAddress>& pairStealthAddress);
    bool AddToStealthQueue(const std::pair<CKeyID, CStealthKeyQueueData>& pairStealthQueue);
    CWalletDB* GetWalletDB();
    bool HaveStealthAddress(const CKeyID& address) const;
    bool MintableCoins();
    bool SelectStakeCoins(std::list<std::unique_ptr<CStakeInput> >& listInputs, const CAmount& nTargetAmount, const int& blockHeight, const bool fPrecompute = false);
    bool CreateCoinStake(const CKeyStore& keystore, const unsigned int& nBits, const int64_t& nSearchInterval, CMutableTransaction& txNew, unsigned int& nTxNewTime);
    bool MultiSend();
    void AutoCombineDust();
};

/** A key allocated from the key pool. */
class CReserveKey : public CReserveScript
{
protected:
    CWallet* pwallet;
    int64_t nIndex;
    CPubKey vchPubKey;
    bool fInternal;

public:
    CReserveKey(CWallet* pwalletIn)
    {
        nIndex = -1;
        pwallet = pwalletIn;
        fInternal = false;
    }

    ~CReserveKey()
    {
        ReturnKey();
    }

    void ReturnKey();
    bool GetReservedKey(CPubKey& pubkey, bool fInternalIn /*= false*/);
    void KeepKey();
    void KeepScript() override { KeepKey(); }
};


/** 
 * Account information.
 * Stored in wallet with key "acc"+string account name.
 */
class CAccount
{
public:
    CPubKey vchPubKey;

    CAccount()
    {
        SetNull();
    }

    void SetNull()
    {
        vchPubKey = CPubKey();
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        int nVersion = s.GetVersion();
        if (!(s.GetType() & SER_GETHASH))
            READWRITE(nVersion);
        READWRITE(vchPubKey);
    }
};

// Helper for producing a bunch of max-sized low-S signatures (eg 72 bytes)
// ContainerType is meant to hold pair<CWalletTx *, int>, and be iterable
// so that each entry corresponds to each vIn, in order.
// Returns true if all inputs could be signed normally, false if any were padded out for sizing purposes.
template <typename ContainerType>
bool CWallet::DummySignTx(CMutableTransaction &txNew, const ContainerType &coins) const
{
    bool allSigned = true;

    // pad past max expected sig length (256)
    const std::string zeros = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    const unsigned char* cstrZeros = (unsigned char*)zeros.c_str();

    // Fill in dummy signatures for fee calculation.
    int nIn = 0;
    for (const auto& coin : coins)
    {
        const CScript& scriptPubKey = coin.txout.scriptPubKey;
        SignatureData sigdata;

        if (!ProduceSignature(DummySignatureCreator(this), scriptPubKey, sigdata))
        {
            // just add dummy 256 bytes as sigdata if this fails (can't necessarily sign for all inputs)
            CScript dummyScript = CScript(cstrZeros, cstrZeros + 256);
            SignatureData dummyData = SignatureData(dummyScript);
            UpdateTransaction(txNew, nIn, dummyData);
            allSigned = false;
        } else {
            UpdateTransaction(txNew, nIn, sigdata);
        }

        nIn++;
    }
    return allSigned;
}

bool RunProcessStealthQueue();

bool IsDataScript(const CScript& data);
bool GetDataFromScript(const CScript& data, std::vector<uint8_t>& vData);

#endif // DYNAMIC_WALLET_WALLET_H
