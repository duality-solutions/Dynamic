// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "validation.h"

#include "alert.h"
#include "arith_uint256.h"
#include "bdap/auditdb.h"
#include "bdap/domainentrydb.h"
#include "bdap/fees.h"
#include "bdap/linking.h"
#include "bdap/linkingdb.h"
#include "bdap/utils.h"
#include "blockencodings.h"
#include "chainparams.h"
#include "checkpoints.h"
#include "checkqueue.h"
#include "consensus/consensus.h"
#include "consensus/merkle.h"
#include "consensus/params.h"
#include "consensus/validation.h"
#include "cuckoocache.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "fluid/banaccount.h"
#include "fluid/fluid.h"
#include "fluid/fluiddb.h"
#include "fluid/fluiddynode.h"
#include "fluid/fluidmining.h"
#include "fluid/fluidmint.h"
#include "fluid/fluidstaking.h"
#include "fs.h"
#include "hash.h"
#include "init.h"
#include "instantsend.h"
#include "keystore.h"
#include "policy/fees.h"
#include "policy/policy.h"
#include "pos/kernel.h"
#include "pos/stakeinput.h"
#include "pow.h"
#include "primitives/block.h"
#include "primitives/transaction.h"
#include "pubkey.h"
#include "random.h"
#include "reverse_iterator.h"
#include "rpc/server.h"
#include "script/script.h"
#include "script/sigcache.h"
#include "script/standard.h"
#include "spork.h"
#include "timedata.h"
#include "tinyformat.h"
#include "txdb.h"
#include "txmempool.h"
#include "ui_interface.h"
#include "undo.h"
#include "util.h"
#include "utilmoneystr.h"
#include "utilstrencodings.h"
#include "validationinterface.h"
#include "versionbits.h"
#include "wallet/wallet.h"
#include "warnings.h"

#include <atomic>
#include <sstream>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/thread.hpp>

#if defined(NDEBUG)
#error "Dynamic cannot be compiled without assertions."
#endif

#define MICRO 0.000001
#define MILLI 0.001

/**
 * Global state
 */

CCriticalSection cs_main;

BlockMap mapBlockIndex;
CChain chainActive;
CBlockIndex* pindexBestHeader = nullptr;
Mutex g_best_block_mutex;
std::condition_variable g_best_block_cv;
uint256 g_best_block;
std::map<uint256, uint256> mapProofOfStake;
std::map<unsigned int, unsigned int> mapHashedBlocks;
int nScriptCheckThreads = 0;
std::atomic_bool fImporting(false);
bool fMessaging = true;
bool fReindex = false;
bool fTxIndex = true;
bool fAssetIndex = false;
bool fAddressIndex = false;
bool fTimestampIndex = false;
bool fSpentIndex = false;
bool fHavePruned = false;
bool fPruneMode = false;
bool fIsBareMultisigStd = DEFAULT_PERMIT_BAREMULTISIG;
bool fRequireStandard = true;
bool fCheckBlockIndex = false;
bool fCheckpointsEnabled = DEFAULT_CHECKPOINTS_ENABLED;
size_t nCoinCacheUsage = 5000 * 300;
uint64_t nPruneTarget = 0;
bool fAlerts = DEFAULT_ALERTS;
int64_t nMaxTipAge = DEFAULT_MAX_TIP_AGE;
bool fEnableReplacement = DEFAULT_ENABLE_REPLACEMENT;
bool fStealthTx = false;
int64_t nReserveBalance = 0;
bool fUnitTest = false;

uint256 hashAssumeValid;

CFeeRate minRelayTxFee = CFeeRate(DEFAULT_MIN_RELAY_TX_FEE);
CAmount maxTxFee = DEFAULT_TRANSACTION_MAXFEE;

CBlockPolicyEstimator feeEstimator;
CTxMemPool mempool(&feeEstimator);

std::map<uint256, int64_t> mapRejectedBlocks GUARDED_BY(cs_main);

static bool IsSuperMajority(int minVersion, const CBlockIndex* pstart, unsigned nRequired, const Consensus::Params& consensusParams);
static void CheckBlockIndex(const Consensus::Params& consensusParams);

/** Constant stuff for coinbase transactions we create: */
CScript COINBASE_FLAGS;

const std::string strMessageMagic = "Dynamic Signed Message:\n";

// Internal stuff
namespace
{
struct CBlockIndexWorkComparator {
    bool operator()(CBlockIndex* pa, CBlockIndex* pb) const
    {
        // First sort by most total work, ...
        if (pa->nChainWork > pb->nChainWork)
            return false;
        if (pa->nChainWork < pb->nChainWork)
            return true;

        // ... then by earliest time received, ...
        if (pa->nSequenceId < pb->nSequenceId)
            return false;
        if (pa->nSequenceId > pb->nSequenceId)
            return true;

        // Use pointer address as tie breaker (should only happen with blocks
        // loaded from disk, as those all have id 0).
        if (pa < pb)
            return false;
        if (pa > pb)
            return true;

        // Identical blocks.
        return false;
    }
};

CBlockIndex* pindexBestInvalid;

/**
     * The set of all CBlockIndex entries with BLOCK_VALID_TRANSACTIONS (for itself and all ancestors) and
     * as good as our current tip or better. Entries may be failed, though, and pruning nodes may be
     * missing the data for the block.
     */
std::set<CBlockIndex*, CBlockIndexWorkComparator> setBlockIndexCandidates;
/** All pairs A->B, where A (or one of its ancestors) misses transactions, but B has transactions.
     * Pruned nodes may have entries where B is missing data.
     */
std::multimap<CBlockIndex*, CBlockIndex*> mapBlocksUnlinked;

CCriticalSection cs_LastBlockFile;
std::vector<CBlockFileInfo> vinfoBlockFile;
int nLastBlockFile = 0;
/** Global flag to indicate we should check to see if there are
     *  block/undo files that should be deleted.  Set on startup
     *  or if we allocate more file space when we're in prune mode
     */
bool fCheckForPruning = false;

/**
     * Every received block is assigned a unique and increasing identifier, so we
     * know which one to give priority in case of a fork.
     */
CCriticalSection cs_nBlockSequenceId;
/** Blocks loaded from disk are assigned id 0, so start the counter at 1. */
int32_t nBlockSequenceId = 1;
/** Decreasing counter (used by subsequent preciousblock calls). */
int32_t nBlockReverseSequenceId = -1;
/** chainwork for the last block that preciousblock has been applied to. */
arith_uint256 nLastPreciousChainwork = 0;

/** Dirty block index entries. */
std::set<CBlockIndex*> setDirtyBlockIndex;

/** Dirty block file entries. */
std::set<int> setDirtyFileInfo;
} // namespace

/* Use this class to start tracking transactions that are removed from the
 * mempool and pass all those transactions through SyncTransaction when the
 * object goes out of scope. This is currently only used to call SyncTransaction
 * on conflicts removed from the mempool during block connection.  Applied in
 * ActivateBestChain around ActivateBestStep which in turn calls:
 * ConnectTip->removeForBlock->removeConflicts
 */
class MemPoolConflictRemovalTracker
{
private:
    std::vector<CTransactionRef> conflictedTxs;
    CTxMemPool& pool;

public:
    MemPoolConflictRemovalTracker(CTxMemPool& _pool) : pool(_pool)
    {
        pool.NotifyEntryRemoved.connect(boost::bind(&MemPoolConflictRemovalTracker::NotifyEntryRemoved, this, _1, _2));
    }

    void NotifyEntryRemoved(CTransactionRef txRemoved, MemPoolRemovalReason reason)
    {
        if (reason == MemPoolRemovalReason::CONFLICT) {
            conflictedTxs.push_back(txRemoved);
        }
    }

    ~MemPoolConflictRemovalTracker()
    {
        pool.NotifyEntryRemoved.disconnect(boost::bind(&MemPoolConflictRemovalTracker::NotifyEntryRemoved, this, _1, _2));
        for (const auto& tx : conflictedTxs) {
            GetMainSignals().SyncTransaction(*tx, nullptr, CMainSignals::SYNC_TRANSACTION_NOT_IN_BLOCK);
        }
        conflictedTxs.clear();
    }
};

CBlockIndex* FindForkInGlobalIndex(const CChain& chain, const CBlockLocator& locator)
{
    // Find the first block the caller has in the main chain
    for (const uint256& hash : locator.vHave) {
        BlockMap::iterator mi = mapBlockIndex.find(hash);
        if (mi != mapBlockIndex.end()) {
            CBlockIndex* pindex = (*mi).second;
            if (chain.Contains(pindex))
                return pindex;
        }
    }
    return chain.Genesis();
}

CCoinsViewDB* pcoinsdbview = nullptr;
CCoinsViewCache* pcoinsTip = nullptr;
CBlockTreeDB* pblocktree = nullptr;

CAssetsDB *passetsdb = nullptr;
CAssetsCache *passets = nullptr;
CLRUCache<std::string, CDatabasedAssetData> *passetsCache = nullptr;
CLRUCache<std::string, CMessage> *pMessagesCache = nullptr;
CLRUCache<std::string, int> *pMessageSubscribedChannelsCache = nullptr;
CLRUCache<std::string, int> *pMessagesSeenAddressCache = nullptr;
CMessageDB *pmessagedb = nullptr;
CMessageChannelDB *pmessagechanneldb = nullptr;
CMyRestrictedDB *pmyrestricteddb = nullptr;
CSnapshotRequestDB *pSnapshotRequestDb = nullptr;
CAssetSnapshotDB *pAssetSnapshotDb = nullptr;
CDistributeSnapshotRequestDB *pDistributeSnapshotDb = nullptr;

CLRUCache<std::string, CNullAssetTxVerifierString> *passetsVerifierCache = nullptr;
CLRUCache<std::string, int8_t> *passetsQualifierCache = nullptr;
CLRUCache<std::string, int8_t> *passetsRestrictionCache = nullptr;
CLRUCache<std::string, int8_t> *passetsGlobalRestrictionCache = nullptr;
CRestrictedDB *prestricteddb = nullptr;

enum FlushStateMode {
    FLUSH_STATE_NONE,
    FLUSH_STATE_IF_NEEDED,
    FLUSH_STATE_PERIODIC,
    FLUSH_STATE_ALWAYS
};

// See definition for documentation
bool static FlushStateToDisk(CValidationState& state, FlushStateMode mode, int nManualPruneHeight = 0);
void FindFilesToPruneManual(std::set<int>& setFilesToPrune, int nManualPruneHeight);

bool IsFinalTx(const CTransaction& tx, int nBlockHeight, int64_t nBlockTime)
{
    if (tx.nLockTime == 0)
        return true;

    if ((int64_t)tx.nLockTime < ((int64_t)tx.nLockTime < LOCKTIME_THRESHOLD ? (int64_t)nBlockHeight : nBlockTime))
        return true;

    if (nBlockHeight >= fluid.FLUID_ACTIVATE_HEIGHT) {
        if (!fluid.ProvisionalCheckTransaction(tx))
            return false;

        CScript scriptFluid;
        if (IsTransactionFluid(tx, scriptFluid)) {
            std::string strErrorMessage;
            if (!fluid.CheckFluidOperationScript(scriptFluid, nBlockTime, strErrorMessage)) {
                return false;
            }
        }
    }

    for (const auto& txin : tx.vin) {
        if (!(txin.nSequence == CTxIn::SEQUENCE_FINAL))
            return false;
    }
    return true;
}

bool CheckFinalTx(const CTransaction& tx, int flags)
{
    AssertLockHeld(cs_main);

    // By convention a negative value for flags indicates that the
    // current network-enforced consensus rules should be used. In
    // a future soft-fork scenario that would mean checking which
    // rules would be enforced for the next block and setting the
    // appropriate flags. At the present time no soft-forks are
    // scheduled, so no flags are set.
    flags = std::max(flags, 0);

    // CheckFinalTx() uses chainActive.Height()+1 to evaluate
    // nLockTime because when IsFinalTx() is called within
    // CBlock::AcceptBlock(), the height of the block *being*
    // evaluated is what is used. Thus if we want to know if a
    // transaction can be part of the *next* block, we need to call
    // IsFinalTx() with one more than chainActive.Height().
    const int nBlockHeight = chainActive.Height() + 1;

    // BIP113 will require that time-locked transactions have nLockTime set to
    // less than the median time of the previous block they're contained in.
    // When the next block is created its previous block will be the current
    // chain tip, so we use that to calculate the median time passed to
    // IsFinalTx() if LOCKTIME_MEDIAN_TIME_PAST is set.
    const int64_t nBlockTime = (flags & LOCKTIME_MEDIAN_TIME_PAST) ? chainActive.Tip()->GetMedianTimePast() : GetAdjustedTime();

    return IsFinalTx(tx, nBlockHeight, nBlockTime);
}

/**
 * Calculates the block height and previous block's median time past at
 * which the transaction will be considered final in the context of BIP 68.
 * Also removes from the vector of input heights any entries which did not
 * correspond to sequence locked inputs as they do not affect the calculation.
 */
static std::pair<int, int64_t> CalculateSequenceLocks(const CTransaction& tx, int flags, std::vector<int>* prevHeights, const CBlockIndex& block)
{
    assert(prevHeights->size() == tx.vin.size());

    // Will be set to the equivalent height- and time-based nLockTime
    // values that would be necessary to satisfy all relative lock-
    // time constraints given our view of block chain history.
    // The semantics of nLockTime are the last invalid height/time, so
    // use -1 to have the effect of any height or time being valid.
    int nMinHeight = -1;
    int64_t nMinTime = -1;

    // tx.nVersion is signed integer so requires cast to unsigned otherwise
    // we would be doing a signed comparison and half the range of nVersion
    // wouldn't support BIP 68.
    bool fEnforceBIP68 = static_cast<uint32_t>(tx.nVersion) >= 2 && flags & LOCKTIME_VERIFY_SEQUENCE;

    // Do not enforce sequence numbers as a relative lock time
    // unless we have been instructed to
    if (!fEnforceBIP68) {
        return std::make_pair(nMinHeight, nMinTime);
    }

    for (size_t txinIndex = 0; txinIndex < tx.vin.size(); txinIndex++) {
        const CTxIn& txin = tx.vin[txinIndex];

        // Sequence numbers with the most significant bit set are not
        // treated as relative lock-times, nor are they given any
        // consensus-enforced meaning at this point.
        if (txin.nSequence & CTxIn::SEQUENCE_LOCKTIME_DISABLE_FLAG) {
            // The height of this input is not relevant for sequence locks
            (*prevHeights)[txinIndex] = 0;
            continue;
        }

        int nCoinHeight = (*prevHeights)[txinIndex];

        if (txin.nSequence & CTxIn::SEQUENCE_LOCKTIME_TYPE_FLAG) {
            int64_t nCoinTime = block.GetAncestor(std::max(nCoinHeight - 1, 0))->GetMedianTimePast();
            // NOTE: Subtract 1 to maintain nLockTime semantics
            // BIP 68 relative lock times have the semantics of calculating
            // the first block or time at which the transaction would be
            // valid. When calculating the effective block time or height
            // for the entire transaction, we switch to using the
            // semantics of nLockTime which is the last invalid block
            // time or height.  Thus we subtract 1 from the calculated
            // time or height.

            // Time-based relative lock-times are measured from the
            // smallest allowed timestamp of the block containing the
            // txout being spent, which is the median time past of the
            // block prior.
            nMinTime = std::max(nMinTime, nCoinTime + (int64_t)((txin.nSequence & CTxIn::SEQUENCE_LOCKTIME_MASK) << CTxIn::SEQUENCE_LOCKTIME_GRANULARITY) - 1);
        } else {
            nMinHeight = std::max(nMinHeight, nCoinHeight + (int)(txin.nSequence & CTxIn::SEQUENCE_LOCKTIME_MASK) - 1);
        }
    }

    return std::make_pair(nMinHeight, nMinTime);
}

static bool EvaluateSequenceLocks(const CBlockIndex& block, std::pair<int, int64_t> lockPair)
{
    assert(block.pprev);
    int64_t nBlockTime = block.pprev->GetMedianTimePast();
    if (lockPair.first >= block.nHeight || lockPair.second >= nBlockTime)
        return false;

    return true;
}

bool SequenceLocks(const CTransaction& tx, int flags, std::vector<int>* prevHeights, const CBlockIndex& block)
{
    return EvaluateSequenceLocks(block, CalculateSequenceLocks(tx, flags, prevHeights, block));
}

bool TestLockPointValidity(const LockPoints* lp)
{
    AssertLockHeld(cs_main);
    assert(lp);
    // If there are relative lock times then the maxInputBlock will be set
    // If there are no relative lock times, the LockPoints don't depend on the chain
    if (lp->maxInputBlock) {
        // Check whether chainActive is an extension of the block at which the LockPoints
        // calculation was valid.  If not LockPoints are no longer valid
        if (!chainActive.Contains(lp->maxInputBlock)) {
            return false;
        }
    }

    // LockPoints still valid
    return true;
}

bool CheckSequenceLocks(const CTransaction& tx, int flags, LockPoints* lp, bool useExistingLockPoints)
{
    AssertLockHeld(cs_main);
    AssertLockHeld(mempool.cs);

    CBlockIndex* tip = chainActive.Tip();
    CBlockIndex index;
    index.pprev = tip;
    // CheckSequenceLocks() uses chainActive.Height()+1 to evaluate
    // height based locks because when SequenceLocks() is called within
    // ConnectBlock(), the height of the block *being*
    // evaluated is what is used.
    // Thus if we want to know if a transaction can be part of the
    // *next* block, we need to use one more than chainActive.Height()
    index.nHeight = tip->nHeight + 1;

    std::pair<int, int64_t> lockPair;
    if (useExistingLockPoints) {
        assert(lp);
        lockPair.first = lp->height;
        lockPair.second = lp->time;
    } else {
        // pcoinsTip contains the UTXO set for chainActive.Tip()
        CCoinsViewMemPool viewMemPool(pcoinsTip, mempool);
        std::vector<int> prevheights;
        prevheights.resize(tx.vin.size());
        for (size_t txinIndex = 0; txinIndex < tx.vin.size(); txinIndex++) {
            const CTxIn& txin = tx.vin[txinIndex];
            Coin coin;
            if (!viewMemPool.GetCoin(txin.prevout, coin)) {
                return error("%s: Missing input", __func__);
            }
            if (coin.nHeight == MEMPOOL_HEIGHT) {
                // Assume all mempool transaction confirm in the next block
                prevheights[txinIndex] = tip->nHeight + 1;
            } else {
                prevheights[txinIndex] = coin.nHeight;
            }
        }
        lockPair = CalculateSequenceLocks(tx, flags, &prevheights, index);
        if (lp) {
            lp->height = lockPair.first;
            lp->time = lockPair.second;
            // Also store the hash of the block with the highest height of
            // all the blocks which have sequence locked prevouts.
            // This hash needs to still be on the chain
            // for these LockPoint calculations to be valid
            // Note: It is impossible to correctly calculate a maxInputBlock
            // if any of the sequence locked inputs depend on unconfirmed txs,
            // except in the special case where the relative lock time/height
            // is 0, which is equivalent to no sequence lock. Since we assume
            // input height of tip+1 for mempool txs and test the resulting
            // lockPair from CalculateSequenceLocks against tip+1.  We know
            // EvaluateSequenceLocks will fail if there was a non-zero sequence
            // lock on a mempool input, so we can use the return value of
            // CheckSequenceLocks to indicate the LockPoints validity
            int maxInputHeight = 0;
            for (int height : prevheights) {
                // Can ignore mempool inputs since we'll fail if they had non-zero locks
                if (height != tip->nHeight + 1) {
                    maxInputHeight = std::max(maxInputHeight, height);
                }
            }
            lp->maxInputBlock = tip->GetAncestor(maxInputHeight);
        }
    }
    return EvaluateSequenceLocks(index, lockPair);
}


unsigned int GetLegacySigOpCount(const CTransaction& tx)
{
    unsigned int nSigOps = 0;
    for (const CTxIn& txin : tx.vin) {
        nSigOps += txin.scriptSig.GetSigOpCount(false);
    }
    for (const CTxOut& txout : tx.vout) {
        nSigOps += txout.scriptPubKey.GetSigOpCount(false);
    }
    return nSigOps;
}

unsigned int GetP2SHSigOpCount(const CTransaction& tx, const CCoinsViewCache& inputs)
{
    if (tx.IsCoinBase())
        return 0;

    unsigned int nSigOps = 0;
    for (unsigned int i = 0; i < tx.vin.size(); i++) {
        const Coin& coin = inputs.AccessCoin(tx.vin[i].prevout);
        assert(!coin.IsSpent());
        const CTxOut& prevout = coin.out;
        if (prevout.scriptPubKey.IsPayToScriptHash())
            nSigOps += prevout.scriptPubKey.GetSigOpCount(tx.vin[i].scriptSig);
    }
    return nSigOps;
}

bool GetUTXOCoin(const COutPoint& outpoint, Coin& coin)
{
    LOCK(cs_main);
    if (!pcoinsTip->GetCoin(outpoint, coin))
        return false;
    if (coin.IsSpent())
        return false;
    return true;
}

int GetUTXOHeight(const COutPoint& outpoint)
{
    // -1 means UTXO is yet unknown or already spent
    Coin coin;
    return GetUTXOCoin(outpoint, coin) ? coin.nHeight : -1;
}

int GetUTXOConfirmations(const COutPoint& outpoint)
{
    // -1 means UTXO is yet unknown or already spent
    LOCK(cs_main);
    int nPrevoutHeight = GetUTXOHeight(outpoint);
    return (nPrevoutHeight > -1 && chainActive.Tip()) ? chainActive.Height() - nPrevoutHeight + 1 : -1;
}

bool CheckTransaction(const CTransaction& tx, CValidationState& state, bool fCheckDuplicateInputs)
{
    // Basic checks that don't depend on any context
    if (tx.vin.empty())
        return state.DoS(10, false, REJECT_INVALID, "bad-txns-vin-empty");
    if (tx.vout.empty())
        return state.DoS(10, false, REJECT_INVALID, "bad-txns-vout-empty");
    // Size limits
    if (::GetSerializeSize(tx, SER_NETWORK, PROTOCOL_VERSION) > MAX_TX_SIZE)
        return state.DoS(100, false, REJECT_INVALID, "bad-txns-oversize");

    // Check for BDAP inputs or outputs so we can validate credit usage
    bool fIsBDAP = false;
    // Check for negative or overflow output values
    CAmount nValueOut = 0;
    CAmount nStandardOut = 0;
    CAmount nCreditsOut = 0;
    CAmount nDataBurned = 0;
    std::set<std::string> setAssetTransferNames;
    std::map<std::pair<std::string, std::string>, int> mapNullDataTxCount; // (asset_name, address) -> int
    std::set<std::string> setNullGlobalAssetChanges;
    bool fContainsNewRestrictedAsset = false;
    bool fContainsRestrictedAssetReissue = false;
    bool fContainsNullAssetVerifierTx = false;
    int nCountAddTagOuts = 0;
    for (const CTxOut& txout : tx.vout) {
        if ((txout.nValue < 0) && !tx.IsCoinBase() && !tx.IsCoinStake())
            return state.DoS(100, error("CheckTransaction(): txout empty for user transaction"));
        if (txout.nValue < 0)
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-vout-negative");
        if (txout.nValue > MAX_MONEY)
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-vout-toolarge");
        nValueOut += txout.nValue;
        if (!MoneyRange(nValueOut))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-txouttotal-toolarge");
        if (IsTransactionFluid(txout.scriptPubKey)) {
            if (fluid.FLUID_TRANSACTION_COST > txout.nValue)
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-fluid-vout-amount-toolow");
            if (!fluid.ValidationProcesses(state, txout.scriptPubKey, txout.nValue))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-fluid-validate-failure");
        }
        if (txout.IsBDAP()) {
            fIsBDAP = true;
            nCreditsOut += txout.nValue;
        } else if (txout.IsData()) {
            nDataBurned += txout.nValue;
        } else {
            nStandardOut += txout.nValue;
        }

        /** ASSET START */
        // Find and handle all new OP_DYN_ASSET null data transactions
        if (txout.scriptPubKey.IsNullAsset()) {
            CNullAssetTxData data;
            std::string address;
            std::string strError = "";

            if (txout.scriptPubKey.IsNullAssetTxDataScript()) {
                if (!AssetNullDataFromScript(txout.scriptPubKey, data, address))
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-null-asset-data-serialization");

                if (!VerifyNullAssetDataFlag(data.flag, strError))
                    return state.DoS(100, false, REJECT_INVALID, strError);

                auto pair = std::make_pair(data.asset_name, address);
                if(!mapNullDataTxCount.count(pair)){
                    mapNullDataTxCount.insert(std::make_pair(pair, 0));
                }

                mapNullDataTxCount.at(pair)++;

                if (mapNullDataTxCount.at(pair) > 1)
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-null-data-only-one-change-per-asset-address");

                // For each qualifier that is added, there is a burn fee
                if (IsAssetNameAQualifier(data.asset_name)) {
                    if (data.flag == (int)QualifierType::ADD_QUALIFIER) {
                        nCountAddTagOuts++;
                    }
                }

            } else if (txout.scriptPubKey.IsNullGlobalRestrictionAssetTxDataScript()) {
                if (!GlobalAssetNullDataFromScript(txout.scriptPubKey, data))
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-null-global-asset-data-serialization");

                if (!VerifyNullAssetDataFlag(data.flag, strError))
                    return state.DoS(100, false, REJECT_INVALID, strError);

                if (setNullGlobalAssetChanges.count(data.asset_name)) {
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-null-data-only-one-global-change-per-asset-name");
                }

                setNullGlobalAssetChanges.insert(data.asset_name);

            } else if (txout.scriptPubKey.IsNullAssetVerifierTxDataScript()) {

                if (!CheckVerifierAssetTxOut(txout, strError))
                    return state.DoS(100, false, REJECT_INVALID, strError);

                if (fContainsNullAssetVerifierTx)
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-null-data-only-one-verifier-per-tx");

                fContainsNullAssetVerifierTx = true;
            }
        }
        /** ASSET END */

        /** ASSET START */
        bool isAsset = false;
        int nType;
        bool fIsOwner;
        if (txout.scriptPubKey.IsAssetScript(nType, fIsOwner))
            isAsset = true;
        
        // Check for transfers that don't meet the assets units only if the assetCache is not null
        if (isAsset) {
            // Get the transfer transaction data from the scriptPubKey
            if (nType == TX_TRANSFER_ASSET) {
                CAssetTransfer transfer;
                std::string address;
                if (!TransferAssetFromScript(txout.scriptPubKey, transfer, address))
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-asset-bad-deserialize");

                // insert into set, so that later on we can check asset null data transactions
                setAssetTransferNames.insert(transfer.strName);

                // Check asset name validity and get type
                AssetType assetType;
                if (!IsAssetNameValid(transfer.strName, assetType)) {
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-asset-name-invalid");
                }

                // If the transfer is an ownership asset. Check to make sure that it is OWNER_ASSET_AMOUNT
                if (IsAssetNameAnOwner(transfer.strName)) {
                    if (transfer.nAmount != OWNER_ASSET_AMOUNT)
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-owner-amount-was-not-1");
                }

                // If the transfer is a unique asset. Check to make sure that it is UNIQUE_ASSET_AMOUNT
                if (assetType == AssetType::UNIQUE) {
                    if (transfer.nAmount != UNIQUE_ASSET_AMOUNT)
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-unique-amount-was-not-1");
                }

                // If the transfer is a restricted channel asset.
                if (assetType == AssetType::RESTRICTED) {
                    // TODO add checks here if any
                }

                // If the transfer is a qualifier channel asset.
                if (assetType == AssetType::QUALIFIER || assetType == AssetType::SUB_QUALIFIER) {
                    if (transfer.nAmount < QUALIFIER_ASSET_MIN_AMOUNT || transfer.nAmount > QUALIFIER_ASSET_MAX_AMOUNT)
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-qualifier-amount-must be between 1 - 100");
                }
                
                // Specific check and error message to go with to make sure the amount is 0
                if (txout.nValue != 0)
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-asset-transfer-amount-isn't-zero");
            } else if (nType == TX_NEW_ASSET) {
                // Specific check and error message to go with to make sure the amount is 0
                if (txout.nValue != 0)
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-asset-issued-amount-isn't-zero");
            }
        }
    }

    // Check for Add Tag Burn Fee
    if (nCountAddTagOuts) {
        if (!tx.CheckAddingTagBurnFee(nCountAddTagOuts))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-tx-doesn't-contain-required-burn-fee-for-adding-tags");
    }

    for (auto entry: mapNullDataTxCount) {
        if (entry.first.first.front() == RESTRICTED_CHAR) {
            std::string ownerToken = entry.first.first.substr(1,  entry.first.first.size()); // $TOKEN into TOKEN
            if (!setAssetTransferNames.count(ownerToken + OWNER_TAG)) {
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-tx-contains-restricted-asset-null-tx-without-asset-transfer");
            }
        } else { // must be a qualifier asset QUALIFIER_CHAR
            if (!setAssetTransferNames.count(entry.first.first)) {
                return state.DoS(100, false, REJECT_INVALID,
                                 "bad-txns-tx-contains-qualifier-asset-null-tx-without-asset-transfer");
            }
        }
    }

    for (auto name: setNullGlobalAssetChanges) {
        if (name.size() == 0)
            return state.DoS(100, false, REJECT_INVALID,"bad-txns-tx-contains-global-asset-null-tx-with-null-asset-name");

        std::string rootName = name.substr(1,  name.size()); // $TOKEN into TOKEN
        if (!setAssetTransferNames.count(rootName + OWNER_TAG)) {
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-tx-contains-global-asset-null-tx-without-asset-transfer");
        }
    }

    /** ASSET END */

    if (fCheckDuplicateInputs) {
        // Check for duplicate inputs
        std::set<COutPoint> vInOutPoints;
        for (const auto& txin : tx.vin)
        {
            if (!vInOutPoints.insert(txin.prevout).second)
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-inputs-duplicate");
        }   
    }

    CAmount nStandardIn = 0;
    CAmount nCreditsIn = 0;
    std::vector<Coin> vBdapCoins;
    if (tx.IsCoinBase()) {
        if (tx.vin[0].scriptSig.size() < 2 || tx.vin[0].scriptSig.size() > 100)
            return state.DoS(100, false, REJECT_INVALID, "bad-cb-length");
    } else {
        for (const CTxIn& txin : tx.vin) {
            if (txin.prevout.IsNull())
                return state.DoS(10, false, REJECT_INVALID, "bad-txns-prevout-null");

            CCoinsViewCache view(pcoinsTip);
            const Coin& coin = view.AccessCoin(txin.prevout);
            if (coin.out.IsBDAP()) {
                vBdapCoins.push_back(coin);
                fIsBDAP = true;
                nCreditsIn += coin.out.nValue;
            } else {
                nStandardIn += coin.out.nValue;
            }
        }
    }

    if (tx.IsCoinBase() && tx.nVersion == BDAP_TX_VERSION)
        return state.DoS(100, false, REJECT_INVALID, "bdap-tx-can-not-be-coinbase");

    // if we find a BDAP txin or txout, then make sure the transaction has the correct version
    if (fIsBDAP && tx.nVersion != BDAP_TX_VERSION)
        return state.DoS(100, false, REJECT_INVALID, "incorrect-bdap-tx-version");

    if (fIsBDAP && !CheckBDAPTxCreditUsage(tx, vBdapCoins, nStandardIn, nCreditsIn, nStandardOut, nCreditsOut, nDataBurned))
        return state.DoS(100, false, REJECT_INVALID, "bad-bdap-credit-use");

    /** ASSET START */
    if (tx.IsNewAsset()) {
        /** Verify the reissue assets data */
        std::string strError = "";
        if(!tx.VerifyNewAsset(strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

        CNewAsset asset;
        std::string strAddress;
        if (!AssetFromTransaction(tx, asset, strAddress))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-asset-from-transaction");

        // Validate the new assets information
        if (!IsNewOwnerTxValid(tx, asset.strName, strAddress, strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

        if(!CheckNewAsset(asset, strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

    } else if (tx.IsReissueAsset()) {

        /** Verify the reissue assets data */
        std::string strError;
        if (!tx.VerifyReissueAsset(strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

        CReissueAsset reissue;
        std::string strAddress;
        if (!ReissueAssetFromTransaction(tx, reissue, strAddress))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-reissue-asset");

        if (!CheckReissueAsset(reissue, strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

        // Get the assetType
        AssetType type;
        IsAssetNameValid(reissue.strName, type);

        // If this is a reissuance of a restricted asset, mark it as such, so we can check to make sure only valid verifier string tx are added to the chain
        if (type == AssetType::RESTRICTED) {
            CNullAssetTxVerifierString new_verifier;
            bool fNotFound = false;

            // Try and get the verifier string if it was changed
            if (!tx.GetVerifierStringFromTx(new_verifier, strError, fNotFound)) {
                // If it return false for any other reason besides not being found, fail the transaction check
                if (!fNotFound) {
                    return state.DoS(100, false, REJECT_INVALID,
                                     "bad-txns-reissue-restricted-verifier-" + strError);
                }
            }

            fContainsRestrictedAssetReissue = true;
        }

    } else if (tx.IsNewUniqueAsset()) {

        /** Verify the unique assets data */
        std::string strError = "";
        if (!tx.VerifyNewUniqueAsset(strError)) {
            return state.DoS(100, false, REJECT_INVALID, strError);
        }


        for (auto out : tx.vout)
        {
            if (IsScriptNewUniqueAsset(out.scriptPubKey))
            {
                CNewAsset asset;
                std::string strAddress;
                if (!AssetFromScript(out.scriptPubKey, asset, strAddress))
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-check-transaction-issue-unique-asset-serialization");

                if (!CheckNewAsset(asset, strError))
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-unique" + strError);
            }
        }
    } else if (tx.IsNewMsgChannelAsset()) {
        /** Verify the msg channel assets data */
        std::string strError = "";
        if(!tx.VerifyNewMsgChannelAsset(strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

        CNewAsset asset;
        std::string strAddress;
        if (!MsgChannelAssetFromTransaction(tx, asset, strAddress))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-msgchannel-from-transaction");

        if (!CheckNewAsset(asset, strError))
            return state.DoS(100, error("%s: %s", __func__, strError), REJECT_INVALID, "bad-txns-issue-msgchannel" + strError);

    } else if (tx.IsNewQualifierAsset()) {
        /** Verify the qualifier channel assets data */
        std::string strError = "";
        if(!tx.VerifyNewQualfierAsset(strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

        CNewAsset asset;
        std::string strAddress;
        if (!QualifierAssetFromTransaction(tx, asset, strAddress))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-qualifier-from-transaction");

        if (!CheckNewAsset(asset, strError))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-qualfier" + strError);

    } else if (tx.IsNewRestrictedAsset()) {
        /** Verify the restricted assets data. */
        std::string strError = "";
        if(!tx.VerifyNewRestrictedAsset(strError))
            return state.DoS(100, false, REJECT_INVALID, strError);

        // Get asset data
        CNewAsset asset;
        std::string strAddress;
        if (!RestrictedAssetFromTransaction(tx, asset, strAddress))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-restricted-from-transaction");

        if (!CheckNewAsset(asset, strError))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-restricted" + strError);

        // Get verifier string
        CNullAssetTxVerifierString verifier;
        if (!tx.GetVerifierStringFromTx(verifier, strError))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-restricted-verifier-search-" + strError);

        // Mark that this transaction has a restricted asset issuance, for checks later with the verifier string tx
        fContainsNewRestrictedAsset = true;
    }
    else {
        // Fail if transaction contains any non-transfer asset scripts and hasn't conformed to one of the
        // above transaction types.  Also fail if it contains OP_DYN_ASSET opcode but wasn't a valid script.
        for (auto out : tx.vout) {
            int nType;
            bool _isOwner;
            if (out.scriptPubKey.IsAssetScript(nType, _isOwner)) {
                if (nType != TX_TRANSFER_ASSET) {
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-bad-asset-transaction");
                }
            } else {
                if (out.scriptPubKey.Find(OP_DYN_ASSET)) {
                    if (out.scriptPubKey[0] != OP_DYN_ASSET) {
                        return state.DoS(100, false, REJECT_INVALID,
                                         "bad-txns-op-dyn-asset-not-in-right-script-location");
                    }
                }
            }
        }
    }

    // Check to make sure that if there is a verifier string, that there is also a issue or reissuance of a restricted asset
    if (fContainsNullAssetVerifierTx && !fContainsRestrictedAssetReissue && !fContainsNewRestrictedAsset)
        return state.DoS(100, false, REJECT_INVALID, "bad-txns-tx-cointains-verifier-string-without-restricted-asset-issuance-or-reissuance");

    // If there is a restricted asset issuance, verify that there is a verifier tx associated with it.
    if (fContainsNewRestrictedAsset && !fContainsNullAssetVerifierTx) {
        return state.DoS(100, false, REJECT_INVALID, "bad-txns-tx-cointains-restricted-asset-issuance-without-verifier");
    }

    // we allow restricted asset reissuance without having a verifier string transaction, we don't force it to be update
    /** ASSET END */

    return true;
}

bool CheckBDAPTxCreditUsage(const CTransaction& tx, const std::vector<Coin>& vBdapCoins, 
                                const CAmount& nStandardIn, const CAmount& nCreditsIn, const CAmount& nStandardOut, const CAmount& nCreditsOut, const CAmount& nDataBurned)
{
    LogPrint("bdap", "%s -- nStandardIn %d, nCreditsIn %d, nStandardOut %d, nCreditsOut %d, nDataBurned %d\n", __func__,
                    FormatMoney(nStandardIn), FormatMoney(nCreditsIn), FormatMoney(nStandardOut), FormatMoney(nCreditsOut), FormatMoney(nDataBurned));
    // when there are no BDAP inputs, we do not need to check how credits are used.
    if (vBdapCoins.size() == 0 || nCreditsIn == 0)
        return true;

    if (nStandardIn > 0 && nStandardOut > 0 && nStandardOut >= nStandardIn) {
        LogPrintf("%s -- Check failed. Invalid use of BDAP credits. Standard DYN output amounts exceeds or equals standard DYN input amount\n", __func__);
        if (ENFORCE_BDAP_CREDIT_USE)
            return false;
    }

    if (nCreditsOut >= nCreditsIn) {
        LogPrintf("%s -- Check failed. Invalid use of BDAP credits. BDAP credits output amount exceeds BDAP credit input amount\n", __func__);
        if (ENFORCE_BDAP_CREDIT_USE)
            return false;
    }

    std::multimap<CDynamicAddress, CServiceCredit> mapInputs;
    std::vector<std::pair<CServiceCredit, CDynamicAddress>> vInputInfo;
    for (const Coin& coin : vBdapCoins) {
        int opCode1 = -1; int opCode2 = -1;
        std::vector<std::vector<unsigned char>> vvchOpParameters;
        coin.out.GetBDAPOpCodes(opCode1, opCode2, vvchOpParameters);
        CDynamicAddress address = GetScriptAddress(coin.out.scriptPubKey);
        std::string strOpType = GetBDAPOpTypeString(opCode1, opCode2);
        CServiceCredit credit(strOpType, coin.out.nValue, vvchOpParameters);
        vInputInfo.push_back(std::make_pair(credit, address));
        mapInputs.insert({address, credit});
        LogPrint("bdap", "%s -- BDAP Input strOpType %s, opCode1 %d, opCode2 %d, nValue %d, address %s\n", __func__, 
            strOpType, opCode1, opCode2, FormatMoney(coin.out.nValue), address.ToString());
    }

    std::multimap<CDynamicAddress, CServiceCredit> mapOutputs;
    for (const CTxOut& txout : tx.vout) {
        if (txout.IsBDAP()) {
            int opCode1 = -1; int opCode2 = -1;
            std::vector<std::vector<unsigned char>> vvchOpParameters;
            txout.GetBDAPOpCodes(opCode1, opCode2, vvchOpParameters);
            CDynamicAddress address = GetScriptAddress(txout.scriptPubKey);
            std::string strOpType = GetBDAPOpTypeString(opCode1, opCode2);
            CServiceCredit credit(strOpType, txout.nValue, vvchOpParameters);
            mapOutputs.insert({address, credit});
            LogPrint("bdap", "%s -- BDAP Output strOpType %s, opCode1 %d, opCode2 %d, nValue %d, address %s\n", __func__, 
                strOpType, opCode1, opCode2, FormatMoney(txout.nValue), address.ToString());
        } else if (txout.IsData()) {
            CDynamicAddress address;
            CServiceCredit credit("data", txout.nValue);
            mapOutputs.insert({address, credit});
            LogPrint("bdap", "%s -- BDAP Output strOpType %s, nValue %d\n", __func__, "data", FormatMoney(txout.nValue));
        } else {
            CDynamicAddress address = GetScriptAddress(txout.scriptPubKey);
            CServiceCredit credit("standard", txout.nValue);
            mapOutputs.insert({address, credit});
        }
    }

    for (const std::pair<CServiceCredit, CDynamicAddress>& credit : vInputInfo) {
        if (credit.first.OpType == "bdap_move_asset") {
            // When an input is a BDAP credit, make sure unconsumed coins go to a BDAP credit change ouput with the same credit input address and parameters
            if (credit.first.vParameters.size() == 2) {
                std::vector<unsigned char> vchMoveSource = credit.first.vParameters[0];
                std::vector<unsigned char> vchMoveDestination = credit.first.vParameters[1];
                if (vchMoveSource != vchFromString(std::string("DYN")) || vchMoveDestination != vchFromString(std::string("BDAP"))) {
                    LogPrintf("%s -- Check failed. Invalid use of BDAP credits. BDAP Credit has incorrect parameter. Move Source %s (should be DYN), Move Destination %s (should be BDAP)\n", __func__, 
                                            stringFromVch(vchMoveSource), stringFromVch(vchMoveDestination));
                    return false;
                }
            } else {
                LogPrintf("%s -- Check failed. Invalid use of BDAP credits. BDAP Credit has incorrect parameter count.\n", __func__);
                return false;
            }
            // make sure all of the credits are spent when we can't find an output address
            CDynamicAddress inputAddress = credit.second;
            std::multimap<CDynamicAddress, CServiceCredit>::iterator it = mapOutputs.find(inputAddress);
            if (it == mapOutputs.end()) {
                LogPrintf("%s -- Check failed. Invalid use of BDAP credits. Can't find credit address %s in outputs\n", __func__, inputAddress.ToString());
                if (ENFORCE_BDAP_CREDIT_USE)
                    return false;

            } else {
                // make sure asset doesn't move to another address, check outputs
                CAmount nInputAmount = 0;
                for (auto itr = mapInputs.find(inputAddress); itr != mapInputs.end(); itr++) {
                    nInputAmount += itr->second.nValue;
                }
                CAmount nOutputAmount = 0;
                for (auto itr = mapOutputs.find(inputAddress); itr != mapOutputs.end(); itr++) {
                    nOutputAmount += itr->second.nValue;
                }
                LogPrint("bdap", "%s -- inputAddress %s, nInputAmount %d, nOutputAmount %d, Diff %d\n", __func__, 
                                inputAddress.ToString(), FormatMoney(nInputAmount), FormatMoney(nOutputAmount), FormatMoney((nInputAmount - nOutputAmount)));

                if (!((nInputAmount - nOutputAmount) == (nCreditsIn - nCreditsOut))) {
                    LogPrintf("%s -- Check failed. Invalid use of BDAP credits. Fuel used %d should equal total fuel used %d\n", __func__, 
                                    FormatMoney((nInputAmount - nOutputAmount)), FormatMoney((nCreditsIn - nCreditsOut)));
                    if (ENFORCE_BDAP_CREDIT_USE)
                        return false;
                }
            }
        } else if (credit.first.OpType == "bdap_new_account" || credit.first.OpType == "bdap_update_account" || 
                        credit.first.OpType == "bdap_new_link_request" || credit.first.OpType == "bdap_new_link_accept" || credit.first.OpType == "bdap_new_audit") {
            // When input is a BDAP account new or update operation, make sure deposit change goes back to input wallet address
            // When input is a BDAP link operation, make sure it is only spent by a link update or delete operations with the same input address and parameters
            CDynamicAddress inputAddress = credit.second;
            std::multimap<CDynamicAddress, CServiceCredit>::iterator it = mapOutputs.find(inputAddress);
            if (it == mapOutputs.end()) {
                LogPrintf("%s -- Check failed. Invalid use of BDAP credits. Can't find account address %s in outputs\n", __func__, inputAddress.ToString());
                if (ENFORCE_BDAP_CREDIT_USE)
                    return false;

            } else {
                // make sure asset doesn't move to another address, check outputs
                CAmount nInputAmount = 0;
                for (auto itr = mapInputs.find(inputAddress); itr != mapInputs.end(); itr++) {
                    nInputAmount += itr->second.nValue;
                }
                CAmount nOutputAmount = 0;
                for (auto itr = mapOutputs.find(inputAddress); itr != mapOutputs.end(); itr++) {
                    nOutputAmount += itr->second.nValue;
                }
                LogPrint("bdap", "%s --inputAddress %s, nInputAmount %d, nOutputAmount %d, Diff %d\n", __func__, 
                                inputAddress.ToString(), FormatMoney(nInputAmount), FormatMoney(nOutputAmount), FormatMoney((nInputAmount - nOutputAmount)));

                if (!((nInputAmount - nOutputAmount) == (nCreditsIn - nCreditsOut))) {
                    LogPrintf("%s -- Check failed. Invalid use of BDAP credits. Fuel used %d should equal total fuel used %d\n", __func__, 
                                    FormatMoney((nInputAmount - nOutputAmount)), FormatMoney((nCreditsIn - nCreditsOut)));
                    if (ENFORCE_BDAP_CREDIT_USE)
                        return false;
                }
            }
        }
    }
    return true;
}

// Returns the script flags which should be checked for a given block
static unsigned int GetBlockScriptFlags(const CBlockIndex* pindex, const Consensus::Params& chainparams);

void LimitMempoolSize(CTxMemPool& pool, size_t limit, unsigned long age)
{
    int expired = pool.Expire(GetTime() - age);
    if (expired != 0)
        LogPrint("mempool", "Expired %i transactions from the memory pool\n", expired);

    std::vector<COutPoint> vNoSpendsRemaining;
    pool.TrimToSize(limit, &vNoSpendsRemaining);
    for (const COutPoint& removed : vNoSpendsRemaining)
        pcoinsTip->Uncache(removed);
}

/** Convert CValidationState to a human-readable message for logging */
std::string FormatStateMessage(const CValidationState& state)
{
    return strprintf("%s%s (code %i)",
        state.GetRejectReason(),
        state.GetDebugMessage().empty() ? "" : ", " + state.GetDebugMessage(),
        state.GetRejectCode());
}

static bool IsCurrentForFeeEstimation()
{
    AssertLockHeld(cs_main);
    if (IsInitialBlockDownload())
        return false;
    if (chainActive.Tip()->GetBlockTime() < (GetTime() - MAX_FEE_ESTIMATION_TIP_AGE))
        return false;
    if (chainActive.Height() < pindexBestHeader->nHeight - 1)
        return false;
    return true;
}

// Check if BDAP entry is valid
bool ValidateBDAPInputs(const CTransactionRef& tx, CValidationState& state, const CCoinsViewCache& inputs, const CBlock& block, bool fJustCheck, int nHeight, bool bSanity)
{
    if (!CheckDomainEntryDB())
        return true;

    std::string statusRpc = "";
    if (fJustCheck && (IsInitialBlockDownload() || RPCIsInWarmup(&statusRpc)))
        return true;

    std::vector<std::vector<unsigned char> > vvchBDAPArgs;
    int op1 = -1;
    int op2 = -1;
    if (nHeight == 0) {
        nHeight = chainActive.Height() + 1;
    }
    bool bValid = false;
    if (tx->nVersion == BDAP_TX_VERSION) {
        CScript scriptOp;
        if (GetBDAPOpScript(tx, scriptOp, vvchBDAPArgs, op1, op2)) {
            std::string errorMessage;
            if (vvchBDAPArgs.size() > 3) {
                errorMessage = "Too many BDAP parameters in operation transactions.";
                return state.DoS(100, false, REJECT_INVALID, errorMessage);
            }
            if (vvchBDAPArgs.size() < 1) {
                errorMessage = "Not enough BDAP parameters in operation transactions.";
                return state.DoS(100, false, REJECT_INVALID, errorMessage);
            }

            std::string strOpType = GetBDAPOpTypeString(op1, op2);
            if (strOpType == "bdap_new_account" || strOpType == "bdap_update_account" || strOpType == "bdap_delete_account") {
                bValid = CheckDomainEntryTx(tx, scriptOp, op1, op2, vvchBDAPArgs, fJustCheck, nHeight, block.nTime, bSanity, errorMessage);
                if (!bValid) {
                    errorMessage = "ValidateBDAPInputs: " + errorMessage;
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
                }
                if (!errorMessage.empty())
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
            }
            else if (strOpType == "bdap_new_link_request") {
                std::vector<unsigned char> vchPubKey = vvchBDAPArgs[0];
                LogPrint("bdap", "%s -- New Link Request vchPubKey = %s\n", __func__, stringFromVch(vchPubKey));
                bValid = CheckLinkTx(tx, op1, op2, vvchBDAPArgs, fJustCheck, nHeight, block.nTime, bSanity, errorMessage);
                if (!bValid) {
                    errorMessage = "ValidateBDAPInputs: CheckLinkTx failed: " + errorMessage;
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
                }
                uint256 txid;
                if (GetLinkIndex(vchPubKey, txid)) {
                    if (txid != tx->GetHash()) {
                        errorMessage = "Link request public key already used.";
                        LogPrintf("%s -- %s\n", __func__, errorMessage);
                        return state.DoS(100, false, REJECT_INVALID, errorMessage);
                    }
                }
                return true;
            }
            else if (strOpType == "bdap_new_link_accept") {
                std::vector<unsigned char> vchPubKey = vvchBDAPArgs[0];
                LogPrint("bdap", "%s -- New Link Accept vchPubKey = %s\n", __func__, stringFromVch(vchPubKey));
                bValid = CheckLinkTx(tx, op1, op2, vvchBDAPArgs, fJustCheck, nHeight, block.nTime, bSanity, errorMessage);
                if (!bValid) {
                    errorMessage = "ValidateBDAPInputs: CheckLinkTx failed: " + errorMessage;
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
                }
                uint256 txid;
                if (GetLinkIndex(vchPubKey, txid)) {
                    if (txid != tx->GetHash()) {
                        errorMessage = "Link accept public key already used.";
                        return state.DoS(100, false, REJECT_INVALID, errorMessage);
                    }
                }
                return true;
            }
            else if (strOpType == "bdap_move_asset") {
                if (!(vvchBDAPArgs.size() == 2)) {
                    errorMessage = "Incorrect number of parameters used for " + strOpType + " transaction.";
                    LogPrintf("%s -- delete link ignored. %s\n", __func__, errorMessage);
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
                }
                LogPrint("bdap", "%s -- BDAP move asset operation. vvchBDAPArgs.size() = %d\n", __func__, vvchBDAPArgs.size());
                return true;
            }
            else if (strOpType == "bdap_new_audit") {
                bValid = CheckAuditTx(tx, scriptOp, op1, op2, vvchBDAPArgs, fJustCheck, nHeight, block.nTime, bSanity, errorMessage);
                if (!bValid) {
                    errorMessage = "ValidateBDAPInputs: " + errorMessage;
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
                }
                if (!errorMessage.empty())
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
                LogPrintf("%s -- CheckAuditTx valid.\n", __func__);
                return true;
            }
            else if (strOpType == "bdap_delete_link_request" || strOpType == "bdap_delete_link_accept") {
                /*
                if (!CheckPreviousLinkInputs(strOpType, scriptOp, vvchBDAPArgs, errorMessage, fJustCheck)) {
                    errorMessage = "ValidateBDAPInputs: Delete link failed" + errorMessage;
                    LogPrintf("%s -- delete link failed. %s\n", __func__, errorMessage);
                    return state.DoS(100, false, REJECT_INVALID, errorMessage);
                }
                */
                // TODO (BDAP): Implement link delete
                errorMessage = "ValidateBDAPInputs: Failed because " + strOpType + " is not implemented yet." + errorMessage;
                LogPrintf("%s -- delete link ignored. %s\n", __func__, errorMessage);
            }
            else if (strOpType == "bdap_update_link_request" || strOpType == "bdap_update_link_accept") {
                // TODO (BDAP): Implement link update, allow for now.
                errorMessage = "ValidateBDAPInputs: Failed because " + strOpType + " is not implemented yet." + errorMessage;
                LogPrintf("%s -- update link ignored. %s\n", __func__, errorMessage);
            }
            else {
                // Do not allow unknown BDAP operations
                errorMessage = strprintf("%s -- Failed, unknown operation found. opcode1 = %d, opcode2 = %d", __func__, op1, op2);
                LogPrintf("%s\n", errorMessage);
                return state.DoS(100, false, REJECT_INVALID, errorMessage);
            }
        }
    }
    return true;
}

/* Make mempool consistent after a reorg, by re-adding or recursively erasing
 * disconnected block transactions from the mempool, and also removing any
 * other transactions from the mempool that are no longer valid given the new
 * tip/height.
 *
 * Note: we assume that disconnectpool only contains transactions that are NOT
 * confirmed in the current chain nor already in the mempool (otherwise,
 * in-mempool descendants of such transactions would be removed).
 *
 * Passing fAddToMempool=false will skip trying to add the transactions back,
 * and instead just erase from the mempool as needed.
 */

void UpdateMempoolForReorg(DisconnectedBlockTransactions &disconnectpool, bool fAddToMempool)
{
    AssertLockHeld(cs_main);
    std::vector<uint256> vHashUpdate;
    // disconnectpool's insertion_order index sorts the entries from
    // oldest to newest, but the oldest entry will be the last tx from the
    // latest mined block that was disconnected.
    // Iterate disconnectpool in reverse, so that we add transactions
    // back to the mempool starting with the earliest transaction that had
    // been previously seen in a block.
    auto it = disconnectpool.queuedTx.get<insertion_order>().rbegin();
    while (it != disconnectpool.queuedTx.get<insertion_order>().rend()) {
        // ignore validation errors in resurrected transactions
        CValidationState stateDummy;
        if (!fAddToMempool || (*it)->IsCoinBase() ||
            !AcceptToMemoryPool(mempool, stateDummy, *it, false /* fLimitFree */, nullptr /* pfMissingInputs */,
                                nullptr /* plTxnReplaced */, true /* fOverrideMempoolLimit */, 0 /* nAbsurdFee */)) {
            // If the transaction doesn't make it in to the mempool, remove any
            // transactions that depend on it (which would now be orphans).
            mempool.removeRecursive(**it, MemPoolRemovalReason::REORG);
        } else if (mempool.exists((*it)->GetHash())) {
            vHashUpdate.push_back((*it)->GetHash());
        }
        ++it;
    }
    disconnectpool.queuedTx.clear();
    // AcceptToMemoryPool/addUnchecked all assume that new mempool entries have
    // no in-mempool children, which is generally not true when adding
    // previously-confirmed transactions back to the mempool.
    // UpdateTransactionsFromBlock finds descendants of any transactions in
    // the disconnectpool that were added back and cleans up the mempool state.
    mempool.UpdateTransactionsFromBlock(vHashUpdate);

    // We also need to remove any now-immature transactions
    mempool.removeForReorg(pcoinsTip, chainActive.Tip()->nHeight + 1, STANDARD_LOCKTIME_VERIFY_FLAGS);
    // Re-limit mempool size, in case we added any transactions
    LimitMempoolSize(mempool, gArgs.GetArg("-maxmempool", DEFAULT_MAX_MEMPOOL_SIZE) * 1000000, gArgs.GetArg("-mempoolexpiry", DEFAULT_MEMPOOL_EXPIRY) * 60 * 60);
}

// Used to avoid mempool polluting consensus critical paths if CCoinsViewMempool
// were somehow broken and returning the wrong scriptPubKeys
static bool CheckInputsFromMempoolAndCache(const CTransaction& tx, CValidationState &state, const CCoinsViewCache &view, CTxMemPool& pool,
                 unsigned int flags, bool cacheSigStore, PrecomputedTransactionData& txdata) {
    AssertLockHeld(cs_main);

    // pool.cs should be locked already, but go ahead and re-take the lock here
    // to enforce that mempool doesn't change between when we check the view
    // and when we actually call through to CheckInputs
    LOCK(pool.cs);

    assert(!tx.IsCoinBase());
    for (const CTxIn& txin : tx.vin) {
        const Coin& coin = view.AccessCoin(txin.prevout);

        // At this point we haven't actually checked if the coins are all
        // available (or shouldn't assume we have, since CheckInputs does).
        // So we just return failure if the inputs are not available here,
        // and then only have to check equivalence for available inputs.
        if (coin.IsSpent()) return false;

        const CTransactionRef& txFrom = pool.get(txin.prevout.hash);
        if (txFrom) {
            assert(txFrom->GetHash() == txin.prevout.hash);
            assert(txFrom->vout.size() > txin.prevout.n);
            assert(txFrom->vout[txin.prevout.n] == coin.out);
        } else {
            const Coin& coinFromDisk = pcoinsTip->AccessCoin(txin.prevout);
            assert(!coinFromDisk.IsSpent());
            assert(coinFromDisk.out == coin.out);
        }
    }

    return CheckInputs(tx, state, view, true, flags, cacheSigStore, true, txdata);
}

bool AcceptToMemoryPoolWorker(CTxMemPool& pool, CValidationState& state, const CTransactionRef& ptx, bool fLimitFree, bool* pfMissingInputs, int64_t nAcceptTime, std::list<CTransactionRef>* plTxnReplaced, bool fOverrideMempoolLimit, const CAmount& nAbsurdFee, std::vector<COutPoint>& coins_to_uncache, bool fDryRun)
{
    const CTransaction& tx = *ptx;
    const uint256 hash = tx.GetHash();
    bool fluidTransaction = false;
    std::vector<std::pair<std::string, uint256>> vReissueAssets;
    AssertLockHeld(cs_main);
    if (pfMissingInputs)
        *pfMissingInputs = false;

    if (!CheckTransaction(tx, state))
        return false; // state filled in by CheckTransaction

    if (!fluid.ProvisionalCheckTransaction(tx))
        return false;

    for (const CTxOut& txout : tx.vout) {
        if (IsTransactionFluid(txout.scriptPubKey)) {
            fluidTransaction = true;
            std::string strErrorMessage;
            // Check if fluid transaction is already in the mempool
            if (fluid.CheckIfExistsInMemPool(pool, txout.scriptPubKey, strErrorMessage)) {
                // fluid transaction is already in the mempool.  Reject tx.
                return state.DoS(100, false, REJECT_INVALID, strErrorMessage);
            }
            std::string strFluidOpScript = ScriptToAsmStr(txout.scriptPubKey);
            std::string verificationWithoutOpCode = GetRidOfScriptStatement(strFluidOpScript);
            std::string strOperationCode = GetRidOfScriptStatement(strFluidOpScript, 0);
            if (strOperationCode == "OP_BDAP_REVOKE" && !sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
                return state.Invalid(false, REJECT_INVALID, "bdap-spork-inactive");

            if (!fluid.ExtractCheckTimestamp(strOperationCode, ScriptToAsmStr(txout.scriptPubKey), GetTime())) {
                return state.DoS(100, false, REJECT_INVALID, "fluid-tx-timestamp-error");
            }
            if (!fluid.CheckFluidOperationScript(txout.scriptPubKey, GetTime(), strErrorMessage, true)) {
                return state.DoS(100, false, REJECT_INVALID, strErrorMessage);
            }
        }
    }
    // Don't relay BDAP transaction until spork is activated
    if (tx.nVersion == BDAP_TX_VERSION && !sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
        return state.DoS(0, false, REJECT_NONSTANDARD, "inactive-spork-bdap-tx");

    bool fIsBDAP = false;
    //TODO: Create a seperate function to check BDAP tx validity.
    if (tx.nVersion == BDAP_TX_VERSION) {
        fIsBDAP = true;
        CScript scriptBDAPOp;
        std::vector<std::vector<unsigned char>> vvch;
        CScript scriptOp;
        int op1, op2;
        if (!GetBDAPOpScript(ptx, scriptBDAPOp, vvch, op1, op2))
            return state.Invalid(false, REJECT_INVALID, "bdap-txn-script-error");

        std::string strErrorMessage;
        vchCharString vvchOpParameters;
        if (!GetBDAPOpScript(ptx, scriptOp, vvchOpParameters, op1, op2)) {
            return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-get-op-failed" + strErrorMessage);
        }
        const std::string strOpType = GetBDAPOpTypeString(op1, op2);
        if (strOpType == "bdap_new_account" || strOpType == "bdap_update_account" || strOpType == "bdap_delete_account") {
            CDomainEntry domainEntry(ptx);
            if (domainEntry.CheckIfExistsInMemPool(pool, strErrorMessage)) {
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-account-txn-already-in-mempool " + strErrorMessage);
            }
            if (strOpType == "bdap_new_account") {
                CDomainEntry findDomainEntry;
                if (GetDomainEntry(domainEntry.vchFullObjectPath(), findDomainEntry))
                {
                    strErrorMessage = "AcceptToMemoryPoolWorker -- The entry " + findDomainEntry.GetFullObjectPath() + " already exists.  Rejected by the tx memory pool!";
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-exists " + strErrorMessage);
                }
            } else if (strOpType == "bdap_update_account") {
                CDomainEntry entry;
                CDomainEntry prevEntry;
                std::vector<unsigned char> vchData;
                std::vector<unsigned char> vchHash;
                int nDataOut;
                bool bData = GetBDAPData(ptx, vchData, vchHash, nDataOut);
                if (bData && !entry.UnserializeFromData(vchData, vchHash)) {
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-get-data-failed" + strErrorMessage);
                }

                if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), prevEntry)) {
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-get-previous-failed" + strErrorMessage);
                }
                CTransactionRef pPrevTx;
                uint256 hashBlock;
                if (!GetTransaction(prevEntry.txHash, pPrevTx, Params().GetConsensus(), hashBlock, true)) {
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-get-previous-tx-failed" + strErrorMessage);
                }
                // Get current wallet address used for BDAP tx
                CScript scriptPubKey = scriptBDAPOp;
                CDynamicAddress txAddress = GetScriptAddress(scriptPubKey);
                // Get previous wallet address used for BDAP tx
                CScript prevScriptPubKey;
                GetBDAPOpScript(pPrevTx, prevScriptPubKey);
                CDynamicAddress prevAddress = GetScriptAddress(prevScriptPubKey);
                if (txAddress.ToString() != prevAddress.ToString()) {
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-incorrect-wallet-address-used" + strErrorMessage);
                }
            } else if (strOpType == "bdap_delete_account") {
                if (!(vvchOpParameters.size() > 0))
                    return state.Invalid(false, REJECT_INVALID, "bdap-delete-account-get-object-path" + strErrorMessage);

                std::vector<unsigned char> vchFullObjectPath = vvchOpParameters[0];
                CDomainEntry prevEntry;
                if (!pDomainEntryDB->GetDomainEntryInfo(vchFullObjectPath, prevEntry)) {
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-get-previous-failed" + strErrorMessage);
                }
                CTransactionRef pPrevTx;
                uint256 hashBlock;
                if (!GetTransaction(prevEntry.txHash, pPrevTx, Params().GetConsensus(), hashBlock, true)) {
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-get-previous-tx-failed" + strErrorMessage);
                }
                // Get current wallet address used for BDAP tx
                CScript scriptPubKey = scriptBDAPOp;
                CDynamicAddress txAddress = GetScriptAddress(scriptPubKey);
                // Get previous wallet address used for BDAP tx
                CScript prevScriptPubKey;
                GetBDAPOpScript(pPrevTx, prevScriptPubKey);
                CDynamicAddress prevAddress = GetScriptAddress(prevScriptPubKey);
                if (txAddress.ToString() != prevAddress.ToString()) {
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-txn-incorrect-wallet-address-used" + strErrorMessage);
                }
            }
        } else if (strOpType == "bdap_new_link_request" || strOpType == "bdap_new_link_accept") {
            if (vvch.size() < 1)
                return state.Invalid(false, REJECT_INVALID, "bdap-txn-pubkey-parameter-not-found");
            if (vvch.size() > 3)
                return state.Invalid(false, REJECT_INVALID, "bdap-txn-too-many-parameters");
            //check for duplicate pubkeys
            std::vector<unsigned char> vchPubKey = vvch[0];
            if (LinkPubKeyExistsInMemPool(pool, vchPubKey, strOpType, strErrorMessage))
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-link-pubkey-txn-already-in-mempool");

            if (LinkPubKeyExists(vchPubKey))
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-link-duplicate-pubkey");

            CDomainEntry prevEntry;
            if (GetDomainEntryPubKey(vchPubKey, prevEntry))
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-link-duplicate-pubkey-entry");

            if (vvch.size() > 1) {
                std::vector<unsigned char> vchSharedPubKey = vvch[1];
                if (LinkPubKeyExists(vchSharedPubKey))
                    return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-link-duplicate-shared-pubkey");

                if (GetDomainEntryPubKey(vchSharedPubKey, prevEntry))
                    return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-link-duplicate-shared-pubkey-entry");
            }
        } else if (strOpType == "bdap_move_asset") {
            if (vvch.size() != 2)
                return state.Invalid(false, REJECT_INVALID, "bdap-move-invalid-parameter-size");
            std::vector<unsigned char> vchMoveSource = vchFromString(std::string("DYN"));
            std::vector<unsigned char> vchMoveDestination = vchFromString(std::string("BDAP"));
            if (vvch[0] != vchMoveSource)
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-move-unknown-source");
            if (vvch[1] != vchMoveDestination)
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-move-unknown-destination");

        } else if (strOpType == "bdap_new_audit") {
            if (!sporkManager.IsSporkActive(SPORK_32_BDAP_V2))
                return state.DoS(0, false, REJECT_NONSTANDARD, "inactive-spork-bdap-v2-tx");

            if (vvch.size() < 1)
                return state.Invalid(false, REJECT_INVALID, "bdap-new-audit-not-enough-parameters");

            if (vvch.size() > 3)
                return state.Invalid(false, REJECT_INVALID, "bdap-new-audit-too-many-parameters");

            if (vvch[0].size() > 10)
                return state.Invalid(false, REJECT_INVALID, "bdap-new-audit-parameter-too-long");

            if (vvch.size() > 1) {
                if (vvch.size() == 2)
                   return state.Invalid(false, REJECT_INVALID, "bdap-new-audit-pubkey-missing");

                if (vvch[1].size() > MAX_OBJECT_FULL_PATH_LENGTH)
                    return state.Invalid(false, REJECT_INVALID, "bdap-new-audit-fqdn-too-long");

                if (vvch[2].size() > 65)
                    return state.Invalid(false, REJECT_INVALID, "bdap-new-audit-pubkey-too-long");

                // check pubkey belongs to bdap account and signature is correct.
                CAudit audit(ptx);
                CDomainEntry findDomainEntry;
                if (!GetDomainEntry(audit.vchOwnerFullObjectPath, findDomainEntry)) {
                    strErrorMessage = "AcceptToMemoryPoolWorker -- The entry " + stringFromVch(audit.vchOwnerFullObjectPath) + " not found.  Rejected by the tx memory pool!";
                    return state.Invalid(false, REJECT_INVALID, "bdap-account-exists " + strErrorMessage);
                }
                CPubKey pubkey(vvch[2]);
                CDynamicAddress address(pubkey.GetID());
                if (findDomainEntry.GetWalletAddress().ToString() != address.ToString()) {
                        strErrorMessage = "AcceptToMemoryPoolWorker -- Public key does not match BDAP account wallet address.  Rejected by the tx memory pool!";
                        return state.Invalid(false, REJECT_INVALID, "bdap-audit-wallet-address-mismatch " + strErrorMessage);
                    }
                if (!audit.CheckSignature(pubkey.Raw())) {
                    strErrorMessage = "AcceptToMemoryPoolWorker -- Invalid signature.  Rejected by the tx memory pool!";
                    return state.Invalid(false, REJECT_INVALID, "bdap-audit-check-signature-failed " + strErrorMessage);
                }
            }
            CAudit audit;
            if (GetAuditTxId(tx.GetHash().GetHex(), audit))
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-audit-already-exists");
        }
        // TODO (BDAP): Implement link delete
        /*
        else if (strOpType == "bdap_delete_link_request" || strOpType == "bdap_delete_link_accept") {
            if (vvch.size() < 1)
                return state.Invalid(false, REJECT_INVALID, "bdap-txn-pubkey-parameter-not-found");
            if (vvch.size() > 2)
                return state.Invalid(false, REJECT_INVALID, "bdap-txn-too-many-parameters");

            std::vector<unsigned char> vchPubKey = vvch[0];
            if (LinkPubKeyExistsInMemPool(pool, vchPubKey, strOpType, strErrorMessage))
                return state.Invalid(false, REJECT_ALREADY_KNOWN, "bdap-link-pubkey-txn-already-in-mempool");
        }
        */
        else {
            // Do not allow unknown BDAP operations
            LogPrintf("%s -- Failed, unknown operation found. opcode1 = %d, opcode2 = %d\n", __func__, op1, op2);
            return state.DoS(100, false, REJECT_INVALID, "bdap-unknown-operation");
        }
    }
    // Coinbase is only valid in a block, not as a loose transaction
    if (tx.IsCoinBase())
        return state.DoS(100, false, REJECT_INVALID, "coinbase");

    //Coinstake is also only valid in a block, not as a loose transaction
    if (tx.IsCoinStake())
        return state.DoS(100, error("AcceptToMemoryPool: coinstake as individual tx. txid=%s", tx.GetHash().GetHex()),
            REJECT_INVALID, "coinstake");

    // Don't relay version 2 transactions until CSV is active, and we can be
    // sure that such transactions will be mined (unless we're on
    // -testnet/-regtest).
    const CChainParams& chainparams = Params();
    if (fRequireStandard && tx.nVersion  > CTransaction::MAX_STANDARD_VERSION && tx.nVersion != BDAP_TX_VERSION && VersionBitsTipState(chainparams.GetConsensus(), Consensus::DEPLOYMENT_CSV) != THRESHOLD_ACTIVE) {
        return state.DoS(0, false, REJECT_NONSTANDARD, "premature-version-tx");
    }

    // Rather not work on nonstandard transactions (unless -testnet/-regtest)
    std::string reason;
    if (fRequireStandard && !fIsBDAP && !IsStandardTx(tx, reason) && !fluidTransaction)
        return state.DoS(0, false, REJECT_NONSTANDARD, reason);

    // Only accept nLockTime-using transactions that can be mined in the next
    // block; we don't want our mempool filled up with transactions that can't
    // be mined yet.
    if (!CheckFinalTx(tx, STANDARD_LOCKTIME_VERIFY_FLAGS))
        return state.DoS(0, false, REJECT_NONSTANDARD, "non-final");

    // is it already in the memory pool?
    if (pool.exists(hash))
        return state.Invalid(false, REJECT_ALREADY_KNOWN, "txn-already-in-mempool");

    // If this is a Transaction Lock Request check to see if it's valid
    if (instantsend.HasTxLockRequest(hash) && !CTxLockRequest(tx).IsValid())
        return state.DoS(10, error("AcceptToMemoryPool : CTxLockRequest %s is invalid", hash.ToString()),
            REJECT_INVALID, "bad-txlockrequest");

    // Check for conflicts with a completed Transaction Lock
    for (const CTxIn& txin : tx.vin) {
        uint256 hashLocked;
        if (instantsend.GetLockedOutPointTxHash(txin.prevout, hashLocked) && hash != hashLocked)
            return state.DoS(10, error("AcceptToMemoryPool : Transaction %s conflicts with completed Transaction Lock %s", hash.ToString(), hashLocked.ToString()),
                REJECT_INVALID, "tx-txlock-conflict");
    }

    // Check for conflicts with in-memory transactions
    std::set<uint256> setConflicts;
    {
        LOCK(pool.cs); // protect pool.mapNextTx
        for (const CTxIn& txin : tx.vin) {
            auto itConflicting = pool.mapNextTx.find(txin.prevout);
            if (itConflicting != pool.mapNextTx.end()) {
                const CTransaction* ptxConflicting = itConflicting->second;
                if (!setConflicts.count(ptxConflicting->GetHash())) {
                    // InstantSend txes are not replacable
                    if (instantsend.HasTxLockRequest(ptxConflicting->GetHash())) {
                        // this tx conflicts with a Transaction Lock Request candidate
                        return state.DoS(0, error("AcceptToMemoryPool : Transaction %s conflicts with Transaction Lock Request %s", hash.ToString(), ptxConflicting->GetHash().ToString()),
                            REJECT_INVALID, "tx-txlockreq-mempool-conflict");
                    } else if (instantsend.HasTxLockRequest(hash)) {
                        // this tx is a tx lock request and it conflicts with a normal tx
                        return state.DoS(0, error("AcceptToMemoryPool : Transaction Lock Request %s conflicts with transaction %s", hash.ToString(), ptxConflicting->GetHash().ToString()),
                            REJECT_INVALID, "txlockreq-tx-mempool-conflict");
                    }
                    // Allow opt-out of transaction replacement by setting
                    // nSequence >= maxint-1 on all inputs.
                    //
                    // maxint-1 is picked to still allow use of nLockTime by
                    // non-replacable transactions. All inputs rather than just one
                    // is for the sake of multi-party protocols, where we don't
                    // want a single party to be able to disable replacement.
                    //
                    // The opt-out ignores descendants as anyone relying on
                    // first-seen mempool behavior should be checking all
                    // unconfirmed ancestors anyway; doing otherwise is hopelessly
                    // insecure.
                    bool fReplacementOptOut = true;
                    if (fEnableReplacement) {
                        for (const CTxIn& _txin : ptxConflicting->vin) {
                            if (_txin.nSequence < std::numeric_limits<unsigned int>::max() - 1) {
                                fReplacementOptOut = false;
                                break;
                            }
                        }
                    }
                    if (fReplacementOptOut)
                        return state.Invalid(false, REJECT_CONFLICT, "txn-mempool-conflict");

                    setConflicts.insert(ptxConflicting->GetHash());
                }
            }
        }
    }

    {
        CCoinsView dummy;
        CCoinsViewCache view(&dummy);

        CAmount nValueIn = 0;
        LockPoints lp;
        {
            LOCK(pool.cs);
            CCoinsViewMemPool viewMemPool(pcoinsTip, pool);
            view.SetBackend(viewMemPool);

            // do we already have it?
            for (size_t out = 0; out < tx.vout.size(); out++) {
                COutPoint outpoint(hash, out);
                bool had_coin_in_cache = pcoinsTip->HaveCoinInCache(outpoint);
                if (view.HaveCoin(outpoint)) {
                    if (!had_coin_in_cache) {
                        coins_to_uncache.push_back(outpoint);
                    }
                    return state.Invalid(false, REJECT_ALREADY_KNOWN, "txn-already-known");
                }
            }

            // do all inputs exist?
            for (const CTxIn txin : tx.vin) {
                if (!pcoinsTip->HaveCoinInCache(txin.prevout)) {
                    coins_to_uncache.push_back(txin.prevout);
                }
                if (!view.HaveCoin(txin.prevout)) {
                    if (pfMissingInputs) {
                        *pfMissingInputs = true;
                    }
                    return false; // fMissingInputs and !state.IsInvalid() is used to detect this condition, don't set state.Invalid()
                }
            }

            // Bring the best block into scope
            view.GetBestBlock();

            nValueIn = view.GetValueIn(tx);

            // we have all inputs cached now, so switch back to dummy, so we don't need to keep lock on mempool
            view.SetBackend(dummy);

            // Only accept BIP68 sequence locked transactions that can be mined in the next
            // block; we don't want our mempool filled up with transactions that can't
            // be mined yet.
            // Must keep pool.cs for this unless we change CheckSequenceLocks to take a
            // CoinsViewCache instead of create its own
            if (!CheckSequenceLocks(tx, STANDARD_LOCKTIME_VERIFY_FLAGS, &lp))
                return state.DoS(0, false, REJECT_NONSTANDARD, "non-BIP68-final");
        }

        /** ASSET START */
        if (!AreAssetsDeployed()) {
            for (auto out : tx.vout) {
                if (out.scriptPubKey.IsAssetScript())
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-contained-asset-when-not-active");
            }
        }

        if (AreAssetsDeployed()) {
            if (!Consensus::CheckTxAssets(tx, state, view, GetCurrentAssetCache(), true, vReissueAssets))
                return error("%s: Consensus::CheckTxAssets: %s, %s", __func__, tx.GetHash().ToString(),
                             FormatStateMessage(state));
        }
        /** ASSET END */

        // Check for non-standard pay-to-script-hash in inputs
        if (fRequireStandard && !AreInputsStandard(tx, view))
            return state.Invalid(false, REJECT_NONSTANDARD, "bad-txns-nonstandard-inputs");

        unsigned int nSigOps = GetLegacySigOpCount(tx);
        nSigOps += GetP2SHSigOpCount(tx, view);

        CAmount nValueOut = tx.GetValueOut();
        CAmount nFees = nValueIn - nValueOut;
        CAmount nBDAPBurn = 0;
        if (tx.nVersion == BDAP_TX_VERSION) {
            // Since fees are burned, count BDAP burn funds into fee calculation
            CAmount nOpCodeAmount;
            ExtractAmountsFromTx(MakeTransactionRef(tx), nBDAPBurn, nOpCodeAmount);
            if (nBDAPBurn > 0)
                nFees += nBDAPBurn;

            LogPrint("bdap", "%s -- BDAP Burn Data Amount %d, BDAP Op Code Amount %d\n", __func__, FormatMoney(nBDAPBurn), FormatMoney(nOpCodeAmount));
        }
        // nModifiedFees includes any fee deltas from PrioritiseTransaction
        CAmount nModifiedFees = nFees;
        double nPriorityDummy = 0;
        pool.ApplyDeltas(hash, nPriorityDummy, nModifiedFees);

        CAmount inChainInputValue;
        double dPriority = view.GetPriority(tx, chainActive.Height(), inChainInputValue);

        // Keep track of transactions that spend a coinbase, which we re-scan
        // during reorgs to ensure COINBASE_MATURITY is still met.
        bool fSpendsCoinbase = false;
        for (const CTxIn& txin : tx.vin) {
            const Coin& coin = view.AccessCoin(txin.prevout);
            if (coin.IsCoinBase()) {
                fSpendsCoinbase = true;
                break;
            }
        }

        CTxMemPoolEntry entry(ptx, nFees, nAcceptTime, dPriority, chainActive.Height(),
            inChainInputValue, fSpendsCoinbase, nSigOps, lp);
        unsigned int nSize = entry.GetTxSize();

        // Check that the transaction doesn't have an excessive number of
        // sigops, making it impossible to mine. Since the coinbase transaction
        // itself can contain sigops MAX_STANDARD_TX_SIGOPS is less than
        // MAX_BLOCK_SIGOPS_COST; we still consider this an invalid rather than
        // merely non-standard transaction.
        if ((nSigOps > MAX_STANDARD_TX_SIGOPS) || (nBytesPerSigOp && nSigOps > nSize / nBytesPerSigOp))
            return state.DoS(0, false, REJECT_NONSTANDARD, "bad-txns-too-many-sigops", false,
                strprintf("%d", nSigOps));

        CAmount mempoolRejectFee = pool.GetMinFee(gArgs.GetArg("-maxmempool", DEFAULT_MAX_MEMPOOL_SIZE) * 1000000).GetFee(nSize);
        if (mempoolRejectFee > 0 && nModifiedFees < mempoolRejectFee) {
            return state.DoS(0, false, REJECT_INSUFFICIENTFEE, "mempool min fee not met", false, strprintf("%d < %d", nFees, mempoolRejectFee));
        } else if (gArgs.GetBoolArg("-relaypriority", DEFAULT_RELAYPRIORITY) && nModifiedFees < ::minRelayTxFee.GetFee(nSize) && !AllowFree(entry.GetPriority(chainActive.Height() + 1))) {
            // Require that free transactions have sufficient priority to be mined in the next block.
            return state.DoS(0, false, REJECT_INSUFFICIENTFEE, "insufficient priority");
        }

        // Continuously rate-limit free (really, very-low-fee) transactions
        // This mitigates 'penny-flooding' -- sending thousands of free transactions just to
        // be annoying or make others' transactions take longer to confirm.
        if (fLimitFree && nModifiedFees < ::minRelayTxFee.GetFee(nSize)) {
            static CCriticalSection csFreeLimiter;
            static double dFreeCount;
            static int64_t nLastTime;
            int64_t nNow = GetTime();

            LOCK(csFreeLimiter);

            // Use an exponentially decaying ~10-minute window:
            dFreeCount *= pow(1.0 - 1.0 / 600.0, (double)(nNow - nLastTime));
            nLastTime = nNow;
            // -limitfreerelay unit is thousand-bytes-per-minute
            // At default rate it would take over a month to fill 1GB
            if (dFreeCount + nSize >= gArgs.GetArg("-limitfreerelay", DEFAULT_LIMITFREERELAY) * 10 * 1000)
                return state.DoS(0, false, REJECT_INSUFFICIENTFEE, "rate limited free transaction");
            LogPrint("mempool", "Rate limit dFreeCount: %g => %g\n", dFreeCount, dFreeCount + nSize);
            dFreeCount += nSize;
        }

        if (nAbsurdFee && nFees - nBDAPBurn > nAbsurdFee)
            return state.Invalid(false,
                REJECT_HIGHFEE, "absurdly-high-fee",
                strprintf("%d > %d", nFees, nAbsurdFee));

        // Calculate in-mempool ancestors, up to a limit.
        CTxMemPool::setEntries setAncestors;
        size_t nLimitAncestors = gArgs.GetArg("-limitancestorcount", DEFAULT_ANCESTOR_LIMIT);
        size_t nLimitAncestorSize = gArgs.GetArg("-limitancestorsize", DEFAULT_ANCESTOR_SIZE_LIMIT) * 1000;
        size_t nLimitDescendants = gArgs.GetArg("-limitdescendantcount", DEFAULT_DESCENDANT_LIMIT);
        size_t nLimitDescendantSize = gArgs.GetArg("-limitdescendantsize", DEFAULT_DESCENDANT_SIZE_LIMIT) * 1000;
        std::string errString;
        if (!pool.CalculateMemPoolAncestors(entry, setAncestors, nLimitAncestors, nLimitAncestorSize, nLimitDescendants, nLimitDescendantSize, errString)) {
            return state.DoS(0, false, REJECT_NONSTANDARD, "too-long-mempool-chain", false, errString);
        }

        // A transaction that spends outputs that would be replaced by it is invalid. Now
        // that we have the set of all ancestors we can detect this
        // pathological case by making sure setConflicts and setAncestors don't
        // intersect.
        for (CTxMemPool::txiter ancestorIt : setAncestors) {
            const uint256& hashAncestor = ancestorIt->GetTx().GetHash();
            if (setConflicts.count(hashAncestor)) {
                return state.DoS(10, false,
                    REJECT_INVALID, "bad-txns-spends-conflicting-tx", false,
                    strprintf("%s spends conflicting transaction %s",
                        hash.ToString(),
                        hashAncestor.ToString()));
            }
        }

        // Check if it's economically rational to mine this transaction rather
        // than the ones it replaces.
        CAmount nConflictingFees = 0;
        size_t nConflictingSize = 0;
        uint64_t nConflictingCount = 0;
        CTxMemPool::setEntries allConflicting;

        // If we don't hold the lock allConflicting might be incomplete; the
        // subsequent RemoveStaged() and addUnchecked() calls don't guarantee
        // mempool consistency for us.
        LOCK(pool.cs);
        const bool fReplacementTransaction = setConflicts.size();
        if (fReplacementTransaction) {
            CFeeRate newFeeRate(nModifiedFees, nSize);
            std::set<uint256> setConflictsParents;
            const int maxDescendantsToVisit = 100;
            CTxMemPool::setEntries setIterConflicting;
            for (const uint256& hashConflicting : setConflicts) {
                CTxMemPool::txiter mi = pool.mapTx.find(hashConflicting);
                if (mi == pool.mapTx.end())
                    continue;

                // Save these to avoid repeated lookups
                setIterConflicting.insert(mi);

                // Don't allow the replacement to reduce the feerate of the
                // mempool.
                //
                // We usually don't want to accept replacements with lower
                // feerates than what they replaced as that would lower the
                // feerate of the next block. Requiring that the feerate always
                // be increased is also an easy-to-reason about way to prevent
                // DoS attacks via replacements.
                //
                // The mining code doesn't (currently) take children into
                // account (CPFP) so we only consider the feerates of
                // transactions being directly replaced, not their indirect
                // descendants. While that does mean high feerate children are
                // ignored when deciding whether or not to replace, we do
                // require the replacement to pay more overall fees too,
                // mitigating most cases.
                CFeeRate oldFeeRate(mi->GetModifiedFee(), mi->GetTxSize());
                if (newFeeRate <= oldFeeRate) {
                    return state.DoS(0, false,
                        REJECT_INSUFFICIENTFEE, "insufficient fee", false,
                        strprintf("rejecting replacement %s; new feerate %s <= old feerate %s",
                            hash.ToString(),
                            newFeeRate.ToString(),
                            oldFeeRate.ToString()));
                }

                for (const CTxIn& txin : mi->GetTx().vin) {
                    setConflictsParents.insert(txin.prevout.hash);
                }

                nConflictingCount += mi->GetCountWithDescendants();
            }
            // This potentially overestimates the number of actual descendants
            // but we just want to be conservative to avoid doing too much
            // work.
            if (nConflictingCount <= maxDescendantsToVisit) {
                // If not too many to replace, then calculate the set of
                // transactions that would have to be evicted
                for (CTxMemPool::txiter it : setIterConflicting) {
                    pool.CalculateDescendants(it, allConflicting);
                }
                for (CTxMemPool::txiter it : allConflicting) {
                    nConflictingFees += it->GetModifiedFee();
                    nConflictingSize += it->GetTxSize();
                }
            } else {
                return state.DoS(0, false,
                    REJECT_NONSTANDARD, "too many potential replacements", false,
                    strprintf("rejecting replacement %s; too many potential replacements (%d > %d)\n",
                        hash.ToString(),
                        nConflictingCount,
                        maxDescendantsToVisit));
            }

            for (unsigned int j = 0; j < tx.vin.size(); j++) {
                // We don't want to accept replacements that require low
                // feerate junk to be mined first. Ideally we'd keep track of
                // the ancestor feerates and make the decision based on that,
                // but for now requiring all new inputs to be confirmed works.
                if (!setConflictsParents.count(tx.vin[j].prevout.hash)) {
                    // Rather than check the UTXO set - potentially expensive -
                    // it's cheaper to just check if the new input refers to a
                    // tx that's in the mempool.
                    if (pool.mapTx.find(tx.vin[j].prevout.hash) != pool.mapTx.end())
                        return state.DoS(0, false,
                            REJECT_NONSTANDARD, "replacement-adds-unconfirmed", false,
                            strprintf("replacement %s adds unconfirmed input, idx %d",
                                hash.ToString(), j));
                }
            }

            // The replacement must pay greater fees than the transactions it
            // replaces - if we did the bandwidth used by those conflicting
            // transactions would not be paid for.
            if (nModifiedFees < nConflictingFees) {
                return state.DoS(0, false,
                    REJECT_INSUFFICIENTFEE, "insufficient fee", false,
                    strprintf("rejecting replacement %s, less fees than conflicting txs; %s < %s",
                        hash.ToString(), FormatMoney(nModifiedFees), FormatMoney(nConflictingFees)));
            }

            // Finally in addition to paying more fees than the conflicts the
            // new transaction must pay for its own bandwidth.
            CAmount nDeltaFees = nModifiedFees - nConflictingFees;
            if (nDeltaFees < ::incrementalRelayFee.GetFee(nSize)) {
                return state.DoS(0, false,
                    REJECT_INSUFFICIENTFEE, "insufficient fee", false,
                    strprintf("rejecting replacement %s, not enough additional fees to relay; %s < %s",
                        hash.ToString(),
                        FormatMoney(nDeltaFees),
                        FormatMoney(::incrementalRelayFee.GetFee(nSize))));
            }
        }

        // If we aren't going to actually accept it but just were verifying it, we are fine already
        if (fDryRun)
            return true;

        unsigned int scriptVerifyFlags = STANDARD_SCRIPT_VERIFY_FLAGS;
        if (!chainparams.RequireStandard()) {
            scriptVerifyFlags = gArgs.GetArg("-promiscuousmempoolflags", scriptVerifyFlags);
        }

        // Check against previous transactions
        // This is done last to help prevent CPU exhaustion denial-of-service attacks.
        PrecomputedTransactionData txdata(tx);
        if (!CheckInputs(tx, state, view, true, scriptVerifyFlags, true, false, txdata)) {
            CValidationState stateDummy; // Want reported failures to be from first CheckInputs
            if (CheckInputs(tx, stateDummy, view, true, scriptVerifyFlags, true, false, txdata) &&
                !CheckInputs(tx, stateDummy, view, true, scriptVerifyFlags, true, false, txdata)) {
                state.SetCorruptionPossible();
            }
            return false; // state filled in by CheckInputs
        }

        // Check again against the current block tip's script verification
        // flags to cache our script execution flags. This is, of course,
        // useless if the next block has different script flags from the
        // previous one, but because the cache tracks script flags for us it
        // will auto-invalidate and we'll just have a few blocks of extra
        // misses on soft-fork activation.
        //
        // This is also useful in case of bugs in the standard flags that cause
        // transactions to pass as valid when they're actually invalid. For
        // instance the STRICTENC flag was incorrectly allowing certain
        // CHECKSIG NOT scripts to pass, even though they were invalid.
        //
        // There is a similar check in CreateNewBlock() to prevent creating
        // invalid blocks (using TestBlockValidity), however allowing such
        // transactions into the mempool can be exploited as a DoS attack.
        unsigned int currentBlockScriptVerifyFlags = GetBlockScriptFlags(chainActive.Tip(), Params().GetConsensus());
        if (!CheckInputsFromMempoolAndCache(tx, state, view, pool, currentBlockScriptVerifyFlags, true, txdata))
        {
            // If we're using promiscuousmempoolflags, we may hit this normally
            // Check if current block has some flags that scriptVerifyFlags
            // does not before printing an ominous warning
            if (!(~scriptVerifyFlags & currentBlockScriptVerifyFlags)) {
                return error("%s: BUG! PLEASE REPORT THIS! ConnectInputs failed against latest-block but not STANDARD flags %s, %s",
                    __func__, hash.ToString(), FormatStateMessage(state));
            } else {
                if (!CheckInputs(tx, state, view, true, MANDATORY_SCRIPT_VERIFY_FLAGS, true, false, txdata)) {
                    return error("%s: ConnectInputs failed against MANDATORY but not STANDARD flags due to promiscuous mempool %s, %s",
                        __func__, hash.ToString(), FormatStateMessage(state));
                } else {
                    LogPrintf("Warning: -promiscuousmempool flags set to not include currently enforced soft forks, this may break mining or otherwise cause instability!\n");
                }
            }
        }

        if (tx.nVersion == BDAP_TX_VERSION && !ValidateBDAPInputs(ptx, state, view, CBlock(), true, chainActive.Height())) {
            return false;
        }

        // Remove conflicting transactions from the mempool
        for (const CTxMemPool::txiter it : allConflicting) {
            LogPrint("mempool", "replacing tx %s with %s for %s %s additional fees, %d delta bytes\n",
                it->GetTx().GetHash().ToString(),
                hash.ToString(),
                FormatMoney(nModifiedFees - nConflictingFees),
                CURRENCY_UNIT,
                (int)nSize - (int)nConflictingSize);
            if (plTxnReplaced)
                plTxnReplaced->push_back(it->GetSharedTx());
        }
        pool.RemoveStaged(allConflicting, false);

        // This transaction should only count for fee estimation if it isn't a
        // BIP 125 replacement transaction (may not be widely supported), the
        // node is not behind, and the transaction is not dependent on any other
        // transactions in the mempool.
        bool validForFeeEstimation = !fReplacementTransaction && !fOverrideMempoolLimit && IsCurrentForFeeEstimation() && pool.HasNoInputsOf(tx);

        // Store transaction in memory
        pool.addUnchecked(hash, entry, setAncestors, validForFeeEstimation);

        // Add memory address index
        if (fAddressIndex) {
            pool.addAddressIndex(entry, view);
        }

        // Add memory spent index
        if (fSpentIndex) {
            pool.addSpentIndex(entry, view);
        }

        // trim mempool and check if tx was trimmed
        if (!fOverrideMempoolLimit) {
            LimitMempoolSize(pool, gArgs.GetArg("-maxmempool", DEFAULT_MAX_MEMPOOL_SIZE) * 1000000, gArgs.GetArg("-mempoolexpiry", DEFAULT_MEMPOOL_EXPIRY) * 60 * 60);
            if (!pool.exists(hash))
                return state.DoS(0, false, REJECT_INSUFFICIENTFEE, "mempool full");
        }

        for (auto out : vReissueAssets) {
            mapReissuedAssets.insert(out);
            mapReissuedTx.insert(std::make_pair(out.second, out.first));
        }

        if (AreAssetsDeployed()) {
            for (auto out : tx.vout) {
                if (out.scriptPubKey.IsAssetScript()) {
                    CAssetOutputEntry data;
                    if (!GetAssetData(out.scriptPubKey, data))
                        continue;
                    if (data.type == TX_NEW_ASSET && !IsAssetNameAnOwner(data.assetName)) {
                        pool.mapAssetToHash[data.assetName] = hash;
                        pool.mapHashToAsset[hash] = data.assetName;
                    }

                    // Keep track of all restricted assets tx that can become invalid if qualifier or verifiers are changed
                    if (AreRestrictedAssetsDeployed()) {
                        if (IsAssetNameAnRestricted(data.assetName)) {
                            std::string address = EncodeDestination(data.destination);
                            pool.mapAddressesQualifiersChanged[address].insert(hash);
                            pool.mapHashQualifiersChanged[hash].insert(address);

                            pool.mapAssetVerifierChanged[data.assetName].insert(hash);
                            pool.mapHashVerifierChanged[hash].insert(data.assetName);
                        }
                    }
                } else if (out.scriptPubKey.IsNullGlobalRestrictionAssetTxDataScript()) {
                    CNullAssetTxData globalNullData;
                    if (GlobalAssetNullDataFromScript(out.scriptPubKey, globalNullData)) {
                        if (globalNullData.flag == 1) {
                            if (pool.mapGlobalFreezingAssetTransactions.count(globalNullData.asset_name)) {
                                return state.DoS(0, false, REJECT_INVALID, "bad-txns-global-freeze-already-in-mempool");
                            } else {
                                pool.mapGlobalFreezingAssetTransactions[globalNullData.asset_name].insert(tx.GetHash());
                                pool.mapHashGlobalFreezingAssetTransactions[tx.GetHash()].insert(globalNullData.asset_name);
                            }
                        } else if (globalNullData.flag == 0) {
                            if (pool.mapGlobalUnFreezingAssetTransactions.count(globalNullData.asset_name)) {
                                return state.DoS(0, false, REJECT_INVALID, "bad-txns-global-unfreeze-already-in-mempool");
                            } else {
                                pool.mapGlobalUnFreezingAssetTransactions[globalNullData.asset_name].insert(tx.GetHash());
                                pool.mapHashGlobalUnFreezingAssetTransactions[tx.GetHash()].insert(globalNullData.asset_name);
                            }
                        }
                    }
                } else if (out.scriptPubKey.IsNullAssetTxDataScript()) {
                    // We need to track all tags that are being adding to address, that live in the mempool
                    // This will allow us to keep the mempool clean, and only allow one tag per address at a time into the mempool
                    CNullAssetTxData addressNullData;
                    std::string address;
                    if (AssetNullDataFromScript(out.scriptPubKey, addressNullData, address)) {
                        if (IsAssetNameAQualifier(addressNullData.asset_name)) {
                            if (addressNullData.flag == (int) QualifierType::ADD_QUALIFIER) {
                                if (pool.mapAddressAddedTag.count(std::make_pair(address, addressNullData.asset_name))) {
                                    return state.DoS(0, false, REJECT_INVALID,
                                                     "bad-txns-adding-tag-already-in-mempool");
                                }
                                // Adding a qualifier to an address
                                pool.mapAddressAddedTag[std::make_pair(address, addressNullData.asset_name)].insert(tx.GetHash());
                                pool.mapHashToAddressAddedTag[tx.GetHash()].insert(std::make_pair(address, addressNullData.asset_name));
                            } else {
                                    if (pool.mapAddressRemoveTag.count(std::make_pair(address, addressNullData.asset_name))) {
                                        return state.DoS(0, false, REJECT_INVALID,
                                                         "bad-txns-remove-tag-already-in-mempool");
                                    }

                                pool.mapAddressRemoveTag[std::make_pair(address, addressNullData.asset_name)].insert(tx.GetHash());
                                pool.mapHashToAddressRemoveTag[tx.GetHash()].insert(std::make_pair(address, addressNullData.asset_name));
                            }
                        }
                    }
                }
            }
        }

        // Keep track of all restricted assets tx that can become invalid if address or assets are marked as frozen
        if (AreRestrictedAssetsDeployed()) {
            for (auto in : tx.vin) {
                const Coin coin = pcoinsTip->AccessCoin(in.prevout);

                if (!coin.IsAsset())
                    continue;

                CAssetOutputEntry data;
                if (GetAssetData(coin.out.scriptPubKey, data)) {

                    if (IsAssetNameAnRestricted(data.assetName)) {
                        pool.mapAssetMarkedGlobalFrozen[data.assetName].insert(hash);
                        pool.mapHashMarkedGlobalFrozen[hash].insert(data.assetName);

                        auto pair = std::make_pair(EncodeDestination(data.destination), data.assetName);
                        pool.mapAddressesMarkedFrozen[pair].insert(hash);
                        pool.mapHashToAddressMarkedFrozen[hash].insert(pair);
                    }
                }
            }
        }
    }

    if (!fDryRun)
        GetMainSignals().SyncTransaction(tx, nullptr, CMainSignals::SYNC_TRANSACTION_NOT_IN_BLOCK);

    GetMainSignals().TransactionAddedToMempool(ptx);

    return true;
}

bool AcceptToMemoryPoolWithTime(CTxMemPool& pool, CValidationState& state, const CTransactionRef& tx, bool fLimitFree, bool* pfMissingInputs, int64_t nAcceptTime, std::list<CTransactionRef>* plTxnReplaced, bool fOverrideMempoolLimit, const CAmount nAbsurdFee, bool fDryRun)
{
    std::vector<COutPoint> coins_to_uncache;
    bool res = AcceptToMemoryPoolWorker(pool, state, tx, fLimitFree, pfMissingInputs, nAcceptTime, plTxnReplaced, fOverrideMempoolLimit, nAbsurdFee, coins_to_uncache, fDryRun);
    bool fluidTimestampCheck = true;

    if (!fluid.ProvisionalCheckTransaction(*tx))
        return false;

    for (const CTxOut& txout : tx->vout) {
        if (IsTransactionFluid(txout.scriptPubKey)) {
            std::string strErrorMessage;
            if (!fluid.CheckFluidOperationScript(txout.scriptPubKey, GetTime(), strErrorMessage)) {
                fluidTimestampCheck = false;
            }
        }
    }

    if (!res || fDryRun || !fluidTimestampCheck) {
        if (!res)
            LogPrint("mempool", "%s: %s %s %s\n", __func__, tx->GetHash().ToString(), state.GetRejectReason(), state.GetDebugMessage());
        for (const COutPoint& hashTx : coins_to_uncache)
            pcoinsTip->Uncache(hashTx);
    }
    // After we've (potentially) uncached entries, ensure our coins cache is still within its size limits
    CValidationState stateDummy;
    FlushStateToDisk(stateDummy, FLUSH_STATE_PERIODIC);
    return res;
}

bool AcceptToMemoryPool(CTxMemPool& pool, CValidationState& state, const CTransactionRef& tx, bool fLimitFree, bool* pfMissingInputs, std::list<CTransactionRef>* plTxnReplaced, bool fOverrideMempoolLimit, const CAmount nAbsurdFee, bool fDryRun)
{
    return AcceptToMemoryPoolWithTime(pool, state, tx, fLimitFree, pfMissingInputs, GetTime(), plTxnReplaced, fOverrideMempoolLimit, nAbsurdFee, fDryRun);
}

bool GetTimestampIndex(const unsigned int &high, const unsigned int &low, const bool fActiveOnly, std::vector<std::pair<uint256, unsigned int> > &hashes)
{
    if (!fTimestampIndex)
        return error("Timestamp index not enabled");

    if (!pblocktree->ReadTimestampIndex(high, low, fActiveOnly, hashes))
        return error("Unable to get hashes for timestamps");

    return true;
}

bool GetSpentIndex(CSpentIndexKey& key, CSpentIndexValue& value)
{
    if (!fSpentIndex)
        return false;

    if (mempool.getSpentIndex(key, value))
        return true;

    if (!pblocktree->ReadSpentIndex(key, value))
        return false;

    return true;
}

bool HashOnchainActive(const uint256 &hash)
{
    CBlockIndex* pblockindex = mapBlockIndex[hash];

    if (!chainActive.Contains(pblockindex)) {
        return false;
    }

    return true;
}

bool GetAddressIndex(uint160 addressHash, int type, std::vector<std::pair<CAddressIndexKey, CAmount> >& addressIndex, int start, int end)
{
    if (!fAddressIndex)
        return error("address index not enabled");

    if (!pblocktree->ReadAddressIndex(addressHash, type, addressIndex, start, end))
        return error("unable to get txids for address");

    return true;
}

bool GetAddressUnspent(uint160 addressHash, int type, std::vector<std::pair<CAddressUnspentKey, CAddressUnspentValue> >& unspentOutputs)
{
    if (!fAddressIndex)
        return error("address index not enabled");

    if (!pblocktree->ReadAddressUnspentIndex(addressHash, type, unspentOutputs))
        return error("unable to get txids for address");

    return true;
}

/** Return transaction in tx, and if it was found inside a block, its hash is placed in hashBlock */
bool GetTransaction(const uint256& hash, CTransactionRef& txOut, const Consensus::Params& consensusParams, uint256& hashBlock, bool fAllowSlow)
{
    CBlockIndex* pindexSlow = nullptr;

    LOCK(cs_main);

    CTransactionRef ptx = mempool.get(hash);
    if (ptx) {
        txOut = ptx;
        return true;
    }

    if (fTxIndex) {
        CDiskTxPos postx;
        if (pblocktree->ReadTxIndex(hash, postx)) {
            CAutoFile file(OpenBlockFile(postx, true), SER_DISK, CLIENT_VERSION);
            if (file.IsNull())
                return error("%s: OpenBlockFile failed", __func__);
            CBlockHeader header;
            try {
                file >> header;
                fseek(file.Get(), postx.nTxOffset, SEEK_CUR);
                file >> txOut;
            } catch (const std::exception& e) {
                return error("%s: Deserialize or I/O error - %s", __func__, e.what());
            }
            hashBlock = header.GetHash();
            if (txOut->GetHash() != hash)
                return error("%s: txid mismatch", __func__);
            return true;
        }
        // transaction not found in index, nothing more can be done
        return false;
    }

    if (fAllowSlow) { // use coin database to locate block that contains transaction, and scan it
        const Coin& coin = AccessByTxid(*pcoinsTip, hash);
        if (!coin.IsSpent())
            pindexSlow = chainActive[coin.nHeight];
    }

    if (pindexSlow) {
        CBlock block;
        if (ReadBlockFromDisk(block, pindexSlow, consensusParams)) {
            for (const auto& tx : block.vtx) {
                if (tx->GetHash() == hash) {
                    txOut = tx;
                    hashBlock = pindexSlow->GetBlockHash();
                    return true;
                }
            }
        }
    }

    return false;
}


//////////////////////////////////////////////////////////////////////////////
//
// CBlock and CBlockIndex
//

bool WriteBlockToDisk(const CBlock& block, CDiskBlockPos& pos, const CMessageHeader::MessageStartChars& messageStart)
{
    // Open history file to append
    CAutoFile fileout(OpenBlockFile(pos), SER_DISK, CLIENT_VERSION);
    if (fileout.IsNull())
        return error("WriteBlockToDisk: OpenBlockFile failed");

    // Write index header
    unsigned int nSize = GetSerializeSize(fileout, block);
    fileout << FLATDATA(messageStart) << nSize;

    // Write block
    long fileOutPos = ftell(fileout.Get());
    if (fileOutPos < 0)
        return error("WriteBlockToDisk: ftell failed");
    pos.nPos = (unsigned int)fileOutPos;
    fileout << block;

    return true;
}

bool ReadBlockFromDisk(CBlock& block, const CDiskBlockPos& pos, const Consensus::Params& consensusParams)
{
    block.SetNull();

    // Open history file to read
    CAutoFile filein(OpenBlockFile(pos, true), SER_DISK, CLIENT_VERSION);
    if (filein.IsNull())
        return error("ReadBlockFromDisk: OpenBlockFile failed for %s", pos.ToString());

    // Read block
    try {
        filein >> block;
    } catch (const std::exception& e) {
        return error("%s: Deserialize or I/O error - %s at %s", __func__, e.what(), pos.ToString());
    }

    // Check the header
    if (block.IsProofOfWork()) {
        if (!CheckProofOfWork(block.GetHash(), block.nBits, consensusParams))
            return error("ReadBlockFromDisk: Errors in block header at %s", pos.ToString());
    }
    return true;
}

bool ReadBlockFromDisk(CBlock& block, const CBlockIndex* pindex, const Consensus::Params& consensusParams)
{
    if (!ReadBlockFromDisk(block, pindex->GetBlockPos(), consensusParams))
        return false;
    if (block.GetHash() != pindex->GetBlockHash())
        return error("ReadBlockFromDisk(CBlock&, CBlockIndex*): GetHash() doesn't match index for %s at %s",
            pindex->ToString(), pindex->GetBlockPos().ToString());
    return true;
}

bool IsInitialBlockDownload()
{
    // Once this function has returned false, it must remain false.
    static std::atomic<bool> latchToFalse{false};
    // Optimization: pre-test latch before taking the lock.
    if (latchToFalse.load(std::memory_order_relaxed))
        return false;

    LOCK(cs_main);
    if (latchToFalse.load(std::memory_order_relaxed))
        return false;
    if (fImporting || fReindex)
        return true;
    const CChainParams& chainParams = Params();
    if (chainActive.Tip() == nullptr)
        return true;
    if (chainActive.Tip()->nChainWork < int64_t(chainParams.GetConsensus().nMinimumChainWork))
        return true;
    if (chainActive.Tip()->GetBlockTime() < (GetTime() - nMaxTipAge))
        return true;
    latchToFalse.store(true, std::memory_order_relaxed);
    return false;
}

CBlockIndex *pindexBestForkTip = nullptr, *pindexBestForkBase = nullptr;

void CheckForkWarningConditions()
{
    AssertLockHeld(cs_main);
    // Before we get past initial download, we cannot reliably alert about forks
    // (we assume we don't get stuck on a fork before the last checkpoint)
    if (IsInitialBlockDownload())
        return;

    // If our best fork is no longer within 72 blocks (+/- 3 hours if no one mines it)
    // of our head, drop it
    if (pindexBestForkTip && chainActive.Height() - pindexBestForkTip->nHeight >= 72)
        pindexBestForkTip = nullptr;

    if (pindexBestForkTip || (pindexBestInvalid && pindexBestInvalid->nChainWork > chainActive.Tip()->nChainWork + (GetBlockProof(*chainActive.Tip()) * 6))) {
        if (!GetfLargeWorkForkFound() && pindexBestForkBase) {
            if (pindexBestForkBase->phashBlock) {
                std::string warning = std::string("'Warning: Large-work fork detected, forking after block ") +
                                      pindexBestForkBase->phashBlock->ToString() + std::string("'");
                CAlert::Notify(warning);
            }
        }
        if (pindexBestForkTip && pindexBestForkBase) {
            if (pindexBestForkBase->phashBlock) {
                LogPrintf("%s: Warning: Large valid fork found\n  forking the chain at height %d (%s)\n  lasting to height %d (%s).\nChain state database corruption likely.\n", __func__,
                    pindexBestForkBase->nHeight, pindexBestForkBase->phashBlock->ToString(),
                    pindexBestForkTip->nHeight, pindexBestForkTip->phashBlock->ToString());
                SetfLargeWorkForkFound(true);
            }
        } else {
            if (pindexBestInvalid->nHeight > chainActive.Height() + 10)
                LogPrintf("%s: Warning: Found invalid chain at least ~10 blocks longer than our best chain.\nChain state database corruption likely.\n", __func__);
            else
                LogPrintf("%s: Warning: Found invalid chain which has higher work (at least ~10 blocks worth of work) than our best chain.\nChain state database corruption likely.\n", __func__);
            SetfLargeWorkInvalidChainFound(true);
        }
    } else {
        SetfLargeWorkForkFound(false);
        SetfLargeWorkInvalidChainFound(false);
    }
}

void CheckForkWarningConditionsOnNewFork(CBlockIndex* pindexNewForkTip)
{
    AssertLockHeld(cs_main);
    // If we are on a fork that is sufficiently large, set a warning flag
    CBlockIndex* pfork = pindexNewForkTip;
    CBlockIndex* plonger = chainActive.Tip();
    while (pfork && pfork != plonger) {
        while (plonger && plonger->nHeight > pfork->nHeight)
            plonger = plonger->pprev;
        if (pfork == plonger)
            break;
        pfork = pfork->pprev;
    }

    // We define a condition where we should warn the user about as a fork of at least 7 blocks
    // with a tip within 72 blocks (+/- 3 hours if no one mines it) of ours
    // or a chain that is entirely longer than ours and invalid (note that this should be detected by both)
    // We use 7 blocks rather arbitrarily as it represents just under 10% of sustained network
    // hash rate operating on the fork.
    // We define it this way because it allows us to only store the highest fork tip (+ base) which meets
    // the 7-block condition and from this always have the most-likely-to-cause-warning fork
    if (pfork && (!pindexBestForkTip || (pindexBestForkTip && pindexNewForkTip->nHeight > pindexBestForkTip->nHeight)) &&
        pindexNewForkTip->nChainWork - pfork->nChainWork > (GetBlockProof(*pfork) * 7) &&
        chainActive.Height() - pindexNewForkTip->nHeight < 72) {
        pindexBestForkTip = pindexNewForkTip;
        pindexBestForkBase = pfork;
    }

    CheckForkWarningConditions();
}

void static InvalidChainFound(CBlockIndex* pindexNew)
{
    if (!pindexBestInvalid || pindexNew->nChainWork > pindexBestInvalid->nChainWork)
        pindexBestInvalid = pindexNew;

    LogPrintf("%s: invalid block=%s  height=%d  log2_work=%.8g  date=%s\n", __func__,
        pindexNew->GetBlockHash().ToString(), pindexNew->nHeight,
        log(pindexNew->nChainWork.getdouble()) / log(2.0), DateTimeStrFormat("%Y-%m-%d %H:%M:%S", pindexNew->GetBlockTime()));
    CBlockIndex* tip = chainActive.Tip();
    assert(tip);
    LogPrintf("%s:  current best=%s  height=%d  log2_work=%.8g  date=%s\n", __func__,
        tip->GetBlockHash().ToString(), chainActive.Height(), log(tip->nChainWork.getdouble()) / log(2.0),
        DateTimeStrFormat("%Y-%m-%d %H:%M:%S", tip->GetBlockTime()));
    CheckForkWarningConditions();
}

void static InvalidBlockFound(CBlockIndex* pindex, const CValidationState& state)
{
    if (!state.CorruptionPossible()) {
        pindex->nStatus |= BLOCK_FAILED_VALID;
        setDirtyBlockIndex.insert(pindex);
        setBlockIndexCandidates.erase(pindex);
        InvalidChainFound(pindex);
    }
}

void UpdateCoins(const CTransaction& tx, CCoinsViewCache& inputs, CTxUndo &txundo, int nHeight, uint256 blockHash, CAssetsCache* assetCache, std::pair<std::string, CBlockAssetUndo>* undoAssetData)
{
    // mark inputs spent
    if (!tx.IsCoinBase()) {
        txundo.vprevout.reserve(tx.vin.size());
        for (const CTxIn &txin : tx.vin) { 
            txundo.vprevout.emplace_back();
            bool is_spent = false;
            if (AreAssetsDeployed()) {
                is_spent = SpendCoinWithAssets(inputs, txin.prevout, &txundo.vprevout.back(), assetCache); /** ASSET START */ /* Pass assetCache into function */ /** ASSET END */
            } else {
                is_spent = inputs.SpendCoin(txin.prevout, &txundo.vprevout.back());
            }
            assert(is_spent);
        }
    }
    // add outputs
    if (AreAssetsDeployed()) {
        AddCoinsWithAssets(inputs, tx, nHeight, blockHash, false, assetCache, undoAssetData); /** ASSET START */ /* Pass assetCache into function */ /** ASSET END */
    } else {
        // add outputs
        AddCoins(inputs, tx, nHeight);
    }
}

void UpdateCoins(const CTransaction& tx, CCoinsViewCache& inputs, int nHeight)
{
    CTxUndo txundo;
    UpdateCoins(tx, inputs, txundo, nHeight, uint256());
}

bool CScriptCheck::operator()()
{
    const CScript& scriptSig = ptxTo->vin[nIn].scriptSig;
    return VerifyScript(scriptSig, m_tx_out.scriptPubKey, nFlags, CachingTransactionSignatureChecker(ptxTo, nIn, m_tx_out.nValue, cacheStore, *txdata), &error);
}

int GetSpendHeight(const CCoinsViewCache& inputs)
{
    LOCK(cs_main);
    CBlockIndex* pindexPrev = mapBlockIndex.find(inputs.GetBestBlock())->second;
    return pindexPrev->nHeight + 1;
}

static CuckooCache::cache<uint256, SignatureCacheHasher> scriptExecutionCache;
static uint256 scriptExecutionCacheNonce(GetRandHash());

void InitScriptExecutionCache() {
    // nMaxCacheSize is unsigned. If -maxsigcachesize is set to zero,
    // setup_bytes creates the minimum possible cache (2 elements).
    size_t nMaxCacheSize = std::min(std::max((int64_t)0, gArgs.GetArg("-maxsigcachesize", DEFAULT_MAX_SIG_CACHE_SIZE) / 2), MAX_MAX_SIG_CACHE_SIZE) * ((size_t) 1 << 20);
    size_t nElems = scriptExecutionCache.setup_bytes(nMaxCacheSize);
    LogPrintf("Using %zu MiB out of %zu/2 requested for script execution cache, able to store %zu elements\n",
            (nElems*sizeof(uint256)) >>20, (nMaxCacheSize*2)>>20, nElems);
}

namespace Consensus
{
bool CheckTxInputs(const CTransaction& tx, CValidationState& state, const CCoinsViewCache& inputs, int nSpendHeight)
{
    // This doesn't trigger the DoS code on purpose; if it did, it would make it easier
    // for an attacker to attempt to split the network.
    if (!inputs.HaveInputs(tx))
        return state.Invalid(false, 0, "", "Inputs unavailable");

    CAmount nValueIn = 0;
    CAmount nFees = 0;
    for (unsigned int i = 0; i < tx.vin.size(); i++) {
        const COutPoint& prevout = tx.vin[i].prevout;
        const Coin& coin = inputs.AccessCoin(prevout);
        assert(!coin.IsSpent());

        // If prev is coinbase, check that it's matured
        if (coin.IsCoinBase() || coin.IsCoinStake()) {
            if (nSpendHeight - coin.nHeight < COINBASE_MATURITY)
                return state.Invalid(false,
                    REJECT_INVALID, "bad-txns-premature-spend-of-coinbase",
                    strprintf("tried to spend coinbase at depth %d", nSpendHeight - coin.nHeight));
        }

        // Check for negative or overflow input values
        nValueIn += coin.out.nValue;
        if (!MoneyRange(coin.out.nValue) || !MoneyRange(nValueIn))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-inputvalues-outofrange");
    }

    if (!tx.IsCoinStake()) {
        if (nValueIn < tx.GetValueOut())
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-in-belowout", false,
                strprintf("value in (%s) < value out (%s)", FormatMoney(nValueIn), FormatMoney(tx.GetValueOut())));
        // Tally transaction fees
        CAmount nTxFee = nValueIn - tx.GetValueOut();
        if (nTxFee < 0)
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-fee-negative");
        nFees += nTxFee;
        if (!MoneyRange(nFees))
            return state.DoS(100, false, REJECT_INVALID, "bad-txns-fee-outofrange");
    }
    return true;
}
//! Check to make sure that the inputs and outputs CAmount match exactly.
bool CheckTxAssets(const CTransaction& tx, CValidationState& state, const CCoinsViewCache& inputs, CAssetsCache* assetCache, bool fCheckMempool, std::vector<std::pair<std::string, uint256> >& vPairReissueAssets, const bool fRunningUnitTests, std::set<CMessage>* setMessages, int64_t nBlocktime,   std::vector<std::pair<std::string, CNullAssetTxData>>* myNullAssetData)
{
    // are the actual inputs available?
    if (!inputs.HaveInputs(tx)) {
        return state.DoS(100, false, REJECT_INVALID, "bad-txns-inputs-missing-or-spent", false,
                         strprintf("%s: inputs missing/spent", __func__), tx.GetHash());
    }

    // Create map that stores the amount of an asset transaction input. Used to verify no assets are burned
    std::map<std::string, CAmount> totalInputs;

    std::map<std::string, std::string> mapAddresses;

    for (unsigned int i = 0; i < tx.vin.size(); ++i) {
        const COutPoint &prevout = tx.vin[i].prevout;
        const Coin& coin = inputs.AccessCoin(prevout);
        assert(!coin.IsSpent());

        if (coin.IsAsset()) {
            CAssetOutputEntry data;
            if (!GetAssetData(coin.out.scriptPubKey, data))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-failed-to-get-asset-from-script", false, "", tx.GetHash());

            // Add to the total value of assets in the inputs
            if (totalInputs.count(data.assetName))
                totalInputs.at(data.assetName) += data.nAmount;
            else
                totalInputs.insert(make_pair(data.assetName, data.nAmount));

            if (AreMessagesDeployed()) {
                mapAddresses.insert(make_pair(data.assetName,EncodeDestination(data.destination)));
            }

            if (IsAssetNameAnRestricted(data.assetName)) {
                if (assetCache->CheckForAddressRestriction(data.assetName, EncodeDestination(data.destination), true)) {
                    return state.DoS(100, false, REJECT_INVALID, "bad-txns-restricted-asset-transfer-from-frozen-address", false, "", tx.GetHash());
                }
            }
        }
    }

    // Create map that stores the amount of an asset transaction output. Used to verify no assets are burned
    std::map<std::string, CAmount> totalOutputs;
    int index = 0;
    int64_t currentTime = GetTime();
    std::string strError = "";
    int i = 0;
    for (const auto& txout : tx.vout) {
        i++;
        bool fIsAsset = false;
        int nType = 0;
        bool fIsOwner = false;
        if (txout.scriptPubKey.IsAssetScript(nType, fIsOwner))
            fIsAsset = true;

        if (assetCache) {
            if (fIsAsset && !AreAssetsDeployed())
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-is-asset-and-asset-not-active");

            if (txout.scriptPubKey.IsNullAsset()) {
                if (!AreRestrictedAssetsDeployed())
                    return state.DoS(100, false, REJECT_INVALID,
                                     "bad-tx-null-asset-data-before-restricted-assets-activated");

                if (txout.scriptPubKey.IsNullAssetTxDataScript()) {
                    if (!ContextualCheckNullAssetTxOut(txout, assetCache, strError, myNullAssetData))
                        return state.DoS(100, false, REJECT_INVALID, strError, false, "", tx.GetHash());
                } else if (txout.scriptPubKey.IsNullGlobalRestrictionAssetTxDataScript()) {
                    if (!ContextualCheckGlobalAssetTxOut(txout, assetCache, strError))
                        return state.DoS(100, false, REJECT_INVALID, strError, false, "", tx.GetHash());
                } else if (txout.scriptPubKey.IsNullAssetVerifierTxDataScript()) {
                    if (!ContextualCheckVerifierAssetTxOut(txout, assetCache, strError))
                        return state.DoS(100, false, REJECT_INVALID, strError, false, "", tx.GetHash());
                } else {
                    return state.DoS(100, false, REJECT_INVALID, "bad-tx-null-asset-data-unknown-type", false, "", tx.GetHash());
                }
            }
        }

        if (nType == TX_TRANSFER_ASSET) {
            CAssetTransfer transfer;
            std::string address = "";
            if (!TransferAssetFromScript(txout.scriptPubKey, transfer, address))
                return state.DoS(100, false, REJECT_INVALID, "bad-tx-asset-transfer-bad-deserialize", false, "", tx.GetHash());

            if (!ContextualCheckTransferAsset(assetCache, transfer, address, strError))
                return state.DoS(100, false, REJECT_INVALID, strError, false, "", tx.GetHash());

            // Add to the total value of assets in the outputs
            if (totalOutputs.count(transfer.strName))
                totalOutputs.at(transfer.strName) += transfer.nAmount;
            else
                totalOutputs.insert(make_pair(transfer.strName, transfer.nAmount));

            if (!fRunningUnitTests) {
                if (IsAssetNameAnOwner(transfer.strName)) {
                    if (transfer.nAmount != OWNER_ASSET_AMOUNT)
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-owner-amount-was-not-1", false, "", tx.GetHash());
                } else {
                    // For all other types of assets, make sure they are sending the right type of units
                    CNewAsset asset;
                    if (!assetCache->GetAssetMetaDataIfExists(transfer.strName, asset))
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-asset-not-exist", false, "", tx.GetHash());

                    if (asset.strName != transfer.strName)
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-asset-database-corrupted", false, "", tx.GetHash());

                    if (!CheckAmountWithUnits(transfer.nAmount, asset.units))
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-transfer-asset-amount-not-match-units", false, "", tx.GetHash());
                }
            }

            /** Get messages from the transaction, only used when getting called from ConnectBlock **/
            // Get the messages from the Tx unless they are expired
            if (AreMessagesDeployed() && fMessaging && setMessages) {
                if (IsAssetNameAnOwner(transfer.strName) || IsAssetNameAnMsgChannel(transfer.strName)) {
                    if (!transfer.message.empty()) {
                        if (transfer.nExpireTime == 0 || transfer.nExpireTime > currentTime) {
                            if (mapAddresses.count(transfer.strName)) {
                                if (mapAddresses.at(transfer.strName) == address) {
                                    COutPoint out(tx.GetHash(), index);
                                    CMessage message(out, transfer.strName, transfer.message,
                                                     transfer.nExpireTime, nBlocktime);
                                    setMessages->insert(message);
                                    LogPrintf("Got message: %s\n", message.ToString()); // TODO remove after testing
                                }
                            }
                        }
                    }
                }
            }
        } else if (nType == TX_REISSUE_ASSET) {
            CReissueAsset reissue;
            std::string address;
            if (!ReissueAssetFromScript(txout.scriptPubKey, reissue, address))
                return state.DoS(100, false, REJECT_INVALID, "bad-tx-asset-reissue-bad-deserialize", false, "", tx.GetHash());

            if (mapReissuedAssets.count(reissue.strName)) {
                if (mapReissuedAssets.at(reissue.strName) != tx.GetHash())
                    return state.DoS(100, false, REJECT_INVALID, "bad-tx-reissue-chaining-not-allowed", false, "", tx.GetHash());
            } else {
                vPairReissueAssets.emplace_back(std::make_pair(reissue.strName, tx.GetHash()));
            }
        }
        index++;
    }

    if (assetCache) {
        if (tx.IsNewAsset()) {
            // Get the asset type
            CNewAsset asset;
            std::string address;
            if (!AssetFromScript(tx.vout[tx.vout.size() - 1].scriptPubKey, asset, address)) {
                error("%s : Failed to get new asset from transaction: %s", __func__, tx.GetHash().GetHex());
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-serialzation-failed", false, "", tx.GetHash());
            }

            AssetType assetType;
            IsAssetNameValid(asset.strName, assetType);

            if (!ContextualCheckNewAsset(assetCache, asset, strError, fCheckMempool))
                return state.DoS(100, false, REJECT_INVALID, strError);

        } else if (tx.IsReissueAsset()) {
            CReissueAsset reissue_asset;
            std::string address;
            if (!ReissueAssetFromScript(tx.vout[tx.vout.size() - 1].scriptPubKey, reissue_asset, address)) {
                error("%s : Failed to get new asset from transaction: %s", __func__, tx.GetHash().GetHex());
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-reissue-serialzation-failed", false, "", tx.GetHash());
            }
            if (!ContextualCheckReissueAsset(assetCache, reissue_asset, strError, tx))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-reissue-contextual-" + strError, false, "", tx.GetHash());
        } else if (tx.IsNewUniqueAsset()) {
            if (!ContextualCheckUniqueAssetTx(assetCache, strError, tx))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-unique-contextual-" + strError, false, "", tx.GetHash());
        } else if (tx.IsNewMsgChannelAsset()) {
            if (!AreMessagesDeployed())
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-msgchannel-before-messaging-is-active", false, "", tx.GetHash());

            CNewAsset asset;
            std::string strAddress;
            if (!MsgChannelAssetFromTransaction(tx, asset, strAddress))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-msgchannel-serialzation-failed", false, "", tx.GetHash());

            if (!ContextualCheckNewAsset(assetCache, asset, strError, fCheckMempool))
                return state.DoS(100, error("%s: %s", __func__, strError), REJECT_INVALID,
                                 "bad-txns-issue-msgchannel-contextual-" + strError);
        } else if (tx.IsNewQualifierAsset()) {
            if (!AreRestrictedAssetsDeployed())
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-qualifier-before-it-is-active", false, "", tx.GetHash());

            CNewAsset asset;
            std::string strAddress;
            if (!QualifierAssetFromTransaction(tx, asset, strAddress))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-qualifier-serialzation-failed", false, "", tx.GetHash());

            if (!ContextualCheckNewAsset(assetCache, asset, strError, fCheckMempool))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-qualfier-contextual" + strError, false, "", tx.GetHash());

        } else if (tx.IsNewRestrictedAsset()) {
            if (!AreRestrictedAssetsDeployed())
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-restricted-before-it-is-active", false, "", tx.GetHash());

            // Get asset data
            CNewAsset asset;
            std::string strAddress;
            if (!RestrictedAssetFromTransaction(tx, asset, strAddress))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-restricted-serialzation-failed", false, "", tx.GetHash());

            if (!ContextualCheckNewAsset(assetCache, asset, strError, fCheckMempool))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-restricted-contextual" + strError, false, "", tx.GetHash());

            // Get verifier string
            CNullAssetTxVerifierString verifier;
            if (!tx.GetVerifierStringFromTx(verifier, strError))
                return state.DoS(100, false, REJECT_INVALID, "bad-txns-issue-restricted-verifier-search-" + strError, false, "", tx.GetHash());

            // Check the verifier string against the destination address
            if (!ContextualCheckVerifierString(assetCache, verifier.verifier_string, strAddress, strError))
                return state.DoS(100, false, REJECT_INVALID, strError, false, "", tx.GetHash());

        } else {
            for (auto out : tx.vout) {
                int nType;
                bool _isOwner;
                if (out.scriptPubKey.IsAssetScript(nType, _isOwner)) {
                    if (nType != TX_TRANSFER_ASSET) {
                        return state.DoS(100, false, REJECT_INVALID, "bad-txns-bad-asset-transaction", false, "", tx.GetHash());
                    }
                } else {
                    if (out.scriptPubKey.Find(OP_DYN_ASSET)) {
                        if (AreRestrictedAssetsDeployed()) {
                            if (out.scriptPubKey[0] != OP_DYN_ASSET) {
                                return state.DoS(100, false, REJECT_INVALID,
                                                 "bad-txns-op-dyn-asset-not-in-right-script-location", false, "", tx.GetHash());
                            }
                        } else {
                            return state.DoS(100, false, REJECT_INVALID, "bad-txns-bad-asset-script", false, "", tx.GetHash());
                        }
                    }
                }
            }
        }
    }

    for (const auto& outValue : totalOutputs) {
        if (!totalInputs.count(outValue.first)) {
            std::string errorMsg;
            errorMsg = strprintf("Bad Transaction - Trying to create outpoint for asset that you don't have: %s", outValue.first);
            return state.DoS(100, false, REJECT_INVALID, "bad-tx-inputs-outputs-mismatch " + errorMsg, false, "", tx.GetHash());
        }

        if (totalInputs.at(outValue.first) != outValue.second) {
            std::string errorMsg;
            errorMsg = strprintf("Bad Transaction - Assets would be burnt %s", outValue.first);
            return state.DoS(100, false, REJECT_INVALID, "bad-tx-inputs-outputs-mismatch " + errorMsg, false, "", tx.GetHash());
        }
    }

    // Check the input size and the output size
    if (totalOutputs.size() != totalInputs.size()) {
        return state.DoS(100, false, REJECT_INVALID, "bad-tx-asset-inputs-size-does-not-match-outputs-size", false, "", tx.GetHash());
    }
    return true;
}
} // namespace Consensus

bool CheckInputs(const CTransaction& tx, CValidationState& state, const CCoinsViewCache& inputs, bool fScriptChecks, unsigned int flags, bool cacheSigStore, bool cacheFullScriptStore, PrecomputedTransactionData& txdata, std::vector<CScriptCheck> *pvChecks)
{
    if (!tx.IsCoinBase()) {
        if (!Consensus::CheckTxInputs(tx, state, inputs, GetSpendHeight(inputs)))
            return false;

        if (pvChecks)
            pvChecks->reserve(tx.vin.size());

        // The first loop above does all the inexpensive checks.
        // Only if ALL inputs pass do we perform expensive ECDSA signature checks.
        // Helps prevent CPU exhaustion attacks.

        // Skip script verification when connecting blocks under the
        // assumevalid block. Assuming the assumevalid block is valid this
        // is safe because block merkle hashes are still computed and checked,
        // Of course, if an assumed valid block is invalid due to false scriptSigs
        // this optimization would allow an invalid chain to be accepted.
        if (fScriptChecks) {
            // First check if script executions have been cached with the same
            // flags. Note that this assumes that the inputs provided are
            // correct (ie that the transaction hash which is in tx's prevouts
            // properly commits to the scriptPubKey in the inputs view of that
            // transaction).
            uint256 hashCacheEntry;
            // We only use the first 19 bytes of nonce to avoid a second SHA
            // round - giving us 19 + 32 + 4 = 55 bytes (+ 8 + 1 = 64)
            static_assert(55 - sizeof(flags) - 32 >= 128/8, "Want at least 128 bits of nonce for script execution cache");
            CSHA256().Write(scriptExecutionCacheNonce.begin(), 55 - sizeof(flags)).Write((unsigned char*)&flags, sizeof(flags)).Finalize(hashCacheEntry.begin());
            AssertLockHeld(cs_main); //TODO: Remove this requirement by making CuckooCache not require external locks
            if (scriptExecutionCache.contains(hashCacheEntry, !cacheFullScriptStore)) {
                return true;
            }

            for (unsigned int i = 0; i < tx.vin.size(); i++) {
                const COutPoint& prevout = tx.vin[i].prevout;
                const Coin& coin = inputs.AccessCoin(prevout);
                assert(!coin.IsSpent());

                // Verify signature
                CScriptCheck check(coin.out, tx, i, flags, cacheSigStore, &txdata);
                if (pvChecks) {
                    pvChecks->push_back(CScriptCheck());
                    check.swap(pvChecks->back());
                } else if (!check()) {
                    if (flags & STANDARD_NOT_MANDATORY_VERIFY_FLAGS) {
                        // Check whether the failure was caused by a
                        // non-mandatory script verification check, such as
                        // non-standard DER encodings or non-null dummy
                        // arguments; if so, don't trigger DoS protection to
                        // avoid splitting the network between upgraded and
                        // non-upgraded nodes.
                        CScriptCheck check2(coin.out, tx, i,
                                flags & ~STANDARD_NOT_MANDATORY_VERIFY_FLAGS, cacheSigStore, &txdata);
                        if (check2())
                            return state.Invalid(false, REJECT_NONSTANDARD, strprintf("non-mandatory-script-verify-flag (%s)", ScriptErrorString(check.GetScriptError())));
                    }
                    // Failures of other flags indicate a transaction that is
                    // invalid in new blocks, e.g. a invalid P2SH. We DoS ban
                    // such nodes as they are not following the protocol. That
                    // said during an upgrade careful thought should be taken
                    // as to the correct behavior - we may want to continue
                    // peering with non-upgraded nodes even after a soft-fork
                    // super-majority vote has passed.
                    return state.DoS(100, false, REJECT_INVALID, strprintf("mandatory-script-verify-flag-failed (%s)", ScriptErrorString(check.GetScriptError())));
                }
            }

            if (cacheFullScriptStore && !pvChecks) {
                // We executed all of the provided scripts, and were told to
                // cache the result. Do so now.
                scriptExecutionCache.insert(hashCacheEntry);
            }
        }
    }

    return true;
}

namespace
{
bool UndoWriteToDisk(const CBlockUndo& blockundo, CDiskBlockPos& pos, const uint256& hashBlock, const CMessageHeader::MessageStartChars& messageStart)
{
    // Open history file to append
    CAutoFile fileout(OpenUndoFile(pos), SER_DISK, CLIENT_VERSION);
    if (fileout.IsNull())
        return error("%s: OpenUndoFile failed", __func__);

    // Write index header
    unsigned int nSize = GetSerializeSize(fileout, blockundo);
    fileout << FLATDATA(messageStart) << nSize;

    // Write undo data
    long fileOutPos = ftell(fileout.Get());
    if (fileOutPos < 0)
        return error("%s: ftell failed", __func__);
    pos.nPos = (unsigned int)fileOutPos;
    fileout << blockundo;

    // calculate & write checksum
    CHashWriter hasher(SER_GETHASH, PROTOCOL_VERSION);
    hasher << hashBlock;
    hasher << blockundo;
    fileout << hasher.GetHash();

    return true;
}

bool UndoReadFromDisk(CBlockUndo& blockundo, const CDiskBlockPos& pos, const uint256& hashBlock)
{
    // Open history file to read
    CAutoFile filein(OpenUndoFile(pos, true), SER_DISK, CLIENT_VERSION);
    if (filein.IsNull())
        return error("%s: OpenUndoFile failed", __func__);

    // Read block
    uint256 hashChecksum;
    CHashVerifier<CAutoFile> verifier(&filein); // We need a CHashVerifier as reserializing may lose data
    try {
        verifier << hashBlock;
        verifier >> blockundo;
        filein >> hashChecksum;
    } catch (const std::exception& e) {
        return error("%s: Deserialize or I/O error - %s", __func__, e.what());
    }

    // Verify checksum
    if (hashChecksum != verifier.GetHash())
        return error("%s: Checksum mismatch", __func__);

    return true;
}

/** Abort with a message */
bool AbortNode(const std::string& strMessage, const std::string& userMessage = "")
{
    SetMiscWarning(strMessage);
    LogPrintf("*** %s\n", strMessage);
    uiInterface.ThreadSafeMessageBox(
        userMessage.empty() ? _("Error: A fatal internal error occurred, see debug.log for details") : userMessage,
        "", CClientUIInterface::MSG_ERROR);
    StartShutdown();
    return false;
}

bool AbortNode(CValidationState& state, const std::string& strMessage, const std::string& userMessage = "")
{
    AbortNode(strMessage, userMessage);
    return state.Error(strMessage);
}

} // namespace

enum DisconnectResult {
    DISCONNECT_OK,      // All good.
    DISCONNECT_UNCLEAN, // Rolled back, but UTXO set was inconsistent with block.
    DISCONNECT_FAILED   // Something else went wrong.
};

/**
 * Restore the UTXO in a Coin at a given COutPoint
 * @param undo The Coin to be restored.
 * @param view The coins view to which to apply the changes.
 * @param out The out point that corresponds to the tx input.
 * @return A DisconnectResult as an int
 */
int ApplyTxInUndo(Coin&& undo, CCoinsViewCache& view, const COutPoint& out, CAssetsCache* assetCache = nullptr)
{
    bool fClean = true;

    /** ASSET START */
    // This is needed because undo, is going to be cleared and moved when AddCoin is called. We need this for undo assets
    Coin tempCoin;
    bool fIsAsset = false;
    if (undo.IsAsset()) {
        fIsAsset = true;
        tempCoin = undo;
    }
    /** ASSET END */

    if (view.HaveCoin(out)) fClean = false; // overwriting transaction output

    if (undo.nHeight == 0) {
        // Missing undo metadata (height and coinbase). Older versions included this
        // information only in undo records for the last spend of a transactions'
        // outputs. This implies that it must be present for some other output of the same tx.
        const Coin& alternate = AccessByTxid(view, out.hash);
        if (!alternate.IsSpent()) {
            undo.nHeight = alternate.nHeight;
            undo.fCoinBase = alternate.fCoinBase;
        } else {
            return DISCONNECT_FAILED; // adding output for transaction without known metadata
        }
    }
    view.AddCoin(out, std::move(undo), !fClean);

    /** ASSET START */
    if (AreAssetsDeployed()) {
        if (assetCache && fIsAsset) {
            if (!assetCache->UndoAssetCoin(tempCoin, out))
                fClean = false;
        }
    }
    /** ASSET END */

    return fClean ? DISCONNECT_OK : DISCONNECT_UNCLEAN;
}

/** Undo the effects of this block (with given index) on the UTXO set represented by coins.
 *  When UNCLEAN or FAILED is returned, view is left in an indeterminate state. */
static DisconnectResult DisconnectBlock(const CBlock& block, CValidationState& state, const CBlockIndex* pindex, CCoinsViewCache& view, int nCheckLevel, CAssetsCache* assetsCache = nullptr, bool ignoreAddressIndex = false, bool databaseMessaging = true)
{
    assert(pindex->GetBlockHash() == view.GetBestBlock());
    bool fClean = true;

    CBlockUndo blockUndo;
    CDiskBlockPos pos = pindex->GetUndoPos();
    if (pos.IsNull()) {
        error("DisconnectBlock(): no undo data available");
        return DISCONNECT_FAILED;
    }
    if (!UndoReadFromDisk(blockUndo, pos, pindex->pprev->GetBlockHash())) {
        error("DisconnectBlock(): failure reading undo data");
        return DISCONNECT_FAILED;
    }

    if (blockUndo.vtxundo.size() + 1 != block.vtx.size()) {
        error("DisconnectBlock(): block and undo data inconsistent");
        return DISCONNECT_FAILED;
    }

    std::vector<std::pair<std::string, CBlockAssetUndo> > vUndoData;
    if (!passetsdb->ReadBlockUndoAssetData(block.GetHash(), vUndoData)) {
        error("DisconnectBlock(): block asset undo data inconsistent");
        return DISCONNECT_FAILED;
    }

    std::vector<std::pair<CAddressIndexKey, CAmount> > addressIndex;
    std::vector<std::pair<CAddressUnspentKey, CAddressUnspentValue> > addressUnspentIndex;
    std::vector<std::pair<CSpentIndexKey, CSpentIndexValue> > spentIndex;

    // undo transactions in reverse order
    CAssetsCache tempCache(*assetsCache);
    for (int i = block.vtx.size() - 1; i >= 0; i--) {
        const CTransaction& tx = *block.vtx[i];
        uint256 hash = tx.GetHash();
        bool is_coinbase = tx.IsCoinBase();

        bool fIsBDAP = tx.nVersion == BDAP_TX_VERSION;
        if (fIsBDAP && !fReindex && nCheckLevel >= 4) {
            LogPrintf("%s -- BDAP tx found. Hash %s\n", __func__, hash.ToString());
            // get BDAP object
            CScript scriptBDAPOp; 
            std::vector<std::vector<unsigned char>> vvchOpParameters;
            int op1, op2;
            CTransactionRef ptx = MakeTransactionRef(tx);
            if (GetBDAPOpScript(ptx, scriptBDAPOp, vvchOpParameters, op1, op2)) {
                LogPrintf("%s -- Found new BDAP object, op1 %d, op2 %d\n", __func__, op1, op2);
                std::string strErrorMessage;
                std::string strOpType = GetBDAPOpTypeString(op1, op2);
                if (strOpType == "bdap_new_account") {
                    CDomainEntry domainEntry(ptx);
                    LogPrintf("%s -- Found new BDAP account %s. Running undo.\n", __func__, domainEntry.GetFullObjectPath());
                    if (!UndoAddDomainEntry(domainEntry)) {
                        LogPrintf("%s -- Failed to undo new BDAP account transaction %s. Disconnect %s transaction failed.\n", __func__, domainEntry.GetFullObjectPath(), hash.ToString());
                    }
                }
                else if (strOpType == "bdap_update_account") {
                    CDomainEntry domainEntry(ptx);
                    if (!UndoUpdateDomainEntry(domainEntry)) {
                        LogPrintf("%s -- Failed to undo update BDAP account transaction %s. Disconnect %s transaction failed.\n", __func__, domainEntry.GetFullObjectPath(), hash.ToString());
                    }
                }
                else if (strOpType == "bdap_delete_account") {
                    CDomainEntry domainEntry(ptx);
                    if (!UndoDeleteDomainEntry(domainEntry)) {
                        LogPrintf("%s -- Failed to undo delete BDAP account transaction %s. Disconnect %s transaction failed.\n", __func__, domainEntry.GetFullObjectPath(), hash.ToString());
                    }
                }
                else if (strOpType == "bdap_new_link_request" || strOpType == "bdap_new_link_accept") {
                    std::vector<unsigned char> vchPubKey, vchSharedPubKey;
                    if (vvchOpParameters.size() > 0)
                        vchPubKey = vvchOpParameters[0];

                    if (vvchOpParameters.size() > 1)
                        vchSharedPubKey = vvchOpParameters[0];

                    LogPrintf("%s -- Found new BDAP link (pubkey %s, sharedpubkey %s). Running undo.\n", __func__, stringFromVch(vchPubKey), stringFromVch(vchSharedPubKey));
                    if (!UndoLinkData(vchPubKey, vchSharedPubKey)) {
                        LogPrintf("%s -- Failed to undo link transaction. Disconnect %s transaction failed.\n", __func__, hash.ToString());
                    }
                }
                else if (strOpType == "bdap_new_audit") {
                    CAudit audit(ptx);
                    if (!UndoAddAudit(audit)) {
                        LogPrintf("%s -- Failed to undo add BDAP audit transaction %s. Disconnect %s transaction failed.\n", __func__, audit.ToString(), hash.ToString());
                    }
                }
                else {
                    LogPrintf("%s -- Failed to undo unknown BDAP transaction (op1 = %d, op2 = %d). Nothing to undo for %s transaction.\n", __func__, op1, op2, hash.ToString());
                }
            }
        }
        std::vector<int> vAssetTxIndex;
        std::vector<int> vNullAssetTxIndex;
        if (fAddressIndex) {
            for (unsigned int k = tx.vout.size(); k-- > 0;) {
                const CTxOut& out = tx.vout[k];

                if (out.scriptPubKey.IsPayToScriptHash()) {
                    // Remove BDAP portion of the script
                    CScript scriptPubKey;
                    CScript scriptPubKeyOut;
                    if (RemoveBDAPScript(out.scriptPubKey, scriptPubKeyOut)) {
                        scriptPubKey = scriptPubKeyOut;
                    } else {
                        scriptPubKey = out.scriptPubKey;
                    }

                    std::vector<unsigned char> hashBytes(scriptPubKey.begin() + 2, scriptPubKey.begin() + 22);

                    // undo receiving activity
                    addressIndex.push_back(std::make_pair(CAddressIndexKey(2, uint160(hashBytes), pindex->nHeight, i, hash, k, false), out.nValue));

                    // undo unspent index
                    addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(2, uint160(hashBytes), hash, k), CAddressUnspentValue()));

                } else if (out.scriptPubKey.IsPayToPublicKeyHash()) {
                    // Remove BDAP portion of the script
                    CScript scriptPubKey;
                    CScript scriptPubKeyOut;
                    if (RemoveBDAPScript(out.scriptPubKey, scriptPubKeyOut)) {
                        scriptPubKey = scriptPubKeyOut;
                    } else {
                        scriptPubKey = out.scriptPubKey;
                    }

                    std::vector<unsigned char> hashBytes(scriptPubKey.begin() + 3, scriptPubKey.begin() + 23);

                    // undo receiving activity
                    addressIndex.push_back(std::make_pair(CAddressIndexKey(1, uint160(hashBytes), pindex->nHeight, i, hash, k, false), out.nValue));

                    // undo unspent index
                    addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(1, uint160(hashBytes), hash, k), CAddressUnspentValue()));

                } else {
                    /** ASSET START */
                    if (AreAssetsDeployed()) {
                        std::string assetName;
                        CAmount assetAmount;
                        uint160 hashBytes;

                        if (ParseAssetScript(out.scriptPubKey, hashBytes, assetName, assetAmount)) {
//                            std::cout << "ConnectBlock(): pushing assets onto addressIndex: " << "1" << ", " << hashBytes.GetHex() << ", " << assetName << ", " << pindex->nHeight
//                                      << ", " << i << ", " << hash.GetHex() << ", " << k << ", " << "true" << ", " << assetAmount << std::endl;

                            // undo receiving activity
                            addressIndex.push_back(std::make_pair(
                                    CAddressIndexKey(1, uint160(hashBytes), assetName, pindex->nHeight, i, hash, k,
                                                     false), assetAmount));

                            // undo unspent index
                            addressUnspentIndex.push_back(
                                    std::make_pair(CAddressUnspentKey(1, uint160(hashBytes), assetName, hash, k),
                                                   CAddressUnspentValue()));
                        } else {
                            continue;
                        }
                    }
                    /** ASSET END */
                }
            }
        }

        /** ASSET START */
        // Check that all outputs are available and match the outputs in the block itself
        // exactly.
        int indexOfRestrictedAssetVerifierString = -1;
        for (size_t o = 0; o < tx.vout.size(); o++) {
            if (!tx.vout[o].scriptPubKey.IsUnspendable()) {
                COutPoint out(hash, o);
                Coin coin;
                bool is_spent = false;
                if (AreAssetsDeployed()) {
                    is_spent = SpendCoinWithAssets(view, out, &coin, &tempCache); /** ASSET START */ /* Pass assetsCache into the SpendCoin function */ /** ASSET END */
                } else {
                    is_spent = view.SpendCoin(out, &coin);
                }
                if (!is_spent || tx.vout[o] != coin.out || pindex->nHeight != coin.nHeight || is_coinbase != coin.fCoinBase) {
                    fClean = false; // transaction output mismatch
                }

                if (AreAssetsDeployed()) {
                    if (assetsCache) {
                        if (IsScriptTransferAsset(tx.vout[o].scriptPubKey))
                            vAssetTxIndex.emplace_back(o);
                    }
                }
            } else {
                if(AreRestrictedAssetsDeployed()) {
                    if (assetsCache) {
                        if (tx.vout[o].scriptPubKey.IsNullAsset()) {
                            if (tx.vout[o].scriptPubKey.IsNullAssetVerifierTxDataScript()) {
                                indexOfRestrictedAssetVerifierString = o;
                            } else {
                                vNullAssetTxIndex.emplace_back(o);
                            }
                        }
                    }
                }
            }
        }

        if (AreAssetsDeployed()) {
            if (assetsCache) {
                if (tx.IsNewAsset()) {
                    // Remove the newly created asset
                    CNewAsset asset;
                    std::string strAddress;
                    if (!AssetFromTransaction(tx, asset, strAddress)) {
                        error("%s : Failed to get asset from transaction. TXID : %s", __func__, tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }
                    if (assetsCache->ContainsAsset(asset)) {
                        if (!assetsCache->RemoveNewAsset(asset, strAddress)) {
                            error("%s : Failed to Remove Asset. Asset Name : %s", __func__, asset.strName);
                            return DISCONNECT_FAILED;
                        }
                    }

                    // Get the owner from the transaction and remove it
                    std::string ownerName;
                    std::string ownerAddress;
                    if (!OwnerFromTransaction(tx, ownerName, ownerAddress)) {
                        error("%s : Failed to get owner from transaction. TXID : %s", __func__, tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }

                    if (!assetsCache->RemoveOwnerAsset(ownerName, ownerAddress)) {
                        error("%s : Failed to Remove Owner from transaction. TXID : %s", __func__, tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }
                } else if (tx.IsReissueAsset()) {
                    CReissueAsset reissue;
                    std::string strAddress;

                    if (!ReissueAssetFromTransaction(tx, reissue, strAddress)) {
                        error("%s : Failed to get reissue asset from transaction. TXID : %s", __func__,
                              tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }

                    if (assetsCache->ContainsAsset(reissue.strName)) {
                        if (!assetsCache->RemoveReissueAsset(reissue, strAddress,
                                                             COutPoint(tx.GetHash(), tx.vout.size() - 1),
                                                             vUndoData)) {
                            error("%s : Failed to Undo Reissue Asset. Asset Name : %s", __func__, reissue.strName);
                            return DISCONNECT_FAILED;
                        }
                    }
                } else if (tx.IsNewUniqueAsset()) {
                    for (int n = 0; n < (int)tx.vout.size(); n++) {
                        auto out = tx.vout[n];
                        CNewAsset asset;
                        std::string strAddress;

                        if (IsScriptNewUniqueAsset(out.scriptPubKey)) {
                            if (!AssetFromScript(out.scriptPubKey, asset, strAddress)) {
                                error("%s : Failed to get unique asset from transaction. TXID : %s, vout: %s", __func__,
                                      tx.GetHash().GetHex(), n);
                                return DISCONNECT_FAILED;
                            }

                            if (assetsCache->ContainsAsset(asset.strName)) {
                                if (!assetsCache->RemoveNewAsset(asset, strAddress)) {
                                    error("%s : Failed to Undo Unique Asset. Asset Name : %s", __func__, asset.strName);
                                    return DISCONNECT_FAILED;
                                }
                            }
                        }
                    }
                } else if (tx.IsNewMsgChannelAsset()) {
                    CNewAsset asset;
                    std::string strAddress;

                    if (!MsgChannelAssetFromTransaction(tx, asset, strAddress)) {
                        error("%s : Failed to get msgchannel asset from transaction. TXID : %s", __func__,
                              tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }

                    if (assetsCache->ContainsAsset(asset.strName)) {
                        if (!assetsCache->RemoveNewAsset(asset, strAddress)) {
                            error("%s : Failed to Undo Msg Channel Asset. Asset Name : %s", __func__, asset.strName);
                            return DISCONNECT_FAILED;
                        }
                    }
                } else if (tx.IsNewQualifierAsset()) {
                    CNewAsset asset;
                    std::string strAddress;

                    if (!QualifierAssetFromTransaction(tx, asset, strAddress)) {
                        error("%s : Failed to get qualifier asset from transaction. TXID : %s", __func__,
                              tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }

                    if (assetsCache->ContainsAsset(asset.strName)) {
                        if (!assetsCache->RemoveNewAsset(asset, strAddress)) {
                            error("%s : Failed to Undo Qualifier Asset. Asset Name : %s", __func__, asset.strName);
                            return DISCONNECT_FAILED;
                        }
                    }
                } else if (tx.IsNewRestrictedAsset()) {
                    CNewAsset asset;
                    std::string strAddress;

                    if (!RestrictedAssetFromTransaction(tx, asset, strAddress)) {
                        error("%s : Failed to get restricted asset from transaction. TXID : %s", __func__,
                              tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }

                    if (assetsCache->ContainsAsset(asset.strName)) {
                        if (!assetsCache->RemoveNewAsset(asset, strAddress)) {
                            error("%s : Failed to Undo Restricted Asset. Asset Name : %s", __func__, asset.strName);
                            return DISCONNECT_FAILED;
                        }
                    }

                    if (indexOfRestrictedAssetVerifierString < 0) {
                        error("%s : Failed to find the restricted asset verifier string index from trasaction. TxID : %s", __func__, tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }

                    CNullAssetTxVerifierString verifier;
                    if (!AssetNullVerifierDataFromScript(tx.vout[indexOfRestrictedAssetVerifierString].scriptPubKey, verifier)) {
                        error("%s : Failed to get the restricted asset verifier string from trasaction. TxID : %s", __func__, tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }

                    if (!assetsCache->RemoveRestrictedVerifier(asset.strName, verifier.verifier_string)){
                        error("%s : Failed to Remove Restricted Verifier from transaction. TXID : %s", __func__, tx.GetHash().GetHex());
                        return DISCONNECT_FAILED;
                    }
                }

                for (auto index : vAssetTxIndex) {
                    CAssetTransfer transfer;
                    std::string strAddress;
                    if (!TransferAssetFromScript(tx.vout[index].scriptPubKey, transfer, strAddress)) {
                        error("%s : Failed to get transfer asset from transaction. CTxOut : %s", __func__,
                              tx.vout[index].ToString());
                        return DISCONNECT_FAILED;
                    }

                    COutPoint out(hash, index);
                    if (!assetsCache->RemoveTransfer(transfer, strAddress, out)) {
                        error("%s : Failed to Remove the transfer of an asset. Asset Name : %s, COutPoint : %s",
                              __func__,
                              transfer.strName, out.ToString());
                        return DISCONNECT_FAILED;
                    }

                    // Undo messages
                    if (AreMessagesDeployed() && fMessaging && databaseMessaging && !transfer.message.empty() &&
                        (IsAssetNameAnOwner(transfer.strName) || IsAssetNameAnMsgChannel(transfer.strName))) {

                        LOCK(cs_messaging);
                        if (IsChannelSubscribed(transfer.strName)) {
                            OrphanMessage(COutPoint(hash, index));
                        }
                    }
                }

                if (AreRestrictedAssetsDeployed()) {
                    // Because of the strict rules for allowing the null asset tx types into a transaction.
                    // We know that if these are in a transaction, that they are valid null asset tx, and can be reversed
                    for (auto index: vNullAssetTxIndex) {
                        CScript script = tx.vout[index].scriptPubKey;

                        if (script.IsNullAssetTxDataScript()) {
                            CNullAssetTxData data;
                            std::string address;
                            if (!AssetNullDataFromScript(script, data, address)) {
                                error("%s : Failed to get null asset data from transaction. CTxOut : %s", __func__,
                                      tx.vout[index].ToString());
                                return DISCONNECT_FAILED;
                            }

                            AssetType type;
                            IsAssetNameValid(data.asset_name, type);

                            // Handle adding qualifiers to addresses
                            if (type == AssetType::QUALIFIER || type == AssetType::SUB_QUALIFIER) {
                                if (!assetsCache->RemoveQualifierAddress(data.asset_name, address, data.flag ? QualifierType::ADD_QUALIFIER : QualifierType::REMOVE_QUALIFIER)) {
                                    error("%s : Failed to remove qualifier from address, Qualifier : %s, Flag Removing : %d, Address : %s",
                                          __func__, data.asset_name, data.flag, address);
                                    return DISCONNECT_FAILED;
                                }
                            // Handle adding restrictions to addresses
                            } else if (type == AssetType::RESTRICTED) {
                                if (!assetsCache->RemoveRestrictedAddress(data.asset_name, address, data.flag ? RestrictedType::FREEZE_ADDRESS : RestrictedType::UNFREEZE_ADDRESS)) {
                                    error("%s : Failed to remove restriction from address, Restriction : %s, Flag Removing : %d, Address : %s",
                                          __func__, data.asset_name, data.flag, address);
                                    return DISCONNECT_FAILED;
                                }
                            }
                        } else if (script.IsNullGlobalRestrictionAssetTxDataScript()) {
                            CNullAssetTxData data;
                            std::string address;
                            if (!GlobalAssetNullDataFromScript(script, data)) {
                                error("%s : Failed to get global null asset data from transaction. CTxOut : %s", __func__,
                                      tx.vout[index].ToString());
                                return DISCONNECT_FAILED;
                            }

                            if (!assetsCache->RemoveGlobalRestricted(data.asset_name, data.flag ? RestrictedType::GLOBAL_FREEZE : RestrictedType::GLOBAL_UNFREEZE)) {
                                error("%s : Failed to remove global restriction from cache. Asset Name: %s, Flag Removing %d", __func__, data.asset_name, data.flag);
                                return DISCONNECT_FAILED;
                            }
                        } else if (script.IsNullAssetVerifierTxDataScript()) {
                            // These are handled in the undo restricted asset issuance, and restricted asset reissuance
                            continue;
                        }
                    }
                }
                /** ASSET END */
            }
        }

        // restore inputs
        if (i > 0) { // not coinbases
            CTxUndo &txundo = blockUndo.vtxundo[i-1];
            if (txundo.vprevout.size() != tx.vin.size()) {
                error("DisconnectBlock(): transaction and undo data inconsistent");
                return DISCONNECT_FAILED;
            }
            for (unsigned int j = tx.vin.size(); j-- > 0;) {
                const COutPoint& out = tx.vin[j].prevout;
                int undoHeight = txundo.vprevout[j].nHeight;
                Coin &undo = txundo.vprevout[j];
                int res = ApplyTxInUndo(std::move(undo), view, out, assetsCache); /* ASSET START */ /* Pass assetsCache into ApplyTxInUndo function */ /* ASSET END */
                if (res == DISCONNECT_FAILED)
                    return DISCONNECT_FAILED;
                fClean = fClean && res != DISCONNECT_UNCLEAN;

                const CTxIn input = tx.vin[j];

                if (fSpentIndex) {
                    // undo and delete the spent index
                    spentIndex.push_back(std::make_pair(CSpentIndexKey(input.prevout.hash, input.prevout.n), CSpentIndexValue()));
                }

                if (fAddressIndex) {
                    const Coin& coin = view.AccessCoin(tx.vin[j].prevout);
                    const CTxOut& prevout = coin.out;
                    if (prevout.scriptPubKey.IsPayToScriptHash()) {
                        // Remove BDAP portion of the script
                        CScript scriptPubKey;
                        CScript scriptPubKeyOut;
                        if (RemoveBDAPScript(prevout.scriptPubKey, scriptPubKeyOut)) {
                            scriptPubKey = scriptPubKeyOut;
                        } else {
                            scriptPubKey = prevout.scriptPubKey;
                        }

                        std::vector<unsigned char> hashBytes(scriptPubKey.begin() + 2, scriptPubKey.begin() + 22);
                        // undo spending activity
                        addressIndex.push_back(std::make_pair(CAddressIndexKey(2, uint160(hashBytes), pindex->nHeight, i, hash, j, true), prevout.nValue * -1));
                        // restore unspent index
                        addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(2, uint160(hashBytes), input.prevout.hash, input.prevout.n), CAddressUnspentValue(prevout.nValue, scriptPubKey, undoHeight)));

                    } else if (prevout.scriptPubKey.IsPayToPublicKeyHash()) {
                        // Remove BDAP portion of the script
                        CScript scriptPubKey;
                        CScript scriptPubKeyOut;
                        if (RemoveBDAPScript(prevout.scriptPubKey, scriptPubKeyOut)) {
                            scriptPubKey = scriptPubKeyOut;
                        } else {
                            scriptPubKey = prevout.scriptPubKey;
                        }

                        std::vector<unsigned char> hashBytes(scriptPubKey.begin() + 3, scriptPubKey.begin() + 23);
                        // undo spending activity
                        addressIndex.push_back(std::make_pair(CAddressIndexKey(1, uint160(hashBytes), pindex->nHeight, i, hash, j, true), prevout.nValue * -1));
                        // restore unspent index
                        addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(1, uint160(hashBytes), input.prevout.hash, input.prevout.n), CAddressUnspentValue(prevout.nValue, scriptPubKey, undoHeight)));
                    } else {

                        /** ASSET START */
                        if (AreAssetsDeployed()) {
                            std::string assetName;
                            CAmount assetAmount;
                            uint160 hashBytes;

                            if (ParseAssetScript(prevout.scriptPubKey, hashBytes, assetName, assetAmount)) {
//                                std::cout << "ConnectBlock(): pushing assets onto addressIndex: " << "1" << ", " << hashBytes.GetHex() << ", " << assetName << ", " << pindex->nHeight
//                                          << ", " << i << ", " << hash.GetHex() << ", " << j << ", " << "true" << ", " << assetAmount * -1 << std::endl;

                                // undo spending activity
                                addressIndex.push_back(std::make_pair(
                                        CAddressIndexKey(1, uint160(hashBytes), assetName, pindex->nHeight, i, hash, j,
                                                         true), assetAmount * -1));

                                // restore unspent index
                                addressUnspentIndex.push_back(std::make_pair(
                                        CAddressUnspentKey(1, uint160(hashBytes), assetName, input.prevout.hash,
                                                           input.prevout.n),
                                        CAddressUnspentValue(assetAmount, prevout.scriptPubKey, undo.nHeight)));
                            } else {
                                continue;
                            }
                        }
                        /** ASSET END */
                    }
                }
            }
            // At this point, all of txundo.vprevout should have been moved out.
        }
    }


    // move best block pointer to prevout block
    view.SetBestBlock(pindex->pprev->GetBlockHash());

    if (fAddressIndex) {
        if (!pblocktree->EraseAddressIndex(addressIndex)) {
            AbortNode(state, "Failed to delete address index");
            return DISCONNECT_FAILED;
        }
        if (!pblocktree->UpdateAddressUnspentIndex(addressUnspentIndex)) {
            AbortNode(state, "Failed to write address unspent index");
            return DISCONNECT_FAILED;
        }
    }

    return fClean ? DISCONNECT_OK : DISCONNECT_UNCLEAN;
}

void static FlushBlockFile(bool fFinalize = false)
{
    LOCK(cs_LastBlockFile);

    CDiskBlockPos posOld(nLastBlockFile, 0);

    FILE* fileOld = OpenBlockFile(posOld);
    if (fileOld) {
        if (fFinalize)
            TruncateFile(fileOld, vinfoBlockFile[nLastBlockFile].nSize);
        FileCommit(fileOld);
        fclose(fileOld);
    }

    fileOld = OpenUndoFile(posOld);
    if (fileOld) {
        if (fFinalize)
            TruncateFile(fileOld, vinfoBlockFile[nLastBlockFile].nUndoSize);
        FileCommit(fileOld);
        fclose(fileOld);
    }
}

bool FindUndoPos(CValidationState& state, int nFile, CDiskBlockPos& pos, unsigned int nAddSize);

static CCheckQueue<CScriptCheck> scriptcheckqueue(128);

void ThreadScriptCheck()
{
    RenameThread("dynamic-scriptch");
    scriptcheckqueue.Thread();
}

// Protected by cs_main
VersionBitsCache versionbitscache;

int32_t ComputeBlockVersion(const CBlockIndex* pindexPrev, const Consensus::Params& params, bool fAssumeDynodeIsUpgraded)
{
    LOCK(cs_main);
    int32_t nVersion = VERSIONBITS_TOP_BITS;

    for (int i = 0; i < (int)Consensus::MAX_VERSION_BITS_DEPLOYMENTS; i++) {
        Consensus::DeploymentPos pos = Consensus::DeploymentPos(i);
        ThresholdState state = VersionBitsState(pindexPrev, params, pos, versionbitscache);
        const struct BIP9DeploymentInfo& vbinfo = VersionBitsDeploymentInfo[pos];
        if (vbinfo.check_dn_protocol && state == THRESHOLD_STARTED && !fAssumeDynodeIsUpgraded) {
            CScript payee;
            dynode_info_t dnInfo;
            if (!dnpayments.GetBlockPayee(pindexPrev->nHeight + 1, payee)) {
                // no votes for this block
                continue;
            }
            if (!dnodeman.GetDynodeInfo(payee, dnInfo)) {
                // unknown dynode
                continue;
            }
        }
        if (state == THRESHOLD_LOCKED_IN || state == THRESHOLD_STARTED) {
            nVersion |= VersionBitsMask(params, (Consensus::DeploymentPos)i);
        }
    }

    return nVersion;
}

bool GetBlockHash(uint256& hashRet, int nBlockHeight)
{
    LOCK(cs_main);
    if (chainActive.Tip() == nullptr)
        return false;
    if (nBlockHeight < -1 || nBlockHeight > chainActive.Height())
        return false;
    if (nBlockHeight == -1)
        nBlockHeight = chainActive.Height();
    hashRet = chainActive[nBlockHeight]->GetBlockHash();
    return true;
}

bool CheckProofOfStakeAmount(const CBlock& block, const CAmount& blockReward, std::string& strErrorRet) {
    CAmount nInputAmount = 0;
    CTransactionRef stakeTxIn = block.vtx[1];
    // Inputs
    std::vector<CTxIn> vInputs;
    for (const CTxIn& stakeIn : stakeTxIn->vin) {
        vInputs.push_back(stakeIn);
    }
    const bool fHasInputs = !vInputs.empty();
    const CCoinsViewCache coins(pcoinsTip);
    for (const CTransactionRef& tx : block.vtx) {
        if(tx->IsCoinStake())
            continue;
        if(fHasInputs) {
            for (const CTxIn& txIn : vInputs){
                const Coin coin = coins.AccessCoin(txIn.prevout);
                nInputAmount += coin.out.nValue;
            }
        }
    }
    CAmount nStakeReward = block.vtx[1]->GetValueOut() - nInputAmount;
    //LogPrintf("%s: out %d, in %d, blockReward %d\n", __func__, FormatMoney(block.vtx[1]->GetValueOut()), FormatMoney(nInputAmount), FormatMoney(blockReward));
    if (nStakeReward > blockReward) {
        strErrorRet = strprintf("coinbase pays too much (actual=%s vs limit=%s), exceeded block reward", FormatMoney(nStakeReward), FormatMoney(blockReward));
        return false;
    } else {
        return true;
    }
}

/**
 * Threshold condition checker that triggers when unknown versionbits are seen on the network.
 */
class WarningBitsConditionChecker : public AbstractThresholdConditionChecker
{
private:
    int bit;

public:
    WarningBitsConditionChecker(int bitIn) : bit(bitIn) {}

    int64_t BeginTime(const Consensus::Params& params) const override { return 0; }
    int64_t EndTime(const Consensus::Params& params) const override { return std::numeric_limits<int64_t>::max(); }
    int Period(const Consensus::Params& params) const override { return params.nMinerConfirmationWindow; }
    int Threshold(const Consensus::Params& params) const override { return params.nRuleChangeActivationThreshold; }

    bool Condition(const CBlockIndex* pindex, const Consensus::Params& params) const override
    {
        return ((pindex->nVersion & VERSIONBITS_TOP_MASK) == VERSIONBITS_TOP_BITS) &&
               ((pindex->nVersion >> bit) & 1) != 0 &&
               ((ComputeBlockVersion(pindex->pprev, params) >> bit) & 1) == 0;
    }
};

// Protected by cs_main
static ThresholdConditionCache warningcache[VERSIONBITS_NUM_BITS];

static unsigned int GetBlockScriptFlags(const CBlockIndex* pindex, const Consensus::Params& consensusparams) {
    AssertLockHeld(cs_main);

    const CChainParams& chainparams = Params();

    // BIP16 didn't become active until Apr 1 2012
    int64_t nBIP16SwitchTime = 1333238400;
    bool fStrictPayToScriptHash = (pindex->GetBlockTime() >= nBIP16SwitchTime);
    int nLockTimeFlags = 0;

    unsigned int flags = fStrictPayToScriptHash ? SCRIPT_VERIFY_P2SH : SCRIPT_VERIFY_NONE;
    flags |= SCRIPT_VERIFY_DERSIG;
    flags |= SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY;
    flags |= SCRIPT_VERIFY_CHECKSEQUENCEVERIFY;
    nLockTimeFlags |= LOCKTIME_VERIFY_SEQUENCE;

    if (VersionBitsState(pindex->pprev, chainparams.GetConsensus(), Consensus::DEPLOYMENT_BIP147, versionbitscache) == THRESHOLD_ACTIVE) {
        flags |= SCRIPT_VERIFY_NULLDUMMY;
    }

    return flags;
}

static int64_t nTimeCheck = 0;
static int64_t nTimeForks = 0;
static int64_t nTimeVerify = 0;
static int64_t nTimeConnect = 0;
static int64_t nTimeIndex = 0;
static int64_t nTimeCallbacks = 0;
static int64_t nTimeTotal = 0;
static int64_t nBlocksTotal = 0;

/** Apply the effects of this block (with given index) on the UTXO set represented by coins.
 *  Validity checks that depend on the UTXO set are also done; ConnectBlock()
 *  can fail if those validity checks fail (among other reasons). */
static bool ConnectBlock(const CBlock& block, CValidationState& state, CBlockIndex* pindex, CCoinsViewCache& view, const CChainParams& chainparams, CAssetsCache* assetsCache = nullptr, bool fJustCheck = false, bool ignoreAddressIndex = false)
{
    AssertLockHeld(cs_main);

    int64_t nTimeStart = GetTimeMicros();

    // Check it again in case a previous version let a bad block in
    if (!CheckBlock(block, state, chainparams.GetConsensus(), !fJustCheck, !fJustCheck))
        return error("%s: Consensus::CheckBlock: %s", __func__, FormatStateMessage(state));

    if (block.IsProofOfStake()) {
        uint256 hashProofOfStake = uint256(0);
        std::unique_ptr<CStakeInput> stake;
        CBlockIndex* pindexPrev = pindex->pprev;
        if (!CheckProofOfStake(block, hashProofOfStake, stake, pindexPrev->nHeight))
            return state.DoS(100, error("%s: proof of stake check failed", __func__));

        if (!stake)
            return error("%s: null stake ptr", __func__);

        uint256 hash = block.GetHash();
        if(!mapProofOfStake.count(hash)) // add to mapProofOfStake
            mapProofOfStake.insert(std::make_pair(hash, hashProofOfStake));
    }

    // verify that the view's current state corresponds to the previous block
    uint256 hashPrevBlock = pindex->pprev == nullptr ? uint256() : pindex->pprev->GetBlockHash();
    assert(hashPrevBlock == view.GetBestBlock());

    // Special case for the genesis block, skipping connection of its transactions
    // (its coinbase is unspendable)
    if (block.GetHash() == chainparams.GetConsensus().hashGenesisBlock) {
        if (!fJustCheck)
            view.SetBestBlock(pindex->GetBlockHash());
        return true;
    }

    nBlocksTotal++;

    bool fScriptChecks = true;
    if (!hashAssumeValid.IsNull()) {
        // We've been configured with the hash of a block which has been externally verified to have a valid history.
        // A suitable default value is included with the software and updated from time to time.  Because validity
        //  relative to a piece of software is an objective fact these defaults can be easily reviewed.
        // This setting doesn't force the selection of any particular chain but makes validating some faster by
        //  effectively caching the result of part of the verification.
        BlockMap::const_iterator it = mapBlockIndex.find(hashAssumeValid);
        if (it != mapBlockIndex.end()) {
            if (it->second->GetAncestor(pindex->nHeight) == pindex &&
                pindexBestHeader->GetAncestor(pindex->nHeight) == pindex &&
                pindexBestHeader->nChainWork >= int64_t(chainparams.GetConsensus().nMinimumChainWork)) {

                // This block is a member of the assumed verified chain and an ancestor of the best header.
                // The equivalent time check discourages hashpower from extorting the network via DOS attack
                // into accepting an invalid block through telling users they must manually set assumevalid.
                // Requiring a software change or burying the invalid block, regardless of the setting, makes
                // it hard to hide the implication of the demand.  This also avoids having release candidates
                // that are hardly doing any signature verification at all in testing without having to
                // artificially set the default assumed verified block further back.
                // The test against nMinimumChainWork prevents the skipping when denied access to any chain at
                // least as good as the expected chain.
                fScriptChecks = (GetBlockProofEquivalentTime(*pindexBestHeader, *pindex, *pindexBestHeader, chainparams.GetConsensus()) <= 60 * 60 * 24 * 7 * 2);
            }
        }
    }

    int64_t nTime1 = GetTimeMicros();
    nTimeCheck += nTime1 - nTimeStart;
    LogPrint("bench", "    - Sanity checks: %.2fms [%.2fs]\n", MILLI * (nTime1 - nTimeStart), nTimeCheck * MICRO);

    // make sure old budget is the real one
    if (pindex->nHeight == chainparams.GetConsensus().nSuperblockStartBlock &&
        chainparams.GetConsensus().nSuperblockStartHash != uint256() &&
        block.GetHash() != chainparams.GetConsensus().nSuperblockStartHash)
        return state.DoS(100, error("ConnectBlock(): invalid superblock start"),
            REJECT_INVALID, "bad-sb-start");

    // Get the script flags for this block
    unsigned int flags = GetBlockScriptFlags(pindex, chainparams.GetConsensus());

    int64_t nTime2 = GetTimeMicros();
    nTimeForks += nTime2 - nTime1;
    LogPrint("bench", "    - Fork checks: %.2fms [%.2fs (%.2fms/blk)]\n", MILLI * (nTime2 - nTime1), nTimeForks * MICRO, nTimeForks * MILLI / nBlocksTotal);

    CBlockUndo blockundo;
    std::vector<std::pair<std::string, CBlockAssetUndo> > vUndoAssetData;

    CCheckQueueControl<CScriptCheck> control(fScriptChecks && nScriptCheckThreads ? &scriptcheckqueue : nullptr);

    std::vector<uint256> vOrphanErase;
    std::vector<int> prevheights;
    CAmount nFees = 0;
    int nInputs = 0;
    unsigned int nSigOps = 0;
    CDiskTxPos pos(pindex->GetBlockPos(), GetSizeOfCompactSize(block.vtx.size()));
    std::vector<std::pair<uint256, CDiskTxPos> > vPos;
    vPos.reserve(block.vtx.size());
    blockundo.vtxundo.reserve(block.vtx.size() - 1);
    std::vector<PrecomputedTransactionData> txdata;
    txdata.reserve(block.vtx.size()); // Required so that pointers to individual PrecomputedTransactionData don't get invalidated

    std::vector<std::pair<CAddressIndexKey, CAmount> > addressIndex;
    std::vector<std::pair<CAddressUnspentKey, CAddressUnspentValue> > addressUnspentIndex;
    std::vector<std::pair<CSpentIndexKey, CSpentIndexValue> > spentIndex;

/* ASSET START */
    std::set<CMessage> setMessages;
    std::vector<std::pair<std::string, CNullAssetTxData>> myNullAssetData;
/* ASSET END */

    for (unsigned int i = 0; i < block.vtx.size(); i++) {
        const CTransaction& tx = *block.vtx[i];
        const uint256 txhash = tx.GetHash();

        nInputs += tx.vin.size();
        nSigOps += GetLegacySigOpCount(tx);
        if (nSigOps > MAX_BLOCK_SIGOPS_COST)
            return state.DoS(100, error("ConnectBlock(): too many sigops"),
                REJECT_INVALID, "bad-blk-sigops");

        if (!tx.IsCoinBase()) {
            if (!view.HaveInputs(tx))
                return state.DoS(100, error("ConnectBlock(): inputs missing/spent"),
                    REJECT_INVALID, "bad-txns-inputs-missingorspent");

            /** ASSET START */
            if (!AreAssetsDeployed()) {
                for (auto out : tx.vout)
                    if (out.scriptPubKey.IsAssetScript())
                        return state.DoS(100, error("%s : Received Block with tx that contained an asset when assets wasn't active", __func__), REJECT_INVALID, "bad-txns-assets-not-active");
                    else if (out.scriptPubKey.IsNullAsset())
                        return state.DoS(100, error("%s : Received Block with tx that contained an null asset data tx when assets wasn't active", __func__), REJECT_INVALID, "bad-txns-null-data-assets-not-active");
            }

            if (AreAssetsDeployed()) {
                std::vector<std::pair<std::string, uint256>> vReissueAssets;
                if (!Consensus::CheckTxAssets(tx, state, view, assetsCache, false, vReissueAssets, false, &setMessages, block.nTime, &myNullAssetData)) {
                    state.SetFailedTransaction(tx.GetHash());
                    return error("%s: Consensus::CheckTxAssets: %s, %s", __func__, tx.GetHash().ToString(),
                                 FormatStateMessage(state));
                }
            }

            /** ASSET END */

            // Check that transaction is BIP68 final
            // BIP68 lock checks (as opposed to nLockTime checks) must
            // be in ConnectBlock because they require the UTXO set
            prevheights.resize(tx.vin.size());
            for (size_t j = 0; j < tx.vin.size(); j++) {
                prevheights[j] = view.AccessCoin(tx.vin[j].prevout).nHeight;
            }

            int nLockTimeFlags = 0;

            if (!SequenceLocks(tx, nLockTimeFlags, &prevheights, *pindex)) {
                return state.DoS(100, error("%s: contains a non-BIP68-final transaction", __func__),
                    REJECT_INVALID, "bad-txns-nonfinal");
            }
            if (fAddressIndex || fSpentIndex) {
                for (size_t j = 0; j < tx.vin.size(); j++) {
                    const CTxIn input = tx.vin[j];
                    const Coin& coin = view.AccessCoin(tx.vin[j].prevout);
                    const CTxOut& prevout = coin.out;
                    uint160 hashBytes;
                    int addressType = 0;
                    bool isAsset = false;
                    std::string assetName;
                    CAmount assetAmount;

                    if (prevout.scriptPubKey.IsPayToScriptHash()) {
                        hashBytes = uint160(std::vector <unsigned char>(prevout.scriptPubKey.begin()+2, prevout.scriptPubKey.begin()+22));
                        addressType = 2;
                    } else if (prevout.scriptPubKey.IsPayToPublicKeyHash()) {
                        hashBytes = uint160(std::vector <unsigned char>(prevout.scriptPubKey.begin()+3, prevout.scriptPubKey.begin()+23));
                        addressType = 1;
                    } else if (prevout.scriptPubKey.IsPayToPublicKey()) {
                        hashBytes = Hash160(prevout.scriptPubKey.begin() + 1, prevout.scriptPubKey.end() - 1);
                        addressType = 1;
                    } else {
                        /* ASSET START */
                        if (AreAssetsDeployed()) {
                            hashBytes.SetNull();
                            addressType = 0;

                            if (ParseAssetScript(prevout.scriptPubKey, hashBytes, assetName, assetAmount)) {
                                addressType = 1;
                                isAsset = true;
                            }
                        }
                        /* ASSET END */
                    }

                    if (fAddressIndex && addressType > 0) {
                    /** ASSET START */
                        if (isAsset) {
//                            std::cout << "ConnectBlock(): pushing assets onto addressIndex: " << "1" << ", " << hashBytes.GetHex() << ", " << assetName << ", " << pindex->nHeight
//                                      << ", " << i << ", " << txhash.GetHex() << ", " << j << ", " << "true" << ", " << assetAmount * -1 << std::endl;

                            // record spending activity
                            addressIndex.push_back(std::make_pair(CAddressIndexKey(addressType, hashBytes, assetName, pindex->nHeight, i, txhash, j, true), assetAmount * -1));

                            // remove address from unspent index
                            addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(addressType, hashBytes, assetName, input.prevout.hash, input.prevout.n), CAddressUnspentValue()));
                        } else {
                            // record spending activity
                            addressIndex.push_back(std::make_pair(CAddressIndexKey(addressType, hashBytes, pindex->nHeight, i, txhash, j, true), prevout.nValue * -1));

                            // remove address from unspent index
                            addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(addressType, hashBytes, input.prevout.hash, input.prevout.n), CAddressUnspentValue()));
                        }
                    }
                    /** ASSET END */

                    if (fSpentIndex) {
                        // add the spent index to determine the txid and input that spent an output
                        // and to find the amount and address from an input
                        spentIndex.push_back(std::make_pair(CSpentIndexKey(input.prevout.hash, input.prevout.n), CSpentIndexValue(txhash, j, pindex->nHeight, prevout.nValue, addressType, hashBytes)));
                    }
                }
            }

            int64_t nBIP16SwitchTime = 1333238400;
            bool fStrictPayToScriptHash = (pindex->GetBlockTime() >= nBIP16SwitchTime);

            if (fStrictPayToScriptHash) {
                // Add in sigops done by pay-to-script-hash inputs;
                // this is to prevent a "rogue miner" from creating
                // an incredibly-expensive-to-validate block.
                nSigOps += GetP2SHSigOpCount(tx, view);
                if (nSigOps > MAX_BLOCK_SIGOPS_COST)
                    return state.DoS(100, error("ConnectBlock(): too many sigops"),
                        REJECT_INVALID, "bad-blk-sigops");
            }

            if (!tx.IsCoinStake())
                nFees += view.GetValueIn(tx) - tx.GetValueOut();

        txdata.emplace_back(tx);
        if (!tx.IsCoinBase())
            {
                std::vector<CScriptCheck> vChecks;
                bool fCacheResults = fJustCheck; /* Don't cache results if we're actually connecting blocks (still consult the cache, though) */
                if (!CheckInputs(tx, state, view, fScriptChecks, flags, fCacheResults, fCacheResults, txdata[i], nScriptCheckThreads ? &vChecks : nullptr))
                    return error("ConnectBlock(): CheckInputs on %s failed with %s",
                        tx.GetHash().ToString(), FormatStateMessage(state));
                control.Add(vChecks);
            }
        }

        if (fAddressIndex) {
            for (unsigned int k = 0; k < tx.vout.size(); k++) {
                const CTxOut& out = tx.vout[k];

                if (out.scriptPubKey.IsPayToScriptHash()) {
                    // Remove BDAP portion of the script
                    CScript scriptPubKey;
                    CScript scriptPubKeyOut;
                    if (RemoveBDAPScript(out.scriptPubKey, scriptPubKeyOut)) {
                        scriptPubKey = scriptPubKeyOut;
                    } else {
                        scriptPubKey = out.scriptPubKey;
                    }

                    std::vector<unsigned char> hashBytes(scriptPubKey.begin() + 2, scriptPubKey.begin() + 22);
                    // record receiving activity
                    addressIndex.push_back(std::make_pair(CAddressIndexKey(2, uint160(hashBytes), pindex->nHeight, i, txhash, k, false), out.nValue));
                    // record unspent output
                    addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(2, uint160(hashBytes), txhash, k), CAddressUnspentValue(out.nValue, scriptPubKey, pindex->nHeight)));
                } else if (out.scriptPubKey.IsPayToPublicKeyHash()) {
                    // Remove BDAP portion of the script
                    CScript scriptPubKey;
                    CScript scriptPubKeyOut;
                    if (RemoveBDAPScript(out.scriptPubKey, scriptPubKeyOut)) {
                        scriptPubKey = scriptPubKeyOut;
                    } else {
                        scriptPubKey = out.scriptPubKey;
                    }

                    std::vector<unsigned char> hashBytes(scriptPubKey.begin() + 3, scriptPubKey.begin() + 23);
                    // record receiving activity
                    addressIndex.push_back(std::make_pair(CAddressIndexKey(1, uint160(hashBytes), pindex->nHeight, i, txhash, k, false), out.nValue));
                    // record unspent output
                    addressUnspentIndex.push_back(std::make_pair(CAddressUnspentKey(1, uint160(hashBytes), txhash, k), CAddressUnspentValue(out.nValue, scriptPubKey, pindex->nHeight)));
                } else {
                    /** ASSET START */
                    if (AreAssetsDeployed()) {
                        std::string assetName;
                        CAmount assetAmount;
                        uint160 hashBytes;

                        if (ParseAssetScript(out.scriptPubKey, hashBytes, assetName, assetAmount)) {
//                            std::cout << "ConnectBlock(): pushing assets onto addressIndex: " << "1" << ", " << hashBytes.GetHex() << ", " << assetName << ", " << pindex->nHeight
//                                      << ", " << i << ", " << txhash.GetHex() << ", " << k << ", " << "true" << ", " << assetAmount << std::endl;

                            // record receiving activity
                            addressIndex.push_back(std::make_pair(
                                    CAddressIndexKey(1, hashBytes, assetName, pindex->nHeight, i, txhash, k, false),
                                    assetAmount));

                            // record unspent output
                            addressUnspentIndex.push_back(
                                    std::make_pair(CAddressUnspentKey(1, hashBytes, assetName, txhash, k),
                                                   CAddressUnspentValue(assetAmount, out.scriptPubKey,
                                                                        pindex->nHeight)));
                        }
                    } else {
                        continue;
                    }
                    /** ASSET END */
                }
            }
        }

        CCoinsViewCache viewCoinCache(pcoinsTip);
        CTransactionRef ptx = MakeTransactionRef(tx);

        if (tx.nVersion == BDAP_TX_VERSION && !ValidateBDAPInputs(ptx, state, viewCoinCache, block, fJustCheck, pindex->nHeight)) {
            return error("ConnectBlock(): ValidateBDAPInputs on block %s failed\n", block.GetHash().ToString());
        }

        CTxUndo undoDummy;
        if (i > 0) {
            blockundo.vtxundo.push_back(CTxUndo());
        }
        /** ASSET START */
        // Create the basic empty string pair for the undoblock
        std::pair<std::string, CBlockAssetUndo> undoPair = std::make_pair("", CBlockAssetUndo());
        std::pair<std::string, CBlockAssetUndo>* undoAssetData = &undoPair;
        /** ASSET END */
        UpdateCoins(tx, view, i == 0 ? undoDummy : blockundo.vtxundo.back(), pindex->nHeight, block.GetHash(), assetsCache, undoAssetData);

        /** ASSET START */
        if (!undoAssetData->first.empty()) {
            vUndoAssetData.emplace_back(*undoAssetData);
        }
        /** ASSET END */

        vPos.push_back(std::make_pair(tx.GetHash(), pos));
        pos.nTxOffset += ::GetSerializeSize(tx, SER_DISK, CLIENT_VERSION);
    }
    int64_t nTime3 = GetTimeMicros();
    nTimeConnect += nTime3 - nTime2;
    LogPrint("bench", "      - Connect %u transactions: %.2fms (%.3fms/tx, %.3fms/txin) [%.2fs]\n", (unsigned)block.vtx.size(), MILLI * (nTime3 - nTime2), MILLI * (nTime3 - nTime2) / block.vtx.size(), nInputs <= 1 ? 0 : MILLI * (nTime3 - nTime2) / (nInputs - 1), nTimeConnect * MICRO);

    // DYN : MODIFIED TO CHECK DYNODE PAYMENTS AND SUPERBLOCKS

    // It's possible that we simply don't have enough data and this could fail
    // (i.e. block itself could be a correct one and we need to store it),
    // that's why this is in ConnectBlock. Could be the other way around however -
    // the peer who sent us this block is missing some data and wasn't able
    // to recognize that block is actually invalid.
    // TODO: resync data (both ways?) and try to reprocess this block later.
    bool fDynodePaid = false;

    if (chainActive.Height() > Params().GetConsensus().nDynodePaymentsStartBlock) {
        fDynodePaid = true;
    } else if (chainActive.Height() <= Params().GetConsensus().nDynodePaymentsStartBlock) {
        fDynodePaid = false;
    }

    // BEGIN FLUID
    CAmount nExpectedBlockValue;
    std::string strError = "";
    {
        CBlockIndex* prevIndex = pindex->pprev;
        CAmount newMiningReward = 0;
        CAmount newDynodeReward = 0;
        if (block.IsProofOfWork()) {
            newMiningReward = GetFluidMiningReward(pindex->nHeight);
            if (fDynodePaid)
                newDynodeReward = GetFluidDynodeReward(pindex->nHeight);
        } else {
            newMiningReward = GetFluidStakingReward(pindex->nHeight);
        }
        CAmount newMintIssuance = 0;
        CDynamicAddress mintAddress;
        if (prevIndex->nHeight + 1 >= fluid.FLUID_ACTIVATE_HEIGHT) {
            CFluidMint fluidMint;
            if (GetMintingInstructions(pindex->nHeight, fluidMint)) {
                newMintIssuance = fluidMint.MintAmount;
                mintAddress = fluidMint.GetDestinationAddress();
                LogPrintf("ConnectBlock, GetMintingInstructions MintAmount = %u\n", fluidMint.MintAmount);
            }
        }
        nExpectedBlockValue = newMintIssuance + newMiningReward + newDynodeReward;

        if (block.IsProofOfWork()) {
            // check Proof-of-Work and Dynode amount paid
            if (!IsBlockValueValid(block, pindex->nHeight, nExpectedBlockValue, strError)) {
                return state.DoS(0, error("ConnectBlock(DYN): %s", strError), REJECT_INVALID, "bad-cb-amount");
            }
        } else {
            // check Proof-of-Stake amount paid
            if (!CheckProofOfStakeAmount(block, nExpectedBlockValue, strError)) {
                return state.DoS(0, error("ConnectBlock(DYN): %s", strError), REJECT_INVALID, "bad-cb-stake-amount");
            }
        }
        // Do not pay Dynodes for Proof-of-Stake blocks
        if (block.IsProofOfWork() && !IsBlockPayeeValid(*block.vtx[0], pindex->nHeight, nExpectedBlockValue)) {
            mapRejectedBlocks.insert(std::make_pair(block.GetHash(), GetTime()));
            return state.DoS(0, error("ConnectBlock(DYN): couldn't find Dynode or Superblock payments"),
                REJECT_INVALID, "bad-cb-payee");
        }
    }
    for (unsigned int i = 0; i < block.vtx.size(); i++) {
        const CTransaction& tx = *block.vtx[i];
        CScript scriptFluid;
        if (IsTransactionFluid(tx, scriptFluid)) {
            int OpCode = GetFluidOpCode(scriptFluid);
            if (OpCode == OP_REWARD_DYNODE) {
                CFluidDynode fluidDynode(scriptFluid);
                fluidDynode.nHeight = pindex->nHeight;
                fluidDynode.txHash = tx.GetHash();
                if (CheckFluidDynodeDB()) {
                    if (!CheckSignatureQuorum(fluidDynode.FluidScript, strError)) {
                        return state.DoS(0, error("ConnectBlock(DYN): %s", strError), REJECT_INVALID, "invalid-fluid-dynode-address-signature");
                    }
                    pFluidDynodeDB->AddFluidDynodeEntry(fluidDynode, OP_REWARD_DYNODE);
                }
            } else if (OpCode == OP_REWARD_MINING) {
                CFluidMining fluidMining(scriptFluid);
                fluidMining.nHeight = pindex->nHeight;
                fluidMining.txHash = tx.GetHash();
                if (CheckFluidMiningDB()) {
                    if (!CheckSignatureQuorum(fluidMining.FluidScript, strError)) {
                        return state.DoS(0, error("ConnectBlock(DYN): %s", strError), REJECT_INVALID, "invalid-fluid-mining-address-signature");
                    }
                    pFluidMiningDB->AddFluidMiningEntry(fluidMining, OP_REWARD_MINING);
                }
            } else if (OpCode == OP_REWARD_STAKE) {
                CFluidStaking fluidStaking(scriptFluid);
                fluidStaking.nHeight = pindex->nHeight;
                fluidStaking.txHash = tx.GetHash();
                if (CheckFluidStakingDB()) {
                    if (!CheckSignatureQuorum(fluidStaking.FluidScript, strError)) {
                        return state.DoS(0, error("ConnectBlock(DYN): %s", strError), REJECT_INVALID, "invalid-fluid-staking-address-signature");
                    }
                    pFluidStakingDB->AddFluidStakingEntry(fluidStaking, OP_REWARD_STAKE);
                }
            } else if (OpCode == OP_MINT) {
                CFluidMint fluidMint(scriptFluid);
                fluidMint.nHeight = pindex->nHeight;
                fluidMint.txHash = tx.GetHash();
                if (CheckFluidMintDB()) {
                    if (!CheckSignatureQuorum(fluidMint.FluidScript, strError)) {
                        return state.DoS(0, error("ConnectBlock(DYN): %s", strError), REJECT_INVALID, "invalid-fluid-mint-address-signature");
                    }
                    pFluidMintDB->AddFluidMintEntry(fluidMint, OP_MINT);
                }
            } else if (OpCode == OP_BDAP_REVOKE) {
                if (!CheckSignatureQuorum(FluidScriptToCharVector(scriptFluid), strError))
                    return state.DoS(0, error("%s: %s", __func__, strError), REJECT_INVALID, "invalid-fluid-ban-address-signature");

                //if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
                //    return state.DoS(0, error("%s: BDAP spork is inactive.", __func__), REJECT_INVALID, "bdap-spork-inactive");

                std::vector<CDomainEntry> vBanAccounts;
                if (!fluid.CheckAccountBanScript(scriptFluid, tx.GetHash(), pindex->nHeight, vBanAccounts, strError))
                    return state.DoS(0, error("%s -- CheckAccountBanScript failed: %s", __func__, strError), REJECT_INVALID, "fluid-ban-script-invalid");

                int64_t nTimeStamp;
                std::vector<std::vector<unsigned char>> vSovereignAddresses;
                if (fluid.ExtractTimestampWithAddresses("OP_BDAP_REVOKE", scriptFluid, nTimeStamp, vSovereignAddresses)) {
                    for (const CDomainEntry& entry : vBanAccounts) {
                        LogPrintf("%s -- Fluid command banning account %s\n", __func__, entry.GetFullObjectPath());
                        if (!DeleteDomainEntry(entry))
                            LogPrintf("%s -- Error deleting account %s\n", __func__, entry.GetFullObjectPath());

                        CBanAccount banAccount(scriptFluid, entry.vchFullObjectPath(), nTimeStamp, vSovereignAddresses, tx.GetHash(), pindex->nHeight);
                        AddBanAccountEntry(banAccount);
                    }
                }
            } else {
                std::string strFluidOpScript = ScriptToAsmStr(scriptFluid);
                std::string strOperationCode = GetRidOfScriptStatement(strFluidOpScript, 0);
                return state.DoS(100, error("%s -- Invalid fluid operation code %s (%d)", __func__, strOperationCode, OpCode), REJECT_INVALID, "invalid-fluid-operation-code");
            }
        }
    }
    // END FLUID

    if (!control.Wait())
        return state.DoS(100, false);

    int64_t nTime4 = GetTimeMicros();
    nTimeVerify += nTime4 - nTime2;
    LogPrint("bench", "    - Verify %u txins: %.2fms (%.3fms/txin) [%.2fs]\n", nInputs - 1, MILLI * (nTime4 - nTime2), nInputs <= 1 ? 0 : MILLI * (nTime4 - nTime2) / (nInputs - 1), nTimeVerify * MICRO);

    if (fJustCheck)
        return true;

    if (block.IsProofOfStake() && pindex->nStakeModifier == 0) {
        pindex->nStakeModifier = ComputeStakeModifier(pindex->pprev, block.vtx[1]->vin[0].prevout.hash);
        pindex->prevoutStake = block.vtx[1]->vin[0].prevout;
    }

    // Write undo information to disk
    if (pindex->GetUndoPos().IsNull() || !pindex->IsValid(BLOCK_VALID_SCRIPTS)) {
        if (pindex->GetUndoPos().IsNull()) {
            CDiskBlockPos _pos;
            if (!FindUndoPos(state, pindex->nFile, _pos, ::GetSerializeSize(blockundo, SER_DISK, CLIENT_VERSION) + 40))
                return error("ConnectBlock(): FindUndoPos failed");
            if (!UndoWriteToDisk(blockundo, _pos, pindex->pprev->GetBlockHash(), chainparams.MessageStart()))
                return AbortNode(state, "Failed to write undo data");

            // update nUndoPos in block index
            pindex->nUndoPos = _pos.nPos;
            pindex->nStatus |= BLOCK_HAVE_UNDO;
        }

        if (vUndoAssetData.size()) {
            if (!passetsdb->WriteBlockUndoAssetData(block.GetHash(), vUndoAssetData))
                return AbortNode(state, "Failed to write asset undo data");
        }

        pindex->RaiseValidity(BLOCK_VALID_SCRIPTS);
        setDirtyBlockIndex.insert(pindex);
    }

    if (fTxIndex)
        if (!pblocktree->WriteTxIndex(vPos))
            return AbortNode(state, "Failed to write transaction index");

    if (!ignoreAddressIndex && fAddressIndex) {
        if (!pblocktree->WriteAddressIndex(addressIndex)) {
            return AbortNode(state, "Failed to write address index");
        }

        if (!pblocktree->UpdateAddressUnspentIndex(addressUnspentIndex)) {
            return AbortNode(state, "Failed to write address unspent index");
        }
    }

    if (!ignoreAddressIndex && fSpentIndex)
        if (!pblocktree->UpdateSpentIndex(spentIndex))
            return AbortNode(state, "Failed to write transaction index");

    if (!ignoreAddressIndex && fTimestampIndex) {
        unsigned int logicalTS = pindex->nTime;
        unsigned int prevLogicalTS = 0;

        // retrieve logical timestamp of the previous block
        if (pindex->pprev)
            if (!pblocktree->ReadTimestampBlockIndex(pindex->pprev->GetBlockHash(), prevLogicalTS))
                LogPrintf("%s: Failed to read previous block's logical timestamp\n", __func__);

        if (logicalTS <= prevLogicalTS) {
            logicalTS = prevLogicalTS + 1;
            LogPrintf("%s: Previous logical timestamp is newer Actual[%d] prevLogical[%d] Logical[%d]\n", __func__, pindex->nTime, prevLogicalTS, logicalTS);
        }

        if (!pblocktree->WriteTimestampIndex(CTimestampIndexKey(logicalTS, pindex->GetBlockHash())))
            return AbortNode(state, "Failed to write timestamp index");

        if (!pblocktree->WriteTimestampBlockIndex(CTimestampBlockIndexKey(pindex->GetBlockHash()), CTimestampBlockIndexValue(logicalTS)))
            return AbortNode(state, "Failed to write blockhash index");
    }

    if (AreMessagesDeployed() && fMessaging && setMessages.size()) {
        LOCK(cs_messaging);
        for (auto message : setMessages) {
            int nHeight = 0;
            if (pindex)
                nHeight = pindex->nHeight;
            message.nBlockHeight = nHeight;

            if (message.nExpiredTime == 0 || GetTime() < message.nExpiredTime)
                GetMainSignals().NewAssetMessage(message);

            if (IsChannelSubscribed(message.strName)) {
                AddMessage(message);
            }
        }
    }
#ifdef ENABLE_WALLET
    if (AreRestrictedAssetsDeployed() && myNullAssetData.size() && pmyrestricteddb) {
        for (auto item : myNullAssetData) {
            if (IsAssetNameAQualifier(item.second.asset_name)) {
                // TODO we can add block height to this data also, and use it to pull more info on when this was tagged/untagged
                pmyrestricteddb->WriteTaggedAddress(item.first, item.second.asset_name, item.second.flag ? true : false, block.nTime);
            } else if (IsAssetNameAnRestricted(item.second.asset_name)) {
                pmyrestricteddb->WriteRestrictedAddress(item.first, item.second.asset_name, item.second.flag ? true : false, block.nTime);
            }


            if (pwalletMain)
                pwalletMain->UpdateMyRestrictedAssets(item.first, item.second.asset_name, item.second.flag, block.nTime);

        }
    }
#endif

    assert(pindex->phashBlock);
    // add this block to the view's block chain
    view.SetBestBlock(pindex->GetBlockHash());

    int64_t nTime5 = GetTimeMicros();
    nTimeIndex += nTime5 - nTime4;
    LogPrint("bench", "    - Index writing: %.2fms [%.2fs]\n", 0.001 * (nTime5 - nTime4), nTimeIndex * 0.000001);

    // Watch for changes to the previous coinbase transaction.
    static uint256 hashPrevBestCoinBase;
    GetMainSignals().UpdatedTransaction(hashPrevBestCoinBase);
    hashPrevBestCoinBase = block.vtx[0]->GetHash();


    int64_t nTime6 = GetTimeMicros();
    nTimeCallbacks += nTime6 - nTime5;
    LogPrint("bench", "    - Callbacks: %.2fms [%.2fs]\n", 0.001 * (nTime6 - nTime5), nTimeCallbacks * 0.000001);

    return true;
}

/**
 * Update the on-disk chain state.
 * The caches and indexes are flushed depending on the mode we're called with
 * if they're too large, if it's been a while since the last write,
 * or always and in all cases if we're in prune mode and are deleting files.
 */
bool static FlushStateToDisk(CValidationState& state, FlushStateMode mode, int nManualPruneHeight)
{
    int64_t nMempoolUsage = mempool.DynamicMemoryUsage();
    const CChainParams& chainparams = Params();
    LOCK2(cs_main, cs_LastBlockFile);
    static int64_t nLastWrite = 0;
    static int64_t nLastFlush = 0;
    static int64_t nLastSetChain = 0;
    std::set<int> setFilesToPrune;
    bool fFlushForPrune = false;
    try {
        if (fPruneMode && (fCheckForPruning || nManualPruneHeight > 0) && !fReindex) {
            if (nManualPruneHeight > 0) {
                FindFilesToPruneManual(setFilesToPrune, nManualPruneHeight);
            } else {
                FindFilesToPrune(setFilesToPrune, chainparams.PruneAfterHeight());
                fCheckForPruning = false;
            }
            if (!setFilesToPrune.empty()) {
                fFlushForPrune = true;
                if (!fHavePruned) {
                    pblocktree->WriteFlag("prunedblockfiles", true);
                    fHavePruned = true;
                }
            }
        }
        int64_t nNow = GetTimeMicros();
        // Avoid writing/flushing immediately after startup.
        if (nLastWrite == 0) {
            nLastWrite = nNow;
        }
        if (nLastFlush == 0) {
            nLastFlush = nNow;
        }
        if (nLastSetChain == 0) {
            nLastSetChain = nNow;
        }

        // Get the size of the memory used by the asset cache.
        int64_t assetDynamicSize = 0;
        int64_t assetDirtyCacheSize = 0;
        size_t assetMapAmountSize = 0;
        if (AreAssetsDeployed()) {
            auto currentActiveAssetCache = GetCurrentAssetCache();
            if (currentActiveAssetCache) {
                assetDynamicSize = currentActiveAssetCache->DynamicMemoryUsage();
                assetDirtyCacheSize = currentActiveAssetCache->GetCacheSizeV2();
                assetMapAmountSize = currentActiveAssetCache->mapAssetsAddressAmount.size();
            }
        }

        int messageCacheSize = 0;

        if (fMessaging) {
                messageCacheSize = GetMessageDirtyCacheSize();
        }

        int64_t nMempoolSizeMax = gArgs.GetArg("-maxmempool", DEFAULT_MAX_MEMPOOL_SIZE) * 1000000;
        int64_t cacheSize = pcoinsTip->DynamicMemoryUsage() + assetDynamicSize + assetDirtyCacheSize + messageCacheSize;
        int64_t nTotalSpace = nCoinCacheUsage + std::max<int64_t>(nMempoolSizeMax - nMempoolUsage, 0);
        // The cache is large and we're within 10% and 10 MiB of the limit, but we have time now (not in the middle of a block processing).
        bool fCacheLarge = mode == FLUSH_STATE_PERIODIC && cacheSize > std::max((9 * nTotalSpace) / 10, nTotalSpace - MAX_BLOCK_COINSDB_USAGE * 1024 * 1024);
        // The cache is over the limit, we have to write now.
        bool fCacheCritical = mode == FLUSH_STATE_IF_NEEDED && (cacheSize > nTotalSpace || assetMapAmountSize > 1000000);
        // It's been a while since we wrote the block index to disk. Do this frequently, so we don't need to redownload after a crash.
        bool fPeriodicWrite = mode == FLUSH_STATE_PERIODIC && nNow > nLastWrite + (int64_t)DATABASE_WRITE_INTERVAL * 1000000;
        // It's been very long since we flushed the cache. Do this infrequently, to optimize cache usage.
        bool fPeriodicFlush = mode == FLUSH_STATE_PERIODIC && nNow > nLastFlush + (int64_t)DATABASE_FLUSH_INTERVAL * 1000000;
        // Combine all conditions that result in a full cache flush.
        bool fDoFullFlush = (mode == FLUSH_STATE_ALWAYS) || fCacheLarge || fCacheCritical || fPeriodicFlush || fFlushForPrune;
        // Write blocks and block index to disk.
        if (fDoFullFlush || fPeriodicWrite) {
            // Depend on nMinDiskSpace to ensure we can write block index
            if (!CheckDiskSpace(0))
                return state.Error("out of disk space");
            // First make sure all block and undo data is flushed to disk.
            FlushBlockFile();
            // Then update all block file information (which may refer to block and undo files).
            {
                std::vector<std::pair<int, const CBlockFileInfo*> > vFiles;
                vFiles.reserve(setDirtyFileInfo.size());
                for (std::set<int>::iterator it = setDirtyFileInfo.begin(); it != setDirtyFileInfo.end();) {
                    vFiles.push_back(std::make_pair(*it, &vinfoBlockFile[*it]));
                    setDirtyFileInfo.erase(it++);
                }
                std::vector<const CBlockIndex*> vBlocks;
                vBlocks.reserve(setDirtyBlockIndex.size());
                for (std::set<CBlockIndex*>::iterator it = setDirtyBlockIndex.begin(); it != setDirtyBlockIndex.end();) {
                    vBlocks.push_back(*it);
                    setDirtyBlockIndex.erase(it++);
                }
                if (!pblocktree->WriteBatchSync(vFiles, nLastBlockFile, vBlocks)) {
                    return AbortNode(state, "Files to write to block index database");
                }
            }
            // Finally remove any pruned files
            if (fFlushForPrune)
                UnlinkPrunedFiles(setFilesToPrune);
            nLastWrite = nNow;
        }
        // Flush best chain related state. This can only be done if the blocks / block index write was also done.
        if (fDoFullFlush) {
            // Typical Coin structures on disk are around 48 bytes in size.
            // Pushing a new one to the database can cause it to be written
            // twice (once in the log, and once in the tables). This is already
            // an overestimation, as most will delete an existing entry or
            // overwrite one. Still, use a conservative safety factor of 2.
            if (!CheckDiskSpace((48 * 2 * 2 * pcoinsTip->GetCacheSize()) + assetDirtyCacheSize * 2)) /** ASSET START */ /** ASSET END */
                return state.Error("out of disk space");
            // Flush the chainstate (which may refer to block index entries).
            if (!pcoinsTip->Flush())
                return AbortNode(state, "Failed to write to coin database");

            /** ASSET START */
            // Flush the assetstate
            if (AreAssetsDeployed()) {
                // Flush the assetstate
                auto currentActiveAssetCache = GetCurrentAssetCache();
                if (currentActiveAssetCache) {
                    if (!currentActiveAssetCache->DumpCacheToDatabase())
                        return AbortNode(state, "Failed to write to asset database");
                }
            }

            // Write the reissue mempool data to database
            if (passetsdb)
                passetsdb->WriteReissuedMempoolState();

            if (fMessaging) {
                if (pmessagedb) {
                    LOCK(cs_messaging);
                    if (!pmessagedb->Flush())
                        return AbortNode(state, "Failed to Flush the message database");
                }

                if (pmessagechanneldb) {
                    LOCK(cs_messaging);
                    if (!pmessagechanneldb->Flush())
                        return AbortNode(state, "Failed to Flush the message channel database");
                }
            }
            /** ASSET END */

            nLastFlush = nNow;
        }
        if (fDoFullFlush || ((mode == FLUSH_STATE_ALWAYS || mode == FLUSH_STATE_PERIODIC) && nNow > nLastSetChain + (int64_t)DATABASE_WRITE_INTERVAL * 1000000)) {
            // Update best block in wallet (so we can detect restored wallets).
            GetMainSignals().SetBestChain(chainActive.GetLocator());
            nLastSetChain = nNow;
        }
    } catch (const std::runtime_error& e) {
        return AbortNode(state, std::string("System error while flushing: ") + e.what());
    }
    return true;
}

void FlushStateToDisk()
{
    CValidationState state;
    FlushStateToDisk(state, FLUSH_STATE_ALWAYS);
}

void PruneAndFlush()
{
    CValidationState state;
    fCheckForPruning = true;
    FlushStateToDisk(state, FLUSH_STATE_NONE);
}

/** Update chainActive and related internal data structures. */
void static UpdateTip(CBlockIndex* pindexNew, const CChainParams& chainParams)
{
    chainActive.SetTip(pindexNew);

    // New best block
    mempool.AddTransactionsUpdated(1);

    {
        LOCK(g_best_block_mutex);
        g_best_block = pindexNew->GetBlockHash();
        g_best_block_cv.notify_all();
    }

    static bool fWarned = false;
    std::vector<std::string> warningMessages;
    if (!IsInitialBlockDownload()) {
        int nUpgraded = 0;
        const CBlockIndex* pindex = chainActive.Tip();
        for (int bit = 0; bit < VERSIONBITS_NUM_BITS; bit++) {
            WarningBitsConditionChecker checker(bit);
            ThresholdState state = checker.GetStateFor(pindex, chainParams.GetConsensus(), warningcache[bit]);
            if (state == THRESHOLD_ACTIVE || state == THRESHOLD_LOCKED_IN) {
                if (state == THRESHOLD_ACTIVE) {
                    std::string strWarning = strprintf(_("Warning: unknown new rules activated (versionbit %i)"), bit);
                    SetMiscWarning(strWarning);
                    if (!fWarned) {
                        CAlert::Notify(strWarning);
                        fWarned = true;
                    }
                } else {
                    warningMessages.push_back(strprintf("unknown new rules are about to activate (versionbit %i)", bit));
                }
            }
        }
        // Check the version of the last 100 blocks to see if we need to upgrade:
        for (int i = 0; i < 100 && pindex != nullptr; i++) {
            int32_t nExpectedVersion = ComputeBlockVersion(pindex->pprev, chainParams.GetConsensus());
            if (pindex->nVersion > VERSIONBITS_LAST_OLD_BLOCK_VERSION && (pindex->nVersion & ~nExpectedVersion) != 0)
                ++nUpgraded;
            pindex = pindex->pprev;
        }
        if (nUpgraded > 0)
            warningMessages.push_back(strprintf("%d of last 100 blocks have unexpected version", nUpgraded));
        if (nUpgraded > 100 / 2) {
            std::string strWarning = _("Warning: Unknown block versions being mined! It's possible unknown rules are in effect");
            // notify GetWarnings(), called by Qt and the JSON-RPC code to warn the user:
            SetMiscWarning(strWarning);
            if (!fWarned) {
                CAlert::Notify(strWarning);
                fWarned = true;
            }
        }
    }
    LogPrint("validation", "%s: new best=%s height=%d version=0x%08x log2_work=%.8f tx=%lu date='%s' progress=%f cache=%.1fMiB(%utxo)\n", __func__,
        chainActive.Tip()->GetBlockHash().ToString(), chainActive.Height(), chainActive.Tip()->nVersion,
        log(chainActive.Tip()->nChainWork.getdouble()) / log(2.0), (unsigned long)chainActive.Tip()->nChainTx,
        DateTimeStrFormat("%Y-%m-%d %H:%M:%S", chainActive.Tip()->GetBlockTime()),
        GuessVerificationProgress(chainParams.TxData(), chainActive.Tip()), pcoinsTip->DynamicMemoryUsage() * (1.0 / (1 << 20)), pcoinsTip->GetCacheSize());
    if (!warningMessages.empty())
        LogPrintf("%s -- warning='%s'\n", __func__, boost::algorithm::join(warningMessages, ", "));
}

/** Disconnect chainActive's tip.
  * After calling, the mempool will be in an inconsistent state, with
  * transactions from disconnected blocks being added to disconnectpool.  You
  * should make the mempool consistent again by calling UpdateMempoolForReorg.
  * with cs_main held.
  *
  * If disconnectpool is nullptr, then no disconnected transactions are added to
  * disconnectpool (note that the caller is responsible for mempool consistency
  * in any case).
  */
bool static DisconnectTip(CValidationState& state, const CChainParams& chainparams, DisconnectedBlockTransactions *disconnectpool)
{

    CBlockIndex* pindexDelete = chainActive.Tip();
    assert(pindexDelete);
    // Read block from disk.
    std::shared_ptr<CBlock> pblock = std::make_shared<CBlock>();
    CBlock& block = *pblock;
    if (!ReadBlockFromDisk(block, pindexDelete, chainparams.GetConsensus()))
        return AbortNode(state, "Failed to read block");
    // Apply the block atomically to the chain state.
    int64_t nStart = GetTimeMicros();
    {
        CCoinsViewCache view(pcoinsTip);
        CAssetsCache assetCache;

        assert(view.GetBestBlock() == pindexDelete->GetBlockHash());
        if (DisconnectBlock(block, state, pindexDelete, view, 4, &assetCache) != DISCONNECT_OK)
            return error("DisconnectTip(): DisconnectBlock %s failed", pindexDelete->GetBlockHash().ToString());
        bool flushed = view.Flush();
        assert(flushed);

        bool assetsFlushed = assetCache.Flush();
        assert(assetsFlushed);
    }
    LogPrint("bench", "- Disconnect block: %.2fms\n", (GetTimeMicros() - nStart) * MILLI);
    
    // Write the chain state to disk, if necessary.
    if (!FlushStateToDisk(state, FLUSH_STATE_IF_NEEDED))
        return false;

    if (disconnectpool) {
        // Save transactions to re-add to mempool at end of reorg
        for (auto it = block.vtx.rbegin(); it != block.vtx.rend(); ++it) {
            disconnectpool->addTransaction(*it);
        }
        while (disconnectpool->DynamicMemoryUsage() > MAX_DISCONNECTED_TX_POOL_SIZE * 1000) {
            // Drop the earliest entry, and remove its children from the mempool.
            auto it = disconnectpool->queuedTx.get<insertion_order>().begin();
            mempool.removeRecursive(**it, MemPoolRemovalReason::REORG);
            disconnectpool->removeEntry(it);
        }
    }

    // Resurrect mempool transactions from the disconnected block.
    std::vector<uint256> vHashUpdate;
    for (const auto& it : block.vtx) {
        const CTransaction& tx = *it;
        // ignore validation errors in resurrected transactions
        CValidationState stateDummy;
        if (tx.IsCoinBase() || tx.IsCoinStake() || !AcceptToMemoryPool(mempool, stateDummy, it, false, nullptr, nullptr, true)) {
            mempool.removeRecursive(tx, MemPoolRemovalReason::REORG);
        } else if (mempool.exists(tx.GetHash())) {
            vHashUpdate.push_back(tx.GetHash());
        }
    }
    // AcceptToMemoryPool/addUnchecked all assume that new mempool entries have
    // no in-mempool children, which is generally not true when adding
    // previously-confirmed transactions back to the mempool.
    // UpdateTransactionsFromBlock finds descendants of any transactions in this
    // block that were added back and cleans up the mempool state.
    mempool.UpdateTransactionsFromBlock(vHashUpdate);
    // Update chainActive and related variables.
    UpdateTip(pindexDelete->pprev, chainparams);
    // Let wallets know transactions went from 1-confirmed to
    // 0-confirmed or conflicted:
    GetMainSignals().BlockDisconnected(pblock);
    return true;
}

static int64_t nTimeReadFromDisk = 0;
static int64_t nTimeConnectTotal = 0;
static int64_t nTimeFlush = 0;
static int64_t nTimeAssetFlush = 0;
static int64_t nTimeAssetTasks = 0;
static int64_t nTimeChainState = 0;
static int64_t nTimePostConnect = 0;

struct PerBlockConnectTrace {
    CBlockIndex* pindex = nullptr;
    std::shared_ptr<const CBlock> pblock;
    std::shared_ptr<std::vector<CTransactionRef>> conflictedTxs;
    PerBlockConnectTrace() : conflictedTxs(std::make_shared<std::vector<CTransactionRef>>()) {}
};
/**
 * Used to track blocks whose transactions were applied to the UTXO state as a
 * part of a single ActivateBestChainStep call.
 *
 * This class also tracks transactions that are removed from the mempool as
 * conflicts (per block) and can be used to pass all those transactions
 * through SyncTransaction.
 *
 * This class assumes (and asserts) that the conflicted transactions for a given
 * block are added via mempool callbacks prior to the BlockConnected() associated
 * with those transactions. If any transactions are marked conflicted, it is
 * assumed that an associated block will always be added.
 *
 * This class is single-use, once you call GetBlocksConnected() you have to throw
 * it away and make a new one.
 */
class ConnectTrace {
private:
    std::vector<PerBlockConnectTrace> blocksConnected;
    CTxMemPool &pool;

public:
    explicit ConnectTrace(CTxMemPool &_pool) : blocksConnected(1), pool(_pool) {
        pool.NotifyEntryRemoved.connect(boost::bind(&ConnectTrace::NotifyEntryRemoved, this, _1, _2));
    }

    ~ConnectTrace() {
        pool.NotifyEntryRemoved.disconnect(boost::bind(&ConnectTrace::NotifyEntryRemoved, this, _1, _2));
    }

    void BlockConnected(CBlockIndex* pindex, std::shared_ptr<const CBlock> pblock) {
        assert(!blocksConnected.back().pindex);
        assert(pindex);
        assert(pblock);
        blocksConnected.back().pindex = pindex;
        blocksConnected.back().pblock = std::move(pblock);
        blocksConnected.emplace_back();
    }

    std::vector<PerBlockConnectTrace>& GetBlocksConnected() {
        // We always keep one extra block at the end of our list because
        // blocks are added after all the conflicted transactions have
        // been filled in. Thus, the last entry should always be an empty
        // one waiting for the transactions from the next block. We pop
        // the last entry here to make sure the list we return is sane.
        assert(!blocksConnected.back().pindex);
        assert(blocksConnected.back().conflictedTxs->empty());
        blocksConnected.pop_back();
        return blocksConnected;
    }

    void NotifyEntryRemoved(CTransactionRef txRemoved, MemPoolRemovalReason reason) {
        assert(!blocksConnected.back().pindex);
        if (reason == MemPoolRemovalReason::CONFLICT) {
            blocksConnected.back().conflictedTxs->emplace_back(std::move(txRemoved));
        }
    }
};

/**
 * Connect a new block to chainActive. pblock is either nullptr or a pointer to a CBlock
 * corresponding to pindexNew, to bypass loading it again from disk.
 *
 * The block is added to connectTrace if connection succeeds.
 */
bool static ConnectTip(CValidationState& state, const CChainParams& chainparams, CBlockIndex* pindexNew, const std::shared_ptr<const CBlock>& pblock, ConnectTrace& connectTrace, DisconnectedBlockTransactions &disconnectpool)
{
    assert(pindexNew->pprev == chainActive.Tip());
    // Read block from disk.
    int64_t nTime1 = GetTimeMicros();
    std::shared_ptr<const CBlock> pthisBlock;
    if (!pblock) {
        std::shared_ptr<CBlock> pblockNew = std::make_shared<CBlock>();
        if (!ReadBlockFromDisk(*pblockNew, pindexNew, chainparams.GetConsensus()))
            return AbortNode(state, "Failed to read block");
        pthisBlock = pblockNew;
    } else {
        pthisBlock = pblock;
    }
    const CBlock& blockConnecting = *pthisBlock;
    // Apply the block atomically to the chain state.
    int64_t nTime2 = GetTimeMicros();
    nTimeReadFromDisk += nTime2 - nTime1;
    int64_t nTime3;
    int64_t nTime4;
    int64_t nTimeAssetsFlush;
    LogPrint("bench", "  - Load block from disk: %.2fms [%.2fs]\n", (nTime2 - nTime1) * MILLI, nTimeReadFromDisk * MICRO);
    /** ASSET START */
    // Initialize sets used from removing asset entries from the mempool
    ConnectedBlockAssetData assetDataFromBlock;
    /** ASSET END */
    {
        CCoinsViewCache view(pcoinsTip);
        /** ASSET START */
        // Create the empty asset cache, that will be sent into the connect block
        // All new data will be added to the cache, and will be flushed back into passets after a successful
        // Connect Block cycle
        CAssetsCache assetCache;
        std::vector<std::pair<std::string, CNullAssetTxData>> myNullAssetData;
        /** ASSET END */
        bool rv = ConnectBlock(blockConnecting, state, pindexNew, view, chainparams, &assetCache);
        GetMainSignals().BlockChecked(blockConnecting, state);
        if (!rv) {
            if (state.IsInvalid())
                InvalidBlockFound(pindexNew, state);
            return error("ConnectTip(): ConnectBlock %s failed", pindexNew->GetBlockHash().ToString());
        }

        /** ASSET START */
        int64_t nTimeAssetsStart = GetTimeMicros();
        // Get the newly created assets, from the connectblock assetCache so we can remove the correct assets from the mempool
        assetDataFromBlock = {assetCache.setNewAssetsToAdd, assetCache.setNewRestrictedVerifierToAdd, assetCache.setNewRestrictedAddressToAdd, assetCache.setNewRestrictedGlobalToAdd, assetCache.setNewQualifierAddressToAdd};

        // Remove all tx hashes, that were marked as reissued script from the mapReissuedTx.
        // Without this check, you wouldn't be able to reissue for those assets again, as this maps block it
        for (auto tx : blockConnecting.vtx) {
            uint256 txHash = tx->GetHash();
            if (mapReissuedTx.count(txHash)) {
                mapReissuedAssets.erase(mapReissuedTx.at(txHash));
                mapReissuedTx.erase(txHash);
            }
        }
        int64_t nTimeAssetsEnd = GetTimeMicros(); nTimeAssetTasks += nTimeAssetsEnd - nTimeAssetsStart;
        LogPrint("bench", "  - Compute Asset Tasks total: %.2fms [%.2fs (%.2fms/blk)]\n", (nTimeAssetsEnd - nTimeAssetsStart) * MILLI, nTimeAssetsEnd * MICRO, nTimeAssetsEnd * MILLI / nBlocksTotal);
        /** ASSET END */

        nTime3 = GetTimeMicros();
        nTimeConnectTotal += nTime3 - nTime2;
        LogPrint("bench", "  - Connect total: %.2fms [%.2fs (%.2fms/blk)]\n", (nTime3 - nTime2) * MILLI, nTimeConnectTotal * MICRO, nTimeConnectTotal * MILLI / nBlocksTotal);
        bool flushed = view.Flush();
        assert(flushed);
        nTime4 = GetTimeMicros(); 
        nTimeFlush += nTime4 - nTime3;
        LogPrint("bench", "  - Flush DYN: %.2fms [%.2fs (%.2fms/blk)]\n", (nTime4 - nTime3) * MILLI, nTimeFlush * MICRO, nTimeFlush * MILLI / nBlocksTotal);

        /** ASSET START */
        nTimeAssetsFlush = GetTimeMicros();
        bool assetFlushed = assetCache.Flush();
        assert(assetFlushed);
        int64_t nTimeAssetFlushFinished = GetTimeMicros(); nTimeAssetFlush += nTimeAssetFlushFinished - nTimeAssetsFlush;
        LogPrint("bench", "  - Flush Assets: %.2fms [%.2fs (%.2fms/blk)]\n", (nTimeAssetFlushFinished - nTimeAssetsFlush) * MILLI, nTimeAssetFlush * MICRO, nTimeAssetFlush * MILLI / nBlocksTotal);
        /** ASSET END */
    }

    // Write the chain state to disk, if necessary.
    if (!FlushStateToDisk(state, FLUSH_STATE_IF_NEEDED))
        return false;
    int64_t nTime5 = GetTimeMicros();
    nTimeChainState += nTime5 - nTime4;
    LogPrint("bench", "  - Writing chainstate: %.2fms [%.2fs (%.2fms/blk)]\n", (nTime5 - nTime4) * MILLI, nTimeChainState * MICRO, nTimeChainState * MILLI / nBlocksTotal);
    // Remove conflicting transactions from the mempool.;
    mempool.removeForBlock(blockConnecting.vtx, pindexNew->nHeight, assetDataFromBlock);
    disconnectpool.removeForBlock(blockConnecting.vtx);
    // Update chainActive & related variables.
    UpdateTip(pindexNew, chainparams);

    int64_t nTime6 = GetTimeMicros();
    nTimePostConnect += nTime6 - nTime5;
    nTimeTotal += nTime6 - nTime1;
    LogPrint("bench", "  - Connect postprocess: %.2fms [%.2fs (%.2fms/blk)]\n", (nTime6 - nTime5) * MILLI, nTimePostConnect * MICRO, nTimePostConnect * MILLI / nBlocksTotal);
    LogPrint("bench", "- Connect block: %.2fms [%.2fs (%.2fms/blk)]\n", (nTime6 - nTime1) * MILLI, nTimeTotal * MICRO, nTimeTotal * MILLI / nBlocksTotal);

    connectTrace.BlockConnected(pindexNew, std::move(pthisBlock));

    /** ASSET START */

    //  Determine if the new block height has any pending snapshot requests,
    //      and if so, capture a snapshot of the relevant target assets.
    if (pSnapshotRequestDb != nullptr) {
        //  Retrieve the scheduled snapshot requests
        std::set<CSnapshotRequestDBEntry> assetsToSnapshot;
        if (pSnapshotRequestDb->RetrieveSnapshotRequestsForHeight("", pindexNew->nHeight, assetsToSnapshot)) {
            //  Loop through them
            for (auto const & assetEntry : assetsToSnapshot) {
                //  Add a snapshot entry for the target asset ownership
                if (!pAssetSnapshotDb->AddAssetOwnershipSnapshot(assetEntry.assetName, pindexNew->nHeight)) {
                   LogPrint("rewards", "ConnectTip: Failed to snapshot owners for '%s' at height %d!\n",
                       assetEntry.assetName.c_str(), pindexNew->nHeight);
                }
            }
        }
        else {
            LogPrint("rewards", "ConnectTip: Failed to load payable Snapshot Requests at height %d!\n", pindexNew->nHeight);
        }
    }

#ifdef ENABLE_WALLET
    if (pwalletMain) {
        CheckRewardDistributions(pwalletMain);
    }
#endif
    /** ASSET END */

    return true;
}

bool DisconnectBlocks(int blocks)
{
    LOCK(cs_main);

    CValidationState state;
    const CChainParams& chainparams = Params();
    DisconnectedBlockTransactions disconnectpool;

    LogPrintf("DisconnectBlocks -- Got command to replay %d blocks\n", blocks);
    for (int i = 0; i < blocks; i++) {
        if (!DisconnectTip(state, chainparams, &disconnectpool) || !state.IsValid()) {
            return false;
        }
    }

    return true;
}

void ReprocessBlocks(int nBlocks)
{
    LOCK(cs_main);

    std::map<uint256, int64_t>::iterator it = mapRejectedBlocks.begin();
    while (it != mapRejectedBlocks.end()) {
        //use a window twice as large as is usual for the nBlocks we want to reset
        if ((*it).second > GetTime() - (nBlocks * 60 * 5)) {
            BlockMap::iterator mi = mapBlockIndex.find((*it).first);
            if (mi != mapBlockIndex.end() && (*mi).second) {
                CBlockIndex* pindex = (*mi).second;
                LogPrintf("ReprocessBlocks -- %s\n", (*it).first.ToString());

                ResetBlockFailureFlags(pindex);
            }
        }
        ++it;
    }

    DisconnectBlocks(nBlocks);

    CValidationState state;
    ActivateBestChain(state, Params());
}

/**
 * Return the tip of the chain with the most work in it, that isn't
 * known to be invalid (it's however far from certain to be valid).
 */
static CBlockIndex* FindMostWorkChain()
{
    do {
        CBlockIndex* pindexNew = nullptr;

        // Find the best candidate header.
        {
            std::set<CBlockIndex*, CBlockIndexWorkComparator>::reverse_iterator it = setBlockIndexCandidates.rbegin();
            if (it == setBlockIndexCandidates.rend())
                return nullptr;
            pindexNew = *it;
        }

        // Check whether all blocks on the path between the currently active chain and the candidate are valid.
        // Just going until the active chain is an optimization, as we know all blocks in it are valid already.
        CBlockIndex* pindexTest = pindexNew;
        bool fInvalidAncestor = false;
        while (pindexTest && !chainActive.Contains(pindexTest)) {
            assert(pindexTest->nChainTx || pindexTest->nHeight == 0);

            // Pruned nodes may have entries in setBlockIndexCandidates for
            // which block files have been deleted.  Remove those as candidates
            // for the most work chain if we come across them; we can't switch
            // to a chain unless we have all the non-active-chain parent blocks.
            bool fFailedChain = pindexTest->nStatus & BLOCK_FAILED_MASK;
            bool fMissingData = !(pindexTest->nStatus & BLOCK_HAVE_DATA);
            if (fFailedChain || fMissingData) {
                // Candidate chain is not usable (either invalid or missing data)
                if (fFailedChain && (pindexBestInvalid == nullptr || pindexNew->nChainWork > pindexBestInvalid->nChainWork))
                    pindexBestInvalid = pindexNew;
                CBlockIndex* pindexFailed = pindexNew;
                // Remove the entire chain from the set.
                while (pindexTest != pindexFailed) {
                    if (fFailedChain) {
                        pindexFailed->nStatus |= BLOCK_FAILED_CHILD;
                    } else if (fMissingData) {
                        // If we're missing data, then add back to mapBlocksUnlinked,
                        // so that if the block arrives in the future we can try adding
                        // to setBlockIndexCandidates again.
                        mapBlocksUnlinked.insert(std::make_pair(pindexFailed->pprev, pindexFailed));
                    }
                    setBlockIndexCandidates.erase(pindexFailed);
                    pindexFailed = pindexFailed->pprev;
                }
                setBlockIndexCandidates.erase(pindexTest);
                fInvalidAncestor = true;
                break;
            }
            pindexTest = pindexTest->pprev;
        }
        if (!fInvalidAncestor)
            return pindexNew;
    } while (true);
}

/** Delete all entries in setBlockIndexCandidates that are worse than the current tip. */
static void PruneBlockIndexCandidates()
{
    // Note that we can't delete the current block itself, as we may need to return to it later in case a
    // reorganization to a better block fails.
    std::set<CBlockIndex*, CBlockIndexWorkComparator>::iterator it = setBlockIndexCandidates.begin();
    while (it != setBlockIndexCandidates.end() && setBlockIndexCandidates.value_comp()(*it, chainActive.Tip())) {
        setBlockIndexCandidates.erase(it++);
    }
    // Either the current tip or a successor of it we're working towards is left in setBlockIndexCandidates.
    assert(!setBlockIndexCandidates.empty());
}

/**
 * Try to make some progress towards making pindexMostWork the active block.
 * pblock is either nullptr or a pointer to a CBlock corresponding to pindexMostWork.
 */
static bool ActivateBestChainStep(CValidationState& state, const CChainParams& chainparams, CBlockIndex* pindexMostWork, const std::shared_ptr<const CBlock>& pblock, bool& fInvalidFound, ConnectTrace& connectTrace)
{
    AssertLockHeld(cs_main);
    const CBlockIndex* pindexOldTip = chainActive.Tip();
    const CBlockIndex* pindexFork = chainActive.FindFork(pindexMostWork);

    // Disconnect active blocks which are no longer in the best chain.
    bool fBlocksDisconnected = false;
    DisconnectedBlockTransactions disconnectpool;
    while (chainActive.Tip() && chainActive.Tip() != pindexFork) {
        if (!DisconnectTip(state, chainparams, &disconnectpool)) {
            // This is likely a fatal error, but keep the mempool consistent,
            // just in case. Only remove from the mempool in this case.
            UpdateMempoolForReorg(disconnectpool, false);
            return false;
        }
        fBlocksDisconnected = true;
    }

    // Build list of new blocks to connect.
    std::vector<CBlockIndex*> vpindexToConnect;
    bool fContinue = true;
    int nHeight = pindexFork ? pindexFork->nHeight : -1;
    while (fContinue && nHeight != pindexMostWork->nHeight) {
        // Don't iterate the entire list of potential improvements toward the best tip, as we likely only need
        // a few blocks along the way.
        int nTargetHeight = std::min(nHeight + 32, pindexMostWork->nHeight);
        vpindexToConnect.clear();
        vpindexToConnect.reserve(nTargetHeight - nHeight);
        CBlockIndex* pindexIter = pindexMostWork->GetAncestor(nTargetHeight);
        while (pindexIter && pindexIter->nHeight != nHeight) {
            vpindexToConnect.push_back(pindexIter);
            pindexIter = pindexIter->pprev;
        }
        nHeight = nTargetHeight;

        // Connect new blocks.
        for (CBlockIndex *pindexConnect : reverse_iterate(vpindexToConnect)) {
            if (!ConnectTip(state, chainparams, pindexConnect, pindexConnect == pindexMostWork ? pblock : std::shared_ptr<const CBlock>(), connectTrace, disconnectpool)) {
                if (state.IsInvalid()) {
                    // The block violates a consensus rule.
                    if (!state.CorruptionPossible())
                        InvalidChainFound(vpindexToConnect.back());
                    state = CValidationState();
                    fInvalidFound = true;
                    fContinue = false;
                    break;
                } else {
                    // A system error occurred (disk space, database error, ...).
                    // Make the mempool consistent with the current tip, just in case
                    // any observers try to use it before shutdown.
                    UpdateMempoolForReorg(disconnectpool, false);
                    return false;
                }
            } else {
                PruneBlockIndexCandidates();
                if (!pindexOldTip || chainActive.Tip()->nChainWork > pindexOldTip->nChainWork) {
                    // We're in a better position than we were. Return temporarily to release the lock.
                    fContinue = false;
                    break;
                }
            }
        }
    }

    if (fBlocksDisconnected) {
        // If any blocks were disconnected, disconnectpool may be non empty.  Add
        // any disconnected transactions back to the mempool.
        UpdateMempoolForReorg(disconnectpool, true);
    }
    mempool.check(pcoinsTip);

    // Callbacks/notifications for a new best chain.
    if (fInvalidFound)
        CheckForkWarningConditionsOnNewFork(vpindexToConnect.back());
    else
        CheckForkWarningConditions();

    return true;
}

static void NotifyHeaderTip()
{
    bool fNotify = false;
    bool fInitialBlockDownload = false;
    static CBlockIndex* pindexHeaderOld = nullptr;
    CBlockIndex* pindexHeader = nullptr;
    {
        LOCK(cs_main);
        pindexHeader = pindexBestHeader;

        if (pindexHeader != pindexHeaderOld) {
            fNotify = true;
            fInitialBlockDownload = IsInitialBlockDownload();
            pindexHeaderOld = pindexHeader;
        }
    }
    // Send block tip changed notifications without cs_main
    if (fNotify) {
        uiInterface.NotifyHeaderTip(fInitialBlockDownload, pindexHeader);
        GetMainSignals().NotifyHeaderTip(pindexHeader, fInitialBlockDownload);
    }
}

/**
 * Make the best chain active, in multiple steps. The result is either failure
 * or an activated best chain. pblock is either nullptr or a pointer to a block
 * that is already loaded (to avoid loading it again from disk).
 */
bool ActivateBestChain(CValidationState& state, const CChainParams& chainparams, std::shared_ptr<const CBlock> pblock)
{
    // Note that while we're often called here from ProcessNewBlock, this is
    // far from a guarantee. Things in the P2P/RPC will often end up calling
    // us in the middle of ProcessNewBlock - do not assume pblock is set
    // sanely for performance or correctness!
    CBlockIndex* pindexMostWork = nullptr;
    CBlockIndex* pindexNewTip = nullptr;
    do {
        boost::this_thread::interruption_point();
        if (ShutdownRequested())
            break;

        const CBlockIndex* pindexFork;
        bool fInitialDownload;
        {
            LOCK(cs_main);
            ConnectTrace connectTrace(mempool); // Destructed before cs_main is unlocked

            MemPoolConflictRemovalTracker mrt(mempool);
            CBlockIndex* pindexOldTip = chainActive.Tip();
            if (pindexMostWork == nullptr) {
                pindexMostWork = FindMostWorkChain();
            }

            // Whether we have anything to do at all.
            if (pindexMostWork == nullptr || pindexMostWork == chainActive.Tip())
                return true;

            bool fInvalidFound = false;
            std::shared_ptr<const CBlock> nullBlockPtr;
            if (!ActivateBestChainStep(state, chainparams, pindexMostWork, pblock && pblock->GetHash() == pindexMostWork->GetBlockHash() ? pblock : nullBlockPtr, fInvalidFound, connectTrace))
                return false;

            if (fInvalidFound) {
                // Wipe cache, we may need another branch now.
                pindexMostWork = nullptr;
            }
            pindexNewTip = chainActive.Tip();
            pindexFork = chainActive.FindFork(pindexOldTip);
            fInitialDownload = IsInitialBlockDownload();

            for (const PerBlockConnectTrace& trace : connectTrace.GetBlocksConnected()) {
                assert(trace.pblock && trace.pindex);
                GetMainSignals().BlockConnected(trace.pblock, trace.pindex, *trace.conflictedTxs);
            }
        }
        // When we reach this point, we switched to a new tip (stored in pindexNewTip).

        // Notifications/callbacks that can run without cs_main

        // Notify external listeners about the new tip.
        GetMainSignals().UpdatedBlockTip(pindexNewTip, pindexFork, fInitialDownload);

        // Always notify the UI if a new block tip was connected
        if (pindexFork != pindexNewTip) {
            uiInterface.NotifyBlockTip(fInitialDownload, pindexNewTip);
        }
    } while (pindexNewTip != pindexMostWork);
    CheckBlockIndex(chainparams.GetConsensus());

    // Write changes periodically to disk, after relay.
    if (!FlushStateToDisk(state, FLUSH_STATE_PERIODIC)) {
        return false;
    }

    return true;
}

bool PreciousBlock(CValidationState& state, const CChainParams& params, CBlockIndex* pindex)
{
    {
        LOCK(cs_main);
        if (pindex->nChainWork < chainActive.Tip()->nChainWork) {
            // Nothing to do, this block is not at the tip.
            return true;
        }
        if (chainActive.Tip()->nChainWork > nLastPreciousChainwork) {
            // The chain has been extended since the last call, reset the counter.
            nBlockReverseSequenceId = -1;
        }
        nLastPreciousChainwork = chainActive.Tip()->nChainWork;
        setBlockIndexCandidates.erase(pindex);
        pindex->nSequenceId = nBlockReverseSequenceId;
        if (nBlockReverseSequenceId > std::numeric_limits<int32_t>::min()) {
            // We can't keep reducing the counter if somebody really wants to
            // call preciousblock 2**31-1 times on the same set of tips...
            nBlockReverseSequenceId--;
        }
        if (pindex->IsValid(BLOCK_VALID_TRANSACTIONS) && pindex->nChainTx) {
            setBlockIndexCandidates.insert(pindex);
            PruneBlockIndexCandidates();
        }
    }

    return ActivateBestChain(state, params);
}

bool InvalidateBlock(CValidationState& state, const CChainParams& chainparams, CBlockIndex* pindex)
{
    AssertLockHeld(cs_main);

    // Mark the block itself as invalid.
    pindex->nStatus |= BLOCK_FAILED_VALID;
    setDirtyBlockIndex.insert(pindex);
    setBlockIndexCandidates.erase(pindex);

    if (pindex == pindexBestHeader) {
        pindexBestInvalid = pindexBestHeader;
        pindexBestHeader = pindexBestHeader->pprev;
    }

    DisconnectedBlockTransactions disconnectpool;
    while (chainActive.Contains(pindex)) {
        CBlockIndex* pindexWalk = chainActive.Tip();
        pindexWalk->nStatus |= BLOCK_FAILED_CHILD;
        setDirtyBlockIndex.insert(pindexWalk);
        setBlockIndexCandidates.erase(pindexWalk);
        // ActivateBestChain considers blocks already in chainActive
        // unconditionally valid already, so force disconnect away from it.
        if (!DisconnectTip(state, chainparams, &disconnectpool)) {
            // It's probably hopeless to try to make the mempool consistent
            // here if DisconnectTip failed, but we can try.
            UpdateMempoolForReorg(disconnectpool, false);
            return false;
        }
        if (pindexWalk == pindexBestHeader) {
            pindexBestInvalid = pindexBestHeader;
            pindexBestHeader = pindexBestHeader->pprev;
        }
    }

    LimitMempoolSize(mempool, gArgs.GetArg("-maxmempool", DEFAULT_MAX_MEMPOOL_SIZE) * 1000000, gArgs.GetArg("-mempoolexpiry", DEFAULT_MEMPOOL_EXPIRY) * 60 * 60);
    
    // DisconnectTip will add transactions to disconnectpool; try to add these
    // back to the mempool.
    UpdateMempoolForReorg(disconnectpool, true);

    // The resulting new best tip may not be in setBlockIndexCandidates anymore, so
    // add it again.
    BlockMap::iterator it = mapBlockIndex.begin();
    while (it != mapBlockIndex.end()) {
        if (it->second->IsValid(BLOCK_VALID_TRANSACTIONS) && it->second->nChainTx && !setBlockIndexCandidates.value_comp()(it->second, chainActive.Tip())) {
            setBlockIndexCandidates.insert(it->second);
        }
        it++;
    }

    InvalidChainFound(pindex);
    mempool.removeForReorg(pcoinsTip, chainActive.Tip()->nHeight + 1, STANDARD_LOCKTIME_VERIFY_FLAGS);
    uiInterface.NotifyBlockTip(IsInitialBlockDownload(), pindex->pprev);
    return true;
}

bool ResetBlockFailureFlags(CBlockIndex* pindex)
{
    AssertLockHeld(cs_main);

    int nHeight = pindex->nHeight;

    // Remove the invalidity flag from this block and all its descendants.
    BlockMap::iterator it = mapBlockIndex.begin();
    while (it != mapBlockIndex.end()) {
        if (!it->second->IsValid() && it->second->GetAncestor(nHeight) == pindex) {
            it->second->nStatus &= ~BLOCK_FAILED_MASK;
            setDirtyBlockIndex.insert(it->second);
            if (it->second->IsValid(BLOCK_VALID_TRANSACTIONS) && it->second->nChainTx && setBlockIndexCandidates.value_comp()(chainActive.Tip(), it->second)) {
                setBlockIndexCandidates.insert(it->second);
            }
            if (it->second == pindexBestInvalid) {
                // Reset invalid block marker if it was pointing to one of those.
                pindexBestInvalid = nullptr;
            }
        }
        it++;
    }

    // Remove the invalidity flag from all ancestors too.
    while (pindex != nullptr) {
        if (pindex->nStatus & BLOCK_FAILED_MASK) {
            pindex->nStatus &= ~BLOCK_FAILED_MASK;
            setDirtyBlockIndex.insert(pindex);
        }
        pindex = pindex->pprev;
    }
    return true;
}

bool ReconsiderBlock(CValidationState& state, CBlockIndex* pindex)
{
    AssertLockHeld(cs_main);

    int nHeight = pindex->nHeight;

    // Remove the invalidity flag from this block and all its descendants.
    BlockMap::iterator it = mapBlockIndex.begin();
    while (it != mapBlockIndex.end()) {
        if (!it->second->IsValid() && it->second->GetAncestor(nHeight) == pindex) {
            it->second->nStatus &= ~BLOCK_FAILED_MASK;
            setDirtyBlockIndex.insert(it->second);
            if (it->second->IsValid(BLOCK_VALID_TRANSACTIONS) && it->second->nChainTx && setBlockIndexCandidates.value_comp()(chainActive.Tip(), it->second)) {
                setBlockIndexCandidates.insert(it->second);
            }
            if (it->second == pindexBestInvalid) {
                // Reset invalid block marker if it was pointing to one of those.
                pindexBestInvalid = nullptr;
            }
        }
        it++;
    }

    // Remove the invalidity flag from all ancestors too.
    while (pindex != nullptr) {
        if (pindex->nStatus & BLOCK_FAILED_MASK) {
            pindex->nStatus &= ~BLOCK_FAILED_MASK;
            setDirtyBlockIndex.insert(pindex);
        }
        pindex = pindex->pprev;
    }
    return true;
}

CBlockIndex* AddToBlockIndex(const CBlock& block)
{
    // Check for duplicate
    uint256 hash = block.GetHash();
    BlockMap::iterator it = mapBlockIndex.find(hash);
    if (it != mapBlockIndex.end())
        return it->second;

    // Construct new block index object
    CBlockIndex* pindexNew = new CBlockIndex(block);
    assert(pindexNew);
    // We assign the sequence id to blocks only when the full data is available,
    // to avoid miners withholding blocks but broadcasting headers, to get a
    // competitive advantage.
    pindexNew->nSequenceId = 0;
    BlockMap::iterator mi = mapBlockIndex.insert(std::make_pair(hash, pindexNew)).first;
    pindexNew->phashBlock = &((*mi).first);
    BlockMap::iterator miPrev = mapBlockIndex.find(block.hashPrevBlock);
    if (miPrev != mapBlockIndex.end()) {
        pindexNew->pprev = (*miPrev).second;
        pindexNew->nHeight = pindexNew->pprev->nHeight + 1;
        pindexNew->BuildSkip();

        // ppcoin: compute chain trust score (doesn't appear to be used)
        // pindexNew->bnChainTrust = (pindexNew->pprev ? pindexNew->pprev->bnChainTrust : 0) + pindexNew->GetBlockTrust();

        // ppcoin: compute stake entropy bit for stake modifier
        if (!pindexNew->SetStakeEntropyBit(pindexNew->GetStakeEntropyBit()))
            LogPrintf("AddToBlockIndex() : SetStakeEntropyBit() failed \n");

        // ppcoin: record proof-of-stake hash value
        if (pindexNew->IsProofOfStake()) {
            if (!mapProofOfStake.count(hash))
                LogPrintf("AddToBlockIndex() : hashProofOfStake not found in map \n");
        }
        // compute v2 stake modifier
        if (block.vtx.size() > 1) {
            pindexNew->nStakeModifier = ComputeStakeModifier(pindexNew->pprev, block.vtx[1]->vin[0].prevout.hash);
            pindexNew->prevoutStake = block.vtx[1]->vin[0].prevout;
        }
    }
    pindexNew->nTimeMax = (pindexNew->pprev ? std::max(pindexNew->pprev->nTimeMax, pindexNew->nTime) : pindexNew->nTime);
    pindexNew->nChainWork = (pindexNew->pprev ? pindexNew->pprev->nChainWork : 0) + GetBlockProof(*pindexNew);
    pindexNew->RaiseValidity(BLOCK_VALID_TREE);
    if (pindexBestHeader == nullptr || pindexBestHeader->nChainWork < pindexNew->nChainWork)
        pindexBestHeader = pindexNew;
    setDirtyBlockIndex.insert(pindexNew);

    return pindexNew;
}

/** Mark a block as having its data received and checked (up to BLOCK_VALID_TRANSACTIONS). */
bool ReceivedBlockTransactions(const CBlock& block, CValidationState& state, CBlockIndex* pindexNew, const CDiskBlockPos& pos)
{
    if (block.IsProofOfStake())
        pindexNew->SetProofOfStake();

    pindexNew->nTx = block.vtx.size();
    pindexNew->nChainTx = 0;
    pindexNew->nFile = pos.nFile;
    pindexNew->nDataPos = pos.nPos;
    pindexNew->nUndoPos = 0;
    pindexNew->nStatus |= BLOCK_HAVE_DATA;
    pindexNew->RaiseValidity(BLOCK_VALID_TRANSACTIONS);
    setDirtyBlockIndex.insert(pindexNew);

    if (pindexNew->pprev == nullptr || pindexNew->pprev->nChainTx) {
        // If pindexNew is the genesis block or all parents are BLOCK_VALID_TRANSACTIONS.
        std::deque<CBlockIndex*> queue;
        queue.push_back(pindexNew);

        // Recursively process any descendant blocks that now may be eligible to be connected.
        while (!queue.empty()) {
            CBlockIndex* pindex = queue.front();
            queue.pop_front();
            pindex->nChainTx = (pindex->pprev ? pindex->pprev->nChainTx : 0) + pindex->nTx;
            {
                LOCK(cs_nBlockSequenceId);
                pindex->nSequenceId = nBlockSequenceId++;
            }
            if (chainActive.Tip() == nullptr || !setBlockIndexCandidates.value_comp()(pindex, chainActive.Tip())) {
                setBlockIndexCandidates.insert(pindex);
            }
            std::pair<std::multimap<CBlockIndex*, CBlockIndex*>::iterator, std::multimap<CBlockIndex*, CBlockIndex*>::iterator> range = mapBlocksUnlinked.equal_range(pindex);
            while (range.first != range.second) {
                std::multimap<CBlockIndex*, CBlockIndex*>::iterator it = range.first;
                queue.push_back(it->second);
                range.first++;
                mapBlocksUnlinked.erase(it);
            }
        }
    } else {
        if (pindexNew->pprev && pindexNew->pprev->IsValid(BLOCK_VALID_TREE)) {
            mapBlocksUnlinked.insert(std::make_pair(pindexNew->pprev, pindexNew));
        }
    }

    return true;
}

bool FindBlockPos(CValidationState& state, CDiskBlockPos& pos, unsigned int nAddSize, unsigned int nHeight, uint64_t nTime, bool fKnown = false)
{
    LOCK(cs_LastBlockFile);

    unsigned int nFile = fKnown ? pos.nFile : nLastBlockFile;
    if (vinfoBlockFile.size() <= nFile) {
        vinfoBlockFile.resize(nFile + 1);
    }

    if (!fKnown) {
        while (vinfoBlockFile[nFile].nSize + nAddSize >= MAX_BLOCKFILE_SIZE) {
            nFile++;
            if (vinfoBlockFile.size() <= nFile) {
                vinfoBlockFile.resize(nFile + 1);
            }
        }
        pos.nFile = nFile;
        pos.nPos = vinfoBlockFile[nFile].nSize;
    }

    if ((int)nFile != nLastBlockFile) {
        if (!fKnown) {
            LogPrintf("Leaving block file %i: %s\n", nLastBlockFile, vinfoBlockFile[nLastBlockFile].ToString());
        }
        FlushBlockFile(!fKnown);
        nLastBlockFile = nFile;
    }

    vinfoBlockFile[nFile].AddBlock(nHeight, nTime);
    if (fKnown)
        vinfoBlockFile[nFile].nSize = std::max(pos.nPos + nAddSize, vinfoBlockFile[nFile].nSize);
    else
        vinfoBlockFile[nFile].nSize += nAddSize;

    if (!fKnown) {
        unsigned int nOldChunks = (pos.nPos + BLOCKFILE_CHUNK_SIZE - 1) / BLOCKFILE_CHUNK_SIZE;
        unsigned int nNewChunks = (vinfoBlockFile[nFile].nSize + BLOCKFILE_CHUNK_SIZE - 1) / BLOCKFILE_CHUNK_SIZE;
        if (nNewChunks > nOldChunks) {
            if (fPruneMode)
                fCheckForPruning = true;
            if (CheckDiskSpace(nNewChunks * BLOCKFILE_CHUNK_SIZE - pos.nPos)) {
                FILE* file = OpenBlockFile(pos);
                if (file) {
                    LogPrintf("Pre-allocating up to position 0x%x in blk%05u.dat\n", nNewChunks * BLOCKFILE_CHUNK_SIZE, pos.nFile);
                    AllocateFileRange(file, pos.nPos, nNewChunks * BLOCKFILE_CHUNK_SIZE - pos.nPos);
                    fclose(file);
                }
            } else
                return state.Error("out of disk space");
        }
    }

    setDirtyFileInfo.insert(nFile);
    return true;
}

bool FindUndoPos(CValidationState& state, int nFile, CDiskBlockPos& pos, unsigned int nAddSize)
{
    pos.nFile = nFile;

    LOCK(cs_LastBlockFile);

    unsigned int nNewSize;
    pos.nPos = vinfoBlockFile[nFile].nUndoSize;
    nNewSize = vinfoBlockFile[nFile].nUndoSize += nAddSize;
    setDirtyFileInfo.insert(nFile);

    unsigned int nOldChunks = (pos.nPos + UNDOFILE_CHUNK_SIZE - 1) / UNDOFILE_CHUNK_SIZE;
    unsigned int nNewChunks = (nNewSize + UNDOFILE_CHUNK_SIZE - 1) / UNDOFILE_CHUNK_SIZE;
    if (nNewChunks > nOldChunks) {
        if (fPruneMode)
            fCheckForPruning = true;
        if (CheckDiskSpace(nNewChunks * UNDOFILE_CHUNK_SIZE - pos.nPos)) {
            FILE* file = OpenUndoFile(pos);
            if (file) {
                LogPrintf("Pre-allocating up to position 0x%x in rev%05u.dat\n", nNewChunks * UNDOFILE_CHUNK_SIZE, pos.nFile);
                AllocateFileRange(file, pos.nPos, nNewChunks * UNDOFILE_CHUNK_SIZE - pos.nPos);
                fclose(file);
            }
        } else
            return state.Error("out of disk space");
    }

    return true;
}

bool CheckBlockHeader(const CBlockHeader& block, CValidationState& state, const Consensus::Params& consensusParams, bool fCheckPOW)
{
    // Check proof of work matches claimed amount
    if (fCheckPOW && !CheckProofOfWork(block.GetHash(), block.nBits, consensusParams))
        return state.DoS(50, false, REJECT_INVALID, "high-hash", false, "proof of work failed");

    // Check timestamp
    if (block.GetBlockTime() > GetAdjustedTime() + consensusParams.nMaxClockDrift)
        return state.Invalid(error("CheckBlockHeader() : block timestamp too far in the future"),
                             REJECT_INVALID, "time-too-new");

    return true;
}

bool CheckBlock(const CBlock& block, CValidationState& state, const Consensus::Params& consensusParams, bool fCheckPOW, bool fCheckMerkleRoot)
{
    // These are checks that are independent of context.
    const bool IsPoS = block.IsProofOfStake();
    if (IsPoS)
        LogPrintf("%s: Proof of stake found. block=%s\n", __func__, block.GetHash().ToString().c_str());

    if (block.fChecked)
        return true;

    // Check that the header is valid (particularly PoW).  This is mostly
    // redundant with the call in AcceptBlockHeader.
    if (!IsPoS && !CheckBlockHeader(block, state, consensusParams, fCheckPOW))
        return false;

    // Check the merkle root.
    if (fCheckMerkleRoot) {
        bool mutated;
        uint256 hashMerkleRoot2 = BlockMerkleRoot(block, &mutated);
        if (block.hashMerkleRoot != hashMerkleRoot2)
            return state.DoS(100, error("CheckBlock(): hashMerkleRoot mismatch"),
                REJECT_INVALID, "bad-txnmrklroot", true);

        // Check for merkle tree malleability (CVE-2012-2459): repeating sequences
        // of transactions in a block without affecting the merkle root of a block,
        // while still invalidating it.
        if (mutated)
            return state.DoS(100, error("CheckBlock(): duplicate transaction"),
                REJECT_INVALID, "bad-txns-duplicate", true);
    }

    // All potential-corruption validation must be done before we do any
    // transaction validation, as otherwise we may mark the header as invalid
    // because we receive the wrong transactions for it.

    // Size limits
    if (block.vtx.empty() || block.vtx.size() > MAX_BLOCK_SIZE || ::GetSerializeSize(block, SER_NETWORK, PROTOCOL_VERSION) > MAX_BLOCK_SIZE)
        return state.DoS(100, false, REJECT_INVALID, "bad-blk-length", false, "size limits failed");

    // First transaction must be coinbase, the rest must not be
    if (block.vtx.empty() || !block.vtx[0]->IsCoinBase())
        return state.DoS(100, false, REJECT_INVALID, "bad-cb-missing", false, "first tx is not coinbase");
    for (unsigned int i = 1; i < block.vtx.size(); i++)
        if (block.vtx[i]->IsCoinBase())
            return state.DoS(100, false, REJECT_INVALID, "bad-cb-multiple", false, "more than one coinbase");

    // Check timestamp
    if (block.GetBlockTime() > GetAdjustedTime() + consensusParams.nMaxClockDrift)
        return state.DoS(50, error("CheckBlock() : coinbase timestamp is too early"), REJECT_INVALID, "bad-cb-time");

    if (IsPoS) {
        // Coinbase output should be empty if proof-of-stake block
        if (block.vtx[0]->vout.size() != 1 || !block.vtx[0]->vout[0].IsEmpty())
            return state.DoS(100, error("%s : coinbase output not empty for proof-of-stake block", __func__));

        // Second transaction must be coinstake, the rest must not be
        if (block.vtx.empty() || !block.vtx[1]->IsCoinStake())
            return state.DoS(100, error("%s : second tx is not coinstake", __func__));
        for (unsigned int i = 2; i < block.vtx.size(); i++)
            if (block.vtx[i]->IsCoinStake())
                return state.DoS(100, error("%s : more than one coinstake", __func__));
    }

    // DYNAMIC : CHECK TRANSACTIONS FOR INSTANTSEND
    if (sporkManager.IsSporkActive(SPORK_3_INSTANTSEND_BLOCK_FILTERING)) {
        // We should never accept block which conflicts with completed transaction lock,
        // that's why this is in CheckBlock unlike coinbase payee/amount.
        // Require other nodes to comply, send them some data in case they are missing it.
        for (const auto& tx : block.vtx) {
            // skip coinbase, it has no inputs
            if (tx->IsCoinBase())
                continue;
            // LOOK FOR TRANSACTION LOCK IN OUR MAP OF OUTPOINTS
            for (const auto& txin : tx->vin) {
                uint256 hashLocked;
                if (instantsend.GetLockedOutPointTxHash(txin.prevout, hashLocked) && hashLocked != tx->GetHash()) {
                    // The node which relayed this will have to swtich later,
                    // relaying instantsend data won't help it.
                    LOCK(cs_main);
                    mapRejectedBlocks.insert(std::make_pair(block.GetHash(), GetTime()));
                    return state.DoS(0, error("CheckBlock(DYN): transaction %s conflicts with transaction lock %s", tx->GetHash().ToString(), hashLocked.ToString()),
                        REJECT_INVALID, "conflict-tx-lock");
                }
            }
        }
    } else {
        LogPrintf("CheckBlock(DYN): spork is off, skipping transaction locking checks\n");
    }

    // END DYNAMIC

    // Check transactions
    for (const auto& tx : block.vtx) {
        if (!CheckTransaction(*tx, state))
            return state.Invalid(false, state.GetRejectCode(), state.GetRejectReason(),
                strprintf("Transaction check failed (tx hash %s) %s", tx->GetHash().ToString(), state.GetDebugMessage()));

        for (const auto& txout : tx->vout) {
            if (IsTransactionFluid(txout.scriptPubKey)) {
                std::string strErrorMessage;
                if (!fluid.CheckFluidOperationScript(txout.scriptPubKey, block.nTime, strErrorMessage)) {
                    return error("CheckBlock(): %s, Block %s failed with %s",
                        strErrorMessage,
                        tx->GetHash().ToString(),
                        FormatStateMessage(state));
                }
            }
        }
        if (!fluid.CheckTransactionToBlock(*tx, block))
            return error("CheckBlock(): Fluid transaction violated filtration rules, offender %s", tx->GetHash().ToString());
    }

    unsigned int nSigOps = 0;
    for (const auto& tx : block.vtx) {
        nSigOps += GetLegacySigOpCount(*tx);
    }
    // sigops limits (relaxed)
    if (nSigOps > MAX_BLOCK_SIGOPS_COST)
        return state.DoS(100, false, REJECT_INVALID, "bad-blk-sigops", false, "out-of-bounds SigOpCount");

    if (fCheckPOW && fCheckMerkleRoot)
        block.fChecked = true;

    return true;
}

static bool CheckIndexAgainstCheckpoint(const CBlockIndex* pindexPrev, CValidationState& state, const CChainParams& chainparams, const uint256& hash)
{
    if (*pindexPrev->phashBlock == chainparams.GetConsensus().hashGenesisBlock)
        return true;

    int nHeight = pindexPrev->nHeight + 1;
    // Don't accept any forks from the main chain prior to last checkpoint
    CBlockIndex* pcheckpoint = Checkpoints::GetLastCheckpoint(chainparams.Checkpoints());
    if (pcheckpoint && nHeight < pcheckpoint->nHeight)
        return state.DoS(100, error("%s: forked chain older than last checkpoint (height %d)", __func__, nHeight));

    return true;
}

bool ContextualCheckBlockHeader(const CBlockHeader& block, CValidationState& state, const Consensus::Params& consensusParams, const CBlockIndex* pindexPrev, int64_t nAdjustedTime, bool fProofOfStake)
{
    int nHeight = (pindexPrev->nHeight + 1);
    uint256 hash = block.GetHash();

    if (hash == Params().GetConsensus().hashGenesisBlock)
        return true;

    if (block.nBits != GetNextWorkRequired(pindexPrev, block, fProofOfStake, consensusParams)) {
        return state.DoS(100, error("%s : incorrect proof of work at %d", __func__, nHeight),
            REJECT_INVALID, "bad-diffbits");
    }

    // Check timestamp against prev
    if (block.GetBlockTime() <= pindexPrev->GetMedianTimePast())
        return state.Invalid(error("%s: block's timestamp is too early", __func__),
            REJECT_INVALID, "time-too-old");

    // Check timestamp
    if (block.GetBlockTime() > nAdjustedTime + MAX_FUTURE_BLOCK_TIME)
        return state.Invalid(false, REJECT_INVALID, "time-too-new", "block timestamp too far in the future");

    return true;
}

bool ContextualCheckBlock(const CBlock& block, CValidationState& state, const Consensus::Params& consensusParams, const CBlockIndex* pindexPrev, CAssetsCache* assetCache)
{
    const int nHeight = pindexPrev == nullptr ? 0 : pindexPrev->nHeight + 1;

    // Start enforcing BIP113 (Median Time Past) using versionbits logic.
    int nLockTimeFlags = 0;
    if (VersionBitsState(pindexPrev, consensusParams, Consensus::DEPLOYMENT_CSV, versionbitscache) == THRESHOLD_ACTIVE) {
        nLockTimeFlags |= LOCKTIME_MEDIAN_TIME_PAST;
    }

    int64_t nLockTimeCutoff = (nLockTimeFlags & LOCKTIME_MEDIAN_TIME_PAST) ? pindexPrev->GetMedianTimePast() : block.GetBlockTime();

    // Check that all transactions are finalized and not over-sized
    // Also count sigops
    for (const auto& tx : block.vtx) {
        if (pindexPrev != nullptr) {
            if (!fluid.CheckTransactionToBlock(*tx, pindexPrev->GetBlockHeader()))
                return state.DoS(10, error("%s: contains an invalid fluid transaction", __func__), REJECT_INVALID, "invalid-fluid-txns");
        }
        if (!IsFinalTx(*tx, nHeight, nLockTimeCutoff)) {
            return state.DoS(10, error("%s: contains a non-final transaction", __func__), REJECT_INVALID, "bad-txns-nonfinal");
        }
    }

    // Enforce block.nVersion=2 rule that the coinbase starts with serialized block height
    // if 750 of the last 1,000 blocks are version 2 or greater (51/100 if testnet):
    if (block.nVersion >= 2 && IsSuperMajority(2, pindexPrev, consensusParams.nMajorityEnforceBlockUpgrade, consensusParams)) {
        CScript expect = CScript() << nHeight;
        if (block.vtx[0]->vin[0].scriptSig.size() < expect.size() ||
            !std::equal(expect.begin(), expect.end(), block.vtx[0]->vin[0].scriptSig.begin())) {
            return state.DoS(100, error("%s: block height mismatch in coinbase", __func__), REJECT_INVALID, "bad-cb-height");
        }
    }

    // If Fluid transaction present, has it been adhered to?
    CDynamicAddress mintAddress;
    CAmount fluidIssuance;

    if (fluid.GetMintingInstructions(pindexPrev, mintAddress, fluidIssuance)) {
        bool found = false;

        CScript script;
        assert(mintAddress.IsValid());
        if (!mintAddress.IsScript()) {
            script = GetScriptForDestination(mintAddress.Get());
        } else {
            CScriptID scriptID = boost::get<CScriptID>(mintAddress.Get());
            script = CScript() << OP_HASH160 << ToByteVector(scriptID) << OP_EQUAL;
        }

        for (const CTxOut& output : block.vtx[0]->vout) {
            if (output.scriptPubKey == script) {
                if (output.nValue == fluidIssuance) {
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            return state.DoS(100, error("%s: fluid issuance not complied to", __func__), REJECT_INVALID, "cb-no-fluid-mint");
        }
    }

    return true;
}

static bool AcceptBlockHeader(const CBlock& block, CValidationState& state, const CChainParams& chainparams, CBlockIndex** ppindex)
{
    AssertLockHeld(cs_main);
    // Check for duplicate
    uint256 hash = block.GetHash();
    BlockMap::iterator miSelf = mapBlockIndex.find(hash);
    CBlockIndex* pindex = nullptr;

    // TODO : ENABLE BLOCK CACHE IN SPECIFIC CASES
    if (hash != chainparams.GetConsensus().hashGenesisBlock) {
        if (miSelf != mapBlockIndex.end() && !miSelf->second) {
            mapBlockIndex.erase(hash);
        }
        if (miSelf != mapBlockIndex.end() && miSelf->second) {
            // Block header is already known.
            pindex = miSelf->second;
            if (ppindex)
                *ppindex = pindex;
            if (pindex->nStatus & BLOCK_FAILED_MASK)
                return state.Invalid(error("%s: block %s is marked invalid", __func__, hash.ToString()), 0, "duplicate");
            return true;
        }
        
        if (!CheckBlockHeader(block, state, chainparams.GetConsensus(), false))
            return error("%s: Consensus::CheckBlockHeader: %s, %s", __func__, hash.ToString(), FormatStateMessage(state));

        // Get prev block index
        CBlockIndex* pindexPrev = nullptr;
        BlockMap::iterator mi = mapBlockIndex.find(block.hashPrevBlock);
        if (mi == mapBlockIndex.end())
            return state.DoS(10, error("%s: prev block not found", __func__), 0, "bad-prevblk");
        pindexPrev = (*mi).second;
        if (pindexPrev->nStatus & BLOCK_FAILED_MASK)
            return state.DoS(100, error("%s: prev block invalid", __func__), REJECT_INVALID, "bad-prevblk");

        assert(pindexPrev);
        if (fCheckpointsEnabled && !CheckIndexAgainstCheckpoint(pindexPrev, state, chainparams, hash))
            return error("%s: CheckIndexAgainstCheckpoint(): %s", __func__, state.GetRejectReason().c_str());

        if (!block.IsHeaderOnly())
            if (!ContextualCheckBlockHeader(block, state, chainparams.GetConsensus(), pindexPrev, GetAdjustedTime(), block.IsProofOfStake()))
                return error("%s: Consensus::ContextualCheckBlockHeader: %s, %s", __func__, hash.ToString(), FormatStateMessage(state));
    }
    if (pindex == nullptr)
        pindex = AddToBlockIndex(block);

    if (ppindex)
        *ppindex = pindex;

    CheckBlockIndex(chainparams.GetConsensus());

    // Notify external listeners about accepted block header
    GetMainSignals().AcceptedBlockHeader(pindex);

    return true;
}

// Exposed wrapper for AcceptBlockHeader
bool ProcessNewBlockHeaders(const std::vector<CBlockHeader>& headers, CValidationState& state, const CChainParams& chainparams, const CBlockIndex** ppindex, CBlockHeader *first_invalid)
{
    {
        LOCK(cs_main);
        for (const CBlockHeader& header : headers) {
            CBlockIndex* pindex = nullptr; // Use a temp pindex instead of ppindex to avoid a const_cast
            if (!AcceptBlockHeader(header, state, chainparams, &pindex)) {
                return false;
            }
            if (ppindex) {
                *ppindex = pindex;
            }
        }
    }
    NotifyHeaderTip();
    return true;
}

static bool CheckWork(const CBlock& block)
{
    if (block.IsProofOfStake())
        return true;

    uint256 hash = block.GetHash();
    arith_uint256 hash_target = arith_uint256().SetCompact(block.nBits);
    if (UintToArith256(hash) > hash_target)
        return false;

    return true;
}

/** Store block on disk. If dbp is non-nullptr, the file is known to already reside on disk */
static bool AcceptBlock(const std::shared_ptr<const CBlock>& pblock, CValidationState& state, const CChainParams& chainparams, CBlockIndex** ppindex, bool fRequested, const CDiskBlockPos* dbp, bool* fNewBlock, bool fFromLoad = false)
{
    const CBlock& block = *pblock;

    if (fNewBlock)
        *fNewBlock = false;
    AssertLockHeld(cs_main);

    // Get prev block index
    CBlockIndex* pindexPrev = nullptr;
    if (block.GetHash() != Params().GenesisBlock().GetHash()) {
        BlockMap::iterator mi = mapBlockIndex.find(block.hashPrevBlock);
        if (mi == mapBlockIndex.end())
            return state.DoS(0, error("%s : prev block %s not found", __func__, block.hashPrevBlock.GetHex()), 0, "bad-prevblk");
        pindexPrev = (*mi).second;
        if (pindexPrev->nStatus & BLOCK_FAILED_MASK) {
            //If this "invalid" block is an exact match from the checkpoints, then reconsider it
            if (Checkpoints::CheckBlock(pindexPrev->nHeight, block.hashPrevBlock, true)) {
                LogPrintf("%s : Reconsidering block %s height %d\n", __func__, pindexPrev->GetBlockHash().GetHex(), pindexPrev->nHeight);
                CValidationState statePrev;
                ReconsiderBlock(statePrev, pindexPrev);
                if (statePrev.IsValid()) {
                    ActivateBestChain(statePrev, chainparams, pblock);
                    return true;
                }
            }
            return state.DoS(100, error("%s : prev block %s is invalid, unable to add block %s", __func__, block.hashPrevBlock.GetHex(), block.GetHash().GetHex()),
                             REJECT_INVALID, "bad-prevblk");
        }
    }

    if (block.GetHash() != Params().GenesisBlock().GetHash() && !CheckWork(block))
        return false;

    CBlockIndex* pindexDummy = nullptr;
    CBlockIndex*& pindex = ppindex ? *ppindex : pindexDummy;
    if (!AcceptBlockHeader(block, state, chainparams, &pindex))
        return false;

    int nHeight = pindex->nHeight;

    // Try to process all requested blocks that we don't have, but only
    // process an unrequested block if it's new and has enough work to
    // advance our tip, and isn't too many blocks ahead.
    bool fAlreadyHave = pindex->nStatus & BLOCK_HAVE_DATA;
    bool fHasMoreWork = (chainActive.Tip() ? pindex->nChainWork > chainActive.Tip()->nChainWork : true);
    // Blocks that are too out-of-order needlessly limit the effectiveness of
    // pruning, because pruning will not delete block files that contain any
    // blocks which are too close in height to the tip.  Apply this test
    // regardless of whether pruning is enabled; it should generally be safe to
    // not process unrequested blocks.
    bool fTooFarAhead = (pindex->nHeight > int(chainActive.Height() + MIN_BLOCKS_TO_KEEP));

    // TODO: deal better with return value and error conditions for duplicate
    // and unrequested blocks.
    if (fAlreadyHave)
        return true;
    if (!fRequested) { // If we didn't ask for it:
        if (pindex->nTx != 0)
            return true; // This is a previously-processed block that was pruned
        if (!fHasMoreWork)
            return true; // Don't process less-work chains
        if (fTooFarAhead)
            return true; // Block height is too high
    }
    if (fNewBlock)
        *fNewBlock = true;

    auto currentActiveAssetCache = GetCurrentAssetCache();
    // Dont force the CheckBlock asset duplciates when checking from this state
    if (!CheckBlock(block, state, chainparams.GetConsensus(), true, true) ||
        !ContextualCheckBlock(block, state, chainparams.GetConsensus(), pindex->pprev, currentActiveAssetCache)) {
        if (fFromLoad && state.GetRejectReason() == "bad-txns-transfer-asset-bad-deserialize") {
            // keep going, we are only loading blocks from database
            CValidationState new_state;
            state = new_state;
        } else {
            if (state.IsInvalid() && !state.CorruptionPossible()) {
                pindex->nStatus |= BLOCK_FAILED_VALID;
                setDirtyBlockIndex.insert(pindex);
            }
            return error("%s: %s", __func__, FormatStateMessage(state));
        }
    }

    if (block.IsProofOfStake()) {
        LOCK(cs_main);

        // Blocks arrives in order, so if prev block is not the tip then we are on a fork.
        // Extra info: duplicated blocks are skipping this checks, so we don't have to worry about those here.
        bool isBlockFromFork = pindexPrev != nullptr && chainActive.Tip() != pindexPrev;

        // Coin stake
        CTransactionRef stakeTxIn = block.vtx[1];
        // Inputs
        std::vector<CTxIn> vInputs;
        for (const CTxIn& stakeIn : stakeTxIn->vin) {
            vInputs.push_back(stakeIn);
        }
        const bool fHasInputs = !vInputs.empty();

        // Check for serial double spent on the same block, TODO: Move this to the proper method..
        for (const CTransactionRef& tx : block.vtx) {
            for (const CTxIn& in: tx->vin) {
                if(tx->IsCoinStake())
                    continue;
                if(fHasInputs) {
                    // Check if coinstake input is double spent inside the same block
                    for (const CTxIn& stakeIn : vInputs){
                        if(stakeIn.prevout == in.prevout){
                            // double spent coinstake input inside block
                            return error("%s: double spent coinstake input inside block", __func__);
                        }
                    }
                }
            }
        }
        // Check whether is a fork or not
        /*
        if (isBlockFromFork) {
            // Start at the block we're adding on to
            CBlockIndex *prev = pindexPrev;

            CBlock bl;
            if (!ReadBlockFromDisk(bl, prev, chainparams.GetConsensus()))
                return error("%s: previous block %s not on disk", __func__, prev->GetBlockHash().GetHex());

            int readBlock = 0;
            // Go backwards on the forked chain up to the split
            while (!chainActive.Contains(prev)) {
                // Increase amount of read blocks
                readBlock++;
                // Check if the forked chain is longer than the max reorg limit
                if (readBlock == chainparams.GetConsensus().MaxReorganizationDepth()) {
                    // TODO: Remove this chain from disk.
                    return error("%s: forked chain longer than maximum reorg limit", __func__);
                }
                // Loop through every input from said block
                for (const CTransactionRef& tx : bl.vtx) {
                    for (const CTxIn &in: tx->vin) {
                        // Loop through every input of the staking tx
                        for (const CTxIn &stakeIn : vInputs) {
                            // if it's already spent

                            // First regular staking check
                            if (fHasInputs) {
                                if (stakeIn.prevout == in.prevout) {
                                    return state.DoS(100, error("%s: input already spent on a previous block",
                                                                __func__));
                                }
                            }
                        }
                    }
                }
                // Prev block
                prev = prev->pprev;
                if (!ReadBlockFromDisk(bl, prev, chainparams.GetConsensus()))
                    // Previous block not on disk
                    return error("%s: previous block %s not on disk", __func__, prev->GetBlockHash().GetHex());

            }
        }*/

        // TODO (PoS): review this code conversion
        // Check if the inputs were spent on the main chain
        const CCoinsViewCache coins(pcoinsTip);
        for (const CTxIn& in: stakeTxIn->vin) {
            const Coin coin = coins.AccessCoin(in.prevout);
            if(coin.IsNull() && !isBlockFromFork){
                // No coins on the main chain
                return error("%s: coin stake inputs not available on main chain, received height %d vs current %d", __func__, nHeight, chainActive.Height());
            }
            if(coin.IsSpent()){
                if(!isBlockFromFork){
                    // Coins not available
                    return error("%s: coin stake inputs already spent in main chain", __func__);
                }
            }
        }
    }

    // Header is valid/has work, merkle tree is good...RELAY NOW
    // (but if it does not build on our best tip, let the SendMessages loop relay it)
    if (!IsInitialBlockDownload() && chainActive.Tip() == pindex->pprev)
        GetMainSignals().NewPoWValidBlock(pindex, pblock);

    // Write block to history file
    try {
        unsigned int nBlockSize = ::GetSerializeSize(block, SER_DISK, CLIENT_VERSION);
        CDiskBlockPos blockPos;
        if (dbp != nullptr)
            blockPos = *dbp;
        if (!FindBlockPos(state, blockPos, nBlockSize + 8, nHeight, block.GetBlockTime(), dbp != nullptr))
            return error("AcceptBlock(): FindBlockPos failed");
        if (dbp == nullptr)
            if (!WriteBlockToDisk(block, blockPos, chainparams.MessageStart()))
                AbortNode(state, "Failed to write block");
        if (!ReceivedBlockTransactions(block, state, pindex, blockPos))
            return error("AcceptBlock(): ReceivedBlockTransactions failed");
    } catch (const std::runtime_error& e) {
        return AbortNode(state, std::string("System error: ") + e.what());
    }

    if (fCheckForPruning)
        FlushStateToDisk(state, FLUSH_STATE_NONE); // we just allocated more disk space for block files

    return true;
}

static bool IsSuperMajority(int minVersion, const CBlockIndex* pstart, unsigned nRequired, const Consensus::Params& consensusParams)
{
    unsigned int nFound = 0;
    for (int i = 0; i < consensusParams.nMajorityWindow && nFound < nRequired && pstart != nullptr; i++) {
        if (pstart->nVersion >= minVersion)
            ++nFound;
        pstart = pstart->pprev;
    }
    return (nFound >= nRequired);
}

bool ProcessNewBlock(const CChainParams& chainparams, const std::shared_ptr<const CBlock> pblock, bool fForceProcessing, bool* fNewBlock)
{

    if (pblock->IsProofOfStake() && !CheckBlockSignature(*pblock))
        return error("ProcessNewBlock() : bad proof-of-stake block signature");

    {
        CBlockIndex* pindex = nullptr;
        if (fNewBlock)
            *fNewBlock = false;
        CValidationState state;
        // Ensure that CheckBlock() passes before calling AcceptBlock, as
        // belt-and-suspenders.
        bool ret = CheckBlock(*pblock, state, chainparams.GetConsensus());

        LOCK(cs_main);

        if (ret) {
            // Store to disk
            ret = AcceptBlock(pblock, state, chainparams, &pindex, fForceProcessing, nullptr, fNewBlock);
        }
        CheckBlockIndex(chainparams.GetConsensus());
        if (!ret) {
            GetMainSignals().BlockChecked(*pblock, state);
            return error("%s: AcceptBlock FAILED", __func__);
        }
    }

    NotifyHeaderTip();

    CValidationState state; // Only used to report errors, not invalidity - ignore it
    if (!ActivateBestChain(state, chainparams, pblock))
        return error("%s: ActivateBestChain failed", __func__);

#ifdef ENABLE_WALLET
extern CWallet* pwalletMain;

    if (pwalletMain) {
        // If turned on MultiSend will send a transaction (or more) on the after maturity of a stake
        if (pwalletMain->isMultiSendEnabled())
            pwalletMain->MultiSend();

        // If turned on Auto Combine will scan wallet for dust to combine
        if (pwalletMain->fCombineDust)
            pwalletMain->AutoCombineDust();
    }
#endif //ENABLE_WALLET

    LogPrint("validation", "%s : ACCEPTED\n", __func__);
    return true;
}

bool TestBlockValidity(CValidationState& state, const CChainParams& chainparams, const CBlock& block, CBlockIndex* pindexPrev, bool fCheckPOW, bool fCheckMerkleRoot)
{
    AssertLockHeld(cs_main);
    assert(pindexPrev && pindexPrev == chainActive.Tip());
    if (fCheckpointsEnabled && !CheckIndexAgainstCheckpoint(pindexPrev, state, chainparams, block.GetHash()))
        return error("%s: CheckIndexAgainstCheckpoint(): %s", __func__, state.GetRejectReason().c_str());

    CCoinsViewCache viewNew(pcoinsTip);
    CBlockIndex indexDummy(block);
    indexDummy.pprev = pindexPrev;
    indexDummy.nHeight = pindexPrev->nHeight + 1;

    /** ASSET START */
    CAssetsCache assetCache = *GetCurrentAssetCache();
    /** ASSET END */

    // NOTE: CheckBlockHeader is called by CheckBlock
    if (!ContextualCheckBlockHeader(block, state, chainparams.GetConsensus(), pindexPrev, GetAdjustedTime(), block.IsProofOfStake()))
        return error("%s: Consensus::ContextualCheckBlockHeader: %s", __func__, FormatStateMessage(state));
    if (!CheckBlock(block, state, chainparams.GetConsensus(), fCheckPOW, fCheckMerkleRoot))
        return error("%s: Consensus::CheckBlock: %s", __func__, FormatStateMessage(state));
    if (!ContextualCheckBlock(block, state, chainparams.GetConsensus(), pindexPrev, &assetCache))
        return error("%s: Consensus::ContextualCheckBlock: %s", __func__, FormatStateMessage(state));
    if (!ConnectBlock(block, state, &indexDummy, viewNew, chainparams, &assetCache, true))
        return false;
    assert(state.IsValid());

    return true;
}

/**
 * BLOCK PRUNING CODE
 */

/* Calculate the amount of disk space the block & undo files currently use */
uint64_t CalculateCurrentUsage()
{
    uint64_t retval = 0;
    for (const CBlockFileInfo& file : vinfoBlockFile) {
        retval += file.nSize + file.nUndoSize;
    }
    return retval;
}

/* Prune a block file (modify associated database entries)*/
void PruneOneBlockFile(const int fileNumber)
{
    for (BlockMap::iterator it = mapBlockIndex.begin(); it != mapBlockIndex.end(); ++it) {
        CBlockIndex* pindex = it->second;
        if (pindex->nFile == fileNumber) {
            pindex->nStatus &= ~BLOCK_HAVE_DATA;
            pindex->nStatus &= ~BLOCK_HAVE_UNDO;
            pindex->nFile = 0;
            pindex->nDataPos = 0;
            pindex->nUndoPos = 0;
            setDirtyBlockIndex.insert(pindex);

            // Prune from mapBlocksUnlinked -- any block we prune would have
            // to be downloaded again in order to consider its chain, at which
            // point it would be considered as a candidate for
            // mapBlocksUnlinked or setBlockIndexCandidates.
            std::pair<std::multimap<CBlockIndex*, CBlockIndex*>::iterator, std::multimap<CBlockIndex*, CBlockIndex*>::iterator> range = mapBlocksUnlinked.equal_range(pindex->pprev);
            while (range.first != range.second) {
                std::multimap<CBlockIndex*, CBlockIndex*>::iterator _it = range.first;
                range.first++;
                if (_it->second == pindex) {
                    mapBlocksUnlinked.erase(_it);
                }
            }
        }
    }

    vinfoBlockFile[fileNumber].SetNull();
    setDirtyFileInfo.insert(fileNumber);
}


void UnlinkPrunedFiles(const std::set<int>& setFilesToPrune)
{
    for (std::set<int>::iterator it = setFilesToPrune.begin(); it != setFilesToPrune.end(); ++it) {
        CDiskBlockPos pos(*it, 0);
        boost::filesystem::remove(GetBlockPosFilename(pos, "blk"));
        boost::filesystem::remove(GetBlockPosFilename(pos, "rev"));
        LogPrintf("Prune: %s deleted blk/rev (%05u)\n", __func__, *it);
    }
}

/* Calculate the block/rev files to delete based on height specified by user with RPC command pruneblockchain */
void FindFilesToPruneManual(std::set<int>& setFilesToPrune, int nManualPruneHeight)
{
    assert(fPruneMode && nManualPruneHeight > 0);

    LOCK2(cs_main, cs_LastBlockFile);
    if (chainActive.Tip() == nullptr)
        return;

    // last block to prune is the lesser of (user-specified height, MIN_BLOCKS_TO_KEEP from the tip)
    unsigned int nLastBlockWeCanPrune = std::min((unsigned)nManualPruneHeight, chainActive.Tip()->nHeight - MIN_BLOCKS_TO_KEEP);
    int count = 0;
    for (int fileNumber = 0; fileNumber < nLastBlockFile; fileNumber++) {
        if (vinfoBlockFile[fileNumber].nSize == 0 || vinfoBlockFile[fileNumber].nHeightLast > nLastBlockWeCanPrune)
            continue;
        PruneOneBlockFile(fileNumber);
        setFilesToPrune.insert(fileNumber);
        count++;
    }
    LogPrintf("Prune (Manual): prune_height=%d removed %d blk/rev pairs\n", nLastBlockWeCanPrune, count);
}

/* This function is called from the RPC code for pruneblockchain */
void PruneBlockFilesManual(int nManualPruneHeight)
{
    CValidationState state;
    FlushStateToDisk(state, FLUSH_STATE_NONE, nManualPruneHeight);
}

/* Calculate the block/rev files that should be deleted to remain under target*/
void FindFilesToPrune(std::set<int>& setFilesToPrune, uint64_t nPruneAfterHeight)
{
    LOCK2(cs_main, cs_LastBlockFile);
    if (chainActive.Tip() == nullptr || nPruneTarget == 0) {
        return;
    }
    if ((uint64_t)chainActive.Tip()->nHeight <= nPruneAfterHeight) {
        return;
    }

    unsigned int nLastBlockWeCanPrune = chainActive.Tip()->nHeight - MIN_BLOCKS_TO_KEEP;
    uint64_t nCurrentUsage = CalculateCurrentUsage();
    // We don't check to prune until after we've allocated new space for files
    // So we should leave a buffer under our target to account for another allocation
    // before the next pruning.
    uint64_t nBuffer = BLOCKFILE_CHUNK_SIZE + UNDOFILE_CHUNK_SIZE;
    uint64_t nBytesToPrune;
    int count = 0;

    if (nCurrentUsage + nBuffer >= nPruneTarget) {
        for (int fileNumber = 0; fileNumber < nLastBlockFile; fileNumber++) {
            nBytesToPrune = vinfoBlockFile[fileNumber].nSize + vinfoBlockFile[fileNumber].nUndoSize;

            if (vinfoBlockFile[fileNumber].nSize == 0)
                continue;

            if (nCurrentUsage + nBuffer < nPruneTarget) // are we below our target?
                break;

            // don't prune files that could have a block within MIN_BLOCKS_TO_KEEP of the main chain's tip but keep scanning
            if (vinfoBlockFile[fileNumber].nHeightLast > nLastBlockWeCanPrune)
                continue;

            PruneOneBlockFile(fileNumber);
            // Queue up the files for removal
            setFilesToPrune.insert(fileNumber);
            nCurrentUsage -= nBytesToPrune;
            count++;
        }
    }

    LogPrint("prune", "Prune: target=%dMiB actual=%dMiB diff=%dMiB max_prune_height=%d removed %d blk/rev pairs\n",
        nPruneTarget / 1024 / 1024, nCurrentUsage / 1024 / 1024,
        ((int64_t)nPruneTarget - (int64_t)nCurrentUsage) / 1024 / 1024,
        nLastBlockWeCanPrune, count);
}

bool CheckDiskSpace(uint64_t nAdditionalBytes)
{
    uint64_t nFreeBytesAvailable = boost::filesystem::space(GetDataDir()).available;

    // Check for nMinDiskSpace bytes (currently 50MB)
    if (nFreeBytesAvailable < nMinDiskSpace + nAdditionalBytes)
        return AbortNode("Disk space is low!", _("Error: Disk space is low!"));

    return true;
}

FILE* OpenDiskFile(const CDiskBlockPos& pos, const char* prefix, bool fReadOnly)
{
    if (pos.IsNull())
        return nullptr;
    boost::filesystem::path path = GetBlockPosFilename(pos, prefix);
    boost::filesystem::create_directories(path.parent_path());
    FILE* file = fopen(path.string().c_str(), "rb+");
    if (!file && !fReadOnly)
        file = fopen(path.string().c_str(), "wb+");
    if (!file) {
        LogPrintf("Unable to open file %s\n", path.string());
        return nullptr;
    }
    if (pos.nPos) {
        if (fseek(file, pos.nPos, SEEK_SET)) {
            LogPrintf("Unable to seek to position %u of %s\n", pos.nPos, path.string());
            fclose(file);
            return nullptr;
        }
    }
    return file;
}

FILE* OpenBlockFile(const CDiskBlockPos& pos, bool fReadOnly)
{
    return OpenDiskFile(pos, "blk", fReadOnly);
}

FILE* OpenUndoFile(const CDiskBlockPos& pos, bool fReadOnly)
{
    return OpenDiskFile(pos, "rev", fReadOnly);
}

boost::filesystem::path GetBlockPosFilename(const CDiskBlockPos& pos, const char* prefix)
{
    return GetDataDir() / "blocks" / strprintf("%s%05u.dat", prefix, pos.nFile);
}

CBlockIndex* InsertBlockIndex(uint256 hash)
{
    if (hash.IsNull())
        return nullptr;

    // Return existing
    BlockMap::iterator mi = mapBlockIndex.find(hash);
    if (mi != mapBlockIndex.end())
        return (*mi).second;

    // Create new
    CBlockIndex* pindexNew = new CBlockIndex();
    if (!pindexNew)
        throw std::runtime_error("LoadBlockIndex(): new CBlockIndex failed");
    mi = mapBlockIndex.insert(std::make_pair(hash, pindexNew)).first;
    pindexNew->phashBlock = &((*mi).first);

    return pindexNew;
}

bool static LoadBlockIndexDB(const CChainParams& chainparams)
{
    if (!pblocktree->LoadBlockIndexGuts(InsertBlockIndex))
        return false;

    boost::this_thread::interruption_point();

    // Calculate nChainWork
    std::vector<std::pair<int, CBlockIndex*> > vSortedByHeight;
    vSortedByHeight.reserve(mapBlockIndex.size());
    for (const std::pair<uint256, CBlockIndex*>& item : mapBlockIndex) {
        CBlockIndex* pindex = item.second;
        vSortedByHeight.push_back(std::make_pair(pindex->nHeight, pindex));
    }
    sort(vSortedByHeight.begin(), vSortedByHeight.end());
    for (const std::pair<int, CBlockIndex*>& item : vSortedByHeight) {
        CBlockIndex* pindex = item.second;
        pindex->nChainWork = (pindex->pprev ? pindex->pprev->nChainWork : 0) + GetBlockProof(*pindex);
        pindex->nTimeMax = (pindex->pprev ? std::max(pindex->pprev->nTimeMax, pindex->nTime) : pindex->nTime);
        // We can link the chain of blocks for which we've received transactions at some point.
        // Pruned nodes may have deleted the block.
        if (pindex->nTx > 0) {
            if (pindex->pprev) {
                if (pindex->pprev->nChainTx) {
                    pindex->nChainTx = pindex->pprev->nChainTx + pindex->nTx;
                } else {
                    pindex->nChainTx = 0;
                    mapBlocksUnlinked.insert(std::make_pair(pindex->pprev, pindex));
                }
            } else {
                pindex->nChainTx = pindex->nTx;
            }
        }
        if (pindex->IsValid(BLOCK_VALID_TRANSACTIONS) && (pindex->nChainTx || pindex->pprev == nullptr))
            setBlockIndexCandidates.insert(pindex);
        if (pindex->nStatus & BLOCK_FAILED_MASK && (!pindexBestInvalid || pindex->nChainWork > pindexBestInvalid->nChainWork))
            pindexBestInvalid = pindex;
        if (pindex->pprev)
            pindex->BuildSkip();
        if (pindex->IsValid(BLOCK_VALID_TREE) && (pindexBestHeader == nullptr || CBlockIndexWorkComparator()(pindexBestHeader, pindex)))
            pindexBestHeader = pindex;
    }

    // Load block file info
    pblocktree->ReadLastBlockFile(nLastBlockFile);
    vinfoBlockFile.resize(nLastBlockFile + 1);
    LogPrintf("%s: last block file = %i\n", __func__, nLastBlockFile);
    for (int nFile = 0; nFile <= nLastBlockFile; nFile++) {
        pblocktree->ReadBlockFileInfo(nFile, vinfoBlockFile[nFile]);
    }
    LogPrintf("%s: last block file info: %s\n", __func__, vinfoBlockFile[nLastBlockFile].ToString());
    for (int nFile = nLastBlockFile + 1; true; nFile++) {
        CBlockFileInfo info;
        if (pblocktree->ReadBlockFileInfo(nFile, info)) {
            vinfoBlockFile.push_back(info);
        } else {
            break;
        }
    }

    // Check presence of blk files
    LogPrintf("Checking all blk files are present...\n");
    std::set<int> setBlkDataFiles;
    for (const std::pair<uint256, CBlockIndex*>& item : mapBlockIndex) {
        CBlockIndex* pindex = item.second;
        if (pindex->nStatus & BLOCK_HAVE_DATA) {
            setBlkDataFiles.insert(pindex->nFile);
        }
    }
    for (std::set<int>::iterator it = setBlkDataFiles.begin(); it != setBlkDataFiles.end(); it++) {
        CDiskBlockPos pos(*it, 0);
        if (CAutoFile(OpenBlockFile(pos, true), SER_DISK, CLIENT_VERSION).IsNull()) {
            return false;
        }
    }

    // Check whether we have ever pruned block & undo files
    pblocktree->ReadFlag("prunedblockfiles", fHavePruned);
    if (fHavePruned)
        LogPrintf("LoadBlockIndexDB(): Block files have previously been pruned\n");

    // Check whether we need to continue reindexing
    bool fReindexing = false;
    pblocktree->ReadReindexing(fReindexing);
    fReindex |= fReindexing;

    // Check whether we have a transaction index
    pblocktree->ReadFlag("txindex", fTxIndex);
    LogPrintf("%s: transaction index %s\n", __func__, fTxIndex ? "enabled" : "disabled");

    // Check whether we have an asset index
    pblocktree->ReadFlag("assetindex", fAssetIndex);
    LogPrintf("%s: asset index %s\n", __func__, fAssetIndex ? "enabled" : "disabled");

    // Check whether we have an address index
    pblocktree->ReadFlag("addressindex", fAddressIndex);
    LogPrintf("%s: address index %s\n", __func__, fAddressIndex ? "enabled" : "disabled");

    // Check whether we have a timestamp index
    pblocktree->ReadFlag("timestampindex", fTimestampIndex);
    LogPrintf("%s: timestamp index %s\n", __func__, fTimestampIndex ? "enabled" : "disabled");

    // Check whether we have a spent index
    pblocktree->ReadFlag("spentindex", fSpentIndex);
    LogPrintf("%s: spent index %s\n", __func__, fSpentIndex ? "enabled" : "disabled");

    // Load pointer to end of best chain
    BlockMap::iterator it = mapBlockIndex.find(pcoinsTip->GetBestBlock());
    if (it == mapBlockIndex.end())
        return true;
    chainActive.SetTip(it->second);

    PruneBlockIndexCandidates();

    LogPrintf("%s: hashBestChain=%s height=%d date=%s progress=%f\n", __func__,
        chainActive.Tip()->GetBlockHash().ToString(), chainActive.Height(),
        DateTimeStrFormat("%Y-%m-%d %H:%M:%S", chainActive.Tip()->GetBlockTime()),
        GuessVerificationProgress(chainparams.TxData(), chainActive.Tip()));

    return true;
}

CVerifyDB::CVerifyDB()
{
    uiInterface.ShowProgress(_("Verifying blocks..."), 0);
}

CVerifyDB::~CVerifyDB()
{
    uiInterface.ShowProgress("", 100);
}

bool CVerifyDB::VerifyDB(const CChainParams& chainparams, CCoinsView* coinsview, int nCheckLevel, int nCheckDepth)
{
    LOCK(cs_main);
    if (chainActive.Tip() == nullptr || chainActive.Tip()->pprev == nullptr)
        return true;

    // Verify blocks in the best chain
    if (nCheckDepth <= 0)
        nCheckDepth = 1000000000; // suffices until the year 19000
    if (nCheckDepth > chainActive.Height())
        nCheckDepth = chainActive.Height();
    nCheckLevel = std::max(0, std::min(4, nCheckLevel));
    LogPrintf("Verifying last %i blocks at level %i\n", nCheckDepth, nCheckLevel);
    CCoinsViewCache coins(coinsview);
    CBlockIndex* pindexState = chainActive.Tip();
    CBlockIndex* pindexFailure = nullptr;
    int nGoodTransactions = 0;
    CValidationState state;
    int reportDone = 0;

    auto currentActiveAssetCache = GetCurrentAssetCache();
    CAssetsCache assetCache(*currentActiveAssetCache);
    LogPrintf("[0%%]...");
    for (CBlockIndex* pindex = chainActive.Tip(); pindex && pindex->pprev; pindex = pindex->pprev) 
    {
        boost::this_thread::interruption_point();
        int percentageDone = std::max(1, std::min(99, (int)(((double)(chainActive.Height() - pindex->nHeight)) / (double)nCheckDepth * (nCheckLevel >= 4 ? 50 : 100))));
        if (reportDone < percentageDone / 10) {
            // report every 10% step
            LogPrintf("[%d%%]...", percentageDone);
            reportDone = percentageDone / 10;
        }
        uiInterface.ShowProgress(_("Verifying blocks..."), percentageDone);
        if (pindex->nHeight < chainActive.Height() - nCheckDepth)
            break;
        if (fPruneMode && !(pindex->nStatus & BLOCK_HAVE_DATA)) {
            // If pruning, only go back as far as we have data.
            LogPrintf("VerifyDB(): block verification stopping at height %d (pruning, no data)\n", pindex->nHeight);
            break;
        }
        CBlock block;
        // check level 0: read from disk
        if (!ReadBlockFromDisk(block, pindex, chainparams.GetConsensus()))
            return error("VerifyDB(): *** ReadBlockFromDisk failed at %d, hash=%s", pindex->nHeight, pindex->GetBlockHash().ToString());
        // check level 1: verify block validity
        if (nCheckLevel >= 1 && !CheckBlock(block, state, chainparams.GetConsensus(), true, true)) // fCheckAssetDuplicate set to false, because we don't want to fail because the asset exists in our database, when loading blocks from our asset databse
            return error("%s: *** found bad block at %d, hash=%s (%s)\n", __func__,
                pindex->nHeight, pindex->GetBlockHash().ToString(), FormatStateMessage(state));
        // check level 2: verify undo validity
        if (nCheckLevel >= 2 && pindex) {
            CBlockUndo undo;
            CDiskBlockPos pos = pindex->GetUndoPos();
            if (!pos.IsNull()) {
                if (!UndoReadFromDisk(undo, pos, pindex->pprev->GetBlockHash()))
                    return error("VerifyDB(): *** found bad undo data at %d, hash=%s\n", pindex->nHeight, pindex->GetBlockHash().ToString());
            }
        }
        // check level 3: check for inconsistencies during memory-only disconnect of tip blocks
        if (nCheckLevel >= 3 && pindex == pindexState && (coins.DynamicMemoryUsage() + pcoinsTip->DynamicMemoryUsage()) <= nCoinCacheUsage) {
            DisconnectResult res = DisconnectBlock(block, state, pindex, coins, nCheckLevel, &assetCache, true, false);
            if (res == DISCONNECT_FAILED) {
                return error("VerifyDB(): *** irrecoverable inconsistency in block data at %d, hash=%s", pindex->nHeight, pindex->GetBlockHash().ToString());
            }
            pindexState = pindex->pprev;
            if (res == DISCONNECT_UNCLEAN) {
                nGoodTransactions = 0;
                pindexFailure = pindex;
            } else {
                nGoodTransactions += block.vtx.size();
            }
        }
        if (ShutdownRequested())
            return true;
    }
    if (pindexFailure)
        return error("VerifyDB(): *** coin database inconsistencies found (last %i blocks, %i good transactions before that)\n", chainActive.Height() - pindexFailure->nHeight + 1, nGoodTransactions);

    // check level 4: try reconnecting blocks
    if (nCheckLevel >= 4) {
        CBlockIndex* pindex = pindexState;
        while (pindex != chainActive.Tip()) {
            boost::this_thread::interruption_point();
            uiInterface.ShowProgress(_("Verifying blocks..."), std::max(1, std::min(99, 100 - (int)(((double)(chainActive.Height() - pindex->nHeight)) / (double)nCheckDepth * 50))));
            pindex = chainActive.Next(pindex);
            CBlock block;
            if (!ReadBlockFromDisk(block, pindex, chainparams.GetConsensus()))
                return error("VerifyDB(): *** ReadBlockFromDisk failed at %d, hash=%s", pindex->nHeight, pindex->GetBlockHash().ToString());
            if (!ConnectBlock(block, state, pindex, coins, chainparams, &assetCache, false, true))
                return error("VerifyDB(): *** found unconnectable block at %d, hash=%s", pindex->nHeight, pindex->GetBlockHash().ToString());
        }
    }

    LogPrintf("[DONE].\n");
    LogPrintf("No coin database inconsistencies in last %i blocks (%i transactions)\n", chainActive.Height() - pindexState->nHeight, nGoodTransactions);

    return true;
}

void UnloadBlockIndex()
{
    LOCK(cs_main);
    setBlockIndexCandidates.clear();
    chainActive.SetTip(nullptr);
    pindexBestInvalid = nullptr;
    pindexBestHeader = nullptr;
    mempool.clear();
    mapBlocksUnlinked.clear();
    vinfoBlockFile.clear();
    nLastBlockFile = 0;
    nBlockSequenceId = 1;
    setDirtyBlockIndex.clear();
    setDirtyFileInfo.clear();
    versionbitscache.Clear();
    for (int b = 0; b < VERSIONBITS_NUM_BITS; b++) {
        warningcache[b].clear();
    }

    for (BlockMap::value_type& entry : mapBlockIndex) {
        delete entry.second;
    }
    mapBlockIndex.clear();
    fHavePruned = false;
}

bool LoadBlockIndex(const CChainParams& chainparams)
{
    // Load block index from databases
    if (!fReindex && !LoadBlockIndexDB(chainparams))
        return false;
    return true;
}

static bool AddGenesisBlock(const CChainParams& chainparams, const CBlock& block, CValidationState& state)
{
    // Start new block file
    unsigned int nBlockSize = ::GetSerializeSize(block, SER_DISK, CLIENT_VERSION);
    CDiskBlockPos blockPos;
    if (!FindBlockPos(state, blockPos, nBlockSize + 8, 0, block.GetBlockTime()))
        return error("%s: FindBlockPos failed", __func__);
    if (!WriteBlockToDisk(block, blockPos, chainparams.MessageStart()))
        return error("%s: writing genesis block to disk failed", __func__);
    CBlockIndex* pindex = AddToBlockIndex(block);
    if (!ReceivedBlockTransactions(block, state, pindex, blockPos))
        return error("%s: genesis block not accepted", __func__);
    return true;
}


bool InitBlockIndex(const CChainParams& chainparams)
{
    LOCK(cs_main);

    // Check whether we're already initialized
    if (chainActive.Genesis() != nullptr)
        return true;

    // Use the provided setting for -txindex in the new database
    fTxIndex = gArgs.GetBoolArg("-txindex", DEFAULT_TXINDEX);
    pblocktree->WriteFlag("txindex", fTxIndex);

    // Use the provided setting for -addressindex in the new database
    fAddressIndex = gArgs.GetBoolArg("-addressindex", DEFAULT_ADDRESSINDEX);
    pblocktree->WriteFlag("addressindex", fAddressIndex);

    // Use the provided setting for -assetindex in the new database
    fAssetIndex = gArgs.GetBoolArg("-assetindex", DEFAULT_ASSETINDEX);
    pblocktree->WriteFlag("assetindex", fAssetIndex);
    LogPrintf("%s: asset index %s\n", __func__, fAssetIndex ? "enabled" : "disabled");

    // Use the provided setting for -timestampindex in the new database
    fTimestampIndex = gArgs.GetBoolArg("-timestampindex", DEFAULT_TIMESTAMPINDEX);
    pblocktree->WriteFlag("timestampindex", fTimestampIndex);

    fSpentIndex = gArgs.GetBoolArg("-spentindex", DEFAULT_SPENTINDEX);
    pblocktree->WriteFlag("spentindex", fSpentIndex);

    LogPrintf("Initializing databases...\n");

    // Only add the genesis block if not reindexing (in which case we reuse the one already on disk)
    if (!fReindex) {
        try {
            CValidationState state;

            if (!AddGenesisBlock(chainparams, chainparams.GenesisBlock(), state))
                return false;

            // Force a chainstate write so that when we VerifyDB in a moment, it doesn't check stale data
            return FlushStateToDisk(state, FLUSH_STATE_ALWAYS);
        } catch (const std::runtime_error& e) {
            return error("%s: failed to initialize block database: %s", __func__, e.what());
        }
    }

    return true;
}

bool LoadExternalBlockFile(const CChainParams& chainparams, FILE* fileIn, CDiskBlockPos* dbp)
{
    // Map of disk positions for blocks with unknown parent (only used for reindex)
    static std::multimap<uint256, CDiskBlockPos> mapBlocksUnknownParent;
    int64_t nStart = GetTimeMillis();

    int nLoaded = 0;
    try {
        // This takes over fileIn and calls fclose() on it in the CBufferedFile destructor
        CBufferedFile blkdat(fileIn, 2 * GetMaxBlockSerializedSize(), GetMaxBlockSerializedSize() + 8, SER_DISK, CLIENT_VERSION);
        uint64_t nRewind = blkdat.GetPos();
        while (!blkdat.eof()) {
            boost::this_thread::interruption_point();

            blkdat.SetPos(nRewind);
            nRewind++;         // start one byte further next time, in case of failure
            blkdat.SetLimit(); // remove former limit
            unsigned int nSize = 0;
            try {
                // locate a header
                unsigned char buf[CMessageHeader::MESSAGE_START_SIZE];
                blkdat.FindByte(chainparams.MessageStart()[0]);
                nRewind = blkdat.GetPos() + 1;
                blkdat >> FLATDATA(buf);
                if (memcmp(buf, chainparams.MessageStart(), CMessageHeader::MESSAGE_START_SIZE))
                    continue;
                // read size
                blkdat >> nSize;
                if (nSize < 80 || nSize > GetMaxBlockSerializedSize())
                    continue;
            } catch (const std::exception&) {
                // no valid block header found; don't complain
                break;
            }
            try {
                // read block
                uint64_t nBlockPos = blkdat.GetPos();
                if (dbp)
                    dbp->nPos = nBlockPos;
                blkdat.SetLimit(nBlockPos + nSize);
                blkdat.SetPos(nBlockPos);
                std::shared_ptr<CBlock> pblock = std::make_shared<CBlock>();
                CBlock& block = *pblock;
                blkdat >> block;
                nRewind = blkdat.GetPos();

                // detect out of order blocks, and store them for later
                uint256 hash = block.GetHash();
                if (hash != chainparams.GetConsensus().hashGenesisBlock && mapBlockIndex.find(block.hashPrevBlock) == mapBlockIndex.end()) {
                    LogPrint("reindex", "%s: Out of order block %s, parent %s not known\n", __func__, hash.ToString(),
                        block.hashPrevBlock.ToString());
                    if (dbp)
                        mapBlocksUnknownParent.insert(std::make_pair(block.hashPrevBlock, *dbp));
                    continue;
                }

                // process in case the block isn't known yet
                if (mapBlockIndex.count(hash) == 0 || (mapBlockIndex[hash]->nStatus & BLOCK_HAVE_DATA) == 0) {
                    LOCK(cs_main);
                    CValidationState state;
                    if (AcceptBlock(pblock, state, chainparams, nullptr, true, dbp, nullptr))
                        nLoaded++;
                    if (state.IsError())
                        break;
                } else if (hash != chainparams.GetConsensus().hashGenesisBlock && mapBlockIndex[hash]->nHeight % 1000 == 0) {
                    LogPrint("reindex", "Block Import: already had block %s at height %d\n", hash.ToString(), mapBlockIndex[hash]->nHeight);
                }

                // Activate the genesis block so normal node progress can continue
                if (hash == chainparams.GetConsensus().hashGenesisBlock) {
                    CValidationState state;
                    if (!ActivateBestChain(state, chainparams)) {
                        break;
                    }
                }

                NotifyHeaderTip();

                // Recursively process earlier encountered successors of this block
                std::deque<uint256> queue;
                queue.push_back(hash);
                while (!queue.empty()) {
                    uint256 head = queue.front();
                    queue.pop_front();
                    std::pair<std::multimap<uint256, CDiskBlockPos>::iterator, std::multimap<uint256, CDiskBlockPos>::iterator> range = mapBlocksUnknownParent.equal_range(head);
                    while (range.first != range.second) {
                        std::multimap<uint256, CDiskBlockPos>::iterator it = range.first;
                        std::shared_ptr<CBlock> pblockrecursive = std::make_shared<CBlock>();
                        if (ReadBlockFromDisk(*pblockrecursive, it->second, chainparams.GetConsensus())) {
                            LogPrint("reindex", "%s: Processing out of order child %s of %s\n", __func__, block.GetHash().ToString(),
                                head.ToString());
                            LOCK(cs_main);
                            CValidationState dummy;
                            if (AcceptBlock(pblockrecursive, dummy, chainparams, nullptr, true, &it->second, nullptr)) {
                                nLoaded++;
                                queue.push_back(pblockrecursive->GetHash());
                            }
                        }
                        range.first++;
                        mapBlocksUnknownParent.erase(it);
                        NotifyHeaderTip();
                    }
                }
            } catch (const std::exception& e) {
                LogPrintf("%s: Deserialize or I/O error - %s\n", __func__, e.what());
            }
        }
    } catch (const std::runtime_error& e) {
        AbortNode(std::string("System error: ") + e.what());
    }
    if (nLoaded > 0)
        LogPrintf("Loaded %i blocks from external file in %dms\n", nLoaded, GetTimeMillis() - nStart);
    return nLoaded > 0;
}

void static CheckBlockIndex(const Consensus::Params& consensusParams)
{
    if (!fCheckBlockIndex) {
        return;
    }

    LOCK(cs_main);

    // During a reindex, we read the genesis block and call CheckBlockIndex before ActivateBestChain,
    // so we have the genesis block in mapBlockIndex but no active chain.  (A few of the tests when
    // iterating the block tree require that chainActive has been initialized.)
    if (chainActive.Height() < 0) {
        assert(mapBlockIndex.size() <= 1);
        return;
    }

    // Build forward-pointing map of the entire block tree.
    std::multimap<CBlockIndex*, CBlockIndex*> forward;
    for (BlockMap::iterator it = mapBlockIndex.begin(); it != mapBlockIndex.end(); it++) {
        forward.insert(std::make_pair(it->second->pprev, it->second));
    }

    assert(forward.size() == mapBlockIndex.size());

    std::pair<std::multimap<CBlockIndex*, CBlockIndex*>::iterator, std::multimap<CBlockIndex*, CBlockIndex*>::iterator> rangeGenesis = forward.equal_range(nullptr);
    CBlockIndex* pindex = rangeGenesis.first->second;
    rangeGenesis.first++;
    assert(rangeGenesis.first == rangeGenesis.second); // There is only one index entry with parent nullptr.

    // Iterate over the entire block tree, using depth-first search.
    // Along the way, remember whether there are blocks on the path from genesis
    // block being explored which are the first to have certain properties.
    size_t nNodes = 0;
    int nHeight = 0;
    CBlockIndex* pindexFirstInvalid = nullptr;              // Oldest ancestor of pindex which is invalid.
    CBlockIndex* pindexFirstMissing = nullptr;              // Oldest ancestor of pindex which does not have BLOCK_HAVE_DATA.
    CBlockIndex* pindexFirstNeverProcessed = nullptr;       // Oldest ancestor of pindex for which nTx == 0.
    CBlockIndex* pindexFirstNotTreeValid = nullptr;         // Oldest ancestor of pindex which does not have BLOCK_VALID_TREE (regardless of being valid or not).
    CBlockIndex* pindexFirstNotTransactionsValid = nullptr; // Oldest ancestor of pindex which does not have BLOCK_VALID_TRANSACTIONS (regardless of being valid or not).
    CBlockIndex* pindexFirstNotChainValid = nullptr;        // Oldest ancestor of pindex which does not have BLOCK_VALID_CHAIN (regardless of being valid or not).
    CBlockIndex* pindexFirstNotScriptsValid = nullptr;      // Oldest ancestor of pindex which does not have BLOCK_VALID_SCRIPTS (regardless of being valid or not).
    while (pindex != nullptr) {
        nNodes++;
        if (pindexFirstInvalid == nullptr && pindex->nStatus & BLOCK_FAILED_VALID)
            pindexFirstInvalid = pindex;
        if (pindexFirstMissing == nullptr && !(pindex->nStatus & BLOCK_HAVE_DATA))
            pindexFirstMissing = pindex;
        if (pindexFirstNeverProcessed == nullptr && pindex->nTx == 0)
            pindexFirstNeverProcessed = pindex;
        if (pindex->pprev != nullptr && pindexFirstNotTreeValid == nullptr && (pindex->nStatus & BLOCK_VALID_MASK) < BLOCK_VALID_TREE)
            pindexFirstNotTreeValid = pindex;
        if (pindex->pprev != nullptr && pindexFirstNotTransactionsValid == nullptr && (pindex->nStatus & BLOCK_VALID_MASK) < BLOCK_VALID_TRANSACTIONS)
            pindexFirstNotTransactionsValid = pindex;
        if (pindex->pprev != nullptr && pindexFirstNotChainValid == nullptr && (pindex->nStatus & BLOCK_VALID_MASK) < BLOCK_VALID_CHAIN)
            pindexFirstNotChainValid = pindex;
        if (pindex->pprev != nullptr && pindexFirstNotScriptsValid == nullptr && (pindex->nStatus & BLOCK_VALID_MASK) < BLOCK_VALID_SCRIPTS)
            pindexFirstNotScriptsValid = pindex;

        // Begin: actual consistency checks.
        if (pindex->pprev == nullptr) {
            // Genesis block checks.
            assert(pindex->GetBlockHash() == consensusParams.hashGenesisBlock); // Genesis block's hash must match.
            assert(pindex == chainActive.Genesis());                            // The current active chain's genesis block must be this block.
        }
        if (pindex->nChainTx == 0)
            assert(pindex->nSequenceId <= 0); // nSequenceId can't be set positive for blocks that aren't linked (negative is used for preciousblock)
        // VALID_TRANSACTIONS is equivalent to nTx > 0 for all nodes (whether or not pruning has occurred).
        // HAVE_DATA is only equivalent to nTx > 0 (or VALID_TRANSACTIONS) if no pruning has occurred.
        if (!fHavePruned) {
            // If we've never pruned, then HAVE_DATA should be equivalent to nTx > 0
            assert(!(pindex->nStatus & BLOCK_HAVE_DATA) == (pindex->nTx == 0));
            assert(pindexFirstMissing == pindexFirstNeverProcessed);
        } else {
            // If we have pruned, then we can only say that HAVE_DATA implies nTx > 0
            if (pindex->nStatus & BLOCK_HAVE_DATA)
                assert(pindex->nTx > 0);
        }
        if (pindex->nStatus & BLOCK_HAVE_UNDO)
            assert(pindex->nStatus & BLOCK_HAVE_DATA);
        assert(((pindex->nStatus & BLOCK_VALID_MASK) >= BLOCK_VALID_TRANSACTIONS) == (pindex->nTx > 0)); // This is pruning-independent.
        // All parents having had data (at some point) is equivalent to all parents being VALID_TRANSACTIONS, which is equivalent to nChainTx being set.
        assert((pindexFirstNeverProcessed != nullptr) == (pindex->nChainTx == 0)); // nChainTx != 0 is used to signal that all parent blocks have been processed (but may have been pruned).
        assert((pindexFirstNotTransactionsValid != nullptr) == (pindex->nChainTx == 0));
        assert(pindex->nHeight == nHeight);                                               // nHeight must be consistent.
        assert(pindex->pprev == nullptr || pindex->nChainWork >= pindex->pprev->nChainWork); // For every block except the genesis block, the chainwork must be larger than the parent's.
        assert(nHeight < 2 || (pindex->pskip && (pindex->pskip->nHeight < nHeight)));     // The pskip pointer must point back for all but the first 2 blocks.
        assert(pindexFirstNotTreeValid == nullptr);                                          // All mapBlockIndex entries must at least be TREE valid
        if ((pindex->nStatus & BLOCK_VALID_MASK) >= BLOCK_VALID_TREE)
            assert(pindexFirstNotTreeValid == nullptr); // TREE valid implies all parents are TREE valid
        if ((pindex->nStatus & BLOCK_VALID_MASK) >= BLOCK_VALID_CHAIN)
            assert(pindexFirstNotChainValid == nullptr); // CHAIN valid implies all parents are CHAIN valid
        if ((pindex->nStatus & BLOCK_VALID_MASK) >= BLOCK_VALID_SCRIPTS)
            assert(pindexFirstNotScriptsValid == nullptr); // SCRIPTS valid implies all parents are SCRIPTS valid
        if (pindexFirstInvalid == nullptr) {
            // Checks for not-invalid blocks.
            assert((pindex->nStatus & BLOCK_FAILED_MASK) == 0); // The failed mask cannot be set for blocks without invalid parents.
        }
        if (!CBlockIndexWorkComparator()(pindex, chainActive.Tip()) && pindexFirstNeverProcessed == nullptr) {
            if (pindexFirstInvalid == nullptr) {
                // If this block sorts at least as good as the current tip and
                // is valid and we have all data for its parents, it must be in
                // setBlockIndexCandidates.  chainActive.Tip() must also be there
                // even if some data has been pruned.
                if (pindexFirstMissing == nullptr || pindex == chainActive.Tip()) {
                    assert(setBlockIndexCandidates.count(pindex));
                }
                // If some parent is missing, then it could be that this block was in
                // setBlockIndexCandidates but had to be removed because of the missing data.
                // In this case it must be in mapBlocksUnlinked -- see test below.
            }
        } else { // If this block sorts worse than the current tip or some ancestor's block has never been seen, it cannot be in setBlockIndexCandidates.
            assert(setBlockIndexCandidates.count(pindex) == 0);
        }
        // Check whether this block is in mapBlocksUnlinked.
        std::pair<std::multimap<CBlockIndex*, CBlockIndex*>::iterator, std::multimap<CBlockIndex*, CBlockIndex*>::iterator> rangeUnlinked = mapBlocksUnlinked.equal_range(pindex->pprev);
        bool foundInUnlinked = false;
        while (rangeUnlinked.first != rangeUnlinked.second) {
            assert(rangeUnlinked.first->first == pindex->pprev);
            if (rangeUnlinked.first->second == pindex) {
                foundInUnlinked = true;
                break;
            }
            rangeUnlinked.first++;
        }
        if (pindex->pprev && (pindex->nStatus & BLOCK_HAVE_DATA) && pindexFirstNeverProcessed != nullptr && pindexFirstInvalid == nullptr) {
            // If this block has block data available, some parent was never received, and has no invalid parents, it must be in mapBlocksUnlinked.
            assert(foundInUnlinked);
        }
        if (!(pindex->nStatus & BLOCK_HAVE_DATA))
            assert(!foundInUnlinked); // Can't be in mapBlocksUnlinked if we don't HAVE_DATA
        if (pindexFirstMissing == nullptr)
            assert(!foundInUnlinked); // We aren't missing data for any parent -- cannot be in mapBlocksUnlinked.
        if (pindex->pprev && (pindex->nStatus & BLOCK_HAVE_DATA) && pindexFirstNeverProcessed == nullptr && pindexFirstMissing != nullptr) {
            // We HAVE_DATA for this block, have received data for all parents at some point, but we're currently missing data for some parent.
            assert(fHavePruned); // We must have pruned.
            // This block may have entered mapBlocksUnlinked if:
            //  - it has a descendant that at some point had more work than the
            //    tip, and
            //  - we tried switching to that descendant but were missing
            //    data for some intermediate block between chainActive and the
            //    tip.
            // So if this block is itself better than chainActive.Tip() and it wasn't in
            // setBlockIndexCandidates, then it must be in mapBlocksUnlinked.
            if (!CBlockIndexWorkComparator()(pindex, chainActive.Tip()) && setBlockIndexCandidates.count(pindex) == 0) {
                if (pindexFirstInvalid == nullptr) {
                    assert(foundInUnlinked);
                }
            }
        }
        // assert(pindex->GetBlockHash() == pindex->GetBlockHeader().GetHash()); // Perhaps too slow
        // End: actual consistency checks.

        // Try descending into the first subnode.
        std::pair<std::multimap<CBlockIndex*, CBlockIndex*>::iterator, std::multimap<CBlockIndex*, CBlockIndex*>::iterator> range = forward.equal_range(pindex);
        if (range.first != range.second) {
            // A subnode was found.
            pindex = range.first->second;
            nHeight++;
            continue;
        }
        // This is a leaf node.
        // Move upwards until we reach a node of which we have not yet visited the last child.
        while (pindex) {
            // We are going to either move to a parent or a sibling of pindex.
            // If pindex was the first with a certain property, unset the corresponding variable.
            if (pindex == pindexFirstInvalid)
                pindexFirstInvalid = nullptr;
            if (pindex == pindexFirstMissing)
                pindexFirstMissing = nullptr;
            if (pindex == pindexFirstNeverProcessed)
                pindexFirstNeverProcessed = nullptr;
            if (pindex == pindexFirstNotTreeValid)
                pindexFirstNotTreeValid = nullptr;
            if (pindex == pindexFirstNotTransactionsValid)
                pindexFirstNotTransactionsValid = nullptr;
            if (pindex == pindexFirstNotChainValid)
                pindexFirstNotChainValid = nullptr;
            if (pindex == pindexFirstNotScriptsValid)
                pindexFirstNotScriptsValid = nullptr;
            // Find our parent.
            CBlockIndex* pindexPar = pindex->pprev;
            // Find which child we just visited.
            std::pair<std::multimap<CBlockIndex*, CBlockIndex*>::iterator, std::multimap<CBlockIndex*, CBlockIndex*>::iterator> rangePar = forward.equal_range(pindexPar);
            while (rangePar.first->second != pindex) {
                assert(rangePar.first != rangePar.second); // Our parent must have at least the node we're coming from as child.
                rangePar.first++;
            }
            // Proceed to the next one.
            rangePar.first++;
            if (rangePar.first != rangePar.second) {
                // Move to the sibling.
                pindex = rangePar.first->second;
                break;
            } else {
                // Move up further.
                pindex = pindexPar;
                nHeight--;
                continue;
            }
        }
    }

    // Check that we actually traversed the entire map.
    assert(nNodes == forward.size());
}

std::string CBlockFileInfo::ToString() const
{
    return strprintf("CBlockFileInfo(blocks=%u, size=%u, heights=%u...%u, time=%s...%s)", nBlocks, nSize, nHeightFirst, nHeightLast, DateTimeStrFormat("%Y-%m-%d", nTimeFirst), DateTimeStrFormat("%Y-%m-%d", nTimeLast));
}

CBlockFileInfo* GetBlockFileInfo(size_t n)
{
    return &vinfoBlockFile.at(n);
}

ThresholdState VersionBitsTipState(const Consensus::Params& params, Consensus::DeploymentPos pos)
{
    AssertLockHeld(cs_main);
    return VersionBitsState(chainActive.Tip(), params, pos, versionbitscache);
}

int VersionBitsTipStateSinceHeight(const Consensus::Params& params, Consensus::DeploymentPos pos)
{
    LOCK(cs_main);
    return VersionBitsStateSinceHeight(chainActive.Tip(), params, pos, versionbitscache);
}

static const uint64_t MEMPOOL_DUMP_VERSION = 1;

bool LoadMempool(void)
{
    int64_t nExpiryTimeout = gArgs.GetArg("-mempoolexpiry", DEFAULT_MEMPOOL_EXPIRY) * 60 * 60;
    FILE* filestr = fopen((GetDataDir() / "mempool.dat").string().c_str(), "r");
    CAutoFile file(filestr, SER_DISK, CLIENT_VERSION);
    if (file.IsNull()) {
        LogPrintf("Failed to open mempool file from disk. Continuing anyway.\n");
        return false;
    }

    int64_t count = 0;
    int64_t skipped = 0;
    int64_t failed = 0;
    int64_t nNow = GetTime();

    try {
        uint64_t version;
        file >> version;
        if (version != MEMPOOL_DUMP_VERSION) {
            return false;
        }
        uint64_t num;
        file >> num;
        double prioritydummy = 0;
        while (num--) {
            CTransactionRef tx;
            int64_t nTime;
            int64_t nFeeDelta;
            file >> tx;
            file >> nTime;
            file >> nFeeDelta;

            CAmount amountdelta = nFeeDelta;
            if (amountdelta) {
                mempool.PrioritiseTransaction(tx->GetHash(), tx->GetHash().ToString(), prioritydummy, amountdelta);
            }
            CValidationState state;
            if (nTime + nExpiryTimeout > nNow) {
                LOCK(cs_main);
                AcceptToMemoryPoolWithTime(mempool, state, tx, true, nullptr, nTime);
                if (state.IsValid()) {
                    ++count;
                } else {
                    ++failed;
                }
            } else {
                ++skipped;
            }
            if (ShutdownRequested())
                return false;
        }
        std::map<uint256, CAmount> mapDeltas;
        file >> mapDeltas;

        for (const auto& i : mapDeltas) {
            mempool.PrioritiseTransaction(i.first, i.first.ToString(), prioritydummy, i.second);
        }
    } catch (const std::exception& e) {
        LogPrintf("Failed to deserialize mempool data on disk: %s. Continuing anyway.\n", e.what());
        return false;
    }

    LogPrintf("Imported mempool transactions from disk: %i successes, %i failed, %i expired\n", count, failed, skipped);
    return true;
}

void DumpMempool(void)
{
    int64_t start = GetTimeMicros();

    std::map<uint256, CAmount> mapDeltas;
    std::vector<TxMempoolInfo> vinfo;

    {
        LOCK(mempool.cs);
        for (const auto& i : mempool.mapDeltas) {
            mapDeltas[i.first] = i.second.second;
        }
        vinfo = mempool.infoAll();
    }

    int64_t mid = GetTimeMicros();

    try {
        FILE* filestr = fopen((GetDataDir() / "mempool.dat.new").string().c_str(), "w");
        if (!filestr) {
            return;
        }

        CAutoFile file(filestr, SER_DISK, CLIENT_VERSION);

        uint64_t version = MEMPOOL_DUMP_VERSION;
        file << version;

        file << (uint64_t)vinfo.size();
        for (const auto& i : vinfo) {
            file << *(i.tx);
            file << (int64_t)i.nTime;
            file << (int64_t)i.nFeeDelta;
            mapDeltas.erase(i.tx->GetHash());
        }

        file << mapDeltas;
        FileCommit(file.Get());
        file.fclose();
        RenameOver(GetDataDir() / "mempool.dat.new", GetDataDir() / "mempool.dat");
        int64_t last = GetTimeMicros();
        LogPrintf("Dumped mempool: %gs to copy, %gs to dump\n", (mid - start) * 0.000001, (last - mid) * 0.000001);
    } catch (const std::exception& e) {
        LogPrintf("Failed to dump mempool: %s. Continuing anyway.\n", e.what());
    }
}

/** ASSET START */
bool AreAssetsDeployed() {

    return sporkManager.IsSporkActive(SPORK_32_BDAP_V2);
}

bool IsMsgRestAssetIsActive()
{
    return AreAssetsDeployed();
}

bool AreMessagesDeployed() {

    return AreAssetsDeployed();
}

bool AreTransferScriptsSizeDeployed() {

    return AreAssetsDeployed();
}

bool AreRestrictedAssetsDeployed() {

    return AreAssetsDeployed();
}

CAssetsCache* GetCurrentAssetCache()
{
    return passets;
}

bool SpendCoinWithAssets(CCoinsViewCache& cache, const COutPoint &outpoint, Coin* moveout, CAssetsCache* assetsCache) 
{
    CCoinsMap::iterator it = cache.FetchCoin(outpoint);
    if (it == cache.CacheCoins()->end())
        return false;

    cache.cachedCoinsUsage -= it->second.coin.DynamicMemoryUsage();

    /** ASSET START */
    Coin tempCoin = it->second.coin;
    /** ASSET END */

    if (moveout) {
        *moveout = std::move(it->second.coin);
    }
    if (it->second.flags & CCoinsCacheEntry::FRESH) {
        cache.CacheCoins()->erase(it);
    } else {
        it->second.flags |= CCoinsCacheEntry::DIRTY;
        it->second.coin.Clear();
    }

    /** ASSET START */
    if (AreAssetsDeployed()) {
        if (assetsCache) {
            if (!assetsCache->TrySpendCoin(outpoint, tempCoin.out)) {
                return error("%s : Failed to try and spend the asset. COutPoint : %s", __func__, outpoint.ToString());
            }
        }
    }
    /** ASSET END */

    return true;
}

void AddCoinsWithAssets(CCoinsViewCache& cache, const CTransaction &tx, int nHeight, uint256 blockHash, bool check, CAssetsCache* assetsCache, std::pair<std::string, CBlockAssetUndo>* undoAssetData) {
    bool fCoinbase = tx.IsCoinBase();
    const uint256& txid = tx.GetHash();

    /** ASSET START */
    if (AreAssetsDeployed()) {
        if (assetsCache) {
            if (tx.IsNewAsset()) { // This works are all new root assets, sub asset, and restricted assets
                CNewAsset asset;
                std::string strAddress;
                AssetFromTransaction(tx, asset, strAddress);

                std::string ownerName;
                std::string ownerAddress;
                OwnerFromTransaction(tx, ownerName, ownerAddress);

                // Add the new asset to cache
                if (!assetsCache->AddNewAsset(asset, strAddress, nHeight, blockHash))
                    error("%s : Failed at adding a new asset to our cache. asset: %s", __func__,
                          asset.strName);

                // Add the owner asset to cache
                if (!assetsCache->AddOwnerAsset(ownerName, ownerAddress))
                    error("%s : Failed at adding a new asset to our cache. asset: %s", __func__,
                          asset.strName);

            } else if (tx.IsReissueAsset()) {
                CReissueAsset reissue;
                std::string strAddress;
                ReissueAssetFromTransaction(tx, reissue, strAddress);

                int reissueIndex = tx.vout.size() - 1;

                // Get the asset before we change it
                CNewAsset asset;
                if (!assetsCache->GetAssetMetaDataIfExists(reissue.strName, asset))
                    error("%s: Failed to get the original asset that is getting reissued. Asset Name : %s",
                          __func__, reissue.strName);

                if (!assetsCache->AddReissueAsset(reissue, strAddress, COutPoint(txid, reissueIndex)))
                    error("%s: Failed to reissue an asset. Asset Name : %s", __func__, reissue.strName);

                // Check to see if we are reissuing a restricted asset
                bool fFoundRestrictedAsset = false;
                AssetType type;
                IsAssetNameValid(asset.strName, type);
                if (type == AssetType::RESTRICTED) {
                    fFoundRestrictedAsset = true;
                }

                // Set the old IPFSHash for the blockundo
                bool fIPFSChanged = !reissue.strIPFSHash.empty();
                bool fUnitsChanged = reissue.nUnits != -1;
                bool fVerifierChanged = false;
                std::string strOldVerifier = "";

                // If we are reissuing a restricted asset, we need to check to see if the verifier string is being reissued
                if (fFoundRestrictedAsset) {
                    CNullAssetTxVerifierString verifier;
                    // Search through all outputs until you find a restricted verifier change.
                    for (auto index: tx.vout) {
                        if (index.scriptPubKey.IsNullAssetVerifierTxDataScript()) {
                            if (!AssetNullVerifierDataFromScript(index.scriptPubKey, verifier)) {
                                error("%s: Failed to get asset null verifier data and add it to the coins CTxOut: %s", __func__,
                                      index.ToString());
                                break;
                            }

                            fVerifierChanged = true;
                            break;
                        }
                    }

                    CNullAssetTxVerifierString oldVerifer{strOldVerifier};
                    if (fVerifierChanged && !assetsCache->GetAssetVerifierStringIfExists(asset.strName, oldVerifer))
                        error("%s : Failed to get asset original verifier string that is getting reissued, Asset Name: %s", __func__, asset.strName);

                    if (fVerifierChanged) {
                        strOldVerifier = oldVerifer.verifier_string;
                    }

                    // Add the verifier to the cache if there was one found
                    if (fVerifierChanged && !assetsCache->AddRestrictedVerifier(asset.strName, verifier.verifier_string))
                        error("%s : Failed at adding a restricted verifier to our cache: asset: %s, verifier : %s",
                              asset.strName, verifier.verifier_string);
                }

                // If any of the following items were changed by reissuing, we need to database the old values so it can be undone correctly
                if (fIPFSChanged || fUnitsChanged || fVerifierChanged) {
                    undoAssetData->first = reissue.strName; // Asset Name
                    undoAssetData->second = CBlockAssetUndo {fIPFSChanged, fUnitsChanged, asset.strIPFSHash, asset.units, ASSET_UNDO_INCLUDES_VERIFIER_STRING, fVerifierChanged, strOldVerifier}; // ipfschanged, unitchanged, Old Assets IPFSHash, old units
                }
            } else if (tx.IsNewUniqueAsset()) {
                for (int n = 0; n < (int)tx.vout.size(); n++) {
                    auto out = tx.vout[n];

                    CNewAsset asset;
                    std::string strAddress;

                    if (IsScriptNewUniqueAsset(out.scriptPubKey)) {
                        AssetFromScript(out.scriptPubKey, asset, strAddress);

                        // Add the new asset to cache
                        if (!assetsCache->AddNewAsset(asset, strAddress, nHeight, blockHash))
                            error("%s : Failed at adding a new asset to our cache. asset: %s", __func__,
                                  asset.strName);
                    }
                }
            } else if (tx.IsNewMsgChannelAsset()) {
                CNewAsset asset;
                std::string strAddress;
                MsgChannelAssetFromTransaction(tx, asset, strAddress);

                // Add the new asset to cache
                if (!assetsCache->AddNewAsset(asset, strAddress, nHeight, blockHash))
                    error("%s : Failed at adding a new asset to our cache. asset: %s", __func__,
                          asset.strName);
            } else if (tx.IsNewQualifierAsset()) {
                CNewAsset asset;
                std::string strAddress;
                QualifierAssetFromTransaction(tx, asset, strAddress);

                // Add the new asset to cache
                if (!assetsCache->AddNewAsset(asset, strAddress, nHeight, blockHash))
                    error("%s : Failed at adding a new qualifier asset to our cache. asset: %s", __func__,
                          asset.strName);
            }  else if (tx.IsNewRestrictedAsset()) {
                CNewAsset asset;
                std::string strAddress;
                RestrictedAssetFromTransaction(tx, asset, strAddress);

                // Add the new asset to cache
                if (!assetsCache->AddNewAsset(asset, strAddress, nHeight, blockHash))
                    error("%s : Failed at adding a new restricted asset to our cache. asset: %s", __func__,
                          asset.strName);

                // Find the restricted verifier string and cache it
                CNullAssetTxVerifierString verifier;
                // Search through all outputs until you find a restricted verifier change.
                for (auto index: tx.vout) {
                    if (index.scriptPubKey.IsNullAssetVerifierTxDataScript()) {
                        CNullAssetTxVerifierString verifier;
                        if (!AssetNullVerifierDataFromScript(index.scriptPubKey, verifier))
                            error("%s: Failed to get asset null data and add it to the coins CTxOut: %s", __func__,
                                  index.ToString());

                        // Add the verifier to the cache
                        if (!assetsCache->AddRestrictedVerifier(asset.strName, verifier.verifier_string))
                            error("%s : Failed at adding a restricted verifier to our cache: asset: %s, verifier : %s",
                                  asset.strName, verifier.verifier_string);

                        break;
                    }
                }
            }
        }
    }
    /** ASSET END */

    for (size_t i = 0; i < tx.vout.size(); ++i) {
        bool overwrite = check ? cache.HaveCoin(COutPoint(txid, i)) : fCoinbase;
        // Always set the possible_overwrite flag to AddCoin for coinbase txn, in order to correctly
        // deal with the pre-BIP30 occurrences of duplicate coinbase transactions.
        cache.AddCoin(COutPoint(txid, i), Coin(tx.vout[i], nHeight, fCoinbase), overwrite);

        /** ASSET START */
        if (AreAssetsDeployed()) {
            if (assetsCache) {
                CAssetOutputEntry assetData;
                if (GetAssetData(tx.vout[i].scriptPubKey, assetData)) {
                    if (assetData.type == TX_TRANSFER_ASSET && !tx.vout[i].scriptPubKey.IsUnspendable()) {
                        CAssetTransfer assetTransfer;
                        std::string address;
                        if (!TransferAssetFromScript(tx.vout[i].scriptPubKey, assetTransfer, address))
                            LogPrintf(
                                    "%s : ERROR - Received a coin that was a Transfer Asset but failed to get the transfer object from the scriptPubKey. CTxOut: %s\n",
                                    __func__, tx.vout[i].ToString());

                        if (!assetsCache->AddTransferAsset(assetTransfer, address, COutPoint(txid, i), tx.vout[i]))
                            LogPrintf("%s : ERROR - Failed to add transfer asset CTxOut: %s\n", __func__,
                                      tx.vout[i].ToString());

                        /** Subscribe to new message channels if they are sent to a new address, or they are the owner token or message channel */
#ifdef ENABLE_WALLET
                        if (fMessaging && pMessageSubscribedChannelsCache) {
                            LOCK(cs_messaging);
                            if (pwalletMain && pwalletMain->IsMine(tx.vout[i]) == ISMINE_SPENDABLE) {
                                AssetType aType;
                                IsAssetNameValid(assetTransfer.strName, aType);

                                if (aType == AssetType::ROOT || aType == AssetType::SUB) {
                                    if (!IsChannelSubscribed(GetParentName(assetTransfer.strName) + OWNER_TAG)) {
                                        if (!IsAddressSeen(address)) {
                                            AddChannel(GetParentName(assetTransfer.strName) + OWNER_TAG);
                                            AddAddressSeen(address);
                                        }
                                    }
                                } else if (aType == AssetType::OWNER || aType == AssetType::MSGCHANNEL) {
                                    AddChannel(assetTransfer.strName);
                                    AddAddressSeen(address);
                                }
                            }
                        }
#endif
                    } else if (assetData.type == TX_NEW_ASSET) {
                        /** Subscribe to new message channels if they are assets you created, or are new msgchannels of channels already being watched */
#ifdef ENABLE_WALLET
                        if (fMessaging && pMessageSubscribedChannelsCache) {
                            LOCK(cs_messaging);
                            if (pwalletMain) {
                                AssetType aType;
                                IsAssetNameValid(assetData.assetName, aType);
                                if (pwalletMain->IsMine(tx.vout[i]) == ISMINE_SPENDABLE) {
                                    if (aType == AssetType::ROOT || aType == AssetType::SUB) {
                                        AddChannel(assetData.assetName + OWNER_TAG);
                                        AddAddressSeen(EncodeDestination(assetData.destination));
                                    } else if (aType == AssetType::OWNER || aType == AssetType::MSGCHANNEL) {
                                        AddChannel(assetData.assetName);
                                        AddAddressSeen(EncodeDestination(assetData.destination));
                                    }
                                } else {
                                    if (aType == AssetType::MSGCHANNEL) {
                                        if (IsChannelSubscribed(GetParentName(assetData.assetName) + OWNER_TAG)) {
                                            AddChannel(assetData.assetName);
                                        }
                                    }
                                }
                            }
                        }
#endif
                    }
                }

                CScript script = tx.vout[i].scriptPubKey;
                if (script.IsNullAsset()) {
                    if (script.IsNullAssetTxDataScript()) {
                        CNullAssetTxData data;
                        std::string address;
                        AssetNullDataFromScript(script, data, address);

                        AssetType type;
                        IsAssetNameValid(data.asset_name, type);

                        if (type == AssetType::RESTRICTED) {
                            assetsCache->AddRestrictedAddress(data.asset_name, address, data.flag ? RestrictedType::FREEZE_ADDRESS : RestrictedType::UNFREEZE_ADDRESS);
                        } else if (type == AssetType::QUALIFIER || type == AssetType::SUB_QUALIFIER) {
                            assetsCache->AddQualifierAddress(data.asset_name, address, data.flag ? QualifierType::ADD_QUALIFIER : QualifierType::REMOVE_QUALIFIER);
                        }
                    } else if (script.IsNullGlobalRestrictionAssetTxDataScript()) {
                        CNullAssetTxData data;
                        GlobalAssetNullDataFromScript(script, data);

                        assetsCache->AddGlobalRestricted(data.asset_name, data.flag ? RestrictedType::GLOBAL_FREEZE : RestrictedType::GLOBAL_UNFREEZE);
                    }
                }
            }
        }
        /** ASSET END */
    }
}

static const Coin coinEmpty; 
static const size_t MAX_OUTPUTS_PER_BLOCK = MAX_BLOCK_SIZE / ::GetSerializeSize(CTxOut(), SER_NETWORK, PROTOCOL_VERSION); // TODO: merge with similar definition in undo.h.

const Coin& AccessByTxid(const CCoinsViewCache& view, const uint256& txid)
{
    COutPoint iter(txid, 0);
    while (iter.n < MAX_OUTPUTS_PER_BLOCK) {
        const Coin& alternate = view.AccessCoin(iter);
        if (!alternate.IsSpent())
            return alternate;
        ++iter.n;
    }
    return coinEmpty;
}
/** ASSET END */

//! Guess how far we are in the verification process at the given block index
double GuessVerificationProgress(const ChainTxData& data, CBlockIndex* pindex)
{
    if (pindex == nullptr)
        return 0.0;

    int64_t nNow = time(nullptr);

    double fTxTotal;

    if (pindex->nChainTx <= data.nTxCount) {
        fTxTotal = data.nTxCount + (nNow - data.nTime) * data.dTxRate;
    } else {
        fTxTotal = pindex->nChainTx + (nNow - pindex->GetBlockTime()) * data.dTxRate;
    }

    return pindex->nChainTx / fTxTotal;
}

class CMainCleanup
{
public:
    CMainCleanup() {}
    ~CMainCleanup()
    {
        // block headers
        BlockMap::iterator it1 = mapBlockIndex.begin();
        for (; it1 != mapBlockIndex.end(); it1++)
            delete (*it1).second;
        mapBlockIndex.clear();
    }
} instance_of_cmaincleanup;

//! Begin Proof-of-Stake
// peercoin: sign block
typedef std::vector<unsigned char> valtype;
bool SignBlock(CBlock& block, const CKeyStore& keystore)
{
    std::vector<valtype> vSolutions;
    txnouttype whichType;
    const CTxOut& txout = block.IsProofOfStake()? block.vtx[1]->vout[1] : block.vtx[0]->vout[0];

    if (!Solver(txout.scriptPubKey, whichType, vSolutions))
        return false;
    if (whichType == TX_PUBKEY)
    {
        // Sign
        const valtype& vchPubKey = vSolutions[0];
        CKey key;
        if (!keystore.GetKey(CKeyID(Hash160(vchPubKey)), key))
            return false;
        if (key.GetPubKey() != CPubKey(vchPubKey))
            return false;
        return key.Sign(block.GetHash(), block.vchBlockSig);
    }
    return false;
}

// peercoin: check block signature
bool CheckBlockSignature(const CBlock& block)
{
    if ((block.GetHash() == Params().GetConsensus().hashGenesisBlock) || block.IsProofOfWork())
        return block.vchBlockSig.empty();

    CPubKey pubkey;
    txnouttype whichType;
    std::vector<valtype> vSolutions;
    const CTxOut& txout = block.vtx[1]->vout[1];
    if (!Solver(txout.scriptPubKey, whichType, vSolutions))
        return false;
    if (whichType == TX_PUBKEY || whichType == TX_PUBKEYHASH) {
        valtype& vchPubKey = vSolutions[0];
        pubkey = CPubKey(vchPubKey);
    }

    if (!pubkey.IsValid())
        return error("%s: invalid pubkey %s", __func__, HexStr(pubkey));

    return pubkey.Verify(block.GetHash(), block.vchBlockSig);
}
//! End Proof-of-Stake