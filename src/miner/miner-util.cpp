// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner/miner-util.h"
#include "consensus/consensus.h"
#include "consensus/merkle.h"
#include "consensus/validation.h"
#include "dynode-payments.h"
#include "fluid/fluiddb.h"
#include "fluid/fluidmining.h"
#include "fluid/fluidmint.h"
#include "governance.h"
#include "policy/policy.h"
#include "pow.h"
#include "primitives/transaction.h"
#include "timedata.h"
#include "txmempool.h"
#include "utilmoneystr.h"
#include "utiltime.h"
#include "validation.h"
#include "wallet/wallet.h"

#include <queue>

bool ProcessBlockFound(const CBlock& block, const CChainParams& chainparams)
{
    LogPrintf("%s\n", block.ToString());
    LogPrintf("generated %s\n", FormatMoney(block.vtx[0]->vout[0].nValue));

    // Found a solution
    {
        LOCK(cs_main);
        if (block.hashPrevBlock != chainActive.Tip()->GetBlockHash()) {
            return error("ProcessBlockFound -- generated block is stale");
        }
    }

    // Inform about the new block
    GetMainSignals().BlockFound(block.GetHash());

    // Process this block the same as if we had received it from another node
    CValidationState state;
    auto shared_pblock = std::make_shared<const CBlock>(block);
    if (!ProcessNewBlock(chainparams, shared_pblock, true, NULL)) {
        return error("ProcessBlockFound -- ProcessNewBlock() failed, block not accepted");
    }

    return true;
}

int64_t UpdateTime(CBlockHeader& block, const Consensus::Params& cparams, const CBlockIndex* indexPrev)
{
    int64_t old_time = block.nTime;
    int64_t new_time = std::max(indexPrev->GetMedianTimePast() + 1, GetAdjustedTime());

    if (old_time < new_time)
        block.nTime = new_time;

    // Updating time can change work required on testnet:
    if (cparams.fPowAllowMinDifficultyBlocks)
        block.nBits = GetNextWorkRequired(indexPrev, block, cparams);

    return new_time - old_time;
}

class ScoreCompare
{
public:
    ScoreCompare() {}

    bool operator()(const CTxMemPool::txiter a, const CTxMemPool::txiter b)
    {
        return CompareTxMemPoolEntryByScore()(*b, *a); // Convert to less than
    }
};

// Unconfirmed transactions in the memory pool often depend on other
// transactions in the memory pool. When we select transactions from the
// pool, we select by highest priority or fee rate, so we might consider
// transactions that depend on transactions that aren't yet in the block.
uint64_t nLastBlockTx = 0;
uint64_t nLastBlockSize = 0;

std::unique_ptr<CBlockTemplate> CreateNewBlock(const CChainParams& chainparams, const CScript* scriptPubKeyIn)
{
    // Create new block
    std::unique_ptr<CBlockTemplate> pblocktemplate(new CBlockTemplate());
    CBlock& block = pblocktemplate->block; // pointer for convenience

    // Create coinbase tx with fluid issuance
    // TODO: Can this be made any more elegant?
    CMutableTransaction txNew;
    txNew.vin.resize(1);
    txNew.vin[0].prevout.SetNull();
    txNew.vout.resize(1);

    // Largest block you're willing to create:
    unsigned int nBlockMaxSize = GetArg("-blockmaxsize", DEFAULT_BLOCK_MAX_SIZE);
    // Limit to between 1K and MAX_BLOCK_SIZE-1K for sanity:
    nBlockMaxSize = std::max((unsigned int)1000, std::min((unsigned int)(MAX_BLOCK_SIZE - 1000), nBlockMaxSize));

    // How much of the block should be dedicated to high-priority transactions,
    // included regardless of the fees they pay
    unsigned int nBlockPrioritySize = GetArg("-blockprioritysize", DEFAULT_BLOCK_PRIORITY_SIZE);
    nBlockPrioritySize = std::min(nBlockMaxSize, nBlockPrioritySize);

    // Minimum block size you want to create; block will be filled with free transactions
    // until there are no more or the block reaches this size:
    unsigned int nBlockMinSize = GetArg("-blockminsize", DEFAULT_BLOCK_MIN_SIZE);
    nBlockMinSize = std::min(nBlockMaxSize, nBlockMinSize);

    // Collect memory pool transactions into the block
    CTxMemPool::setEntries inBlock;
    CTxMemPool::setEntries waitSet;

    // This vector will be sorted into a priority queue:
    std::vector<TxCoinAgePriority> vecPriority;
    TxCoinAgePriorityCompare pricomparer;
    std::map<CTxMemPool::txiter, double, CTxMemPool::CompareIteratorByHash> waitPriMap;
    typedef std::map<CTxMemPool::txiter, double, CTxMemPool::CompareIteratorByHash>::iterator waitPriIter;
    double actualPriority = -1;

    std::priority_queue<CTxMemPool::txiter, std::vector<CTxMemPool::txiter>, ScoreCompare> clearedTxs;
    bool fPrintPriority = GetBoolArg("-printpriority", DEFAULT_PRINTPRIORITY);
    uint64_t nBlockSize = 1000;
    uint64_t nBlockTx = 0;
    unsigned int nBlockSigOps = 100;
    int lastFewTxs = 0;
    CAmount nFees = 0;

    {
        LOCK2(cs_main, governance.cs);
        LOCK(mempool.cs);
        CBlockIndex* indexPrev = chainActive.Tip();
        const int nHeight = indexPrev->nHeight + 1;
        block.nTime = GetAdjustedTime();
        const int64_t nMedianTimePast = indexPrev->GetMedianTimePast();

        // Add our coinbase tx as first transaction
        block.vtx.emplace_back();
        pblocktemplate->vTxFees.push_back(-1);   // updated at end
        pblocktemplate->vTxSigOps.push_back(-1); // updated at end
        block.nVersion = ComputeBlockVersion(indexPrev, chainparams.GetConsensus());
        // -regtest only: allow overriding block.nVersion with
        // -blockversion=N to test forking scenarios
        if (chainparams.MineBlocksOnDemand())
            block.nVersion = GetArg("-blockversion", block.nVersion);

        int64_t nLockTimeCutoff = (STANDARD_LOCKTIME_VERIFY_FLAGS & LOCKTIME_MEDIAN_TIME_PAST) ? nMedianTimePast : block.GetBlockTime();


        bool fPriorityBlock = nBlockPrioritySize > 0;
        if (fPriorityBlock) {
            vecPriority.reserve(mempool.mapTx.size());
            for (CTxMemPool::indexed_transaction_set::iterator mi = mempool.mapTx.begin(); mi != mempool.mapTx.end(); ++mi) {
                double dPriority = mi->GetPriority(nHeight);
                CAmount dummy;
                mempool.ApplyDeltas(mi->GetTx().GetHash(), dPriority, dummy);
                vecPriority.push_back(TxCoinAgePriority(dPriority, mi));
            }
            std::make_heap(vecPriority.begin(), vecPriority.end(), pricomparer);
        }

        CTxMemPool::indexed_transaction_set::index<mining_score>::type::iterator mi = mempool.mapTx.get<mining_score>().begin();
        CTxMemPool::txiter iter;

        while (mi != mempool.mapTx.get<mining_score>().end() || !clearedTxs.empty()) {
            bool priorityTx = false;
            if (fPriorityBlock && !vecPriority.empty()) { // add a tx from priority queue to fill the blockprioritysize
                priorityTx = true;
                iter = vecPriority.front().second;
                actualPriority = vecPriority.front().first;
                std::pop_heap(vecPriority.begin(), vecPriority.end(), pricomparer);
                vecPriority.pop_back();
            } else if (clearedTxs.empty()) { // add tx with next highest score
                iter = mempool.mapTx.project<0>(mi);
                mi++;
            } else { // try to add a previously postponed child tx
                iter = clearedTxs.top();
                clearedTxs.pop();
            }

            if (inBlock.count(iter))
                continue; // could have been added to the priorityBlock

            const CTransaction& tx = iter->GetTx();

            bool fOrphan = false;
            BOOST_FOREACH (CTxMemPool::txiter parent, mempool.GetMemPoolParents(iter)) {
                if (!inBlock.count(parent)) {
                    fOrphan = true;
                    break;
                }
            }
            if (fOrphan) {
                if (priorityTx)
                    waitPriMap.insert(std::make_pair(iter, actualPriority));
                else
                    waitSet.insert(iter);
                continue;
            }

            unsigned int nTxSize = iter->GetTxSize();
            if (fPriorityBlock && (nBlockSize + nTxSize >= nBlockPrioritySize || !AllowFree(actualPriority))) {
                fPriorityBlock = false;
                waitPriMap.clear();
            }
            if (!priorityTx && (iter->GetModifiedFee() < ::minRelayTxFee.GetFee(nTxSize) && nBlockSize >= nBlockMinSize)) {
                break;
            }
            if (nBlockSize + nTxSize >= nBlockMaxSize) {
                if (nBlockSize > nBlockMaxSize - 100 || lastFewTxs > 50) {
                    break;
                }
                // Once we're within 1000 bytes of a full block, only look at 50 more txs
                // to try to fill the remaining space.
                if (nBlockSize > nBlockMaxSize - 1000) {
                    lastFewTxs++;
                }
                continue;
            }

            if (!IsFinalTx(tx, nHeight, nLockTimeCutoff))
                continue;

            unsigned int nTxSigOps = iter->GetSigOpCount();
            if (nBlockSigOps + nTxSigOps >= MAX_BLOCK_SIGOPS) {
                if (nBlockSigOps > MAX_BLOCK_SIGOPS - 2) {
                    break;
                }
                continue;
            }

            block.vtx.emplace_back(iter->GetSharedTx());

            pblocktemplate->vTxFees.push_back(iter->GetFee());
            pblocktemplate->vTxSigOps.push_back(iter->GetSigOpCount());
            nBlockSize += iter->GetTxSize();
            ++nBlockTx;
            nBlockSigOps += iter->GetSigOpCount();
            nFees += iter->GetFee();

            if (fPrintPriority) {
                double dPriority = iter->GetPriority(nHeight);
                CAmount dummy;
                mempool.ApplyDeltas(tx.GetHash(), dPriority, dummy);
                LogPrintf("priority %.1f fee %s txid %s\n", dPriority, CFeeRate(iter->GetModifiedFee(), nTxSize).ToString(),
                    tx.GetHash().ToString());
            }

            inBlock.insert(iter);

            // Add transactions that depend on this one to the priority queue
            BOOST_FOREACH (CTxMemPool::txiter child, mempool.GetMemPoolChildren(iter)) {
                if (fPriorityBlock) {
                    waitPriIter wpiter = waitPriMap.find(child);
                    if (wpiter != waitPriMap.end()) {
                        vecPriority.push_back(TxCoinAgePriority(wpiter->second, child));
                        std::push_heap(vecPriority.begin(), vecPriority.end(), pricomparer);
                        waitPriMap.erase(wpiter);
                    }
                } else {
                    if (waitSet.count(child)) {
                        clearedTxs.push(child);
                        waitSet.erase(child);
                    }
                }
            }
        }

        CAmount blockReward = GetFluidMiningReward(nHeight);
        CDynamicAddress mintAddress;
        CAmount fluidIssuance = 0;
        CFluidMint fluidMint;
        bool areWeMinting = GetMintingInstructions(nHeight, fluidMint);

        // Compute regular coinbase transaction.
        // HACK: move outside for split-miner
        if (scriptPubKeyIn) {
            txNew.vout[0].scriptPubKey = *scriptPubKeyIn;
        }
        txNew.vin[0].scriptSig = CScript() << nHeight << OP_0;

        if (areWeMinting) {
            mintAddress = fluidMint.GetDestinationAddress();
            fluidIssuance = fluidMint.MintAmount;
            txNew.vout[0].nValue = blockReward + fluidIssuance;
        } else {
            txNew.vout[0].nValue = blockReward;
        }

        CScript script;
        if (areWeMinting) {
            // Pick out the amount of issuance
            txNew.vout[0].nValue -= fluidIssuance;

            assert(mintAddress.IsValid());
            if (!mintAddress.IsScript()) {
                script = GetScriptForDestination(mintAddress.Get());
            } else {
                CScriptID fluidScriptID = boost::get<CScriptID>(mintAddress.Get());
                script = CScript() << OP_HASH160 << ToByteVector(fluidScriptID) << OP_EQUAL;
            }
            txNew.vout.push_back(CTxOut(fluidIssuance, script));
            LogPrintf("CreateNewBlock(): Generated Fluid Issuance Transaction:\n%s\n", txNew.ToString());
        }

        // Update coinbase transaction with additional info about dynode and governance payments,
        // get some info back to pass to getblocktemplate
        FillBlockPayments(txNew, nHeight, blockReward, pblocktemplate->txoutDynode, pblocktemplate->voutSuperblock);
        // LogPrintf("CreateNewBlock -- nBlockHeight %d blockReward %lld txoutDynode %s txNew %s",
        //             nHeight, blockReward, block.txoutDynode.ToString(), txNew.ToString());

        nLastBlockTx = nBlockTx;
        nLastBlockSize = nBlockSize;
        LogPrintf("CreateNewBlock(): total size %u txs: %u fees: %ld sigops %d\n", nBlockSize, nBlockTx, nFees, nBlockSigOps);

        CAmount blockAmount = blockReward + fluidIssuance;
        LogPrintf("CreateNewBlock(): Computed Miner Block Reward is %ld DYN\n", FormatMoney(blockAmount));

        // Update block coinbase
        block.vtx[0] = MakeTransactionRef(std::move(txNew));
        pblocktemplate->vTxFees[0] = -nFees;

        // Fill in header
        block.hashPrevBlock = indexPrev->GetBlockHash();
        UpdateTime(block, chainparams.GetConsensus(), indexPrev);
        block.nBits = GetNextWorkRequired(indexPrev, block, chainparams.GetConsensus());
        block.nNonce = 0;
        pblocktemplate->vTxSigOps[0] = GetLegacySigOpCount(*block.vtx[0]);

        CValidationState state;
        if (!TestBlockValidity(state, chainparams, block, indexPrev, false, false)) {
            LogPrintf("CreateNewBlock(): Generated Transaction:\n%s\n", txNew.ToString());
            throw std::runtime_error(tfm::format("%s: TestBlockValidity failed: %s", __func__, FormatStateMessage(state)));
        }
    }

    return pblocktemplate;
}

std::unique_ptr<CBlockTemplate> CreateNewBlock(const CChainParams& chainparams, const CScript& scriptPubKeyIn)
{
    return CreateNewBlock(chainparams, &scriptPubKeyIn);
}

void IncrementExtraNonce(CBlock& block, const CBlockIndex* indexPrev, unsigned int& nExtraNonce)
{
    // Update nExtraNonce
    static uint256 hashPrevBlock;
    if (hashPrevBlock != block.hashPrevBlock) {
        nExtraNonce = 0;
        hashPrevBlock = block.hashPrevBlock;
    }
    // Increment extra nonce
    ++nExtraNonce;
    // Height first in coinbase required for block.version=2
    unsigned int nHeight = indexPrev->nHeight + 1;
    // Create copied transaction
    CMutableTransaction txCoinbase(*block.vtx[0]);
    // Set extra nonce in script
    txCoinbase.vin[0].scriptSig = (CScript() << nHeight << CScriptNum(nExtraNonce)) + COINBASE_FLAGS;
    // Make sure script size is correct
    // NOTE: `100` should be a constant
    assert(txCoinbase.vin[0].scriptSig.size() <= 100);
    // Set new transaction in block
    block.vtx[0] = MakeTransactionRef(std::move(txCoinbase));
    // Generate merkle root hash
    block.hashMerkleRoot = BlockMerkleRoot(block);
}

void SetBlockPubkeyScript(CBlock& block, const CScript& scriptPubKeyIn)
{
    // Create copied transaction
    CMutableTransaction txCoinbase(*block.vtx[0]);
    // Set coinbase out address
    txCoinbase.vout[0].scriptPubKey = scriptPubKeyIn;
    //It should be added to the block
    block.vtx[0] = MakeTransactionRef(std::move(txCoinbase));
    // Generate merkle root hash
    block.hashMerkleRoot = BlockMerkleRoot(block);
}
