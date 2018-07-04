// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2018 The Dash Core Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "miner.h"

#include "amount.h"
#include "chain.h"
#include "chainparams.h"
#include "coins.h"
#include "consensus/consensus.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "governance-classes.h"
#include "hash.h"
#include "validation.h"
#include "consensus/merkle.h"
#include "net.h"
#include "policy/policy.h"
#include "pow.h"
#include "script/standard.h"
#include "timedata.h"
#include "primitives/transaction.h"
#include "txmempool.h"
#include "util.h"
#include "utilmoneystr.h"
#include "consensus/validation.h"
#include "validationinterface.h"
#include "wallet/wallet.h"
#include "fluid.h"

#include <queue>
#include <utility>

#include <openssl/sha.h>

#include <boost/thread.hpp>
#include <boost/tuple/tuple.hpp>

#include "miner-gpu.h"

//////////////////////////////////////////////////////////////////////////////
//
// DynamicMiner
//

//
// Unconfirmed transactions in the memory pool often depend on other
// transactions in the memory pool. When we select transactions from the
// pool, we select by highest priority or fee rate, so we might consider
// transactions that depend on transactions that aren't yet in the block.

uint64_t nLastBlockTx = 0;
uint64_t nLastBlockSize = 0;

class ScoreCompare
{
public:
    ScoreCompare() {}

    bool operator()(const CTxMemPool::txiter a, const CTxMemPool::txiter b)
    {
        return CompareTxMemPoolEntryByScore()(*b,*a); // Convert to less than
    }
};

int64_t UpdateTime(CBlockHeader* pblock, const Consensus::Params& consensusParams, const CBlockIndex* pindexPrev)
{
    int64_t nOldTime = pblock->nTime;
    int64_t nNewTime = std::max(pindexPrev->GetMedianTimePast()+1, GetAdjustedTime());

    if (nOldTime < nNewTime)
        pblock->nTime = nNewTime;

    // Updating time can change work required on testnet:
    if (consensusParams.fPowAllowMinDifficultyBlocks)
        pblock->nBits = GetNextWorkRequired(pindexPrev, pblock, consensusParams);

    return nNewTime - nOldTime;
}

#ifdef ENABLE_WALLET
bool CheckWork(const CChainParams& chainparams, CBlock* pblock, CWallet& wallet, CReserveKey& reservekey, CConnman* connman)
{
    uint256 hash = pblock->GetHash();
    arith_uint256 hashTarget = arith_uint256().SetCompact(pblock->nBits);

    if (UintToArith256(hash) > hashTarget)
        return false;

    // Found a solution
    {
        LOCK(cs_main);
        if (pblock->hashPrevBlock != chainActive.Tip()->GetBlockHash())
            return error("Generated block is stale!");

        // Remove key from key pool
        reservekey.KeepKey();

        // Track how many getdata requests this block gets
        {
            LOCK(wallet.cs_wallet);
            wallet.mapRequestCount[pblock->GetHash()] = 0;
        }

        // Process this block the same as if we had received it from another node
        CValidationState state;
        if (!ProcessNewBlock(chainparams, pblock, true, NULL, NULL))
            return error("ProcessBlock, block not accepted");
    }

    return true;
}
#endif //ENABLE_WALLET

std::unique_ptr<CBlockTemplate> CreateNewBlock(const CChainParams& chainparams, const CScript& scriptPubKeyIn)
{
    // Create new block
    std::unique_ptr<CBlockTemplate> pblocktemplate(new CBlockTemplate());
    if(!pblocktemplate.get())
        return nullptr;
    CBlock *pblock = &pblocktemplate->block; // pointer for convenience

    // Create coinbase tx with fluid issuance
    // TODO: Can this be made any more elegant?
    CMutableTransaction txNew;
    txNew.vin.resize(1);
    txNew.vin[0].prevout.SetNull();
    txNew.vout.resize(1);

    // Largest block you're willing to create:
    unsigned int nBlockMaxSize = GetArg("-blockmaxsize", DEFAULT_BLOCK_MAX_SIZE);
    // Limit to between 1K and MAX_BLOCK_SIZE-1K for sanity:
    nBlockMaxSize = std::max((unsigned int)1000, std::min((unsigned int)(MAX_BLOCK_SIZE-1000), nBlockMaxSize));

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
        CBlockIndex* pindexPrev = chainActive.Tip();
        const int nHeight = pindexPrev->nHeight + 1;
        pblock->nTime = GetAdjustedTime();
        const int64_t nMedianTimePast = pindexPrev->GetMedianTimePast();

        // Add our coinbase tx as first transaction
        pblock->vtx.push_back(txNew);
        pblocktemplate->vTxFees.push_back(-1); // updated at end
        pblocktemplate->vTxSigOps.push_back(-1); // updated at end
        pblock->nVersion = ComputeBlockVersion(pindexPrev, chainparams.GetConsensus());
        // -regtest only: allow overriding block.nVersion with
        // -blockversion=N to test forking scenarios
        if (chainparams.MineBlocksOnDemand())
            pblock->nVersion = GetArg("-blockversion", pblock->nVersion);

        int64_t nLockTimeCutoff = (STANDARD_LOCKTIME_VERIFY_FLAGS & LOCKTIME_MEDIAN_TIME_PAST)
                                  ? nMedianTimePast
                                  : pblock->GetBlockTime();


        bool fPriorityBlock = nBlockPrioritySize > 0;
        if (fPriorityBlock) {
            vecPriority.reserve(mempool.mapTx.size());
            for (CTxMemPool::indexed_transaction_set::iterator mi = mempool.mapTx.begin();
                    mi != mempool.mapTx.end(); ++mi)
            {
                double dPriority = mi->GetPriority(nHeight);
                CAmount dummy;
                mempool.ApplyDeltas(mi->GetTx().GetHash(), dPriority, dummy);
                vecPriority.push_back(TxCoinAgePriority(dPriority, mi));
            }
            std::make_heap(vecPriority.begin(), vecPriority.end(), pricomparer);
        }

        CTxMemPool::indexed_transaction_set::index<mining_score>::type::iterator mi = mempool.mapTx.get<mining_score>().begin();
        CTxMemPool::txiter iter;

        while (mi != mempool.mapTx.get<mining_score>().end() || !clearedTxs.empty())
        {
            bool priorityTx = false;
            if (fPriorityBlock && !vecPriority.empty()) { // add a tx from priority queue to fill the blockprioritysize
                priorityTx = true;
                iter = vecPriority.front().second;
                actualPriority = vecPriority.front().first;
                std::pop_heap(vecPriority.begin(), vecPriority.end(), pricomparer);
                vecPriority.pop_back();
            }
            else if (clearedTxs.empty()) { // add tx with next highest score
                iter = mempool.mapTx.project<0>(mi);
                mi++;
            }
            else {  // try to add a previously postponed child tx
                iter = clearedTxs.top();
                clearedTxs.pop();
            }

            if (inBlock.count(iter))
                continue; // could have been added to the priorityBlock

            const CTransaction& tx = iter->GetTx();

            bool fOrphan = false;
            BOOST_FOREACH(CTxMemPool::txiter parent, mempool.GetMemPoolParents(iter))
            {
                if (!inBlock.count(parent)) {
                    fOrphan = true;
                    break;
                }
            }
            if (fOrphan) {
                if (priorityTx)
                    waitPriMap.insert(std::make_pair(iter,actualPriority));
                else
                    waitSet.insert(iter);
                continue;
            }

            unsigned int nTxSize = iter->GetTxSize();
            if (fPriorityBlock &&
                    (nBlockSize + nTxSize >= nBlockPrioritySize || !AllowFree(actualPriority))) {
                fPriorityBlock = false;
                waitPriMap.clear();
            }
            if (!priorityTx &&
                    (iter->GetModifiedFee() < ::minRelayTxFee.GetFee(nTxSize) && nBlockSize >= nBlockMinSize)) {
                break;
            }
            if (nBlockSize + nTxSize >= nBlockMaxSize) {
                if (nBlockSize >  nBlockMaxSize - 100 || lastFewTxs > 50) {
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

            CAmount nTxFees = iter->GetFee();
            pblock->vtx.push_back(tx);

            pblocktemplate->vTxFees.push_back(nTxFees);
            pblocktemplate->vTxSigOps.push_back(nTxSigOps);
            nBlockSize += nTxSize;
            ++nBlockTx;
            nBlockSigOps += nTxSigOps;
            nFees += nTxFees;

            if (fPrintPriority)
            {
                double dPriority = iter->GetPriority(nHeight);
                CAmount dummy;
                mempool.ApplyDeltas(tx.GetHash(), dPriority, dummy);
                LogPrintf("priority %.1f fee %s txid %s\n",
                          dPriority, CFeeRate(iter->GetModifiedFee(), nTxSize).ToString(), tx.GetHash().ToString());
            }

            inBlock.insert(iter);

            // Add transactions that depend on this one to the priority queue
            BOOST_FOREACH(CTxMemPool::txiter child, mempool.GetMemPoolChildren(iter))
            {
                if (fPriorityBlock) {
                    waitPriIter wpiter = waitPriMap.find(child);
                    if (wpiter != waitPriMap.end()) {
                        vecPriority.push_back(TxCoinAgePriority(wpiter->second,child));
                        std::push_heap(vecPriority.begin(), vecPriority.end(), pricomparer);
                        waitPriMap.erase(wpiter);
                    }
                }
                else {
                    if (waitSet.count(child)) {
                        clearedTxs.push(child);
                        waitSet.erase(child);
                    }
                }
            }
        }

        CDynamicAddress address;
        CFluidEntry prevFluidIndex = pindexPrev->fluidParams;
		CAmount fluidIssuance = 0, blockReward = getBlockSubsidyWithOverride(nHeight, prevFluidIndex.blockReward);
        bool areWeMinting = fluid.GetMintingInstructions(pindexPrev, address, fluidIssuance);

        // Compute regular coinbase transaction.
        txNew.vout[0].scriptPubKey = scriptPubKeyIn;

        if (areWeMinting) {
            txNew.vout[0].nValue = blockReward + fluidIssuance;
        } else {
            txNew.vout[0].nValue = blockReward;
        }
		
        txNew.vin[0].scriptSig = CScript() << nHeight << OP_0;

        CScript script;
        
        if (areWeMinting) {		
            // Pick out the amount of issuance
            txNew.vout[0].nValue -= fluidIssuance;

            assert(address.IsValid());
            if (!address.IsScript()) {
                script = GetScriptForDestination(address.Get());
            } else {
                CScriptID fluidScriptID = boost::get<CScriptID>(address.Get());
                script = CScript() << OP_HASH160 << ToByteVector(fluidScriptID) << OP_EQUAL;
            }

            txNew.vout.push_back(CTxOut(fluidIssuance, script));
            LogPrintf("CreateNewBlock(): Generated Fluid Issuance Transaction:\n%s\n", txNew.ToString());
        }

        // Update coinbase transaction with additional info about dynode and governance payments,
        // get some info back to pass to getblocktemplate
        FillBlockPayments(txNew, nHeight, pblock->txoutDynode, pblock->voutSuperblock);
        // LogPrintf("CreateNewBlock -- nBlockHeight %d blockReward %lld txoutDynode %s txNew %s",
        //             nHeight, blockReward, pblock->txoutDynode.ToString(), txNew.ToString());

        nLastBlockTx = nBlockTx;
        nLastBlockSize = nBlockSize;
        LogPrintf("CreateNewBlock(): total size %u txs: %u fees: %ld sigops %d\n", nBlockSize, nBlockTx, nFees, nBlockSigOps);
 
        CAmount blockAmount = blockReward + fluidIssuance;
		LogPrintf("CreateNewBlock(): Computed Miner Block Reward is %ld DYN\n", FormatMoney(blockAmount));

        // Update block coinbase
        pblock->vtx[0] = txNew;
        pblocktemplate->vTxFees[0] = -nFees;

        // Fill in header
        pblock->hashPrevBlock  = pindexPrev->GetBlockHash();
        UpdateTime(pblock, chainparams.GetConsensus(), pindexPrev);
        pblock->nBits = GetNextWorkRequired(pindexPrev, pblock, chainparams.GetConsensus());
        pblock->nNonce = 0;
        pblocktemplate->vTxSigOps[0] = GetLegacySigOpCount(pblock->vtx[0]);
		CValidationState state;
        if (!TestBlockValidity(state, chainparams, *pblock, pindexPrev, false, false)) {
            LogPrintf("CreateNewBlock(): Generated Transaction:\n%s\n", txNew.ToString());
            throw std::runtime_error(strprintf("%s: TestBlockValidity failed: %s", __func__, FormatStateMessage(state)));
        }
    }

    return pblocktemplate;
}

void IncrementExtraNonce(CBlock* pblock, const CBlockIndex* pindexPrev, unsigned int& nExtraNonce)
{
    // Update nExtraNonce
    static uint256 hashPrevBlock;
    if (hashPrevBlock != pblock->hashPrevBlock)
    {
        nExtraNonce = 0;
        hashPrevBlock = pblock->hashPrevBlock;
    }
    ++nExtraNonce;
    unsigned int nHeight = pindexPrev->nHeight+1; // Height first in coinbase required for block.version=2
    CMutableTransaction txCoinbase(pblock->vtx[0]);
    txCoinbase.vin[0].scriptSig = (CScript() << nHeight << CScriptNum(nExtraNonce)) + COINBASE_FLAGS;
    assert(txCoinbase.vin[0].scriptSig.size() <= 100);

    pblock->vtx[0] = txCoinbase;
    pblock->hashMerkleRoot = BlockMerkleRoot(*pblock);
}

//////////////////////////////////////////////////////////////////////////////
//
// Internal miner
//
//
double dCPUHashesPerSec = 0.0;
double dGPUHashesPerSec = 0.0;
int64_t nHPSTimerStart = 0;

int64_t GetHashRate()
{
    return GetCPUHashRate() + GetGPUHashRate();
}

int64_t GetCPUHashRate()
{
    if (GetTimeMillis() - nHPSTimerStart > 8000)
        return (int64_t)0;
    return (int64_t)(dCPUHashesPerSec);
}

int64_t GetGPUHashRate()
{
    if (GetTimeMillis() - nHPSTimerStart > 8000)
        return (int64_t)0;
    return (int64_t)(dGPUHashesPerSec);
}

// ScanHash scans nonces looking for a hash with at least some zero bits.
// The nonce is usually preserved between calls, but periodically or if the
// nonce is 0xffff0000 or above, the block is rebuilt and nNonce starts over at
// zero.
//
//bool static ScanHash(const CBlockHeader *pblock, uint32_t& nNonce, uint256 *phash)
//{
    // Write the first 76 bytes of the block header to a double-SHA256 state.
//    CHash256 hasher;
//    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
//    ss << *pblock;
//    assert(ss.size() == 80);
//    hasher.Write((unsigned char*)&ss[0], 76);

//    while (true) {
//        nNonce++;

        // Write the last 4 bytes of the block header (the nonce) to a copy of
        // the double-SHA256 state, and compute the result.
//        CHash256(hasher).Write((unsigned char*)&nNonce, 4).Finalize((unsigned char*)phash);

        // Return the nonce if the hash has at least some zero bits,
        // caller will check if it has enough to reach the target
//        if (((uint16_t*)phash)[15] == 0)
//            return true;

        // If nothing found after trying for a while, return -1
//        if ((nNonce & 0xfff) == 0)
//            return false;
//    }/
//}

static bool ProcessBlockFound(const CBlock* pblock, const CChainParams& chainparams, CConnman* connman)
{
    LogPrintf("%s\n", pblock->ToString());
    LogPrintf("generated %s\n", FormatMoney(pblock->vtx[0].vout[0].nValue));

    // Found a solution
    {
        LOCK(cs_main);
        if (pblock->hashPrevBlock != chainActive.Tip()->GetBlockHash())
            return error("ProcessBlockFound -- generated block is stale");
    }

    // Inform about the new block
    GetMainSignals().BlockFound(pblock->GetHash());

    // Process this block the same as if we had received it from another node
    CValidationState state;
    if (!ProcessNewBlock(chainparams, pblock, true, NULL, NULL))
        return error("ProcessBlockFound -- ProcessNewBlock() failed, block not accepted");

    return true;
}

static void WaitForNetworkInit(const CChainParams& chainparams, CConnman& connman)
{
    if (chainparams.MiningRequiresPeers()) {
        // Busy-wait for the network to come online so we don't waste time mining
        // on an obsolete chain. In regtest mode we expect to fly solo.
        while (true) {
            bool fvNodesEmpty = connman.GetNodeCount(CConnman::CONNECTIONS_ALL) == 0;
            if (!fvNodesEmpty && !IsInitialBlockDownload()) {
                break;
            }
            MilliSleep(1000);
        }
    }
}

#ifdef ENABLE_GPU
static Argon2GPU GetProcessingUnit(std::size_t nDeviceIndex, bool fGPU) {
    if (!fGPU) {
        Argon2GPU processingUnit(nullptr, nullptr, nullptr, 1, false, false);
        return processingUnit;
    }
    else {
        // Argon2GPU processingUnit = GetGPUProcessingUnit(nDeviceIndex);
        Argon2GPUContext global;
        auto& devices = global.getAllDevices();
        auto& device = devices[nDeviceIndex];
        Argon2GPUProgramContext context(&global, {device}, argon2gpu::ARGON2_D, argon2gpu::ARGON2_VERSION_10);
        Argon2GPUParams params((std::size_t)OUTPUT_BYTES, 2, 500, 8);
        Argon2GPU processingUnit(&context, &params, &device, 1, false, false);
        return processingUnit;
    }
}
#endif // ENABLE_GPU

static void DynamicMiner(const CChainParams& chainparams, CConnman& connman, std::size_t nDeviceIndex, bool fGPU) {

    std::string dev = fGPU ? "GPU" : "CPU";
    LogPrintf("DynamicMiner -- started #%u@%s\n", nDeviceIndex, dev);
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    if (fGPU) {
        RenameThread("dynamic-gpu-miner");
    }
    else {
        RenameThread("dynamic-cpu-miner");
    }

    double* dHashesPerSec = fGPU ? &dGPUHashesPerSec : &dCPUHashesPerSec;
    unsigned int nExtraNonce = 0;

    boost::shared_ptr<CReserveScript> coinbaseScript;
    GetMainSignals().ScriptForMining(coinbaseScript);
#ifdef ENABLE_GPU
    Argon2GPU processingUnit = GetProcessingUnit(nDeviceIndex, fGPU);
#endif // ENABLE_GPU
    try {
        // Throw an error if no script was provided.  This can happen
        // due to some internal error but also if the keypool is empty.
        // In the latter case, already the pointer is NULL.
        if (!coinbaseScript || coinbaseScript->reserveScript.empty())
            throw std::runtime_error("No coinbase script available (mining requires a wallet)");

        while (true) {
            // Wait for blocks if required
            WaitForNetworkInit(chainparams, connman);

            //
            // Create new block
            //
            unsigned int nTransactionsUpdatedLast = mempool.GetTransactionsUpdated();
            CBlockIndex* pindexPrev = chainActive.Tip();
            std::unique_ptr<CBlockTemplate> pblocktemplate;
            if(!pindexPrev) break;
            
            pblocktemplate = std::unique_ptr<CBlockTemplate> (CreateNewBlock(chainparams, coinbaseScript->reserveScript));

            if (!pblocktemplate.get())
            {
                LogPrintf("DynamicMiner%s -- Keypool ran out, please call keypoolrefill before restarting the mining thread\n", dev);
                return;
            }
            CBlock *pblock = &pblocktemplate->block;
            IncrementExtraNonce(pblock, pindexPrev, nExtraNonce);

            LogPrintf("DynamicMiner%s -- Running miner with %u transactions in block (%u bytes)\n", dev, pblock->vtx.size(),
                ::GetSerializeSize(*pblock, SER_NETWORK, PROTOCOL_VERSION));
            
            //
            // Search
            //
            int64_t nStart = GetTime();
            arith_uint256 hashTarget = arith_uint256().SetCompact(pblock->nBits);
            uint256 hash;
            while (true)
            {
                unsigned int nHashesDone = 0;

                uint256 hash;
                while (true) {
#ifdef ENABLE_GPU
                    if (fGPU)
                        hash = GetBlockHashGPU(pblock, &processingUnit);
                    else
                        hash = pblock->GetHash();
#else
                    hash = pblock->GetHash();
#endif // ENABLE_GPU
                    if (UintToArith256(hash) <= hashTarget)
                    {
                        // Found a solution
                        //pblock->nNonce = nNonce;
                        //assert(hash == pblock->GetHash());

                        SetThreadPriority(THREAD_PRIORITY_NORMAL);
                        LogPrintf("DynamicMiner%s:\n proof-of-work found  \n  hash: %s  \ntarget: %s\n", dev, hash.GetHex(), hashTarget.GetHex());
                        ProcessBlockFound(pblock, chainparams, &connman);
                        SetThreadPriority(THREAD_PRIORITY_LOWEST);
                        coinbaseScript->KeepScript();

                        // In regression test mode, stop mining after a block is found.
                        if (chainparams.MineBlocksOnDemand())
                            throw boost::thread_interrupted();

                        break;
                    }
                    pblock->nNonce += 1;
                    nHashesDone += 1;
                    if ((pblock->nNonce & 0xFF) == 0)
                        break;
                }
                
                // Meter hashes/seconds
                static int64_t nHashCounter = 0;
                static int64_t nLogTime = 0;

                if (nHPSTimerStart == 0)
                {
                    nHPSTimerStart = GetTimeMillis();
                    nHashCounter = 0;
                }
                else
                    nHashCounter += nHashesDone;
                if (GetTimeMillis() - nHPSTimerStart > 4000)
                {
                    static CCriticalSection cs;        
                    {
                        LOCK(cs);
                        if (GetTimeMillis() - nHPSTimerStart > 4000)
                        {
                            *dHashesPerSec = 1000.0 * nHashCounter / (GetTimeMillis() - nHPSTimerStart);
                            nHPSTimerStart = GetTimeMillis();
                            nHashCounter = 0;
                            if (GetTime() - nLogTime > 30 * 60)
                            {
                                nLogTime = GetTime();
                                LogPrintf("hashmeter %6.0f khash/s\n", *dHashesPerSec / 1000.0);
                            }
                        }
                    }
                }

                // Check for stop or if block needs to be rebuilt
                boost::this_thread::interruption_point();
                // Regtest mode doesn't require peers
                if (connman.GetNodeCount(CConnman::CONNECTIONS_ALL) == 0 && chainparams.MiningRequiresPeers())
                    break;
                if (pblock->nNonce >= 0xffff0000)
                    break;
                if (mempool.GetTransactionsUpdated() != nTransactionsUpdatedLast && GetTime() - nStart > 60)
                    break;
                if (pindexPrev != chainActive.Tip())
                    break;

                // Update nTime every few seconds
                if (UpdateTime(pblock, chainparams.GetConsensus(), pindexPrev) < 0)
                    break; // Recreate the block if the clock has run backwards,
                           // so that we can use the correct time.
                if (chainparams.GetConsensus().fPowAllowMinDifficultyBlocks)
                {
                    // Changing pblock->nTime can change work required on testnet:
                    hashTarget.SetCompact(pblock->nBits);
                }
            }
        }
    }
    catch (const boost::thread_interrupted&)
    {
        LogPrintf("DynamicMiner -- terminated\n");
        throw;
    }
    catch (const std::runtime_error &e)
    {
        LogPrintf("DynamicMiner -- runtime error: %s\n", e.what());
        return;
    }
}

static boost::thread_group* GetCPUMinerThreads(bool fInit = true, bool fRestart = true)
{
    static boost::thread_group* minerThreadsCPU = NULL;
    if (fRestart && minerThreadsCPU != NULL) {
        minerThreadsCPU->interrupt_all();
        delete minerThreadsCPU;
        minerThreadsCPU = NULL;
    }
    if (fInit && minerThreadsCPU == NULL) {
        minerThreadsCPU = new boost::thread_group();
    }
    return minerThreadsCPU;
}

static boost::thread_group* GetGPUMinerThreads(bool fInit = true, bool fRestart = true)
{
    static boost::thread_group* minerThreadsGPU = NULL;
    if (fRestart && minerThreadsGPU != NULL) {
        minerThreadsGPU->interrupt_all();
        delete minerThreadsGPU;
        minerThreadsGPU = NULL;
    }
    if (fInit && minerThreadsGPU == NULL) {
        minerThreadsGPU = new boost::thread_group();
    }
    return minerThreadsGPU;
}

static boost::thread_group* GetMinerThreads(bool fGPU = false, bool fInit = true, bool fRestart = true)
{
    if (fGPU) {
        return GetGPUMinerThreads(fInit, fRestart);
    }
    return GetCPUMinerThreads(fInit, fRestart);
}

void AutoTuneDeviceThreads(const CChainParams& chainparams, CConnman& connman, std::size_t nDeviceIndex = -1, bool fGPU = false)
{
    // Threads
    boost::thread* lastThread = NULL;
    boost::thread_group* minerThreads = GetMinerThreads(fGPU);

    // Metrics
    double nLastHashesPerSec = 0.0;
    double* dHashesPerSec = fGPU ? &dGPUHashesPerSec : &dCPUHashesPerSec;

    // Device name
    std::string dev = fGPU ? "GPU" : "CPU";

    LogPrintf("Autotune hashmeter: %6.0f\n", *dHashesPerSec / 1000.0);

    // Start Dynamin miner threads on device
    while (true) {
        LogPrintf("Starting %s Miner thread #%u %6.0f khash/s\n", dev, minerThreads->size(), *dHashesPerSec / 1000.0);
        std::size_t nThread = minerThreads->size();
        std::size_t device = nDeviceIndex == -1 ? nThread : nDeviceIndex;
        lastThread = minerThreads->create_thread(boost::bind(&DynamicMiner, boost::cref(chainparams), boost::ref(connman), boost::cref(device), boost::cref(fGPU)));
        MilliSleep(10000);

        if (*dHashesPerSec > nLastHashesPerSec) {
            nLastHashesPerSec = *dHashesPerSec;
            continue;
        }

        LogPrintf("Removing %s Miner thread #%u %6.0f khash/s\n", dev, minerThreads->size(), *dHashesPerSec / 1000.0);
        minerThreads->remove_thread(lastThread);
        break;
    }

    LogPrintf("Autotune %s finished with hashrate %6.0f\n khash/s", dev, *dHashesPerSec / 1000.0);
}

void GenerateAutoTuneDevice(const CChainParams& chainparams, CConnman& connman, bool fGPU)
{
    if (!fGPU) {
        AutoTuneDeviceThreads(chainparams, connman);
    } else {
        // Start GPU threads
        for (std::size_t device = 0; device < GetGPUDeviceCount(); device++) {
            AutoTuneDeviceThreads(chainparams, connman, device, true);
        }
    }
}

void GenerateDynamicsAutoTune(const CChainParams& chainparams, CConnman& connman)
{
    // Wait for blocks if required
    WaitForNetworkInit(chainparams, connman);
    MilliSleep(30000);

    GenerateAutoTuneDevice(chainparams, connman, false);
    GenerateAutoTuneDevice(chainparams, connman, true);
}

void GenerateDynamics(int nCPUThreads, int nGPUThreads, const CChainParams& chainparams, CConnman& connman, bool fAutotune)
{
    if (fAutotune) {
        return GenerateDynamicsAutoTune(chainparams, connman);
    }
    std::size_t devices = 0;
#ifdef ENABLE_GPU
    devices = GetGPUDeviceCount();
    LogPrintf("DynamicMiner -- GPU Devices: %u\n", devices);
#else
    LogPrintf("DynamicMiner -- GPU no support\n");
#endif

    if (nCPUThreads == 0 && (devices == 0 || nGPUThreads == 0)) {
        LogPrintf("DynamicMiner -- disabled -- CPU Threads set to zero\n");
        return;
    }

    boost::thread_group* cpuMinerThreads = GetCPUMinerThreads();

    if (nCPUThreads < 0)
        nCPUThreads = GetNumCores();

    // Start CPU threads
    while (cpuMinerThreads->size() < nCPUThreads) {
        std::size_t nThread = cpuMinerThreads->size() -1;
        LogPrintf("Starting CPU Miner thread #%u\n", nThread);
        cpuMinerThreads->create_thread(boost::bind(&DynamicMiner, boost::cref(chainparams), boost::ref(connman), nThread, false));
    }

    if (nGPUThreads < 0)
        nGPUThreads = 1;

    // Start GPU threads
    boost::thread_group* gpuMinerThreads = GetGPUMinerThreads();
    //for (std::size_t device = 0; device < devices; device++) {
        for (std::size_t i = 0; i < nGPUThreads || i < devices; i++) {
            LogPrintf("Starting GPU Miner thread %u on device %u\n", i, i);
            gpuMinerThreads->create_thread(boost::bind(&DynamicMiner, boost::cref(chainparams), boost::ref(connman), i, true));
        }
    //}
}

void ShutdownCPUMiners()
{
    GetCPUMinerThreads(false);
}

void ShutdownGPUMiners()
{
    GetGPUMinerThreads(false);
}

void ShutdownMiners()
{
    ShutdownCPUMiners();
    ShutdownGPUMiners();
}
