// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin developers
// Copyright (c) 2014-2019 The Dash developers
// Copyright (c) 2015-2019 The PIVX developers
// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.


#include "staker.h"

#ifdef ENABLE_WALLET
#include "chain.h"
#include "dynode-sync.h"
#include "miner/miner-util.h"
#include "net.h"
#include "pow.h"
#include "primitives/block.h"
#include "spork.h"
#include "timedata.h"
#include "util.h"
#include "utilmoneystr.h"
#include "utiltime.h"
#include "wallet/wallet.h"
#include "validation.h"
#include "validationinterface.h"

#include <stdint.h>
#include <boost/thread.hpp>

//////////////////////////////////////////////////////////////////////////////
//
// Internal stake minter
//
bool ProcessStakeBlockFound(CBlock* pblock, CWallet& wallet, CReserveKey& reservekey)
{
    LogPrintf("%s: Proof-of-Stake block found: %s\n", __func__, pblock->ToString());
    LogPrintf("%s: Generated %s\n", __func__, FormatMoney(pblock->vtx[0].vout[0].nValue));

    // Found a solution
    {
        LOCK(cs_main);
        if (pblock->hashPrevBlock != chainActive.Tip()->GetBlockHash())
            return error("Dynamic Staker : generated block is stale");
    }

    // Remove key from key pool
    reservekey.KeepKey();

    // Track how many getdata requests this block gets
    {
        LOCK(wallet.cs_wallet);
        wallet.mapRequestCount[pblock->GetHash()] = 0;
    }

    // Inform about the new block
    GetMainSignals().BlockFound(pblock->GetHash());

    // Process this block the same as if we had received it from another node
    CValidationState state;
    if (!ProcessNewBlock(state, NULL, pblock)) {
        return error("PIVXMiner : ProcessNewBlock, block not accepted");
    }

    for (CNode* node : vNodes) {
        node->PushInventory(CInv(MSG_BLOCK, pblock->GetHash()));
    }

    return true;
}

void DynamicStakeMinter(CWallet* pwallet)
{
    LogPrintf("Dynamic stake minter started\n");
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread("dyn-minter");
    bool fProofOfStake = true;
    // Each thread has its own key and counter
    CReserveKey reservekey(pwallet);
    unsigned int nExtraNonce = 0;
    bool fLastLoopOrphan = false;
    while (fProofOfStake) {
        //control the amount of times the client will check for mintable coins
        if ((GetTime() - nMintableLastCheck > 5 * 60)) // 5 minute check time
        {
            nMintableLastCheck = GetTime();
            fMintableCoins = pwallet->MintableCoins();
        }

        while (vNodes.empty() || pwallet->IsLocked() || !fMintableCoins ||
               (pwallet->GetBalance() > 0 && nReserveBalance >= pwallet->GetBalance()) || 
               !dynodeSync.IsSynced() || !sporkManager.IsSporkActive(SPORK_31_PROOF_OF_STAKE_ENABLED )) 
       	{
            nLastCoinStakeSearchInterval = 0;
            MilliSleep(5000);
            // Do a separate 1 minute check here to ensure fMintableCoins is updated
            if (!fMintableCoins && (GetTime() - nMintableLastCheck > 1 * 60)) // 1 minute check time
            {
                nMintableLastCheck = GetTime();
                fMintableCoins = pwallet->MintableCoins();
            }
        }

        //search our map of hashed blocks, see if bestblock has been hashed yet
        if (mapHashedBlocks.count(chainActive.Tip()->nHeight) && !fLastLoopOrphan)
        {
            // wait half of the nHashDrift with max wait of 3 minutes
            if (GetTime() - mapHashedBlocks[chainActive.Tip()->nHeight] < std::max(pwallet->nHashInterval, (unsigned int)1))
            {
                MilliSleep(5000);
                continue;
            }
        }
        //
        // Create new block
        //
        unsigned int nTransactionsUpdatedLast = mempool.GetTransactionsUpdated();
        CBlockIndex* pindexPrev = chainActive.Tip();
        if (!pindexPrev)
            continue;

        std::unique_ptr<CBlockTemplate> pblocktemplate(CreateNewBlock(CScript(), pwallet, fProofOfStake));
        if (!pblocktemplate.get())
            continue;

        CBlock* pblock = &pblocktemplate->block;
        IncrementExtraNonce(pblock, pindexPrev, nExtraNonce);

        //Stake miner main
        LogPrintf("%s : proof-of-stake block found %s \n", __func__, pblock->GetHash().ToString().c_str());
        if (!SignBlock(*pblock, *pwallet)) {
            LogPrintf("%s: Signing new block with UTXO key failed \n", __func__);
            continue;
        }

        LogPrintf("%s : proof-of-stake block was signed %s \n",  __func__, pblock->GetHash().ToString().c_str());
        SetThreadPriority(THREAD_PRIORITY_NORMAL);
        if (!ProcessStakeBlockFound(pblock, *pwallet, reservekey)) {
            fLastLoopOrphan = true;
            continue;
        }
        SetThreadPriority(THREAD_PRIORITY_LOWEST);

        // Check for stop or if block needs to be rebuilt
        boost::this_thread::interruption_point();

        continue;
    }
}

// ppcoin: stake minter thread
void ThreadStakeMinter()
{
    boost::this_thread::interruption_point();
    LogPrintf("ThreadStakeMinter started\n");
    CWallet* pwallet = pwalletMain;
    try {
        DynamicStakeMinter(pwallet);
        boost::this_thread::interruption_point();
    } catch (std::exception& e) {
        LogPrintf("ThreadStakeMinter() exception \n");
    } catch (...) {
        LogPrintf("ThreadStakeMinter() error \n");
    }
    LogPrintf("ThreadStakeMinter exiting,\n");
}

#endif // ENABLE_WALLET