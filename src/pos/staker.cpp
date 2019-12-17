// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin developers
// Copyright (c) 2014-2019 The Dash developers
// Copyright (c) 2015-2019 The PIVX developers
// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.


#include "staker.h"

#include "chain.h"
#include "chainparams.h"
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
bool ProcessStakeBlockFound(const std::shared_ptr<const CBlock> pblock, CWallet& wallet, CReserveKey& reservekey)
{
    LogPrintf("%s: Proof-of-Stake block found, vchBlockSig size %d:\n%s\n", __func__, pblock->vchBlockSig.size(), pblock->ToString());

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

    bool fNewBlock = false;
    // Process this block the same as if we had received it from another node
    if (!ProcessNewBlock(Params(), pblock, true, &fNewBlock))
        return error("Dynamic Staker: ProcessStakeBlockFound, block not accepted");

    CConnman& connman = *g_connman;
    connman.ForEachNode([pblock](CNode* pnode) {
        if (pnode->nVersion != 0)
        {
        	pnode->PushInventory(CInv(MSG_BLOCK, pblock->GetHash()));
        }
    });

    return true;
}

bool fMintableCoins = false;
int nMintableLastCheck = 0;

void DynamicStakeMinter(CWallet* pwallet)
{
    LogPrintf("%s: Dynamic stake minter started\n", __func__);
    SetThreadPriority(THREAD_PRIORITY_LOWEST);
    RenameThread("dyn-minter");
    bool fProofOfStake = true;
    // Each thread has its own key and counter
    CReserveKey reservekey(pwallet);
    unsigned int nExtraNonce = 0;
    bool fLastLoopOrphan = false;
    CConnman& connman = *g_connman;
    while (fProofOfStake) {
        //control the amount of times the client will check for mintable coins
        if ((GetTime() - nMintableLastCheck > 5 * 60)) // 5 minute check time
        {
            nMintableLastCheck = GetTime();
            fMintableCoins = pwallet->MintableCoins();
        }
        while (connman.GetNodeCount(CConnman::CONNECTIONS_ALL) == 0 || pwallet->IsLocked() || !fMintableCoins ||
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
        LogPrintf("%s: Dynamic stake minter initialized.\n", __func__);
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
        CBlockIndex* pindexPrev = chainActive.Tip();
        if (!pindexPrev)
            continue;

        CScript scriptStake;
        std::unique_ptr<CBlockTemplate> pblocktemplate(CreateNewBlock(Params(), &scriptStake, pwallet, fProofOfStake));
        if (!pblocktemplate.get())
            continue;

        IncrementExtraNonce(pblocktemplate->block, pindexPrev, nExtraNonce);

        //Stake miner main
        if (!SignBlock(pblocktemplate->block, *pwallet)) {
            LogPrintf("%s: Signing new block with UTXO key failed \n", __func__);
            continue;
        }

        SetThreadPriority(THREAD_PRIORITY_NORMAL);
        const std::shared_ptr<const CBlock> pblock = std::make_shared<const CBlock>(pblocktemplate->block);
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
        LogPrintf("ThreadStakeMinter() exception: %s \n", e.what());
    } catch (...) {
        LogPrintf("ThreadStakeMinter() error \n");
    }
    LogPrintf("ThreadStakeMinter exiting,\n");
}