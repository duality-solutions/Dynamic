// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2019 The PIVX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "pos/stakeinput.h"

#include "chain.h"
#include "chainparams.h"
#include "pos/kernel.h"
#include "validation.h"
#include "wallet/wallet.h"

typedef std::vector<unsigned char> valtype;

//!Dynamic Stake
bool CDynamicStake::SetInput(const CTransaction& txPrev, unsigned int n)
{
    this->txFrom = txPrev;
    this->nPosition = n;
    return true;
}

bool CDynamicStake::GetTxFrom(CTransaction& tx)
{
    tx = txFrom;
    return true;
}

bool CDynamicStake::CreateTxIn(CWallet* pwallet, CTxIn& txIn, uint256 hashTxOut)
{
    txIn = CTxIn(txFrom.GetHash(), nPosition);
    return true;
}

CAmount CDynamicStake::GetValue()
{
    return txFrom.vout[nPosition].nValue;
}

bool CDynamicStake::CreateTxOuts(CWallet* pwallet, std::vector<CTxOut>& vout, CAmount nTotal)
{
    std::vector<valtype> vSolutions;
    txnouttype whichType;
    CScript scriptPubKeyKernel = txFrom.vout[nPosition].scriptPubKey;
    if (!Solver(scriptPubKeyKernel, whichType, vSolutions)) {
        LogPrintf("CreateCoinStake : failed to parse kernel\n");
        return false;
    }

    if (whichType != TX_PUBKEY && whichType != TX_PUBKEYHASH)
        return false; // only support pay to public key and pay to address

    CScript scriptPubKey;
    if (whichType == TX_PUBKEYHASH) // pay to address type
    {
        //convert to pay to public key type
        CKey key;
        CKeyID keyID = CKeyID(uint160(vSolutions[0]));
        if (!pwallet->GetKey(keyID, key))
            return false;

        scriptPubKey << key.GetPubKey() << OP_CHECKSIG;
    } else
        scriptPubKey = scriptPubKeyKernel;

    vout.emplace_back(CTxOut(0, scriptPubKey));

    // Calculate if we need to split the output
    if (nTotal / 2 > (CAmount)(pwallet->nStakeSplitThreshold * COIN))
        vout.emplace_back(CTxOut(0, scriptPubKey));

    return true;
}

CDataStream CDynamicStake::GetUniqueness()
{
    //The unique identifier for a DYN stake is the outpoint
    CDataStream ss(SER_NETWORK, 0);
    ss << nPosition << txFrom.GetHash();
    return ss;
}

//The block that the UTXO was added to the chain
CBlockIndex* CDynamicStake::GetIndexFrom()
{
    if (pindexFrom)
        return pindexFrom;
    uint256 hashBlock = uint256Zero;
    CTransactionRef ptx;
    if (GetTransaction(txFrom.GetHash(), ptx, Params().GetConsensus(), hashBlock, true)) {
        // If the index is in the chain, then set it as the "index from"
        if (mapBlockIndex.count(hashBlock)) {
            CBlockIndex* pindex = mapBlockIndex.at(hashBlock);
            if (chainActive.Contains(pindex))
                pindexFrom = pindex;
        }
    } else {
        LogPrintf("%s : failed to find tx %s\n", __func__, txFrom.GetHash().GetHex());
    }

    return pindexFrom;
}
