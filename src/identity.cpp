// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "identity.h"

#include "fluid.h"
#include "validation.h"

CIdentityDB *pIdentityDB = NULL;

bool IsIdentityTransaction(CScript txOut) {
    return (txOut.IsIdentityScript(IDENTITY_NEW_TX)
            || txOut.IsIdentityScript(IDENTITY_UPDATE_TX)
            || txOut.IsIdentityScript(IDENTITY_DELETE_TX)
            || txOut.IsIdentityScript(IDENTITY_ACTIVATE_TX)
           );
}

bool GetIdentityTransaction(int nHeight, const uint256 &hash, CTransaction &txOut, const Consensus::Params& consensusParams)
{
    if(nHeight < 0 || nHeight > chainActive.Height())
        return false;
    CBlockIndex *pindexSlow = NULL; 
    LOCK(cs_main);
    pindexSlow = chainActive[nHeight];
    if (pindexSlow) {
        CBlock block;
        if (ReadBlockFromDisk(block, pindexSlow, consensusParams)) {
            BOOST_FOREACH(const CTransaction &tx, block.vtx) {
                if (tx.GetHash() == hash) {
                    txOut = tx;
                    return true;
                }
            }
        }
    }
    return false;
}

/*
std::vector<std::pair<std::string, std::vector<unsigned char>>> CIdentityParameters::InitialiseCoreIdentities()
{


}
*/