// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "directory.h"

#include "fluid.h"
#include "validation.h"

CDirectoryDB *pDirectoryDB = NULL;

bool IsDirectoryTransaction(CScript txOut) {
    return (txOut.IsDirectoryScript(DIRECTORY_NEW_TX)
            || txOut.IsDirectoryScript(DIRECTORY_UPDATE_TX)
            || txOut.IsDirectoryScript(DIRECTORY_DELETE_TX)
            || txOut.IsDirectoryScript(DIRECTORY_ACTIVATE_TX)
           );
}

bool GetDirectoryTransaction(int nHeight, const uint256 &hash, CTransaction &txOut, const Consensus::Params& consensusParams)
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
std::vector<std::pair<std::string, std::vector<unsigned char>>> CIdentityParameters::InitialiseAdminOwners()
{


}
*/