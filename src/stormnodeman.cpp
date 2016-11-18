// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "stormnodeman.h"
#include "activestormnode.h"
#include "sandstorm.h"
#include "governance.h"
#include "stormnode.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"
#include "netfulfilledman.h"
#include "util.h"
#include "addrman.h"
#include "spork.h"
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

/** Stormnode manager */
CStormnodeMan snodeman;

const std::string CStormnodeMan::SERIALIZATION_VERSION_STRING = "CStormnodeMan-Version-1";

struct CompareLastPaidBlock
{
    bool operator()(const std::pair<int, CStormnode*>& t1,
                    const std::pair<int, CStormnode*>& t2) const
    {
        return (t1.first != t2.first) ? (t1.first < t2.first) : (t1.second->vin < t2.second->vin);
    }
};

struct CompareScoreSN
{
    bool operator()(const std::pair<int64_t, CStormnode*>& t1,
                    const std::pair<int64_t, CStormnode*>& t2) const
    {
        return (t1.first != t2.first) ? (t1.first < t2.first) : (t1.second->vin < t2.second->vin);
    }
};

CStormnodeIndex::CStormnodeIndex()
    : nSize(0),
      mapIndex(),
      mapReverseIndex()
{}

bool CStormnodeIndex::Get(int nIndex, CTxIn& vinStormnode) const
{
    rindex_m_cit it = mapReverseIndex.find(nIndex);
    if(it == mapReverseIndex.end()) {
        return false;
    }
    vinStormnode = it->second;
    return true;
}

int CStormnodeIndex::GetStormnodeIndex(const CTxIn& vinStormnode) const
{
    index_m_cit it = mapIndex.find(vinStormnode);
    if(it == mapIndex.end()) {
        return -1;
    }
    return it->second;
}

void CStormnodeIndex::AddStormnodeVIN(const CTxIn& vinStormnode)
{
    index_m_it it = mapIndex.find(vinStormnode);
    if(it != mapIndex.end()) {
        return;
    }
    int nNextIndex = nSize;
    mapIndex[vinStormnode] = nNextIndex;
    mapReverseIndex[nNextIndex] = vinStormnode;
    ++nSize;
}

void CStormnodeIndex::Clear()
{
    mapIndex.clear();
    mapReverseIndex.clear();
    nSize = 0;
}
struct CompareByAddr

{
    bool operator()(const CStormnode* t1,
                    const CStormnode* t2) const
    {
        return t1->addr < t2->addr;
    }
};

void CStormnodeIndex::RebuildIndex()
{
    nSize = mapIndex.size();
    for(index_m_it it = mapIndex.begin(); it != mapIndex.end(); ++it) {
        mapReverseIndex[it->second] = it->first;
    }
}

CStormnodeMan::CStormnodeMan()
: cs(),
  vStormnodes(),
  mAskedUsForStormnodeList(),
  mWeAskedForStormnodeList(),
  mWeAskedForStormnodeListEntry(),
  nLastIndexRebuildTime(0),
  indexStormnodes(),
  indexStormnodesOld(),
  fIndexRebuilt(false),
  fStormnodesAdded(false),
  fStormnodesRemoved(false),
  vecDirtyGovernanceObjectHashes(),
  nLastWatchdogVoteTime(0),
  mapSeenStormnodeBroadcast(),
  mapSeenStormnodePing(),
  nSsqCount(0)
{}

bool CStormnodeMan::Add(CStormnode &sn)
{
    LOCK(cs);

    if (!sn.IsEnabled() && !sn.IsPreEnabled())
        return false;

    CStormnode *psn = Find(sn.vin);
    if (psn == NULL) {
        LogPrint("stormnode", "CStormnodeMan::Add -- Adding new Stormnode: addr=%s, %i now\n", sn.addr.ToString(), size() + 1);
        sn.nTimeLastWatchdogVote = sn.sigTime;
        vStormnodes.push_back(sn);
        indexStormnodes.AddStormnodeVIN(sn.vin);
        fStormnodesAdded = true;
        return true;
    }

    return false;
}

void CStormnodeMan::AskForSN(CNode* pnode, const CTxIn &vin)
{
    if(!pnode) return;

    std::map<COutPoint, int64_t>::iterator it = mWeAskedForStormnodeListEntry.find(vin.prevout);
    if (it != mWeAskedForStormnodeListEntry.end() && GetTime() < (*it).second) {
        // we've asked recently, should not repeat too often or we could get banned
        return;
    }

    // ask for the snb info once from the node that sent snp

    LogPrintf("CStormnodeMan::AskForSN -- Asking node for missing stormnode entry: %s\n", vin.prevout.ToStringShort());
    pnode->PushMessage(NetMsgType::SSEG, vin);
    mWeAskedForStormnodeListEntry[vin.prevout] = GetTime() + SSEG_UPDATE_SECONDS;;
}

void CStormnodeMan::Check()
{
    LOCK(cs);

    LogPrint("stormnode", "CStormnodeMan::Check nLastWatchdogVoteTime = %d, IsWatchdogActive() = %d\n", nLastWatchdogVoteTime, IsWatchdogActive());

    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        sn.Check();
    }
}

void CStormnodeMan::CheckAndRemove()
{
    LogPrintf("CStormnodeMan::CheckAndRemove\n");

    Check();

    {
        LOCK(cs);

        // Remove inactive and outdated stormnodes
        std::vector<CStormnode>::iterator it = vStormnodes.begin();
        while(it != vStormnodes.end()) {
            bool fRemove =  // If it's marked to be removed from the list by CStormnode::Check for whatever reason ...
                    (*it).nActiveState == CStormnode::STORMNODE_REMOVE ||
                    // or collateral was spent ...
                    (*it).nActiveState == CStormnode::STORMNODE_OUTPOINT_SPENT;

            if (fRemove) {
                LogPrint("stormnode", "CStormnodeMan::CheckAndRemove -- Removing Stormnode: %s  addr=%s  %i now\n", (*it).GetStatus(), (*it).addr.ToString(), size() - 1);

                // erase all of the broadcasts we've seen from this txin, ...
                mapSeenStormnodeBroadcast.erase(CStormnodeBroadcast(*it).GetHash());
                // allow us to ask for this stormnode again if we see another ping ...
                mWeAskedForStormnodeListEntry.erase((*it).vin.prevout);

                // and finally remove it from the list
                it = vStormnodes.erase(it);
                fStormnodesRemoved = true;
            } else {
                ++it;
            }
        }

        // check who's asked for the Stormnode list
        std::map<CNetAddr, int64_t>::iterator it1 = mAskedUsForStormnodeList.begin();
        while(it1 != mAskedUsForStormnodeList.end()){
            if((*it1).second < GetTime()) {
                mAskedUsForStormnodeList.erase(it1++);
            } else {
                ++it1;
            }
        }

        // check who we asked for the Stormnode list
        it1 = mWeAskedForStormnodeList.begin();
        while(it1 != mWeAskedForStormnodeList.end()){
            if((*it1).second < GetTime()){
                mWeAskedForStormnodeList.erase(it1++);
            } else {
                ++it1;
            }
        }

        // check which Stormnodes we've asked for
        std::map<COutPoint, int64_t>::iterator it2 = mWeAskedForStormnodeListEntry.begin();
        while(it2 != mWeAskedForStormnodeListEntry.end()){
            if((*it2).second < GetTime()){
                mWeAskedForStormnodeListEntry.erase(it2++);
            } else {
                ++it2;
            }
        }

        std::map<CNetAddr, CStormnodeVerification>::iterator itv1 = mWeAskedForVerification.begin();
        while(itv1 != mWeAskedForVerification.end()){
            if(itv1->second.nBlockHeight < pCurrentBlockIndex->nHeight - MAX_POSE_BLOCKS) {
                mWeAskedForVerification.erase(itv1++);
            } else {
                ++itv1;
            }
        }

        // remove expired mapSeenStormnodeBroadcast
        std::map<uint256, CStormnodeBroadcast>::iterator it3 = mapSeenStormnodeBroadcast.begin();
        while(it3 != mapSeenStormnodeBroadcast.end()){
            if((*it3).second.lastPing.sigTime < GetTime() - STORMNODE_REMOVAL_SECONDS*2){
                LogPrint("stormnode", "CStormnodeMan::CheckAndRemove -- Removing expired Stormnode broadcast: hash=%s\n", (*it3).second.GetHash().ToString());
                mapSeenStormnodeBroadcast.erase(it3++);
            } else {
                ++it3;
            }
        }

        // remove expired mapSeenStormnodePing
        std::map<uint256, CStormnodePing>::iterator it4 = mapSeenStormnodePing.begin();
        while(it4 != mapSeenStormnodePing.end()){
            if((*it4).second.sigTime < GetTime() - STORMNODE_REMOVAL_SECONDS*2){
                LogPrint("stormnode", "CStormnodeMan::CheckAndRemove -- Removing expired Stormnode ping: hash=%s\n", (*it4).second.GetHash().ToString());
                mapSeenStormnodePing.erase(it4++);
            } else {
                ++it4;
            }
        }

        // remove expired mapSeenStormnodeVerification
        std::map<uint256, CStormnodeVerification>::iterator itv2 = mapSeenStormnodeVerification.begin();
        while(itv2 != mapSeenStormnodeVerification.end()){
            if((*itv2).second.nBlockHeight < pCurrentBlockIndex->nHeight - MAX_POSE_BLOCKS){
                LogPrint("stormnode", "CStormnodeMan::CheckAndRemove -- Removing expired Stormnode verification: hash=%s\n", (*itv2).first.ToString());
                mapSeenStormnodeVerification.erase(itv2++);
            } else {
                ++itv2;
            }
        }

        LogPrintf("CStormnodeMan::CheckAndRemove -- %s\n", ToString());

        if(fStormnodesRemoved) {
            CheckAndRebuildStormnodeIndex();
        }
    }

    if(fStormnodesRemoved) {
        NotifyStormnodeUpdates();
    }
}

void CStormnodeMan::Clear()
{
    LOCK(cs);
    vStormnodes.clear();
    mAskedUsForStormnodeList.clear();
    mWeAskedForStormnodeList.clear();
    mWeAskedForStormnodeListEntry.clear();
    mapSeenStormnodeBroadcast.clear();
    mapSeenStormnodePing.clear();
    nSsqCount = 0;
    nLastWatchdogVoteTime = 0;
    indexStormnodes.Clear();
    indexStormnodesOld.Clear();
}

int CStormnodeMan::CountStormnodes(int nProtocolVersion)
{
    LOCK(cs);
    int nCount = 0;
    nProtocolVersion = nProtocolVersion == -1 ? snpayments.GetMinStormnodePaymentsProto() : nProtocolVersion;

    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        if(sn.nProtocolVersion < nProtocolVersion) continue;
        nCount++;
    }

    return nCount;
}

int CStormnodeMan::CountEnabled(int nProtocolVersion)
{
    LOCK(cs);
    int nCount = 0;
    nProtocolVersion = nProtocolVersion == -1 ? snpayments.GetMinStormnodePaymentsProto() : nProtocolVersion;

    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        sn.Check();
        if(sn.nProtocolVersion < nProtocolVersion || !sn.IsEnabled()) continue;
        nCount++;
    }

    return nCount;
}

/* Only IPv4 stormnodes are allowed in 12.1, saving this for later
int CStormnodeMan::CountByIP(int nNetworkType)
{
    LOCK(cs);
    int nNodeCount = 0;

    BOOST_FOREACH(CStormnode& sn, vStormnodes)
        if ((nNetworkType == NET_IPV4 && sn.addr.IsIPv4()) ||
            (nNetworkType == NET_TOR  && sn.addr.IsTor())  ||
            (nNetworkType == NET_IPV6 && sn.addr.IsIPv6())) {
                nNodeCount++;
        }

    return nNodeCount;
}
*/

void CStormnodeMan::SsegUpdate(CNode* pnode)
{
    LOCK(cs);

    if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if(!(pnode->addr.IsRFC1918() || pnode->addr.IsLocal())) {
            std::map<CNetAddr, int64_t>::iterator it = mWeAskedForStormnodeList.find(pnode->addr);
            if(it != mWeAskedForStormnodeList.end() && GetTime() < (*it).second) {
                LogPrintf("CStormnodeMan::SsegUpdate -- we already asked %s for the list; skipping...\n", pnode->addr.ToString());
                return;
            }
        }
    }
    
    pnode->PushMessage(NetMsgType::SSEG, CTxIn());
    int64_t askAgain = GetTime() + SSEG_UPDATE_SECONDS;
    mWeAskedForStormnodeList[pnode->addr] = askAgain;

    LogPrint("stormnode", "CStormnodeMan::SsegUpdate -- asked %s for the list\n", pnode->addr.ToString());
}

CStormnode* CStormnodeMan::Find(const CScript &payee)
{
    LOCK(cs);

    BOOST_FOREACH(CStormnode& sn, vStormnodes)
    {
        if(GetScriptForDestination(sn.pubKeyCollateralAddress.GetID()) == payee)
            return &sn;
    }
    return NULL;
}

CStormnode* CStormnodeMan::Find(const CTxIn &vin)
{
    LOCK(cs);

    BOOST_FOREACH(CStormnode& sn, vStormnodes)
    {
        if(sn.vin.prevout == vin.prevout)
            return &sn;
    }
    return NULL;
}

CStormnode* CStormnodeMan::Find(const CPubKey &pubKeyStormnode)
{
    LOCK(cs);

    BOOST_FOREACH(CStormnode& sn, vStormnodes)
    {
        if(sn.pubKeyStormnode == pubKeyStormnode)
            return &sn;
    }
    return NULL;
}

bool CStormnodeMan::Get(const CPubKey& pubKeyStormnode, CStormnode& stormnode)
{
    // Theses mutexes are recursive so double locking by the same thread is safe.
    LOCK(cs);
    CStormnode* pSN = Find(pubKeyStormnode);
    if(!pSN)  {
        return false;
    }
    stormnode = *pSN;
    return true;
}

bool CStormnodeMan::Get(const CTxIn& vin, CStormnode& stormnode)
{
    // Theses mutexes are recursive so double locking by the same thread is safe.
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN)  {
        return false;
    }
    stormnode = *pSN;
    return true;
}

stormnode_info_t CStormnodeMan::GetStormnodeInfo(const CTxIn& vin)
{
    stormnode_info_t info;
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN)  {
        return info;
    }
    info = pSN->GetInfo();
    return info;
}

stormnode_info_t CStormnodeMan::GetStormnodeInfo(const CPubKey& pubKeyStormnode)
{
    stormnode_info_t info;
    LOCK(cs);
    CStormnode* pSN = Find(pubKeyStormnode);
    if(!pSN)  {
        return info;
    }
    info = pSN->GetInfo();
    return info;
}

bool CStormnodeMan::Has(const CTxIn& vin)
{
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    return (pSN != NULL);
}

//
// Deterministically select the oldest/best stormnode to pay on the network
//
CStormnode* CStormnodeMan::GetNextStormnodeInQueueForPayment(int nBlockHeight, bool fFilterSigTime, int& nCount)
{
    // Need LOCK2 here to ensure consistent locking order because the GetBlockHash call below locks cs_main
    LOCK2(cs_main,cs);

    CStormnode *pBestStormnode = NULL;
    std::vector<std::pair<int, CStormnode*> > vecStormnodeLastPaid;

    /*
        Make a vector with all of the last paid times
    */

    int nSnCount = CountEnabled();
    BOOST_FOREACH(CStormnode &sn, vStormnodes)
    {
        sn.Check();
        if(!sn.IsEnabled()) continue;

        // //check protocol version
        if(sn.nProtocolVersion < snpayments.GetMinStormnodePaymentsProto()) continue;

        //it's in the list (up to 8 entries ahead of current block to allow propagation) -- so let's skip it
        if(snpayments.IsScheduled(sn, nBlockHeight)) continue;

        //it's too new, wait for a cycle
        if(fFilterSigTime && sn.sigTime + (nSnCount*2.6*60) > GetAdjustedTime()) continue;

        //make sure it has at least as many confirmations as there are stormnodes
        if(sn.GetCollateralAge() < nSnCount) continue;

        vecStormnodeLastPaid.push_back(std::make_pair(sn.GetLastPaidBlock(), &sn));
    }

    nCount = (int)vecStormnodeLastPaid.size();

    //when the network is in the process of upgrading, don't penalize nodes that recently restarted
    if(fFilterSigTime && nCount < nSnCount/3) return GetNextStormnodeInQueueForPayment(nBlockHeight, false, nCount);

    // Sort them low to high
    sort(vecStormnodeLastPaid.begin(), vecStormnodeLastPaid.end(), CompareLastPaidBlock());

    uint256 blockHash;
    if(!GetBlockHash(blockHash, nBlockHeight - 101)) {
        LogPrintf("CStormnode::GetNextStormnodeInQueueForPayment -- ERROR: GetBlockHash() failed at nBlockHeight %d\n", nBlockHeight - 101);
        return NULL;
    }
    // Look at 1/10 of the oldest nodes (by last payment), calculate their scores and pay the best one
    //  -- This doesn't look at who is being paid in the +8-10 blocks, allowing for double payments very rarely
    //  -- 1/100 payments should be a double payment on mainnet - (1/(3000/10))*2
    //  -- (chance per block * chances before IsScheduled will fire)
    int nTenthNetwork = CountEnabled()/10;
    int nCountTenth = 0;
    arith_uint256 nHighest = 0;
    BOOST_FOREACH (PAIRTYPE(int, CStormnode*)& s, vecStormnodeLastPaid){
        arith_uint256 nScore = s.second->CalculateScore(blockHash);
        if(nScore > nHighest){
            nHighest = nScore;
            pBestStormnode = s.second;
        }
        nCountTenth++;
        if(nCountTenth >= nTenthNetwork) break;
    }
    return pBestStormnode;
}

CStormnode* CStormnodeMan::FindRandomNotInVec(const std::vector<CTxIn> &vecToExclude, int nProtocolVersion)
{
    LOCK(cs);

    nProtocolVersion = nProtocolVersion == -1 ? snpayments.GetMinStormnodePaymentsProto() : nProtocolVersion;

    int nCountEnabled = CountEnabled(nProtocolVersion);
    int nCountNotExcluded = nCountEnabled - vecToExclude.size();

    LogPrintf("CStormnodeMan::FindRandomNotInVec -- %d enabled stormnodes, %d stormnodes to choose from\n", nCountEnabled, nCountNotExcluded);
    if(nCountNotExcluded < 1) return NULL;

    // fill a vector of pointers
    std::vector<CStormnode*> vpStormnodesShuffled;
    BOOST_FOREACH(CStormnode &sn, vStormnodes) {
        vpStormnodesShuffled.push_back(&sn);
    }

    // shuffle pointers
    std::random_shuffle(vpStormnodesShuffled.begin(), vpStormnodesShuffled.end(), GetInsecureRand);
    bool fExclude;

    // loop through
    BOOST_FOREACH(CStormnode* psn, vpStormnodesShuffled) {
        if(psn->nProtocolVersion < nProtocolVersion || !psn->IsEnabled()) continue;
        fExclude = false;
        BOOST_FOREACH(const CTxIn &txinToExclude, vecToExclude) {
            if(psn->vin.prevout == txinToExclude.prevout) {
                fExclude = true;
                break;
            }
        }
        if(fExclude) continue;
        // found the one not in vecToExclude
        LogPrint("stormnode", "CStormnodeMan::FindRandomNotInVec -- found, stormnode=%s\n", psn->vin.prevout.ToStringShort());
        return psn;
    }

    LogPrint("stormnode", "CStormnodeMan::FindRandomNotInVec -- failed\n");
    return NULL;
}

int CStormnodeMan::GetStormnodeRank(const CTxIn& vin, int nBlockHeight, int nMinProtocol, bool fOnlyActive)
{
    std::vector<std::pair<int64_t, CStormnode*> > vecStormnodeScores;

    //make sure we know about this block
    uint256 blockHash = uint256();
    if(!GetBlockHash(blockHash, nBlockHeight)) return -1;

    LOCK(cs);

    // scan for winner
    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        if(sn.nProtocolVersion < nMinProtocol) continue;
        if(fOnlyActive) {
            sn.Check();
            if(!sn.IsEnabled()) continue;
        }
        int64_t nScore = sn.CalculateScore(blockHash).GetCompact(false);

        vecStormnodeScores.push_back(std::make_pair(nScore, &sn));
    }

    sort(vecStormnodeScores.rbegin(), vecStormnodeScores.rend(), CompareScoreSN());

    int nRank = 0;
    BOOST_FOREACH (PAIRTYPE(int64_t, CStormnode*)& scorePair, vecStormnodeScores) {
        nRank++;
        if(scorePair.second->vin.prevout == vin.prevout) return nRank;
    }

    return -1;
}

std::vector<std::pair<int, CStormnode> > CStormnodeMan::GetStormnodeRanks(int nBlockHeight, int nMinProtocol)
{
    std::vector<std::pair<int64_t, CStormnode*> > vecStormnodeScores;
    std::vector<std::pair<int, CStormnode> > vecStormnodeRanks;

    //make sure we know about this block
    uint256 blockHash = uint256();
    if(!GetBlockHash(blockHash, nBlockHeight)) return vecStormnodeRanks;

    LOCK(cs);

    // scan for winner
    BOOST_FOREACH(CStormnode& sn, vStormnodes) {

        sn.Check();

        if(sn.nProtocolVersion < nMinProtocol || !sn.IsEnabled()) continue;

        int64_t nScore = sn.CalculateScore(blockHash).GetCompact(false);

        vecStormnodeScores.push_back(std::make_pair(nScore, &sn));
    }

    sort(vecStormnodeScores.rbegin(), vecStormnodeScores.rend(), CompareScoreSN());

    int nRank = 0;
    BOOST_FOREACH (PAIRTYPE(int64_t, CStormnode*)& s, vecStormnodeScores) {
        nRank++;
        vecStormnodeRanks.push_back(std::make_pair(nRank, *s.second));
    }

    return vecStormnodeRanks;
}

CStormnode* CStormnodeMan::GetStormnodeByRank(int nRank, int nBlockHeight, int nMinProtocol, bool fOnlyActive)
{
    std::vector<std::pair<int64_t, CStormnode*> > vecStormnodeScores;

    LOCK(cs);

    uint256 blockHash;
    if(!GetBlockHash(blockHash, nBlockHeight)) {
        LogPrintf("CStormnode::GetStormnodeByRank -- ERROR: GetBlockHash() failed at nBlockHeight %d\n", nBlockHeight);
        return NULL;
    }

    // Fill scores
    BOOST_FOREACH(CStormnode& sn, vStormnodes) {

        if(sn.nProtocolVersion < nMinProtocol) continue;
        if(fOnlyActive) {
            sn.Check();
            if(!sn.IsEnabled()) continue;
        }

        int64_t nScore = sn.CalculateScore(blockHash).GetCompact(false);

        vecStormnodeScores.push_back(std::make_pair(nScore, &sn));
    }

    sort(vecStormnodeScores.rbegin(), vecStormnodeScores.rend(), CompareScoreSN());

    int rank = 0;
    BOOST_FOREACH (PAIRTYPE(int64_t, CStormnode*)& s, vecStormnodeScores){
        rank++;
        if(rank == nRank) {
            return s.second;
        }
    }

    return NULL;
}

void CStormnodeMan::ProcessStormnodeConnections()
{
    //we don't care about this for regtest
    if(Params().NetworkIDString() == CBaseChainParams::REGTEST) return;

    LOCK(cs_vNodes);
    BOOST_FOREACH(CNode* pnode, vNodes) {
        if(pnode->fStormnode) {
            if(sandStormPool.pSubmittedToStormnode != NULL && pnode->addr == sandStormPool.pSubmittedToStormnode->addr) continue;
            LogPrintf("Closing Stormnode connection: peer=%d, addr=%s\n", pnode->id, pnode->addr.ToString());
            pnode->fDisconnect = true;
        }
    }
}

void CStormnodeMan::ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv)
{
    if(fLiteMode) return; // disable all Dash specific functionality
    if(!stormnodeSync.IsBlockchainSynced()) return;

    if (strCommand == NetMsgType::SNANNOUNCE) { //Stormnode Broadcast

        {
            LOCK(cs);

            CStormnodeBroadcast snb;
            vRecv >> snb;

            int nDos = 0;

            if (CheckSnbAndUpdateStormnodeList(snb, nDos)) {
                // use announced Stormnode as a peer
                addrman.Add(CAddress(snb.addr), pfrom->addr, 2*60*60);
            } else if(nDos > 0) {
                Misbehaving(pfrom->GetId(), nDos);
            }
        }
        if(fStormnodesAdded) {
            NotifyStormnodeUpdates();
        }
    } else if (strCommand == NetMsgType::SNPING) { //Stormnode Ping
        // ignore stormnode pings until stormnode list is synced
        if (!stormnodeSync.IsStormnodeListSynced()) return;

        CStormnodePing snp;
        vRecv >> snp;

        LogPrint("stormnode", "SNPING -- Stormnode ping, stormnode=%s\n", snp.vin.prevout.ToStringShort());

        LOCK(cs);

        if(mapSeenStormnodePing.count(snp.GetHash())) return; //seen
        mapSeenStormnodePing.insert(std::make_pair(snp.GetHash(), snp));

        LogPrint("stormnode", "SNPING -- Stormnode ping, stormnode=%s new\n", snp.vin.prevout.ToStringShort());

        int nDos = 0;
        if(snp.CheckAndUpdate(nDos, false)) return;

        if(nDos > 0) {
            // if anything significant failed, mark that node
            Misbehaving(pfrom->GetId(), nDos);
        } else {
            // if nothing significant failed, search existing Stormnode list
            CStormnode* psn = Find(snp.vin);
            // if it's known, don't ask for the snb, just return
            if(psn != NULL) return;
        }

        // something significant is broken or sn is unknown,
        // we might have to ask for a stormnode entry once
        AskForSN(pfrom, snp.vin);

    } else if (strCommand == NetMsgType::SSEG) { //Get Stormnode list or specific entry
        // Ignore such requests until we are fully synced.
        // We could start processing this after stormnode list is synced
        // but this is a heavy one so it's better to finish sync first.
        if (!stormnodeSync.IsSynced()) return;

        CTxIn vin;
        vRecv >> vin;

        LogPrint("stormnode", "SSEG -- Stormnode list, stormnode=%s\n", vin.prevout.ToStringShort());

        LOCK(cs);

        if(vin == CTxIn()) { //only should ask for this once
            //local network
            bool isLocal = (pfrom->addr.IsRFC1918() || pfrom->addr.IsLocal());

            if(!isLocal && Params().NetworkIDString() == CBaseChainParams::MAIN) {
                std::map<CNetAddr, int64_t>::iterator i = mAskedUsForStormnodeList.find(pfrom->addr);
                if (i != mAskedUsForStormnodeList.end()){
                    int64_t t = (*i).second;
                    if (GetTime() < t) {
                        Misbehaving(pfrom->GetId(), 34);
                        LogPrintf("SSEG -- peer already asked me for the list, peer=%d\n", pfrom->id);
                        return;
                    }
                }
                int64_t askAgain = GetTime() + SSEG_UPDATE_SECONDS;
                mAskedUsForStormnodeList[pfrom->addr] = askAgain;
            }
        } //else, asking for a specific node which is ok

        int nInvCount = 0;

        BOOST_FOREACH(CStormnode& sn, vStormnodes) {
            if (vin != CTxIn() && vin != sn.vin) continue; // asked for specific vin but we are not there yet
            if (sn.addr.IsRFC1918() || sn.addr.IsLocal()) continue; // do not send local network stormnode
            if (!sn.IsEnabled()) continue;

            LogPrint("stormnode", "SSEG -- Sending Stormnode entry: stormnode=%s  addr=%s\n", sn.vin.prevout.ToStringShort(), sn.addr.ToString());
            CStormnodeBroadcast snb = CStormnodeBroadcast(sn);
            uint256 hash = snb.GetHash();
            pfrom->PushInventory(CInv(MSG_STORMNODE_ANNOUNCE, hash));
            nInvCount++;

            if (!mapSeenStormnodeBroadcast.count(hash)) {
                mapSeenStormnodeBroadcast.insert(std::make_pair(hash, snb));
            }

            if (vin == sn.vin) {
                LogPrintf("SSEG -- Sent 1 Stormnode inv to peer %d\n", pfrom->id);
                return;
            }
        }

        if(vin == CTxIn()) {
            pfrom->PushMessage(NetMsgType::SYNCSTATUSCOUNT, STORMNODE_SYNC_LIST, nInvCount);
            LogPrintf("SSEG -- Sent %d Stormnode invs to peer %d\n", nInvCount, pfrom->id);
            return;
        }
        // smth weird happen - someone asked us for vin we have no idea about?
        LogPrint("stormnode", "SSEG -- No invs sent to peer %d\n", pfrom->id);

    } else if (strCommand == NetMsgType::SNVERIFY) { // Stormnode Verify

        LOCK(cs);

        CStormnodeVerification snv;
        vRecv >> snv;

        if(snv.vchSig1.empty()) {
            // CASE 1: someone asked me to verify myself /IP we are using/
            SendVerifyReply(pfrom, snv);
        } else if (snv.vchSig2.empty()) {
            // CASE 2: we _probably_ got verification we requested from some stormnode
            ProcessVerifyReply(pfrom, snv);
        } else {
            // CASE 3: we _probably_ got verification broadcast signed by some stormnode which verified another one
            ProcessVerifyBroadcast(pfrom, snv);
        }
    }
}

// Verification of stormnode via unique direct requests.

void CStormnodeMan::DoFullVerificationStep()
{
    if(activeStormnode.vin == CTxIn()) return;

    std::vector<std::pair<int, CStormnode> > vecStormnodeRanks = GetStormnodeRanks(pCurrentBlockIndex->nHeight - 1, MIN_POSE_PROTO_VERSION);

    LOCK(cs);

    int nCount = 0;
    int nCountMax = std::max(10, (int)vStormnodes.size() / 100); // verify at least 10 stormnode at once but at most 1% of all known stormnodes

    int nMyRank = -1;
    int nRanksTotal = (int)vecStormnodeRanks.size();

    // send verify requests only if we are in top MAX_POSE_RANK
    std::vector<std::pair<int, CStormnode> >::iterator it = vecStormnodeRanks.begin();
    while(it != vecStormnodeRanks.end()) {
        if(it->first > MAX_POSE_RANK) {
            LogPrint("stormnode", "CStormnodeMan::DoFullVerificationStep -- Must be in top %d to send verify request\n",
                        (int)MAX_POSE_RANK);
            return;
        }
        if(it->second.vin == activeStormnode.vin) {
            nMyRank = it->first;
            LogPrint("stormnode", "CStormnodeMan::DoFullVerificationStep -- Found self at rank %d/%d, verifying up to %d stormnodes\n",
                        nMyRank, nRanksTotal, nCountMax);
            break;
        }
        ++it;
    }

    // edge case: list is too short and this stormnode is not enabled
    if(nMyRank == -1) return;

    // send verify requests to up to nCountMax stormnodes starting from
    // (MAX_POSE_RANK + nCountMax * (nMyRank - 1) + 1)
    int nOffset = MAX_POSE_RANK + nCountMax * (nMyRank - 1);
    if(nOffset >= (int)vecStormnodeRanks.size()) return;

    std::vector<CStormnode*> vSortedByAddr;
    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        vSortedByAddr.push_back(&sn);
    }

    sort(vSortedByAddr.begin(), vSortedByAddr.end(), CompareByAddr());

    it = vecStormnodeRanks.begin() + nOffset;
    while(it != vecStormnodeRanks.end()) {
        if(it->second.IsPoSeVerified() || it->second.IsPoSeBanned()) {
            LogPrint("stormnode", "CStormnodeMan::DoFullVerificationStep -- Already %s%s%s stormnode %s address %s, skipping...\n",
                        it->second.IsPoSeVerified() ? "verified" : "",
                        it->second.IsPoSeVerified() && it->second.IsPoSeBanned() ? " and " : "",
                        it->second.IsPoSeBanned() ? "banned" : "",
                        it->second.vin.prevout.ToStringShort(), it->second.addr.ToString());
            ++it;
            continue;
        }
        LogPrint("stormnode", "CStormnodeMan::DoFullVerificationStep -- Verifying stormnode %s rank %d/%d address %s\n",
                    it->second.vin.prevout.ToStringShort(), it->first, nRanksTotal, it->second.addr.ToString());
        if(SendVerifyRequest((CAddress)it->second.addr, vSortedByAddr)) {
            nCount++;
            if(nCount >= nCountMax) break;
        }
        ++it;
    }

    LogPrint("stormnode", "CStormnodeMan::DoFullVerificationStep -- Sent verification requests to %d stormnodes\n", nCount);
}

// This function tries to find stormnodes with the same addr,
// find a verified one and ban all the other. If there are many nodes
// with the same addr but none of them is verified yet, then none of them are banned.
// It could take many times to run this before most of the duplicate nodes are banned.

void CStormnodeMan::CheckSameAddr()
{
    if(!stormnodeSync.IsSynced() || vStormnodes.empty()) return;

    std::vector<CStormnode*> vBan;
    std::vector<CStormnode*> vSortedByAddr;

    {
        LOCK(cs);

        CStormnode* pprevStormnode = NULL;
        CStormnode* pverifiedStormnode = NULL;

        BOOST_FOREACH(CStormnode& sn, vStormnodes) {
            vSortedByAddr.push_back(&sn);
        }

        sort(vSortedByAddr.begin(), vSortedByAddr.end(), CompareByAddr());

        BOOST_FOREACH(CStormnode* psn, vSortedByAddr) {
            // check only (pre)enabled stormnodes
            if(!psn->IsEnabled() && !psn->IsPreEnabled()) continue;
            // initial step
            if(!pprevStormnode) {
                pprevStormnode = psn;
                pverifiedStormnode = psn->IsPoSeVerified() ? psn : NULL;
                continue;
            }
            // second+ step
            if(psn->addr == pprevStormnode->addr) {
                if(pverifiedStormnode) {
                    // another stormnode with the same ip is verified, ban this one
                    vBan.push_back(psn);
                } else if(psn->IsPoSeVerified()) {
                    // this stormnode with the same ip is verified, ban previous one
                    vBan.push_back(pprevStormnode);
                    // and keep a reference to be able to ban following stormnodes with the same ip
                    pverifiedStormnode = psn;
                }
            } else {
                pverifiedStormnode = psn->IsPoSeVerified() ? psn : NULL;
            }
            pprevStormnode = psn;
        }
    }

    // ban duplicates
    BOOST_FOREACH(CStormnode* psn, vBan) {
        LogPrintf("CStormnodeMan::CheckSameAddr -- increasing PoSe ban score for stormnode %s\n", psn->vin.prevout.ToStringShort());
        psn->IncreasePoSeBanScore();
    }
}

bool CStormnodeMan::SendVerifyRequest(const CAddress& addr, const std::vector<CStormnode*>& vSortedByAddr)
{
    if(netfulfilledman.HasFulfilledRequest(addr, strprintf("%s", NetMsgType::SNVERIFY)+"-request")) {
        // we already asked for verification, not a good idea to do this too often, skip it
        LogPrint("stormnode", "CStormnodeMan::SendVerifyRequest -- too many requests, skipping... addr=%s\n", addr.ToString());
        return false;
    }

    CNode* pnode = ConnectNode(addr, NULL, true);
    if(pnode != NULL) {
        netfulfilledman.AddFulfilledRequest(addr, strprintf("%s", NetMsgType::SNVERIFY)+"-request");
        // use random nonce, store it and require node to reply with correct one later
        CStormnodeVerification snv(addr, GetInsecureRand(999999), pCurrentBlockIndex->nHeight - 1);
        mWeAskedForVerification[addr] = snv;
        LogPrintf("CStormnodeMan::SendVerifyRequest -- verifying using nonce %d addr=%s\n", snv.nonce, addr.ToString());
        pnode->PushMessage(NetMsgType::SNVERIFY, snv);
        return true;
    } else {
        // can't connect, add some PoSe "ban score" to all stormnodes with given addr
        bool fFound = false;
        BOOST_FOREACH(CStormnode* psn, vSortedByAddr) {
            if(psn->addr != addr) {
                if(fFound) break;
                continue;
            }
            fFound = true;
            psn->IncreasePoSeBanScore();
        }
        return false;
    }
}

void CStormnodeMan::SendVerifyReply(CNode* pnode, CStormnodeVerification& snv)
{
    // only stormnodes can sign this, why would someone ask regular node?
    if(!fStormNode) {
        // do not ban, malicious node might be using my IP
        // and trying to confuse the node which tries to verify it
        return;
    }

    if(netfulfilledman.HasFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::SNVERIFY)+"-reply")) {
        // peer should not ask us that often
        LogPrintf("StormnodeMan::SendVerifyReply -- ERROR: peer already asked me recently, peer=%d\n", pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    uint256 blockHash;
    if(!GetBlockHash(blockHash, snv.nBlockHeight)) {
        LogPrintf("StormnodeMan::SendVerifyReply -- can't get block hash for unknown block height %d, peer=%d\n", snv.nBlockHeight, pnode->id);
        return;
    }

    std::string strMessage = strprintf("%s%d%s", activeStormnode.service.ToString(false), snv.nonce, blockHash.ToString());

    if(!sandStormSigner.SignMessage(strMessage, snv.vchSig1, activeStormnode.keyStormnode)) {
        LogPrintf("StormnodeMan::SendVerifyReply -- SignMessage() failed\n");
        return;
    }

    std::string strError;

    if(!sandStormSigner.VerifyMessage(activeStormnode.pubKeyStormnode, snv.vchSig1, strMessage, strError)) {
        LogPrintf("StormnodeMan::SendVerifyReply -- VerifyMessage() failed, error: %s\n", strError);
        return;
    }

    pnode->PushMessage(NetMsgType::SNVERIFY, snv);
    netfulfilledman.AddFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::SNVERIFY)+"-reply");
}

void CStormnodeMan::ProcessVerifyReply(CNode* pnode, CStormnodeVerification& snv)
{
    std::string strError;

    // did we even ask for it? if that's the case we should have matching fulfilled request
    if(!netfulfilledman.HasFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::SNVERIFY)+"-request")) {
        LogPrintf("CStormnodeMan::ProcessVerifyReply -- ERROR: we didn't ask for verification of %s, peer=%d\n", pnode->addr.ToString(), pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    // Received nonce for a known address must match the one we sent
    if(mWeAskedForVerification[pnode->addr].nonce != snv.nonce) {
        LogPrintf("CStormnodeMan::ProcessVerifyReply -- ERROR: wrong nounce: requested=%d, received=%d, peer=%d\n",
                    mWeAskedForVerification[pnode->addr].nonce, snv.nonce, pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    // Received nBlockHeight for a known address must match the one we sent
    if(mWeAskedForVerification[pnode->addr].nBlockHeight != snv.nBlockHeight) {
        LogPrintf("CStormnodeMan::ProcessVerifyReply -- ERROR: wrong nBlockHeight: requested=%d, received=%d, peer=%d\n",
                    mWeAskedForVerification[pnode->addr].nBlockHeight, snv.nBlockHeight, pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    uint256 blockHash;
    if(!GetBlockHash(blockHash, snv.nBlockHeight)) {
        // this shouldn't happen...
        LogPrintf("StormnodeMan::ProcessVerifyReply -- can't get block hash for unknown block height %d, peer=%d\n", snv.nBlockHeight, pnode->id);
        return;
    }

    // we already verified this address, why node is spamming?
    if(netfulfilledman.HasFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::SNVERIFY)+"-done")) {
        LogPrintf("CStormnodeMan::ProcessVerifyReply -- ERROR: already verified %s recently\n", pnode->addr.ToString());
        Misbehaving(pnode->id, 20);
        return;
    }

    {
        LOCK(cs);

        CStormnode* prealStormnode = NULL;
        std::vector<CStormnode*> vpStormnodesToBan;
        std::vector<CStormnode>::iterator it = vStormnodes.begin();
        std::string strMessage1 = strprintf("%s%d%s", pnode->addr.ToString(false), snv.nonce, blockHash.ToString());
        while(it != vStormnodes.end()) {
            if((CAddress)it->addr == pnode->addr) {
                if(sandStormSigner.VerifyMessage(it->pubKeyStormnode, snv.vchSig1, strMessage1, strError)) {
                    // found it!
                    prealStormnode = &(*it);
                    if(!it->IsPoSeVerified()) {
                        it->DecreasePoSeBanScore();
                    }
                    netfulfilledman.AddFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::SNVERIFY)+"-done");

                    // we can only broadcast it if we are an activated stormnode
                    if(activeStormnode.vin == CTxIn()) continue;
                    // update ...
                    snv.addr = it->addr;
                    snv.vin1 = it->vin;
                    snv.vin2 = activeStormnode.vin;
                    std::string strMessage2 = strprintf("%s%d%s%s%s", snv.addr.ToString(false), snv.nonce, blockHash.ToString(),
                                            snv.vin1.prevout.ToStringShort(), snv.vin2.prevout.ToStringShort());
                    // ... and sign it
                    if(!sandStormSigner.SignMessage(strMessage2, snv.vchSig2, activeStormnode.keyStormnode)) {
                        LogPrintf("StormnodeMan::ProcessVerifyReply -- SignMessage() failed\n");
                        return;
                    }

                    std::string strError;

                    if(!sandStormSigner.VerifyMessage(activeStormnode.pubKeyStormnode, snv.vchSig2, strMessage2, strError)) {
                        LogPrintf("StormnodeMan::ProcessVerifyReply -- VerifyMessage() failed, error: %s\n", strError);
                        return;
                    }

                    mWeAskedForVerification[pnode->addr] = snv;
                    snv.Relay();

                } else {
                    vpStormnodesToBan.push_back(&(*it));
                }
            }
            ++it;
        }
        // no real stormnode found?...
        if(!prealStormnode) {
            // this should never be the case normally,
            // only if someone is trying to game the system in some way or smth like that
            LogPrintf("CStormnodeMan::ProcessVerifyReply -- ERROR: no real stormnode found for addr %s\n", pnode->addr.ToString());
            Misbehaving(pnode->id, 20);
            return;
        }
        LogPrintf("CStormnodeMan::ProcessVerifyReply -- verified real stormnode %s for addr %s\n",
                    prealStormnode->vin.prevout.ToStringShort(), pnode->addr.ToString());
        // increase ban score for everyone else
        BOOST_FOREACH(CStormnode* psn, vpStormnodesToBan) {
            psn->IncreasePoSeBanScore();
            LogPrint("stormnode", "CStormnodeMan::ProcessVerifyBroadcast -- increased PoSe ban score for %s addr %s, new score %d\n",
                        prealStormnode->vin.prevout.ToStringShort(), pnode->addr.ToString(), psn->nPoSeBanScore);
        }
        LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- PoSe score incresed for %d fake Stormnodes, addr %s\n",
                    (int)vpStormnodesToBan.size(), pnode->addr.ToString());
    }
}

void CStormnodeMan::ProcessVerifyBroadcast(CNode* pnode, const CStormnodeVerification& snv)
{
    std::string strError;

    if(mapSeenStormnodeVerification.find(snv.GetHash()) != mapSeenStormnodeVerification.end()) {
        // we already have one
        return;
    }
    mapSeenStormnodeVerification[snv.GetHash()] = snv;

    // we don't care about history
    if(snv.nBlockHeight < pCurrentBlockIndex->nHeight - MAX_POSE_BLOCKS) {
        LogPrint("stormnode", "StormnodeMan::ProcessVerifyBroadcast -- Outdated: current block %d, verification block %d, peer=%d\n",
                    pCurrentBlockIndex->nHeight, snv.nBlockHeight, pnode->id);
        return;
    }

    if(snv.vin1.prevout == snv.vin2.prevout) {
        LogPrint("stormnode", "StormnodeMan::ProcessVerifyBroadcast -- ERROR: same vins %s, peer=%d\n",
                    snv.vin1.prevout.ToStringShort(), pnode->id);
        // that was NOT a good idea to cheat and verify itself,
        // ban the node we received such message from
        Misbehaving(pnode->id, 100);
        return;
    }

    uint256 blockHash;
    if(!GetBlockHash(blockHash, snv.nBlockHeight)) {
        // this shouldn't happen...
        LogPrintf("StormnodeMan::ProcessVerifyBroadcast -- Can't get block hash for unknown block height %d, peer=%d\n", snv.nBlockHeight, pnode->id);
        return;
    }

    int nRank = GetStormnodeRank(snv.vin2, snv.nBlockHeight, MIN_POSE_PROTO_VERSION);
    if(nRank < MAX_POSE_RANK) {
        LogPrint("stormnode", "StormnodeMan::ProcessVerifyBroadcast -- Stormnode is not in top %d, current rank %d, peer=%d\n",
                    (int)MAX_POSE_RANK, nRank, pnode->id);
        return;
    }

    {
        LOCK(cs);

        std::string strMessage1 = strprintf("%s%d%s", snv.addr.ToString(false), snv.nonce, blockHash.ToString());
        std::string strMessage2 = strprintf("%s%d%s%s%s", snv.addr.ToString(false), snv.nonce, blockHash.ToString(),
                                snv.vin1.prevout.ToStringShort(), snv.vin2.prevout.ToStringShort());

        CStormnode* psn1 = Find(snv.vin1);
        if(!psn1) {
            LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- can't find stormnode1 %s\n", snv.vin1.prevout.ToStringShort());
            return;
        }

        CStormnode* psn2 = Find(snv.vin2);
        if(!psn2) {
            LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- can't find stormnode %s\n", snv.vin2.prevout.ToStringShort());
            return;
        }

        if(psn1->addr != snv.addr) {
            LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- addr %s do not match %s\n", snv.addr.ToString(), pnode->addr.ToString());
            return;
        }

        if(sandStormSigner.VerifyMessage(psn1->pubKeyStormnode, snv.vchSig1, strMessage1, strError)) {
            LogPrintf("StormnodeMan::ProcessVerifyBroadcast -- VerifyMessage() for stormnode1 failed, error: %s\n", strError);
            return;
        }

        if(sandStormSigner.VerifyMessage(psn2->pubKeyStormnode, snv.vchSig2, strMessage2, strError)) {
            LogPrintf("StormnodeMan::ProcessVerifyBroadcast -- VerifyMessage() for stormnode2 failed, error: %s\n", strError);
            return;
        }

        if(!psn1->IsPoSeVerified()) {
            psn1->DecreasePoSeBanScore();
        }
        snv.Relay();

        LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- verified stormnode %s for addr %s\n",
                    psn1->vin.prevout.ToStringShort(), pnode->addr.ToString());

        // increase ban score for everyone else with the same addr
        int nCount = 0;
        BOOST_FOREACH(CStormnode& sn, vStormnodes) {
            if(sn.addr != snv.addr || sn.vin.prevout == snv.vin1.prevout) continue;
            sn.IncreasePoSeBanScore();
            nCount++;
            LogPrint("stormnode", "CStormnodeMan::ProcessVerifyBroadcast -- increased PoSe ban score for %s addr %s, new score %d\n",
                        sn.vin.prevout.ToStringShort(), sn.addr.ToString(), sn.nPoSeBanScore);
        }
        LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- PoSe score incresed for %d fake stormnodes, addr %s\n",
                    nCount, pnode->addr.ToString());
    }
}

std::string CStormnodeMan::ToString() const
{
    std::ostringstream info;

    info << "Stormnodes: " << (int)vStormnodes.size() <<
            ", peers who asked us for Stormnode list: " << (int)mAskedUsForStormnodeList.size() <<
            ", peers we asked for Stormnode list: " << (int)mWeAskedForStormnodeList.size() <<
            ", entries in Stormnode list we asked for: " << (int)mWeAskedForStormnodeListEntry.size() <<
            ", nSsqCount: " << (int)nSsqCount;

    return info.str();
}

int CStormnodeMan::GetEstimatedStormnodes(int nBlock)
{
    /*
        Stormnodes = (Coins/1000)*X on average

        *X = nPercentage, starting at 0.52
        nPercentage goes up 0.01 each period
        Period starts at 35040, which has exponential slowing growth

    */

    int nPercentage = 52; //0.52
    int nPeriod = 35040;
    int nCollateral = 1000;

    for (int i = nPeriod; i <= nBlock; i += nPeriod) {
        nPercentage++;
        nPeriod*=2;
    }
    return (GetTotalCoinEstimate(nBlock)/100*nPercentage/nCollateral);
}

void CStormnodeMan::UpdateStormnodeList(CStormnodeBroadcast snb)
{
    LOCK(cs);
    mapSeenStormnodePing.insert(std::make_pair(snb.lastPing.GetHash(), snb.lastPing));
    mapSeenStormnodeBroadcast.insert(std::make_pair(snb.GetHash(), snb));

    LogPrintf("CStormnodeMan::UpdateStormnodeList -- stormnode=%s  addr=%s\n", snb.vin.prevout.ToStringShort(), snb.addr.ToString());

    CStormnode* psn = Find(snb.vin);
    if(psn == NULL) {
        CStormnode sn(snb);
        if(Add(sn)) {
            stormnodeSync.AddedStormnodeList();
        }
    } else if(psn->UpdateFromNewBroadcast(snb)) {
        stormnodeSync.AddedStormnodeList();
    }
}

bool CStormnodeMan::CheckSnbAndUpdateStormnodeList(CStormnodeBroadcast snb, int& nDos)
{
    LOCK(cs);

    nDos = 0;
    LogPrint("stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- stormnode=%s\n", snb.vin.prevout.ToStringShort());

    if(mapSeenStormnodeBroadcast.count(snb.GetHash())) { //seen
        return true;
    }
    mapSeenStormnodeBroadcast.insert(std::make_pair(snb.GetHash(), snb));

    LogPrint("stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- stormnode=%s new\n", snb.vin.prevout.ToStringShort());

    if(!snb.SimpleCheck(nDos)) {
        LogPrint("stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- SimpleCheck() failed, stormnode=%s\n", snb.vin.prevout.ToStringShort());
        return false;
    }

    // search Stormnode list
    CStormnode* psn = Find(snb.vin);
    if(psn) {
        if(!snb.Update(psn, nDos)) {
            LogPrint("stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Update() failed, stormnode=%s\n", snb.vin.prevout.ToStringShort());
            return false;
        }
    } else {
        if(snb.CheckOutpoint(nDos)) {
            Add(snb);
            stormnodeSync.AddedStormnodeList();
            // if it matches our Stormnode privkey...
            if(fStormNode && snb.pubKeyStormnode == activeStormnode.pubKeyStormnode) {
                snb.nPoSeBanScore = -STORMNODE_POSE_BAN_MAX_SCORE;
                if(snb.nProtocolVersion == PROTOCOL_VERSION) {
                    // ... and PROTOCOL_VERSION, then we've been remotely activated ...
                    LogPrintf("CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Got NEW Stormnode entry: stormnode=%s  sigTime=%lld  addr=%s\n",
                                snb.vin.prevout.ToStringShort(), snb.sigTime, snb.addr.ToString());
                    activeStormnode.ManageState();
                } else {
                    // ... otherwise we need to reactivate our node, do not add it to the list and do not relay
                    // but also do not ban the node we get this message from
                    LogPrintf("CStormnodeMan::CheckSnbAndUpdateStormnodeList -- wrong PROTOCOL_VERSION, re-activate your SN: message nProtocolVersion=%d  PROTOCOL_VERSION=%d\n", snb.nProtocolVersion, PROTOCOL_VERSION);
                    return false;
                }
            }
            snb.Relay();
        } else {
            LogPrintf("CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Rejected Stormnode entry: %s  addr=%s\n", snb.vin.prevout.ToStringShort(), snb.addr.ToString());
            return false;
        }
    }

    return true;
}

void CStormnodeMan::UpdateLastPaid(const CBlockIndex *pindex)
{
    LOCK(cs);

    if(fLiteMode) return;

    static bool IsFirstRun = true;
    // Do full scan on first run or if we are not a stormnode
    // (MNs should update this info on every block, so limited scan should be enough for them)
    int nMaxBlocksToScanBack = (IsFirstRun || !fStormNode) ? snpayments.GetStorageLimit() : LAST_PAID_SCAN_BLOCKS;

    // LogPrint("snpayments", "CStormnodeMan::UpdateLastPaid -- nHeight=%d, nMaxBlocksToScanBack=%d, IsFirstRun=%s\n",
    //                         pindex->nHeight, nMaxBlocksToScanBack, IsFirstRun ? "true" : "false");

    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        sn.UpdateLastPaid(pindex, nMaxBlocksToScanBack);
    }

    // every time is like the first time if winners list is not synced
    IsFirstRun = !stormnodeSync.IsWinnersListSynced();
}

void CStormnodeMan::CheckAndRebuildStormnodeIndex()
{
    LOCK(cs);

    if(GetTime() - nLastIndexRebuildTime < MIN_INDEX_REBUILD_TIME) {
        return;
    }

    if(indexStormnodes.GetSize() <= MAX_EXPECTED_INDEX_SIZE) {
        return;
    }

    if(indexStormnodes.GetSize() <= int(vStormnodes.size())) {
        return;
    }

    indexStormnodesOld = indexStormnodes;
    indexStormnodes.Clear();
    for(size_t i = 0; i < vStormnodes.size(); ++i) {
        indexStormnodes.AddStormnodeVIN(vStormnodes[i].vin);
    }

    fIndexRebuilt = true;
    nLastIndexRebuildTime = GetTime();
}

void CStormnodeMan::UpdateWatchdogVoteTime(const CTxIn& vin)
{
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN)  {
        return;
    }
    pSN->UpdateWatchdogVoteTime();
    nLastWatchdogVoteTime = GetTime();
}

bool CStormnodeMan::IsWatchdogActive()
{
    LOCK(cs);
    // Check if any stormnodes have voted recently, otherwise return false
    return (GetTime() - nLastWatchdogVoteTime) <= STORMNODE_WATCHDOG_MAX_SECONDS;
}

void CStormnodeMan::AddGovernanceVote(const CTxIn& vin, uint256 nGovernanceObjectHash)
{
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN)  {
        return;
    }
    pSN->AddGovernanceVote(nGovernanceObjectHash);
}

void CStormnodeMan::RemoveGovernanceObject(uint256 nGovernanceObjectHash)
{
    LOCK(cs);
    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        sn.RemoveGovernanceObject(nGovernanceObjectHash);
    }
}

void CStormnodeMan::CheckStormnode(const CTxIn& vin, bool fForce)
{
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN)  {
        return;
    }
    pSN->Check(fForce);
}

void CStormnodeMan::CheckStormnode(const CPubKey& pubKeyStormnode, bool fForce)
{
    LOCK(cs);
    CStormnode* pSN = Find(pubKeyStormnode);
    if(!pSN)  {
        return;
    }
    pSN->Check(fForce);
}

int CStormnodeMan::GetStormnodeState(const CTxIn& vin)
{
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN)  {
        return CStormnode::STORMNODE_REMOVE;
    }
    return pSN->nActiveState;
}

int CStormnodeMan::GetStormnodeState(const CPubKey& pubKeyStormnode)
{
    LOCK(cs);
    CStormnode* pSN = Find(pubKeyStormnode);
    if(!pSN)  {
        return CStormnode::STORMNODE_REMOVE;
    }
    return pSN->nActiveState;
}

bool CStormnodeMan::IsStormnodePingedWithin(const CTxIn& vin, int nSeconds, int64_t nTimeToCheckAt)
{
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN) {
        return false;
    }
    return pSN->IsPingedWithin(nSeconds, nTimeToCheckAt);
}

void CStormnodeMan::SetStormnodeLastPing(const CTxIn& vin, const CStormnodePing& snp)
{
    LOCK(cs);
    CStormnode* pSN = Find(vin);
    if(!pSN)  {
        return;
    }
    pSN->lastPing = snp;
    mapSeenStormnodePing.insert(std::make_pair(snp.GetHash(), snp));

    CStormnodeBroadcast snb(*pSN);
    uint256 hash = snb.GetHash();
    if(mapSeenStormnodeBroadcast.count(hash)) {
        mapSeenStormnodeBroadcast[hash].lastPing = snp;
    }
}

void CStormnodeMan::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
    LogPrint("stormnode", "CStormnodeMan::UpdatedBlockTip -- pCurrentBlockIndex->nHeight=%d\n", pCurrentBlockIndex->nHeight);

    CheckSameAddr();

    if(fStormNode) {
        DoFullVerificationStep();
        // normal wallet does not need to update this every block, doing update on rpc call should be enough
        UpdateLastPaid(pindex);
    }
}

void CStormnodeMan::NotifyStormnodeUpdates()
{
    // Avoid double locking
    bool fStormnodesAddedLocal = false;
    bool fStormnodesRemovedLocal = false;
    {
        LOCK(cs);
        fStormnodesAddedLocal = fStormnodesAdded;
        fStormnodesRemovedLocal = fStormnodesRemoved;
    }

    if(fStormnodesAddedLocal) {
        governance.CheckStormnodeOrphanObjects();
        governance.CheckStormnodeOrphanVotes();
    }
    if(fStormnodesRemovedLocal) {
        governance.UpdateCachesAndClean();
    }

    LOCK(cs);
    fStormnodesAdded = false;
    fStormnodesRemoved = false;
}
