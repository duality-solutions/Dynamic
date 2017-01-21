// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "activestormnode.h"
#include "addrman.h"
#include "sandstorm.h"
#include "governance.h"
#include "stormnode-payments.h"
#include "stormnode-sync.h"
#include "stormnodeman.h"
#include "netfulfilledman.h"
#include "util.h"

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

    CStormnode *psn = Find(sn.vin);
    if (psn == NULL) {
        LogPrint("Stormnode", "CStormnodeMan::Add -- Adding new Stormnode: addr=%s, %i now\n", sn.addr.ToString(), size() + 1);
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

    std::map<COutPoint, std::map<CNetAddr, int64_t> >::iterator it1 = mWeAskedForStormnodeListEntry.find(vin.prevout);
    if (it1 != mWeAskedForStormnodeListEntry.end()) {
        std::map<CNetAddr, int64_t>::iterator it2 = it1->second.find(pnode->addr);
        if (it2 != it1->second.end()) {
            if (GetTime() < it2->second) {
                // we've asked recently, should not repeat too often or we could get banned
                return;
            }
            // we asked this node for this outpoint but it's ok to ask again already
            LogPrintf("CStormnodeMan::AskForSN -- Asking same peer %s for missing Stormnode entry again: %s\n", pnode->addr.ToString(), vin.prevout.ToStringShort());
        } else {
            // we already asked for this outpoint but not this node
            LogPrintf("CStormnodeMan::AskForSN -- Asking new peer %s for missing Stormnode entry: %s\n", pnode->addr.ToString(), vin.prevout.ToStringShort());
        }
    } else {
        // we never asked any node for this outpoint
        LogPrintf("CStormnodeMan::AskForSN -- Asking peer %s for missing Stormnode entry for the first time: %s\n", pnode->addr.ToString(), vin.prevout.ToStringShort());
    }
    mWeAskedForStormnodeListEntry[vin.prevout][pnode->addr] = GetTime() + SSEG_UPDATE_SECONDS;

    pnode->PushMessage(NetMsgType::SSEG, vin);
}

void CStormnodeMan::Check()
{
    LOCK(cs);

    LogPrint("Stormnode", "CStormnodeMan::Check -- nLastWatchdogVoteTime=%d, IsWatchdogActive()=%d\n", nLastWatchdogVoteTime, IsWatchdogActive());

    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        sn.Check();
    }
}

void CStormnodeMan::CheckAndRemove()
{
    if(!stormnodeSync.IsStormnodeListSynced()) return;

    LogPrintf("CStormnodeMan::CheckAndRemove\n");

    {
        // Need LOCK2 here to ensure consistent locking order because code below locks cs_main
        // through GetHeight() signal in ConnectNode and in CheckSnbAndUpdateStormnodeList()
        LOCK2(cs_main, cs);

        Check();

        // Remove spent Stormnodes, prepare structures and make requests to reasure the state of inactive ones
        std::vector<CStormnode>::iterator it = vStormnodes.begin();
        std::vector<std::pair<int, CStormnode> > vecStormnodeRanks;
        bool fAskedForSnbRecovery = false; // ask for one sn at a time
        while(it != vStormnodes.end()) {
            CStormnodeBroadcast snb = CStormnodeBroadcast(*it);
            uint256 hash = snb.GetHash();
            // If collateral was spent ...
            if ((*it).IsOutpointSpent()) {
                LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- Removing Stormnode: %s  addr=%s  %i now\n", (*it).GetStateString(), (*it).addr.ToString(), size() - 1);
                // erase all of the broadcasts we've seen from this txin, ...
                mapSeenStormnodeBroadcast.erase(hash);
                mWeAskedForStormnodeListEntry.erase((*it).vin.prevout);
                // and finally remove it from the list
                it->FlagGovernanceItemsAsDirty();
                it = vStormnodes.erase(it);
                fStormnodesRemoved = true;
            } else {
                bool fAsk = pCurrentBlockIndex &&
                            !fAskedForSnbRecovery &&
                            stormnodeSync.IsSynced() &&
                            it->IsNewStartRequired() &&
                            !IsSnbRecoveryRequested(hash);
                if(fAsk) {
                    // this sn is in a non-recoverable state and we haven't asked other nodes yet
                    std::set<CNetAddr> setRequested;
                    // calulate only once and only when it's needed
                    if(vecStormnodeRanks.empty()) {
                        int nRandomBlockHeight = GetRandInt(pCurrentBlockIndex->nHeight);
                        vecStormnodeRanks = GetStormnodeRanks(nRandomBlockHeight);
                    }
                    // ask first SNB_RECOVERY_QUORUM_TOTAL sn's we can connect to and we haven't asked recently
                    for(int i = 0; setRequested.size() < SNB_RECOVERY_QUORUM_TOTAL && i < (int)vecStormnodeRanks.size(); i++) {
                        // avoid banning
                        if(mWeAskedForStormnodeListEntry.count(it->vin.prevout) && mWeAskedForStormnodeListEntry[it->vin.prevout].count(vecStormnodeRanks[i].second.addr)) continue;
                        // didn't ask recently, ok to ask now
                        CService addr = vecStormnodeRanks[i].second.addr;
                        CNode* pnode = ConnectNode(CAddress(addr), NULL, true);
                        if(pnode) {
                            LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- asking for snb of %s, addr=%s\n", it->vin.prevout.ToStringShort(), addr.ToString());
                            setRequested.insert(addr);
                            // can't use AskForSN here, inv system is way too smart, request data directly instead
                            std::vector<CInv> vToFetch;
                            vToFetch.push_back(CInv(MSG_STORMNODE_ANNOUNCE, hash));
                            pnode->PushMessage(NetMsgType::GETDATA, vToFetch);
                            fAskedForSnbRecovery = true;
                        } else {
                            LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- can't connect to node to ask for snb, addr=%s\n", addr.ToString());
                        }
                    }
                    // wait for snb recovery replies for SNB_RECOVERY_WAIT_SECONDS seconds
                    mSnbRecoveryRequests[hash] = std::make_pair(GetTime() + SNB_RECOVERY_WAIT_SECONDS, setRequested);
                }
                ++it;
            }
        }
        // proces replies for STORMNODE_NEW_START_REQUIRED Stormnodes
        LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- mSnbRecoveryGoodReplies size=%d\n", (int)mSnbRecoveryGoodReplies.size());
        std::map<uint256, std::vector<CStormnodeBroadcast> >::iterator itSnbReplies = mSnbRecoveryGoodReplies.begin();
        while(itSnbReplies != mSnbRecoveryGoodReplies.end()){
            if(mSnbRecoveryRequests[itSnbReplies->first].first < GetTime()) {
                // all nodes we asked should have replied now
                if(itSnbReplies->second.size() >= SNB_RECOVERY_QUORUM_REQUIRED) {
                    // majority of nodes we asked agrees that this sn doesn't require new snb, reprocess one of new snbs
                    LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- reprocessing snb, Stormnode=%s\n", itSnbReplies->second[0].vin.prevout.ToStringShort());
                    // mapSeenStormnodeBroadcast.erase(itSnbReplies->first);
                    int nDos;
                    itSnbReplies->second[0].fRecovery = true;
                    CheckSnbAndUpdateStormnodeList(NULL, itSnbReplies->second[0], nDos);
                }
                LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- removing snb recovery reply, Stormnode=%s, size=%d\n", itSnbReplies->second[0].vin.prevout.ToStringShort(), (int)itSnbReplies->second.size());
                mSnbRecoveryGoodReplies.erase(itSnbReplies++);
            } else {
                ++itSnbReplies;
            }
        }
    }
    {
        // no need for cm_main below
        LOCK(cs);

        std::map<uint256, std::pair< int64_t, std::set<CNetAddr> > >::iterator itSnbRequest = mSnbRecoveryRequests.begin();
        while(itSnbRequest != mSnbRecoveryRequests.end()){
            // Allow this snb to be re-verified again after SNB_RECOVERY_RETRY_SECONDS seconds
            // if sn is still in STORMNODE_NEW_START_REQUIRED state.
            if(GetTime() - itSnbRequest->second.first > SNB_RECOVERY_RETRY_SECONDS) {
                mSnbRecoveryRequests.erase(itSnbRequest++);
            } else {
                ++itSnbRequest;
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
        std::map<COutPoint, std::map<CNetAddr, int64_t> >::iterator it2 = mWeAskedForStormnodeListEntry.begin();
        while(it2 != mWeAskedForStormnodeListEntry.end()){
            std::map<CNetAddr, int64_t>::iterator it3 = it2->second.begin();
            while(it3 != it2->second.end()){
                if(it3->second < GetTime()){
                    it2->second.erase(it3++);
                } else {
                    ++it3;
                }
            }
            if(it2->second.empty()) {
                mWeAskedForStormnodeListEntry.erase(it2++);
            } else {
                ++it2;
            }
        }

        std::map<CNetAddr, CStormnodeVerification>::iterator it3 = mWeAskedForVerification.begin();
        while(it3 != mWeAskedForVerification.end()){
            if(it3->second.nBlockHeight < pCurrentBlockIndex->nHeight - MAX_POSE_BLOCKS) {
                mWeAskedForVerification.erase(it3++);
            } else {
                ++it3;
            }
        }

        // NOTE: do not expire mapSeenStormnodeBroadcast entries here, clean them on snb updates!

        // remove expired mapSeenStormnodePing
        std::map<uint256, CStormnodePing>::iterator it4 = mapSeenStormnodePing.begin();
        while(it4 != mapSeenStormnodePing.end()){
            if((*it4).second.IsExpired()) {
                LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- Removing expired Stormnode ping: hash=%s\n", (*it4).second.GetHash().ToString());
                mapSeenStormnodePing.erase(it4++);
            } else {
                ++it4;
            }
        }

        // remove expired mapSeenStormnodeVerification
        std::map<uint256, CStormnodeVerification>::iterator itv2 = mapSeenStormnodeVerification.begin();
        while(itv2 != mapSeenStormnodeVerification.end()){
            if((*itv2).second.nBlockHeight < pCurrentBlockIndex->nHeight - MAX_POSE_BLOCKS){
                LogPrint("Stormnode", "CStormnodeMan::CheckAndRemove -- Removing expired Stormnode verification: hash=%s\n", (*itv2).first.ToString());
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
        if(sn.nProtocolVersion < nProtocolVersion || !sn.IsEnabled()) continue;
        nCount++;
    }

    return nCount;
}

/* Only IPv4 Stormnodes are allowed, saving this for later
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

    if(!(pnode->addr.IsRFC1918() || pnode->addr.IsLocal())) {
        std::map<CNetAddr, int64_t>::iterator it = mWeAskedForStormnodeList.find(pnode->addr);
        if(it != mWeAskedForStormnodeList.end() && GetTime() < (*it).second) {
            LogPrintf("CStormnodeMan::SsegUpdate -- we already asked %s for the list; skipping...\n", pnode->addr.ToString());
            return;
        }
    }

    pnode->PushMessage(NetMsgType::SSEG, CTxIn());
    int64_t askAgain = GetTime() + SSEG_UPDATE_SECONDS;
    mWeAskedForStormnodeList[pnode->addr] = askAgain;

    LogPrint("Stormnode", "CStormnodeMan::SsegUpdate -- asked %s for the list\n", pnode->addr.ToString());
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
// Deterministically select the oldest/best Stormnode to pay on the network
//
CStormnode* CStormnodeMan::GetNextStormnodeInQueueForPayment(bool fFilterSigTime, int& nCount)
{
    if(!pCurrentBlockIndex) {
        nCount = 0;
        return NULL;
    }
    return GetNextStormnodeInQueueForPayment(pCurrentBlockIndex->nHeight, fFilterSigTime, nCount);
}

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
        if(!sn.IsValidForPayment()) continue;

        // //check protocol version
        if(sn.nProtocolVersion < snpayments.GetMinStormnodePaymentsProto()) continue;

        //it's in the list (up to 8 entries ahead of current block to allow propagation) -- so let's skip it
        if(snpayments.IsScheduled(sn, nBlockHeight)) continue;

        //it's too new, wait for a cycle
        if(fFilterSigTime && sn.sigTime + (nSnCount*2.6*60) > GetAdjustedTime()) continue;

        //make sure it has at least as many confirmations as there are Stormnodes
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
    int nTenthNetwork = nSnCount/10;
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

    LogPrintf("CStormnodeMan::FindRandomNotInVec -- %d enabled Stormnodes, %d Stormnodes to choose from\n", nCountEnabled, nCountNotExcluded);
    if(nCountNotExcluded < 1) return NULL;

    // fill a vector of pointers
    std::vector<CStormnode*> vpStormnodesShuffled;
    BOOST_FOREACH(CStormnode &sn, vStormnodes) {
        vpStormnodesShuffled.push_back(&sn);
    }

    InsecureRand insecureRand;

    // shuffle pointers
    std::random_shuffle(vpStormnodesShuffled.begin(), vpStormnodesShuffled.end(), insecureRand);
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
        LogPrint("Stormnode", "CStormnodeMan::FindRandomNotInVec -- found, Stormnode=%s\n", psn->vin.prevout.ToStringShort());
        return psn;
    }

    LogPrint("Stormnode", "CStormnodeMan::FindRandomNotInVec -- failed\n");
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
            if(!sn.IsEnabled()) continue;
        }
        else {
            if(!sn.IsValidForPayment()) continue;
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
        if(fOnlyActive && !sn.IsEnabled()) continue;

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
    if(fLiteMode) return; // disable all DarkSilk specific functionality
    if(!stormnodeSync.IsBlockchainSynced()) return;

    if (strCommand == NetMsgType::SNANNOUNCE) { //Stormnode Broadcast

        CStormnodeBroadcast snb;
        vRecv >> snb;

        pfrom->setAskFor.erase(snb.GetHash());

        LogPrint("Stormnode", "SNANNOUNCE -- Stormnode announce, Stormnode=%s\n", snb.vin.prevout.ToStringShort());

            int nDos = 0;

        if (CheckSnbAndUpdateStormnodeList(pfrom, snb, nDos)) {
                // use announced Stormnode as a peer
                addrman.Add(CAddress(snb.addr), pfrom->addr, 2*60*60);
            } else if(nDos > 0) {
                Misbehaving(pfrom->GetId(), nDos);
        }
        if(fStormnodesAdded) {
            NotifyStormnodeUpdates();
        }
    } else if (strCommand == NetMsgType::SNPING) { //Stormnode Ping

        CStormnodePing snp;
        vRecv >> snp;

        uint256 nHash = snp.GetHash();

        pfrom->setAskFor.erase(nHash);

        LogPrint("Stormnode", "SNPING -- Stormnode ping, Stormnode=%s\n", snp.vin.prevout.ToStringShort());

        // Need LOCK2 here to ensure consistent locking order because the CheckAndUpdate call below locks cs_main
        LOCK2(cs_main, cs);

        if(mapSeenStormnodePing.count(nHash)) return; //seen
        mapSeenStormnodePing.insert(std::make_pair(nHash, snp));

        LogPrint("Stormnode", "SNPING -- Stormnode ping, Stormnode=%s new\n", snp.vin.prevout.ToStringShort());

        // see if we have this Stormnode
        CStormnode* psn = snodeman.Find(snp.vin);

        // too late, new SNANNOUNCE is required
        if(psn && psn->IsNewStartRequired()) return;

        int nDos = 0;
        if(snp.CheckAndUpdate(psn, false, nDos)) return;

        if(nDos > 0) {
            // if anything significant failed, mark that node
            Misbehaving(pfrom->GetId(), nDos);
        } else if(psn != NULL) {
            // nothing significant failed, sn is a known one too
            return;
        }

        // something significant is broken or sn is unknown,
        // we might have to ask for a Stormnode entry once
        AskForSN(pfrom, snp.vin);

    } else if (strCommand == NetMsgType::SSEG) { //Get Stormnode list or specific entry
        // Ignore such requests until we are fully synced.
        // We could start processing this after Stormnode list is synced
        // but this is a heavy one so it's better to finish sync first.
        if (!stormnodeSync.IsSynced()) return;

        CTxIn vin;
        vRecv >> vin;

        LogPrint("Stormnode", "SSEG -- Stormnode list, Stormnode=%s\n", vin.prevout.ToStringShort());

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
            if (sn.addr.IsRFC1918() || sn.addr.IsLocal()) continue; // do not send local network Stormnode

            LogPrint("Stormnode", "SSEG -- Sending Stormnode entry: Stormnode=%s  addr=%s\n", sn.vin.prevout.ToStringShort(), sn.addr.ToString());
            CStormnodeBroadcast snb = CStormnodeBroadcast(sn);
            uint256 hash = snb.GetHash();
            pfrom->PushInventory(CInv(MSG_STORMNODE_ANNOUNCE, hash));
            pfrom->PushInventory(CInv(MSG_STORMNODE_PING, sn.lastPing.GetHash()));
            nInvCount++;

            if (!mapSeenStormnodeBroadcast.count(hash)) {
                mapSeenStormnodeBroadcast.insert(std::make_pair(hash, std::make_pair(GetTime(), snb)));
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
        LogPrint("Stormnode", "SSEG -- No invs sent to peer %d\n", pfrom->id);

    } else if (strCommand == NetMsgType::SNVERIFY) { // Stormnode Verify

        // Need LOCK2 here to ensure consistent locking order because the all functions below call GetBlockHash which locks cs_main
        LOCK2(cs_main, cs);

        CStormnodeVerification snv;
        vRecv >> snv;

        if(snv.vchSig1.empty()) {
            // CASE 1: someone asked me to verify myself /IP we are using/
            SendVerifyReply(pfrom, snv);
        } else if (snv.vchSig2.empty()) {
            // CASE 2: we _probably_ got verification we requested from some Stormnode
            ProcessVerifyReply(pfrom, snv);
        } else {
            // CASE 3: we _probably_ got verification broadcast signed by some Stormnode which verified another one
            ProcessVerifyBroadcast(pfrom, snv);
        }
    }
}

// Verification of Stormnode via unique direct requests.

void CStormnodeMan::DoFullVerificationStep()
{
    if(activeStormnode.vin == CTxIn()) return;
    if(!stormnodeSync.IsSynced()) return;

    std::vector<std::pair<int, CStormnode> > vecStormnodeRanks = GetStormnodeRanks(pCurrentBlockIndex->nHeight - 1, MIN_POSE_PROTO_VERSION);

    // Need LOCK2 here to ensure consistent locking order because the SendVerifyRequest call below locks cs_main
    // through GetHeight() signal in ConnectNode
    LOCK2(cs_main, cs);

    int nCount = 0;
    int nCountMax = std::max(10, (int)vStormnodes.size() / 100); // verify at least 10 Stormnode at once but at most 1% of all known Stormnodes

    int nMyRank = -1;
    int nRanksTotal = (int)vecStormnodeRanks.size();

    // send verify requests only if we are in top MAX_POSE_RANK
    std::vector<std::pair<int, CStormnode> >::iterator it = vecStormnodeRanks.begin();
    while(it != vecStormnodeRanks.end()) {
        if(it->first > MAX_POSE_RANK) {
            LogPrint("Stormnode", "CStormnodeMan::DoFullVerificationStep -- Must be in top %d to send verify request\n",
                        (int)MAX_POSE_RANK);
            return;
        }
        if(it->second.vin == activeStormnode.vin) {
            nMyRank = it->first;
            LogPrint("Stormnode", "CStormnodeMan::DoFullVerificationStep -- Found self at rank %d/%d, verifying up to %d Stormnodes\n",
                        nMyRank, nRanksTotal, nCountMax);
            break;
        }
        ++it;
    }

    // edge case: list is too short and this Stormnode is not enabled
    if(nMyRank == -1) return;

    // send verify requests to up to nCountMax Stormnodes starting from
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
            LogPrint("Stormnode", "CStormnodeMan::DoFullVerificationStep -- Already %s%s%s Stormnode %s address %s, skipping...\n",
                        it->second.IsPoSeVerified() ? "verified" : "",
                        it->second.IsPoSeVerified() && it->second.IsPoSeBanned() ? " and " : "",
                        it->second.IsPoSeBanned() ? "banned" : "",
                        it->second.vin.prevout.ToStringShort(), it->second.addr.ToString());
            ++it;
            continue;
        }
        LogPrint("Stormnode", "CStormnodeMan::DoFullVerificationStep -- Verifying Stormnode %s rank %d/%d address %s\n",
                    it->second.vin.prevout.ToStringShort(), it->first, nRanksTotal, it->second.addr.ToString());
        if(SendVerifyRequest((CAddress)it->second.addr, vSortedByAddr)) {
            nCount++;
            if(nCount >= nCountMax) break;
        }
        ++it;
    }

    LogPrint("Stormnode", "CStormnodeMan::DoFullVerificationStep -- Sent verification requests to %d Stormnodes\n", nCount);
}

// This function tries to find Stormnodes with the same addr,
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
            // check only (pre)enabled Stormnodes
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
                    // another Stormnode with the same ip is verified, ban this one
                    vBan.push_back(psn);
                } else if(psn->IsPoSeVerified()) {
                    // this Stormnode with the same ip is verified, ban previous one
                    vBan.push_back(pprevStormnode);
                    // and keep a reference to be able to ban following Stormnodes with the same ip
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
        LogPrintf("CStormnodeMan::CheckSameAddr -- increasing PoSe ban score for Stormnode %s\n", psn->vin.prevout.ToStringShort());
        psn->IncreasePoSeBanScore();
    }
}

bool CStormnodeMan::SendVerifyRequest(const CAddress& addr, const std::vector<CStormnode*>& vSortedByAddr)
{
    if(netfulfilledman.HasFulfilledRequest(addr, strprintf("%s", NetMsgType::SNVERIFY)+"-request")) {
        // we already asked for verification, not a good idea to do this too often, skip it
        LogPrint("Stormnode", "CStormnodeMan::SendVerifyRequest -- too many requests, skipping... addr=%s\n", addr.ToString());
        return false;
    }

    CNode* pnode = ConnectNode(addr, NULL, true);
    if(pnode == NULL) {
        LogPrintf("CStormnodeMan::SendVerifyRequest -- can't connect to node to verify it, addr=%s\n", addr.ToString());
        return false;
    }

    netfulfilledman.AddFulfilledRequest(addr, strprintf("%s", NetMsgType::SNVERIFY)+"-request");
    // use random nonce, store it and require node to reply with correct one later
    CStormnodeVerification snv(addr, GetRandInt(999999), pCurrentBlockIndex->nHeight - 1);
    mWeAskedForVerification[addr] = snv;
    LogPrintf("CStormnodeMan::SendVerifyRequest -- verifying node using nonce %d addr=%s\n", snv.nonce, addr.ToString());
    pnode->PushMessage(NetMsgType::SNVERIFY, snv);

    return true;
}

void CStormnodeMan::SendVerifyReply(CNode* pnode, CStormnodeVerification& snv)
{
    // only Stormnodes can sign this, why would someone ask regular node?
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

                    // we can only broadcast it if we are an activated Stormnode
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
        // no real Stormnode found?...
        if(!prealStormnode) {
            // this should never be the case normally,
            // only if someone is trying to game the system in some way or smth like that
            LogPrintf("CStormnodeMan::ProcessVerifyReply -- ERROR: no real Stormnode found for addr %s\n", pnode->addr.ToString());
            Misbehaving(pnode->id, 20);
            return;
        }
        LogPrintf("CStormnodeMan::ProcessVerifyReply -- verified real Stormnode %s for addr %s\n",
                    prealStormnode->vin.prevout.ToStringShort(), pnode->addr.ToString());
        // increase ban score for everyone else
        BOOST_FOREACH(CStormnode* psn, vpStormnodesToBan) {
            psn->IncreasePoSeBanScore();
            LogPrint("Stormnode", "CStormnodeMan::ProcessVerifyBroadcast -- increased PoSe ban score for %s addr %s, new score %d\n",
                        prealStormnode->vin.prevout.ToStringShort(), pnode->addr.ToString(), psn->nPoSeBanScore);
        }
        LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- PoSe score increased for %d fake Stormnodes, addr %s\n",
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
        LogPrint("Stormnode", "StormnodeMan::ProcessVerifyBroadcast -- Outdated: current block %d, verification block %d, peer=%d\n",
                    pCurrentBlockIndex->nHeight, snv.nBlockHeight, pnode->id);
        return;
    }

    if(snv.vin1.prevout == snv.vin2.prevout) {
        LogPrint("Stormnode", "StormnodeMan::ProcessVerifyBroadcast -- ERROR: same vins %s, peer=%d\n",
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
        LogPrint("Stormnode", "StormnodeMan::ProcessVerifyBroadcast -- Stormnode is not in top %d, current rank %d, peer=%d\n",
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
            LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- can't find Stormnode1 %s\n", snv.vin1.prevout.ToStringShort());
            return;
        }

        CStormnode* psn2 = Find(snv.vin2);
        if(!psn2) {
            LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- can't find Stormnode %s\n", snv.vin2.prevout.ToStringShort());
            return;
        }

        if(psn1->addr != snv.addr) {
            LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- addr %s do not match %s\n", snv.addr.ToString(), pnode->addr.ToString());
            return;
        }

        if(sandStormSigner.VerifyMessage(psn1->pubKeyStormnode, snv.vchSig1, strMessage1, strError)) {
            LogPrintf("StormnodeMan::ProcessVerifyBroadcast -- VerifyMessage() for Stormnode1 failed, error: %s\n", strError);
            return;
        }

        if(sandStormSigner.VerifyMessage(psn2->pubKeyStormnode, snv.vchSig2, strMessage2, strError)) {
            LogPrintf("StormnodeMan::ProcessVerifyBroadcast -- VerifyMessage() for Stormnode2 failed, error: %s\n", strError);
            return;
        }

        if(!psn1->IsPoSeVerified()) {
            psn1->DecreasePoSeBanScore();
        }
        snv.Relay();

        LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- verified Stormnode %s for addr %s\n",
                    psn1->vin.prevout.ToStringShort(), pnode->addr.ToString());

        // increase ban score for everyone else with the same addr
        int nCount = 0;
        BOOST_FOREACH(CStormnode& sn, vStormnodes) {
            if(sn.addr != snv.addr || sn.vin.prevout == snv.vin1.prevout) continue;
            sn.IncreasePoSeBanScore();
            nCount++;
            LogPrint("Stormnode", "CStormnodeMan::ProcessVerifyBroadcast -- increased PoSe ban score for %s addr %s, new score %d\n",
                        sn.vin.prevout.ToStringShort(), sn.addr.ToString(), sn.nPoSeBanScore);
        }
        LogPrintf("CStormnodeMan::ProcessVerifyBroadcast -- PoSe score incresed for %d fake Stormnodes, addr %s\n",
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

void CStormnodeMan::UpdateStormnodeList(CStormnodeBroadcast snb)
{
    LOCK(cs);
    mapSeenStormnodePing.insert(std::make_pair(snb.lastPing.GetHash(), snb.lastPing));
    mapSeenStormnodeBroadcast.insert(std::make_pair(snb.GetHash(), std::make_pair(GetTime(), snb)));

    LogPrintf("CStormnodeMan::UpdateStormnodeList -- Stormnode=%s  addr=%s\n", snb.vin.prevout.ToStringShort(), snb.addr.ToString());

    CStormnode* psn = Find(snb.vin);
    if(psn == NULL) {
        CStormnode sn(snb);
        if(Add(sn)) {
            stormnodeSync.AddedStormnodeList();
        }
    } else {
        CStormnodeBroadcast snbOld = mapSeenStormnodeBroadcast[CStormnodeBroadcast(*psn).GetHash()].second;
        if(psn->UpdateFromNewBroadcast(snb)) {
            stormnodeSync.AddedStormnodeList();
            mapSeenStormnodeBroadcast.erase(snbOld.GetHash());
        }
    }
}

bool CStormnodeMan::CheckSnbAndUpdateStormnodeList(CNode* pfrom, CStormnodeBroadcast snb, int& nDos)
{
    // Need LOCK2 here to ensure consistent locking order because the SimpleCheck call below locks cs_main
    LOCK2(cs_main, cs);

    nDos = 0;
    LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Stormnode=%s\n", snb.vin.prevout.ToStringShort());

    uint256 hash = snb.GetHash();
    if(mapSeenStormnodeBroadcast.count(hash) && !snb.fRecovery) { //seen
        LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Stormnode=%s seen\n", snb.vin.prevout.ToStringShort());
        // less then 2 pings left before this SN goes into non-recoverable state, bump sync timeout
        if(GetTime() - mapSeenStormnodeBroadcast[hash].first > STORMNODE_NEW_START_REQUIRED_SECONDS - STORMNODE_MIN_SNP_SECONDS * 2) {
            LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Stormnode=%s seen update\n", snb.vin.prevout.ToStringShort());
            mapSeenStormnodeBroadcast[hash].first = GetTime();
            stormnodeSync.AddedStormnodeList();
        }
        // did we ask this node for it?
        if(pfrom && IsSnbRecoveryRequested(hash) && GetTime() < mSnbRecoveryRequests[hash].first) {
            LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- snb=%s seen request\n", hash.ToString());
            if(mSnbRecoveryRequests[hash].second.count(pfrom->addr)) {
                LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- snb=%s seen request, addr=%s\n", hash.ToString(), pfrom->addr.ToString());
                // do not allow node to send same snb multiple times in recovery mode
                mSnbRecoveryRequests[hash].second.erase(pfrom->addr);
                // does it have newer lastPing?
                if(snb.lastPing.sigTime > mapSeenStormnodeBroadcast[hash].second.lastPing.sigTime) {
                    // simulate Check
                    CStormnode snTemp = CStormnode(snb);
                    snTemp.Check();
                    LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- snb=%s seen request, addr=%s, better lastPing: %d min ago, projected sn state: %s\n", hash.ToString(), pfrom->addr.ToString(), (GetTime() - snb.lastPing.sigTime)/60, snTemp.GetStateString());
                    if(snTemp.IsValidStateForAutoStart(snTemp.nActiveState)) {
                        // this node thinks it's a good one
                        LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Stormnode=%s seen good\n", snb.vin.prevout.ToStringShort());
                        mSnbRecoveryGoodReplies[hash].push_back(snb);
                    }
                }
            }
        }
        return true;
    }
    mapSeenStormnodeBroadcast.insert(std::make_pair(hash, std::make_pair(GetTime(), snb)));

    LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Stormnode=%s new\n", snb.vin.prevout.ToStringShort());

    if(!snb.SimpleCheck(nDos)) {
        LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- SimpleCheck() failed, Stormnode=%s\n", snb.vin.prevout.ToStringShort());
        return false;
    }

    // search Stormnode list
    CStormnode* psn = Find(snb.vin);
    if(psn) {
        CStormnodeBroadcast snbOld = mapSeenStormnodeBroadcast[CStormnodeBroadcast(*psn).GetHash()].second;
        if(!snb.Update(psn, nDos)) {
            LogPrint("Stormnode", "CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Update() failed, Stormnode=%s\n", snb.vin.prevout.ToStringShort());
            return false;
        }
        if(hash != snbOld.GetHash()) {
            mapSeenStormnodeBroadcast.erase(snbOld.GetHash());
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
                    LogPrintf("CStormnodeMan::CheckSnbAndUpdateStormnodeList -- Got NEW Stormnode entry: Stormnode=%s  sigTime=%lld  addr=%s\n",
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

void CStormnodeMan::UpdateLastPaid()
{
    LOCK(cs);

    if(fLiteMode) return;
    if(!pCurrentBlockIndex) return;

    static bool IsFirstRun = true;
    // Do full scan on first run or if we are not a Stormnode
    // (MNs should update this info on every block, so limited scan should be enough for them)
    int nMaxBlocksToScanBack = (IsFirstRun || !fStormNode) ? snpayments.GetStorageLimit() : LAST_PAID_SCAN_BLOCKS;

    // pCurrentBlockIndex->nHeight, nMaxBlocksToScanBack, IsFirstRun ? "true" : "false");

    BOOST_FOREACH(CStormnode& sn, vStormnodes) {
        sn.UpdateLastPaid(pCurrentBlockIndex, nMaxBlocksToScanBack);
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
    // Check if any Stormnodes have voted recently, otherwise return false
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
        return CStormnode::STORMNODE_NEW_START_REQUIRED;
    }
    return pSN->nActiveState;
}

int CStormnodeMan::GetStormnodeState(const CPubKey& pubKeyStormnode)
{
    LOCK(cs);
    CStormnode* pSN = Find(pubKeyStormnode);
    if(!pSN)  {
        return CStormnode::STORMNODE_NEW_START_REQUIRED;
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
        mapSeenStormnodeBroadcast[hash].second.lastPing = snp;
    }
}

void CStormnodeMan::UpdatedBlockTip(const CBlockIndex *pindex)
{
    pCurrentBlockIndex = pindex;
    LogPrint("Stormnode", "CStormnodeMan::UpdatedBlockTip -- pCurrentBlockIndex->nHeight=%d\n", pCurrentBlockIndex->nHeight);

    CheckSameAddr();

    if(fStormNode) {
        DoFullVerificationStep();
        // normal wallet does not need to update this every block, doing update on rpc call should be enough
        UpdateLastPaid();
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
