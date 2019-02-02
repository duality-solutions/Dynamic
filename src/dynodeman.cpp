// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynodeman.h"

#include "activedynode.h"
#include "addrman.h"
#include "alert.h"
#include "clientversion.h"
#include "dynode-payments.h"
#include "dynode-sync.h"
#include "governance.h"
#include "init.h"
#include "messagesigner.h"
#include "netfulfilledman.h"
#include "netmessagemaker.h"
#ifdef ENABLE_WALLET
#include "privatesend-client.h"
#endif // ENABLE_WALLET
#include "script/standard.h"
#include "ui_interface.h"
#include "util.h"
#include "warnings.h"

/** Dynode manager */
CDynodeMan dnodeman;

const std::string CDynodeMan::SERIALIZATION_VERSION_STRING = "CDynodeMan-Version-3";
const int CDynodeMan::LAST_PAID_SCAN_BLOCKS = 100;

struct CompareLastPaidBlock {
    bool operator()(const std::pair<int, const CDynode*>& t1,
        const std::pair<int, const CDynode*>& t2) const
    {
        return (t1.first != t2.first) ? (t1.first < t2.first) : (t1.second->outpoint < t2.second->outpoint);
    }
};

struct CompareScoreDN {
    bool operator()(const std::pair<arith_uint256, const CDynode*>& t1,
        const std::pair<arith_uint256, const CDynode*>& t2) const
    {
        return (t1.first != t2.first) ? (t1.first < t2.first) : (t1.second->outpoint < t2.second->outpoint);
    }
};

struct CompareByAddr

{
    bool operator()(const CDynode* t1,
        const CDynode* t2) const
    {
        return t1->addr < t2->addr;
    }
};

CDynodeMan::CDynodeMan()
    : cs(),
      mapDynodes(),
      mAskedUsForDynodeList(),
      mWeAskedForDynodeList(),
      mWeAskedForDynodeListEntry(),
      mWeAskedForVerification(),
      mDnbRecoveryRequests(),
      mDnbRecoveryGoodReplies(),
      listScheduledDnbRequestConnections(),
      fDynodesAdded(false),
      fDynodesRemoved(false),
      vecDirtyGovernanceObjectHashes(),
      nLastSentinelPingTime(0),
      mapSeenDynodeBroadcast(),
      mapSeenDynodePing(),
      nPsqCount(0)
{
}

bool CDynodeMan::Add(CDynode& dn)
{
    LOCK(cs);

    if (Has(dn.outpoint))
        return false;

    LogPrint("dynode", "CDynodeMan::Add -- Adding new Dynode: addr=%s, %i now\n", dn.addr.ToString(), size() + 1);
    mapDynodes[dn.outpoint] = dn;
    fDynodesAdded = true;
    return true;
}

void CDynodeMan::AskForDN(CNode* pnode, const COutPoint& outpoint, CConnman& connman)
{
    if (!pnode)
        return;

    CNetMsgMaker msgMaker(pnode->GetSendVersion());
    LOCK(cs);

    CService addrSquashed = Params().AllowMultiplePorts() ? (CService)pnode->addr : CService(pnode->addr, 0);
    auto it1 = mWeAskedForDynodeListEntry.find(outpoint);
    if (it1 != mWeAskedForDynodeListEntry.end()) {
        auto it2 = it1->second.find(addrSquashed);
        if (it2 != it1->second.end()) {
            if (GetTime() < it2->second) {
                // we've asked recently, should not repeat too often or we could get banned
                return;
            }
            // we asked this node for this outpoint but it's ok to ask again already
            LogPrintf("CDynodeMan::AskForDN -- Asking same peer %s for missing Dynode entry again: %s\n", addrSquashed.ToString(), outpoint.ToStringShort());
        } else {
            // we already asked for this outpoint but not this node
            LogPrintf("CDynodeMan::AskForDN -- Asking new peer %s for missing Dynode entry: %s\n", addrSquashed.ToString(), outpoint.ToStringShort());
        }
    } else {
        // we never asked any node for this outpoint
        LogPrintf("CDynodeMan::AskForDN -- Asking peer %s for missing Dynode entry for the first time: %s\n", addrSquashed.ToString(), outpoint.ToStringShort());
    }
    mWeAskedForDynodeListEntry[outpoint][addrSquashed] = GetTime() + PSEG_UPDATE_SECONDS;

    if (pnode->GetSendVersion() == 70900) {
        connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PSEG, CTxIn(outpoint)));
    } else {
        connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PSEG, outpoint));
    }
}

bool CDynodeMan::AllowMixing(const COutPoint& outpoint)
{
    LOCK(cs);
    CDynode* pdn = Find(outpoint);
    if (!pdn) {
        return false;
    }
    nPsqCount++;
    pdn->nLastPsq = nPsqCount;
    pdn->fAllowMixingTx = true;

    return true;
}

bool CDynodeMan::DisallowMixing(const COutPoint& outpoint)
{
    LOCK(cs);
    CDynode* pdn = Find(outpoint);
    if (!pdn) {
        return false;
    }
    pdn->fAllowMixingTx = false;

    return true;
}

bool CDynodeMan::PoSeBan(const COutPoint& outpoint)
{
    LOCK(cs);
    CDynode* pdn = Find(outpoint);
    if (!pdn) {
        return false;
    }
    pdn->PoSeBan();

    return true;
}

void CDynodeMan::Check()
{
    LOCK2(cs_main, cs);

    LogPrint("dynode", "CDynodeMan::Check -- nLastSentinelPingTime=%d, IsSentinelPingActive()=%d\n", nLastSentinelPingTime, IsSentinelPingActive());

    for (auto& dnpair : mapDynodes) {
        // NOTE: internally it checks only every DYNODE_CHECK_SECONDS seconds
        // since the last time, so expect some DNs to skip this
        dnpair.second.Check();
    }
}


void CDynodeMan::CheckAndRemove(CConnman& connman)
{
    if (!dynodeSync.IsDynodeListSynced())
        return;

    LogPrint("dynode", "CDynodeMan::CheckAndRemove\n");

    {
        // Need LOCK2 here to ensure consistent locking order because code below locks cs_main
        // in CheckDnbAndUpdateDynodeList()
        LOCK2(cs_main, cs);

        Check();

        // Remove spent Dynodes, prepare structures and make requests to reasure the state of inactive ones
        rank_pair_vec_t vecDynodeRanks;
        // ask for up to DNB_RECOVERY_MAX_ASK_ENTRIES Dynode entries at a time
        int nAskForDnbRecovery = DNB_RECOVERY_MAX_ASK_ENTRIES;
        std::map<COutPoint, CDynode>::iterator it = mapDynodes.begin();
        while (it != mapDynodes.end()) {
            CDynodeBroadcast dnb = CDynodeBroadcast(it->second);
            uint256 hash = dnb.GetHash();
            // If collateral was spent ...
            if (it->second.IsOutpointSpent()) {
                LogPrint("dynode", "CDynodeMan::CheckAndRemove -- Removing Dynode: %s  addr=%s  %i now\n", it->second.GetStateString(), it->second.addr.ToString(), size() - 1);
                // erase all of the broadcasts we've seen from this txin, ...
                mapSeenDynodeBroadcast.erase(hash);
                mWeAskedForDynodeListEntry.erase(it->first);
                // and finally remove it from the list
                it->second.FlagGovernanceItemsAsDirty();
                mapDynodes.erase(it++);
                fDynodesRemoved = true;
            } else {
                bool fAsk = (nAskForDnbRecovery > 0) &&
                            dynodeSync.IsSynced() &&
                            it->second.IsNewStartRequired() &&
                            !IsDnbRecoveryRequested(hash) &&
                            !IsArgSet("-connect");
                if (fAsk) {
                    // this DN is in a non-recoverable state and we haven't asked other nodes yet
                    std::set<CService> setRequested;
                    // calulate only once and only when it's needed
                    if (vecDynodeRanks.empty()) {
                        int nRandomBlockHeight = GetRandInt(nCachedBlockHeight);
                        GetDynodeRanks(vecDynodeRanks, nRandomBlockHeight);
                    }
                    bool fAskedForDnbRecovery = false;
                    // ask first DNB_RECOVERY_QUORUM_TOTAL Dynodes we can connect to and we haven't asked recently
                    for (int i = 0; setRequested.size() < DNB_RECOVERY_QUORUM_TOTAL && i < (int)vecDynodeRanks.size(); i++) {
                        // avoid banning
                        if (mWeAskedForDynodeListEntry.count(it->first) && mWeAskedForDynodeListEntry[it->first].count(vecDynodeRanks[i].second.addr))
                            continue;
                        // didn't ask recently, ok to ask now
                        CService addr = vecDynodeRanks[i].second.addr;
                        setRequested.insert(addr);
                        listScheduledDnbRequestConnections.push_back(std::make_pair(addr, hash));
                        fAskedForDnbRecovery = true;
                    }
                    if (fAskedForDnbRecovery) {
                        LogPrint("dynode", "CDynodeMan::CheckAndRemove -- Recovery initiated, dynode=%s\n", it->first.ToStringShort());
                        nAskForDnbRecovery--;
                    }
                    // wait for dnb recovery replies for DNB_RECOVERY_WAIT_SECONDS seconds
                    mDnbRecoveryRequests[hash] = std::make_pair(GetTime() + DNB_RECOVERY_WAIT_SECONDS, setRequested);
                }
                ++it;
            }
        }
        // proces replies for DYNODE_NEW_START_REQUIRED Dynodes
        LogPrint("dynode", "CDynodeMan::CheckAndRemove -- mDnbRecoveryGoodReplies size=%d\n", (int)mDnbRecoveryGoodReplies.size());
        std::map<uint256, std::vector<CDynodeBroadcast> >::iterator itDnbReplies = mDnbRecoveryGoodReplies.begin();
        while (itDnbReplies != mDnbRecoveryGoodReplies.end()) {
            if (mDnbRecoveryRequests[itDnbReplies->first].first < GetTime()) {
                // all nodes we asked should have replied now
                if (itDnbReplies->second.size() >= DNB_RECOVERY_QUORUM_REQUIRED) {
                    // majority of nodes we asked agrees that this DN doesn't require new dnb, reprocess one of new dnbs
                    LogPrint("dynode", "CDynodeMan::CheckAndRemove -- reprocessing dnb, Dynode=%s\n", itDnbReplies->second[0].outpoint.ToStringShort());
                    // mapSeenDynodeBroadcast.erase(itDnbReplies->first);
                    int nDos;
                    itDnbReplies->second[0].fRecovery = true;
                    CheckDnbAndUpdateDynodeList(nullptr, itDnbReplies->second[0], nDos, connman);
                }
                LogPrint("dynode", "CDynodeMan::CheckAndRemove -- removing dnb recovery reply, Dynode=%s, size=%d\n", itDnbReplies->second[0].outpoint.ToStringShort(), (int)itDnbReplies->second.size());
                mDnbRecoveryGoodReplies.erase(itDnbReplies++);
            } else {
                ++itDnbReplies;
            }
        }
    }
    {
        // no need for cm_main below
        LOCK(cs);

        auto itDnbRequest = mDnbRecoveryRequests.begin();
        while (itDnbRequest != mDnbRecoveryRequests.end()) {
            // Allow this dnb to be re-verified again after DNB_RECOVERY_RETRY_SECONDS seconds
            // if DN is still in DYNODE_NEW_START_REQUIRED state.
            if (GetTime() - itDnbRequest->second.first > DNB_RECOVERY_RETRY_SECONDS) {
                mDnbRecoveryRequests.erase(itDnbRequest++);
            } else {
                ++itDnbRequest;
            }
        }

        // check who's asked for the Dynode list
        auto it1 = mAskedUsForDynodeList.begin();
        while (it1 != mAskedUsForDynodeList.end()) {
            if ((*it1).second < GetTime()) {
                mAskedUsForDynodeList.erase(it1++);
            } else {
                ++it1;
            }
        }

        // check who we asked for the Dynode list
        it1 = mWeAskedForDynodeList.begin();
        while (it1 != mWeAskedForDynodeList.end()) {
            if ((*it1).second < GetTime()) {
                mWeAskedForDynodeList.erase(it1++);
            } else {
                ++it1;
            }
        }

        // check which Dynodes we've asked for
        auto it2 = mWeAskedForDynodeListEntry.begin();
        while (it2 != mWeAskedForDynodeListEntry.end()) {
            auto it3 = it2->second.begin();
            while (it3 != it2->second.end()) {
                if (it3->second < GetTime()) {
                    it2->second.erase(it3++);
                } else {
                    ++it3;
                }
            }
            if (it2->second.empty()) {
                mWeAskedForDynodeListEntry.erase(it2++);
            } else {
                ++it2;
            }
        }

        auto it3 = mWeAskedForVerification.begin();
        while (it3 != mWeAskedForVerification.end()) {
            if (it3->second.nBlockHeight < nCachedBlockHeight - MAX_POSE_BLOCKS) {
                mWeAskedForVerification.erase(it3++);
            } else {
                ++it3;
            }
        }

        // NOTE: do not expire mapSeenDynodeBroadcast entries here, clean them on dnb updates!

        // remove expired mapSeenDynodePing
        std::map<uint256, CDynodePing>::iterator it4 = mapSeenDynodePing.begin();
        while (it4 != mapSeenDynodePing.end()) {
            if ((*it4).second.IsExpired()) {
                LogPrint("dynode", "CDynodeMan::CheckAndRemove -- Removing expired Dynode ping: hash=%s\n", (*it4).second.GetHash().ToString());
                mapSeenDynodePing.erase(it4++);
            } else {
                ++it4;
            }
        }

        // remove expired mapSeenDynodeVerification
        std::map<uint256, CDynodeVerification>::iterator itv2 = mapSeenDynodeVerification.begin();
        while (itv2 != mapSeenDynodeVerification.end()) {
            if ((*itv2).second.nBlockHeight < nCachedBlockHeight - MAX_POSE_BLOCKS) {
                LogPrint("dynode", "CDynodeMan::CheckAndRemove -- Removing expired Dynode verification: hash=%s\n", (*itv2).first.ToString());
                mapSeenDynodeVerification.erase(itv2++);
            } else {
                ++itv2;
            }
        }

        LogPrint("dynode", "CDynodeMan::CheckAndRemove -- %s\n", ToString());
    }

    if (fDynodesRemoved) {
        NotifyDynodeUpdates(connman);
    }
}

void CDynodeMan::Clear()
{
    LOCK(cs);
    mapDynodes.clear();
    mAskedUsForDynodeList.clear();
    mWeAskedForDynodeList.clear();
    mWeAskedForDynodeListEntry.clear();
    mapSeenDynodeBroadcast.clear();
    mapSeenDynodePing.clear();
    nPsqCount = 0;
    nLastSentinelPingTime = 0;
}

int CDynodeMan::CountDynodes(int nProtocolVersion)
{
    LOCK(cs);
    int nCount = 0;
    nProtocolVersion = nProtocolVersion == -1 ? dnpayments.GetMinDynodePaymentsProto() : nProtocolVersion;

    for (auto& dnpair : mapDynodes) {
        if (dnpair.second.nProtocolVersion < nProtocolVersion)
            continue;
        nCount++;
    }

    return nCount;
}

int CDynodeMan::CountEnabled(int nProtocolVersion)
{
    LOCK(cs);
    int nCount = 0;
    nProtocolVersion = nProtocolVersion == -1 ? dnpayments.GetMinDynodePaymentsProto() : nProtocolVersion;

    for (auto& dnpair : mapDynodes) {
        if (dnpair.second.nProtocolVersion < nProtocolVersion || !dnpair.second.IsEnabled())
            continue;
        nCount++;
    }

    return nCount;
}

/* Only IPv4 Dynodes are allowed, saving this for later
int CDynodeMan::CountByIP(int nNetworkType)
{
    LOCK(cs);
    int nNodeCount = 0;

    for (auto& dnpair : mapDynodes)
        if ((nNetworkType == NET_IPV4 && dnpair.second.addr.IsIPv4()) ||
            (nNetworkType == NET_TOR  && dnpair.second.addr.IsTor())  ||
            (nNetworkType == NET_IPV6 && dnpair.second.addr.IsIPv6())) {
                nNodeCount++;
        }

    return nNodeCount;
}
*/

void CDynodeMan::PsegUpdate(CNode* pnode, CConnman& connman)
{
    CNetMsgMaker msgMaker(pnode->GetSendVersion());
    LOCK(cs);

    CService addrSquashed = Params().AllowMultiplePorts() ? (CService)pnode->addr : CService(pnode->addr, 0);
    if (Params().NetworkIDString() == CBaseChainParams::MAIN) {
        if (!(pnode->addr.IsRFC1918() || pnode->addr.IsLocal())) {
            auto it = mWeAskedForDynodeList.find(addrSquashed);
            if (it != mWeAskedForDynodeList.end() && GetTime() < (*it).second) {
                LogPrintf("CDynodeMan::PsegUpdate -- we already asked %s for the list; skipping...\n", addrSquashed.ToString());
                return;
            }
        }
    }

    if (pnode->GetSendVersion() == 70900) {
        connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PSEG, CTxIn()));
    } else {
        connman.PushMessage(pnode, msgMaker.Make(NetMsgType::PSEG, COutPoint()));
    }
    int64_t askAgain = GetTime() + PSEG_UPDATE_SECONDS;
    mWeAskedForDynodeList[pnode->addr] = askAgain;

    LogPrint("dynode", "CDynodeMan::PsegUpdate -- asked %s for the list\n", pnode->addr.ToString());
}

CDynode* CDynodeMan::Find(const COutPoint& outpoint)
{
    LOCK(cs);
    auto it = mapDynodes.find(outpoint);
    return it == mapDynodes.end() ? nullptr : &(it->second);
}

bool CDynodeMan::Get(const COutPoint& outpoint, CDynode& dynodeRet)
{
    // Theses mutexes are recursive so double locking by the same thread is safe.
    LOCK(cs);
    auto it = mapDynodes.find(outpoint);
    if (it == mapDynodes.end()) {
        return false;
    }

    dynodeRet = it->second;
    return true;
}

bool CDynodeMan::GetDynodeInfo(const COutPoint& outpoint, dynode_info_t& dnInfoRet)
{
    LOCK(cs);
    auto it = mapDynodes.find(outpoint);
    if (it == mapDynodes.end()) {
        return false;
    }
    dnInfoRet = it->second.GetInfo();
    return true;
}

bool CDynodeMan::GetDynodeInfo(const CPubKey& pubKeyDynode, dynode_info_t& dnInfoRet)
{
    LOCK(cs);
    for (auto& dnpair : mapDynodes) {
        if (dnpair.second.pubKeyDynode == pubKeyDynode) {
            dnInfoRet = dnpair.second.GetInfo();
            return true;
        }
    }
    return false;
}

bool CDynodeMan::GetDynodeInfo(const CScript& payee, dynode_info_t& dnInfoRet)
{
    LOCK(cs);
    for (const auto& dnpair : mapDynodes) {
        CScript scriptCollateralAddress = GetScriptForDestination(dnpair.second.pubKeyCollateralAddress.GetID());
        if (scriptCollateralAddress == payee) {
            dnInfoRet = dnpair.second.GetInfo();
            return true;
        }
    }
    return false;
}

bool CDynodeMan::Has(const COutPoint& outpoint)
{
    LOCK(cs);
    return mapDynodes.find(outpoint) != mapDynodes.end();
}

//
// Deterministically select the oldest/best Dynode to pay on the network
//
bool CDynodeMan::GetNextDynodeInQueueForPayment(bool fFilterSigTime, int& nCountRet, dynode_info_t& dnInfoRet)
{
    return GetNextDynodeInQueueForPayment(nCachedBlockHeight, fFilterSigTime, nCountRet, dnInfoRet);
}

bool CDynodeMan::GetNextDynodeInQueueForPayment(int nBlockHeight, bool fFilterSigTime, int& nCountRet, dynode_info_t& dnInfoRet)
{
    dnInfoRet = dynode_info_t();
    nCountRet = 0;

    if (!dynodeSync.IsWinnersListSynced()) {
        // without winner list we can't reliably find the next winner anyway
        return false;
    }

    // Need LOCK2 here to ensure consistent locking order because the GetBlockHash call below locks cs_main
    LOCK2(cs_main, cs);

    std::vector<std::pair<int, const CDynode*> > vecDynodeLastPaid;

    /*
        Make a vector with all of the last paid times
    */

    int nDnCount = CountDynodes();

    for (const auto& dnpair : mapDynodes) {
        if (!dnpair.second.IsValidForPayment())
            continue;

        // //check protocol version
        if (dnpair.second.nProtocolVersion < dnpayments.GetMinDynodePaymentsProto())
            continue;

        //it's in the list (up to 8 entries ahead of current block to allow propagation) -- so let's skip it
        if (dnpayments.IsScheduled(dnpair.second, nBlockHeight))
            continue;

        //it's too new, wait for a cycle
        if (fFilterSigTime && dnpair.second.sigTime + (nDnCount * 2.6 * 60) > GetAdjustedTime())
            continue;

        //make sure it has at least as many confirmations as there are Dynodes
        if (GetUTXOConfirmations(dnpair.first) < nDnCount)
            continue;

        vecDynodeLastPaid.push_back(std::make_pair(dnpair.second.GetLastPaidBlock(), &dnpair.second));
    }

    nCountRet = (int)vecDynodeLastPaid.size();

    //when the network is in the process of upgrading, don't penalize nodes that recently restarted
    if (fFilterSigTime && nCountRet < nDnCount / 3)
        return GetNextDynodeInQueueForPayment(nBlockHeight, false, nCountRet, dnInfoRet);

    // Sort them low to high
    sort(vecDynodeLastPaid.begin(), vecDynodeLastPaid.end(), CompareLastPaidBlock());

    uint256 blockHash;
    if (!GetBlockHash(blockHash, nBlockHeight - 101)) {
        LogPrintf("CDynode::GetNextDynodeInQueueForPayment -- ERROR: GetBlockHash() failed at nBlockHeight %d\n", nBlockHeight - 101);
        return false;
    }
    // Look at 1/10 of the oldest nodes (by last payment), calculate their scores and pay the best one
    //  -- This doesn't look at who is being paid in the +8-10 blocks, allowing for double payments very rarely
    //  -- 1/100 payments should be a double payment on mainnet - (1/(3000/10))*2
    //  -- (chance per block * chances before IsScheduled will fire)
    int nTenthNetwork = nDnCount / 10;
    int nCountTenth = 0;
    arith_uint256 nHighest = 0;
    const CDynode* pBestDynode = nullptr;
    for (const auto& s : vecDynodeLastPaid) {
        arith_uint256 nScore = s.second->CalculateScore(blockHash);
        if (nScore > nHighest) {
            nHighest = nScore;
            pBestDynode = s.second;
        }
        nCountTenth++;
        if (nCountTenth >= nTenthNetwork)
            break;
    }
    if (pBestDynode) {
        dnInfoRet = pBestDynode->GetInfo();
    }
    return dnInfoRet.fInfoValid;
}

dynode_info_t CDynodeMan::FindRandomNotInVec(const std::vector<COutPoint>& vecToExclude, int nProtocolVersion)
{
    LOCK(cs);

    nProtocolVersion = nProtocolVersion == -1 ? dnpayments.GetMinDynodePaymentsProto() : nProtocolVersion;

    int nCountEnabled = CountEnabled(nProtocolVersion);
    int nCountNotExcluded = nCountEnabled - vecToExclude.size();

    LogPrintf("CDynodeMan::FindRandomNotInVec -- %d enabled Dynodes, %d Dynodes to choose from\n", nCountEnabled, nCountNotExcluded);
    if (nCountNotExcluded < 1)
        return dynode_info_t();

    // fill a vector of pointers
    std::vector<const CDynode*> vpDynodesShuffled;
    for (const auto& dnpair : mapDynodes) {
        vpDynodesShuffled.push_back(&dnpair.second);
    }

    FastRandomContext insecure_rand;
    // shuffle pointers
    std::random_shuffle(vpDynodesShuffled.begin(), vpDynodesShuffled.end(), insecure_rand);
    bool fExclude;

    // loop through
    for (const auto& pdn : vpDynodesShuffled) {
        if (pdn->nProtocolVersion < nProtocolVersion || !pdn->IsEnabled())
            continue;
        fExclude = false;
        for (const auto& outpointToExclude : vecToExclude) {
            if (pdn->outpoint == outpointToExclude) {
                fExclude = true;
                break;
            }
        }
        if (fExclude)
            continue;
        // found the one not in vecToExclude
        LogPrint("dynode", "CDynodeMan::FindRandomNotInVec -- found, Dynode=%s\n", pdn->outpoint.ToStringShort());
        return pdn->GetInfo();
    }

    LogPrint("dynode", "CDynodeMan::FindRandomNotInVec -- failed\n");
    return dynode_info_t();
}

bool CDynodeMan::GetDynodeScores(const uint256& nBlockHash, CDynodeMan::score_pair_vec_t& vecDynodeScoresRet, int nMinProtocol)
{
    vecDynodeScoresRet.clear();

    if (!dynodeSync.IsDynodeListSynced())
        return false;

    AssertLockHeld(cs);

    if (mapDynodes.empty())
        return false;

    // calculate scores
    for (const auto& dnpair : mapDynodes) {
        if (dnpair.second.nProtocolVersion >= nMinProtocol) {
            vecDynodeScoresRet.push_back(std::make_pair(dnpair.second.CalculateScore(nBlockHash), &dnpair.second));
        }
    }

    sort(vecDynodeScoresRet.rbegin(), vecDynodeScoresRet.rend(), CompareScoreDN());
    return !vecDynodeScoresRet.empty();
}

bool CDynodeMan::GetDynodeRank(const COutPoint& outpoint, int& nRankRet, int nBlockHeight, int nMinProtocol)
{
    nRankRet = -1;

    if (!dynodeSync.IsDynodeListSynced())
        return false;

    // make sure we know about this block
    uint256 nBlockHash = uint256();
    if (!GetBlockHash(nBlockHash, nBlockHeight)) {
        LogPrintf("CDynodeMan::%s -- ERROR: GetBlockHash() failed at nBlockHeight %d\n", __func__, nBlockHeight);
        return false;
    }

    LOCK(cs);

    score_pair_vec_t vecDynodeScores;
    if (!GetDynodeScores(nBlockHash, vecDynodeScores, nMinProtocol))
        return false;

    int nRank = 0;
    for (const auto& scorePair : vecDynodeScores) {
        nRank++;
        if (scorePair.second->outpoint == outpoint) {
            nRankRet = nRank;
            return true;
        }
    }

    return false;
}

bool CDynodeMan::GetDynodeRanks(CDynodeMan::rank_pair_vec_t& vecDynodeRanksRet, int nBlockHeight, int nMinProtocol)
{
    vecDynodeRanksRet.clear();

    if (!dynodeSync.IsDynodeListSynced())
        return false;

    // make sure we know about this block
    uint256 nBlockHash = uint256();
    if (!GetBlockHash(nBlockHash, nBlockHeight)) {
        LogPrintf("CDynodeMan::%s -- ERROR: GetBlockHash() failed at nBlockHeight %d\n", __func__, nBlockHeight);
        return false;
    }

    LOCK(cs);

    score_pair_vec_t vecDynodeScores;
    if (!GetDynodeScores(nBlockHash, vecDynodeScores, nMinProtocol))
        return false;

    int nRank = 0;
    for (const auto& scorePair : vecDynodeScores) {
        nRank++;
        vecDynodeRanksRet.push_back(std::make_pair(nRank, *scorePair.second));
    }

    return true;
}

void CDynodeMan::ProcessDynodeConnections(CConnman& connman)
{
    //we don't care about this for regtest
    if (Params().NetworkIDString() == CBaseChainParams::REGTEST)
        return;

    std::vector<dynode_info_t> vecDnInfo; // will be empty when no wallet
#ifdef ENABLE_WALLET
    privateSendClient.GetMixingDynodesInfo(vecDnInfo);
#endif // ENABLE_WALLET

    connman.ForEachNode(CConnman::AllNodes, [&vecDnInfo](CNode* pnode) {
        if (pnode->fDynode) {
#ifdef ENABLE_WALLET
            bool fFound = false;
            for (const auto& dnInfo : vecDnInfo) {
                if (pnode->addr == dnInfo.addr) {
                    fFound = true;
                    break;
                }
            }
            if (fFound)
                return; // do NOT disconnect mixing dynodes
#endif                  // ENABLE_WALLET
            LogPrintf("Closing Dynode connection: peer=%d, addr=%s\n", pnode->id, pnode->addr.ToString());
            pnode->fDisconnect = true;
        }
    });
}


std::pair<CService, std::set<uint256> > CDynodeMan::PopScheduledDnbRequestConnection()
{
    LOCK(cs);
    if (listScheduledDnbRequestConnections.empty()) {
        return std::make_pair(CService(), std::set<uint256>());
    }

    std::set<uint256> setResult;

    listScheduledDnbRequestConnections.sort();
    std::pair<CService, uint256> pairFront = listScheduledDnbRequestConnections.front();

    // squash hashes from requests with the same CService as the first one into setResult
    std::list<std::pair<CService, uint256> >::iterator it = listScheduledDnbRequestConnections.begin();
    while (it != listScheduledDnbRequestConnections.end()) {
        if (pairFront.first == it->first) {
            setResult.insert(it->second);
            it = listScheduledDnbRequestConnections.erase(it);
        } else {
            // since list is sorted now, we can be sure that there is no more hashes left
            // to ask for from this addr
            break;
        }
    }
    return std::make_pair(pairFront.first, setResult);
}

void CDynodeMan::ProcessPendingDnbRequests(CConnman& connman)
{
    std::pair<CService, std::set<uint256> > p = PopScheduledDnbRequestConnection();
    if (!(p.first == CService() || p.second.empty())) {
        if (connman.IsDynodeOrDisconnectRequested(p.first))
            return;
        mapPendingDNB.insert(std::make_pair(p.first, std::make_pair(GetTime(), p.second)));
        connman.AddPendingDynode(p.first);
    }

    std::map<CService, std::pair<int64_t, std::set<uint256> > >::iterator itPendingDNB = mapPendingDNB.begin();
    while (itPendingDNB != mapPendingDNB.end()) {
        bool fDone = connman.ForNode(itPendingDNB->first, [&](CNode* pnode) {
            // compile request vector
            std::vector<CInv> vToFetch;
            for (auto& nHash : itPendingDNB->second.second) {
                if (nHash != uint256()) {
                    vToFetch.push_back(CInv(MSG_DYNODE_ANNOUNCE, nHash));
                    LogPrint("dynode", "-- asking for dnb %s from addr=%s\n", nHash.ToString(), pnode->addr.ToString());
                }
            }

            // ask for data
            CNetMsgMaker msgMaker(pnode->GetSendVersion());
            connman.PushMessage(pnode, msgMaker.Make(NetMsgType::GETDATA, vToFetch));
            return true;
        });

        int64_t nTimeAdded = itPendingDNB->second.first;
        if (fDone || (GetTime() - nTimeAdded > 15)) {
            if (!fDone) {
                LogPrint("dynode", "CDynodeMan::%s -- failed to connect to %s\n", __func__, itPendingDNB->first.ToString());
            }
            mapPendingDNB.erase(itPendingDNB++);
        } else {
            ++itPendingDNB;
        }
    }
}

void CDynodeMan::ProcessMessage(CNode* pfrom, const std::string& strCommand, CDataStream& vRecv, CConnman& connman)
{
    if (fLiteMode)
        return; // disable all Dynamic specific functionality

    if (strCommand == NetMsgType::DNANNOUNCE) { //Dynode Broadcast

        CDynodeBroadcast dnb;
        vRecv >> dnb;

        pfrom->setAskFor.erase(dnb.GetHash());

        if (!dynodeSync.IsBlockchainSynced())
            return;

        LogPrint("dynode", "DNANNOUNCE -- Dynode announce, Dynode=%s\n", dnb.outpoint.ToStringShort());

        int nDos = 0;

        if (CheckDnbAndUpdateDynodeList(pfrom, dnb, nDos, connman)) {
            // use announced Dynode as a peer
            connman.AddNewAddress(CAddress(dnb.addr, NODE_NETWORK), pfrom->addr, 2 * 60 * 60);
        } else if (nDos > 0) {
            LOCK(cs_main);
            Misbehaving(pfrom->GetId(), nDos);
        }

        if (fDynodesAdded) {
            NotifyDynodeUpdates(connman);
        }
    } else if (strCommand == NetMsgType::DNPING) { //Dynode Ping

        CDynodePing dnp;
        vRecv >> dnp;

        uint256 nHash = dnp.GetHash();

        pfrom->setAskFor.erase(nHash);

        if (!dynodeSync.IsBlockchainSynced())
            return;

        LogPrint("dynode", "DNPING -- Dynode ping, Dynode=%s\n", dnp.dynodeOutpoint.ToStringShort());

        // Need LOCK2 here to ensure consistent locking order because the CheckAndUpdate call below locks cs_main
        LOCK2(cs_main, cs);

        if (mapSeenDynodePing.count(nHash))
            return; //seen
        mapSeenDynodePing.insert(std::make_pair(nHash, dnp));

        LogPrint("dynode", "DNPING -- Dynode ping, Dynode=%s new\n", dnp.dynodeOutpoint.ToStringShort());

        // see if we have this Dynode
        CDynode* pdn = Find(dnp.dynodeOutpoint);

        if (pdn && dnp.fSentinelIsCurrent)
            UpdateLastSentinelPingTime();

        // too late, new DNANNOUNCE is required
        if (pdn && pdn->IsNewStartRequired())
            return;

        int nDos = 0;
        if (dnp.CheckAndUpdate(pdn, false, nDos, connman))
            return;

        if (nDos > 0) {
            // if anything significant failed, mark that node
            Misbehaving(pfrom->GetId(), nDos);
        } else if (pdn != NULL) {
            // nothing significant failed, dn is a known one too
            return;
        }

        // something significant is broken or dn is unknown,
        // we might have to ask for a Dynode entry once
        AskForDN(pfrom, dnp.dynodeOutpoint, connman);

    } else if (strCommand == NetMsgType::PSEG) { //Get Dynode list or specific entry
        // Ignore such requests until we are fully synced.
        // We could start processing this after Dynode list is synced
        // but this is a heavy one so it's better to finish sync first.
        if (!dynodeSync.IsSynced())
            return;

        COutPoint dynodeOutpoint;

        if (pfrom->nVersion == 70900) {
            CTxIn vin;
            vRecv >> vin;
            dynodeOutpoint = vin.prevout;
        } else {
            vRecv >> dynodeOutpoint;
        }

        LogPrint("dynode", "PSEG -- Dynode list, Dynode=%s\n", dynodeOutpoint.ToStringShort());

        if (dynodeOutpoint.IsNull()) {
            SyncAll(pfrom, connman);
        } else {
            SyncSingle(pfrom, dynodeOutpoint, connman);
        }

    } else if (strCommand == NetMsgType::DNVERIFY) { // Dynode Verify

        // Need LOCK2 here to ensure consistent locking order because the all functions below call GetBlockHash which locks cs_main
        LOCK2(cs_main, cs);

        CDynodeVerification dnv;
        vRecv >> dnv;

        pfrom->setAskFor.erase(dnv.GetHash());

        if (!dynodeSync.IsDynodeListSynced())
            return;

        if (dnv.vchSig1.empty()) {
            // CASE 1: someone asked me to verify myself /IP we are using/
            SendVerifyReply(pfrom, dnv, connman);
        } else if (dnv.vchSig2.empty()) {
            // CASE 2: we _probably_ got verification we requested from some Dynode
            ProcessVerifyReply(pfrom, dnv);
        } else {
            // CASE 3: we _probably_ got verification broadcast signed by some Dynode which verified another one
            ProcessVerifyBroadcast(pfrom, dnv);
        }
    }
}

void CDynodeMan::SyncSingle(CNode* pnode, const COutPoint& outpoint, CConnman& connman)
{
    // do not provide any data until our node is synced
    if (!dynodeSync.IsSynced())
        return;

    LOCK(cs);

    auto it = mapDynodes.find(outpoint);

    if (it != mapDynodes.end()) {
        if (it->second.addr.IsRFC1918() || it->second.addr.IsLocal())
            return; // do not send local network Dynode
        // NOTE: send Dynode regardless of its current state, the other node will need it to verify old votes.
        LogPrint("dynode", "CDynodeMan::%s -- Sending Dynode entry: Dynode=%s  addr=%s\n", __func__, outpoint.ToStringShort(), it->second.addr.ToString());
        PushPsegInvs(pnode, it->second);
        LogPrintf("CDynodeMan::%s -- Sent 1 Dynode inv to peer=%d\n", __func__, pnode->id);
    }
}

void CDynodeMan::SyncAll(CNode* pnode, CConnman& connman)
{
    // do not provide any data until our node is synced
    if (!dynodeSync.IsSynced())
        return;

    // local network
    bool isLocal = (pnode->addr.IsRFC1918() || pnode->addr.IsLocal());

    CService addrSquashed = Params().AllowMultiplePorts() ? (CService)pnode->addr : CService(pnode->addr, 0);
    // should only ask for this once
    if (!isLocal && Params().NetworkIDString() == CBaseChainParams::MAIN) {
        LOCK2(cs_main, cs);
        auto it = mAskedUsForDynodeList.find(addrSquashed);
        if (it != mAskedUsForDynodeList.end() && it->second > GetTime()) {
            Misbehaving(pnode->GetId(), 34);
            LogPrintf("CDynodeMan::%s -- peer already asked me for the list, peer=%d\n", __func__, pnode->id);
            return;
        }
        int64_t askAgain = GetTime() + PSEG_UPDATE_SECONDS;
        mAskedUsForDynodeList[addrSquashed] = askAgain;
    }

    int nInvCount = 0;

    LOCK(cs);

    for (const auto& dnpair : mapDynodes) {
        if (Params().RequireRoutableExternalIP() &&
            (dnpair.second.addr.IsRFC1918() || dnpair.second.addr.IsLocal()))
            continue; // do not send local network Dynode
        // NOTE: send Dynode regardless of its current state, the other node will need it to verify old votes.
        LogPrint("dynode", "CDynodeMan::%s -- Sending Dynode entry: Dynode=%s  addr=%s\n", __func__, dnpair.first.ToStringShort(), dnpair.second.addr.ToString());
        PushPsegInvs(pnode, dnpair.second);
        nInvCount++;
    }

    connman.PushMessage(pnode, CNetMsgMaker(pnode->GetSendVersion()).Make(NetMsgType::SYNCSTATUSCOUNT, DYNODE_SYNC_LIST, nInvCount));
    LogPrintf("CDynodeMan::%s -- Sent %d Dynode invs to peer=%d\n", __func__, nInvCount, pnode->id);
}

void CDynodeMan::PushPsegInvs(CNode* pnode, const CDynode& dn)
{
    AssertLockHeld(cs);

    CDynodeBroadcast dnb(dn);
    CDynodePing dnp = dnb.lastPing;
    uint256 hashDNB = dnb.GetHash();
    uint256 hashDNP = dnp.GetHash();
    pnode->PushInventory(CInv(MSG_DYNODE_ANNOUNCE, hashDNB));
    pnode->PushInventory(CInv(MSG_DYNODE_PING, hashDNP));
    mapSeenDynodeBroadcast.insert(std::make_pair(hashDNB, std::make_pair(GetTime(), dnb)));
    mapSeenDynodePing.insert(std::make_pair(hashDNP, dnp));
}

// Verification of Dynode via unique direct requests.

void CDynodeMan::DoFullVerificationStep(CConnman& connman)
{
    if (activeDynode.outpoint == COutPoint())
        return;
    if (!dynodeSync.IsSynced())
        return;

    rank_pair_vec_t vecDynodeRanks;
    GetDynodeRanks(vecDynodeRanks, nCachedBlockHeight - 1, MIN_POSE_PROTO_VERSION);

    LOCK(cs);

    int nCount = 0;

    int nMyRank = -1;
    int nRanksTotal = (int)vecDynodeRanks.size();

    // send verify requests only if we are in top MAX_POSE_RANK
    for (auto& rankPair : vecDynodeRanks) {
        if (rankPair.first > MAX_POSE_RANK) {
            LogPrint("dynode", "CDynodeMan::DoFullVerificationStep -- Must be in top %d to send verify request\n",
                (int)MAX_POSE_RANK);
            return;
        }
        if (rankPair.second.outpoint == activeDynode.outpoint) {
            nMyRank = rankPair.first;
            LogPrint("dynode", "CDynodeMan::DoFullVerificationStep -- Found self at rank %d/%d, verifying up to %d Dynodes\n",
                nMyRank, nRanksTotal, (int)MAX_POSE_CONNECTIONS);
            break;
        }
    }

    // edge case: list is too short and this Dynode is not enabled
    if (nMyRank == -1)
        return;

    // send verify requests to up to MAX_POSE_CONNECTIONS Dynodes
    // starting from MAX_POSE_RANK + nMyRank and using MAX_POSE_CONNECTIONS as a step
    int nOffset = MAX_POSE_RANK + nMyRank - 1;
    if (nOffset >= (int)vecDynodeRanks.size())
        return;

    std::vector<const CDynode*> vSortedByAddr;
    for (const auto& dnpair : mapDynodes) {
        vSortedByAddr.push_back(&dnpair.second);
    }

    sort(vSortedByAddr.begin(), vSortedByAddr.end(), CompareByAddr());

    auto it = vecDynodeRanks.begin() + nOffset;
    while (it != vecDynodeRanks.end()) {
        if (it->second.IsPoSeVerified() || it->second.IsPoSeBanned()) {
            LogPrint("dynode", "CDynodeMan::DoFullVerificationStep -- Already %s%s%s Dynode %s address %s, skipping...\n",
                it->second.IsPoSeVerified() ? "verified" : "",
                it->second.IsPoSeVerified() && it->second.IsPoSeBanned() ? " and " : "",
                it->second.IsPoSeBanned() ? "banned" : "",
                it->second.outpoint.ToStringShort(), it->second.addr.ToString());
            nOffset += MAX_POSE_CONNECTIONS;
            if (nOffset >= (int)vecDynodeRanks.size())
                break;
            it += MAX_POSE_CONNECTIONS;
            continue;
        }
        LogPrint("dynode", "CDynodeMan::DoFullVerificationStep -- Verifying Dynode %s rank %d/%d address %s\n",
            it->second.outpoint.ToStringShort(), it->first, nRanksTotal, it->second.addr.ToString());
        if (SendVerifyRequest(CAddress(it->second.addr, NODE_NETWORK), vSortedByAddr, connman)) {
            nCount++;
            if (nCount >= MAX_POSE_CONNECTIONS)
                break;
        }
        nOffset += MAX_POSE_CONNECTIONS;
        if (nOffset >= (int)vecDynodeRanks.size())
            break;
        it += MAX_POSE_CONNECTIONS;
    }

    LogPrint("dynode", "CDynodeMan::DoFullVerificationStep -- Sent verification requests to %d Dynodes\n", nCount);
}

// This function tries to find Dynodes with the same addr,
// find a verified one and ban all the other. If there are many nodes
// with the same addr but none of them is verified yet, then none of them are banned.
// It could take many times to run this before most of the duplicate nodes are banned.

void CDynodeMan::CheckSameAddr()
{
    if (!dynodeSync.IsSynced() || mapDynodes.empty())
        return;

    std::vector<CDynode*> vBan;
    std::vector<CDynode*> vSortedByAddr;

    {
        LOCK(cs);

        CDynode* pprevDynode = nullptr;
        CDynode* pverifiedDynode = nullptr;

        for (auto& dnpair : mapDynodes) {
            vSortedByAddr.push_back(&dnpair.second);
        }

        sort(vSortedByAddr.begin(), vSortedByAddr.end(), CompareByAddr());

        for (const auto& pdn : vSortedByAddr) {
            // check only (pre)enabled Dynodes
            if (!pdn->IsEnabled() && !pdn->IsPreEnabled())
                continue;
            // initial step
            if (!pprevDynode) {
                pprevDynode = pdn;
                pverifiedDynode = pdn->IsPoSeVerified() ? pdn : nullptr;
                continue;
            }
            // second+ step
            if (pdn->addr == pprevDynode->addr) {
                if (pverifiedDynode) {
                    // another Dynode with the same ip is verified, ban this one
                    vBan.push_back(pdn);
                } else if (pdn->IsPoSeVerified()) {
                    // this Dynode with the same ip is verified, ban previous one
                    vBan.push_back(pprevDynode);
                    // and keep a reference to be able to ban following Dynodes with the same ip
                    pverifiedDynode = pdn;
                }
            } else {
                pverifiedDynode = pdn->IsPoSeVerified() ? pdn : nullptr;
            }
            pprevDynode = pdn;
        }
    }

    // ban duplicates
    for (auto& pdn : vBan) {
        LogPrintf("CDynodeMan::CheckSameAddr -- increasing PoSe ban score for Dynode %s\n", pdn->outpoint.ToStringShort());
        pdn->IncreasePoSeBanScore();
    }
}

bool CDynodeMan::SendVerifyRequest(const CAddress& addr, const std::vector<const CDynode*>& vSortedByAddr, CConnman& connman)
{
    if (netfulfilledman.HasFulfilledRequest(addr, strprintf("%s", NetMsgType::DNVERIFY) + "-request")) {
        // we already asked for verification, not a good idea to do this too often, skip it
        LogPrint("dynode", "CDynodeMan::SendVerifyRequest -- too many requests, skipping... addr=%s\n", addr.ToString());
        return false;
    }

    if (connman.IsDynodeOrDisconnectRequested(addr))
        return false;

    connman.AddPendingDynode(addr);
    // use random nonce, store it and require node to reply with correct one later
    CDynodeVerification dnv(addr, GetRandInt(999999), nCachedBlockHeight - 1);
    LOCK(cs_mapPendingDNV);
    mapPendingDNV.insert(std::make_pair(addr, std::make_pair(GetTime(), dnv)));
    LogPrintf("CDynodeMan::SendVerifyRequest -- verifying node using nonce %d addr=%s\n", dnv.nonce, addr.ToString());
    return true;
}

void CDynodeMan::ProcessPendingDnvRequests(CConnman& connman)
{
    LOCK(cs_mapPendingDNV);

    std::map<CService, std::pair<int64_t, CDynodeVerification> >::iterator itPendingDNV = mapPendingDNV.begin();

    while (itPendingDNV != mapPendingDNV.end()) {
        bool fDone = connman.ForNode(itPendingDNV->first, [&](CNode* pnode) {
            netfulfilledman.AddFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::DNVERIFY) + "-request");
            // use random nonce, store it and require node to reply with correct one later
            mWeAskedForVerification[pnode->addr] = itPendingDNV->second.second;
            LogPrint("dynode", "-- verifying node using nonce %d addr=%s\n", itPendingDNV->second.second.nonce, pnode->addr.ToString());
            CNetMsgMaker msgMaker(pnode->GetSendVersion()); // TODO this gives a warning about version not being set (we should wait for VERSION exchange)
            connman.PushMessage(pnode, msgMaker.Make(NetMsgType::DNVERIFY, itPendingDNV->second.second));
            return true;
        });

        int64_t nTimeAdded = itPendingDNV->second.first;
        if (fDone || (GetTime() - nTimeAdded > 15)) {
            if (!fDone) {
                LogPrint("dynode", "CDynodeMan::%s -- failed to connect to %s\n", __func__, itPendingDNV->first.ToString());
            }
            mapPendingDNV.erase(itPendingDNV++);
        } else {
            ++itPendingDNV;
        }
    }
}

void CDynodeMan::SendVerifyReply(CNode* pnode, CDynodeVerification& dnv, CConnman& connman)
{
    AssertLockHeld(cs_main);

    // only Dynodes can sign this, why would someone ask regular node?
    if (!fDynodeMode) {
        // do not ban, malicious node might be using my IP
        // and trying to confuse the node which tries to verify it
        return;
    }

    if (netfulfilledman.HasFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::DNVERIFY) + "-reply")) {
        // peer should not ask us that often
        LogPrintf("CDynodeMan::SendVerifyReply -- ERROR: peer already asked me recently, peer=%d\n", pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    uint256 blockHash;
    if (!GetBlockHash(blockHash, dnv.nBlockHeight)) {
        LogPrintf("CDynodeMan::SendVerifyReply -- can't get block hash for unknown block height %d, peer=%d\n", dnv.nBlockHeight, pnode->id);
        return;
    }

    std::string strError;

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = dnv.GetSignatureHash1(blockHash);

        if (!CHashSigner::SignHash(hash, activeDynode.keyDynode, dnv.vchSig1)) {
            LogPrintf("CDynodeMan::SendVerifyReply -- SignHash() failed\n");
            return;
        }

        if (!CHashSigner::VerifyHash(hash, activeDynode.pubKeyDynode, dnv.vchSig1, strError)) {
            LogPrintf("CDynodeMan::SendVerifyReply -- VerifyHash() failed, error: %s\n", strError);
            return;
        }
    } else {
        std::string strMessage = strprintf("%s%d%s", activeDynode.service.ToString(false), dnv.nonce, blockHash.ToString());

        if (!CMessageSigner::SignMessage(strMessage, dnv.vchSig1, activeDynode.keyDynode)) {
            LogPrintf("CDynodeMan::SendVerifyReply -- SignMessage() failed\n");
            return;
        }

        if (!CMessageSigner::VerifyMessage(activeDynode.pubKeyDynode, dnv.vchSig1, strMessage, strError)) {
            LogPrintf("CDynodeMan::SendVerifyReply -- VerifyMessage() failed, error: %s\n", strError);
            return;
        }
    }

    CNetMsgMaker msgMaker(pnode->GetSendVersion());
    connman.PushMessage(pnode, msgMaker.Make(NetMsgType::DNVERIFY, dnv));
    netfulfilledman.AddFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::DNVERIFY) + "-reply");
}

void CDynodeMan::ProcessVerifyReply(CNode* pnode, CDynodeVerification& dnv)
{
    AssertLockHeld(cs_main);

    std::string strError;

    // did we even ask for it? if that's the case we should have matching fulfilled request
    if (!netfulfilledman.HasFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::DNVERIFY) + "-request")) {
        LogPrintf("CDynodeMan::ProcessVerifyReply -- ERROR: we didn't ask for verification of %s, peer=%d\n", pnode->addr.ToString(), pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    // Received nonce for a known address must match the one we sent
    if (mWeAskedForVerification[pnode->addr].nonce != dnv.nonce) {
        LogPrintf("CDynodeMan::ProcessVerifyReply -- ERROR: wrong nounce: requested=%d, received=%d, peer=%d\n",
            mWeAskedForVerification[pnode->addr].nonce, dnv.nonce, pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    // Received nBlockHeight for a known address must match the one we sent
    if (mWeAskedForVerification[pnode->addr].nBlockHeight != dnv.nBlockHeight) {
        LogPrintf("CDynodeMan::ProcessVerifyReply -- ERROR: wrong nBlockHeight: requested=%d, received=%d, peer=%d\n",
            mWeAskedForVerification[pnode->addr].nBlockHeight, dnv.nBlockHeight, pnode->id);
        Misbehaving(pnode->id, 20);
        return;
    }

    uint256 blockHash;
    if (!GetBlockHash(blockHash, dnv.nBlockHeight)) {
        // this shouldn't happen...
        LogPrintf("DynodeMan::ProcessVerifyReply -- can't get block hash for unknown block height %d, peer=%d\n", dnv.nBlockHeight, pnode->id);
        return;
    }

    // we already verified this address, why node is spamming?
    if (netfulfilledman.HasFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::DNVERIFY) + "-done")) {
        LogPrintf("CDynodeMan::ProcessVerifyReply -- ERROR: already verified %s recently\n", pnode->addr.ToString());
        Misbehaving(pnode->id, 20);
        return;
    }

    {
        LOCK(cs);

        CDynode* prealDynode = nullptr;
        std::vector<CDynode*> vpDynodesToBan;

        uint256 hash1 = dnv.GetSignatureHash1(blockHash);
        std::string strMessage1 = strprintf("%s%d%s", pnode->addr.ToString(false), dnv.nonce, blockHash.ToString());

        for (auto& dnpair : mapDynodes) {
            if (CAddress(dnpair.second.addr, NODE_NETWORK) == pnode->addr) {
                bool fFound = false;
                if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
                    fFound = CHashSigner::VerifyHash(hash1, dnpair.second.pubKeyDynode, dnv.vchSig1, strError);
                    // we don't care about dnv with signature in old format
                } else {
                    fFound = CMessageSigner::VerifyMessage(dnpair.second.pubKeyDynode, dnv.vchSig1, strMessage1, strError);
                }
                if (fFound) {
                    // found it!
                    prealDynode = &dnpair.second;
                    if (!dnpair.second.IsPoSeVerified()) {
                        dnpair.second.DecreasePoSeBanScore();
                    }
                    netfulfilledman.AddFulfilledRequest(pnode->addr, strprintf("%s", NetMsgType::DNVERIFY) + "-done");

                    // we can only broadcast it if we are an activated dynode
                    if (activeDynode.outpoint.IsNull())
                        continue;
                    // update ...
                    dnv.addr = dnpair.second.addr;
                    dnv.dynodeOutpoint1 = dnpair.second.outpoint;
                    dnv.dynodeOutpoint2 = activeDynode.outpoint;
                    // ... and sign it
                    std::string strError;

                    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
                        uint256 hash2 = dnv.GetSignatureHash2(blockHash);

                        if (!CHashSigner::SignHash(hash2, activeDynode.keyDynode, dnv.vchSig2)) {
                            LogPrintf("DynodeMan::ProcessVerifyReply -- SignHash() failed\n");
                            return;
                        }

                        if (!CHashSigner::VerifyHash(hash2, activeDynode.pubKeyDynode, dnv.vchSig2, strError)) {
                            LogPrintf("DynodeMan::ProcessVerifyReply -- VerifyHash() failed, error: %s\n", strError);
                            return;
                        }
                    } else {
                        std::string strMessage2 = strprintf("%s%d%s%s%s", dnv.addr.ToString(false), dnv.nonce, blockHash.ToString(),
                            dnv.dynodeOutpoint1.ToStringShort(), dnv.dynodeOutpoint2.ToStringShort());

                        if (!CMessageSigner::SignMessage(strMessage2, dnv.vchSig2, activeDynode.keyDynode)) {
                            LogPrintf("DynodeMan::ProcessVerifyReply -- SignMessage() failed\n");
                            return;
                        }

                        if (!CMessageSigner::VerifyMessage(activeDynode.pubKeyDynode, dnv.vchSig2, strMessage2, strError)) {
                            LogPrintf("DynodeMan::ProcessVerifyReply -- VerifyMessage() failed, error: %s\n", strError);
                            return;
                        }
                    }

                    mWeAskedForVerification[pnode->addr] = dnv;
                    mapSeenDynodeVerification.insert(std::make_pair(dnv.GetHash(), dnv));
                    dnv.Relay();

                } else {
                    vpDynodesToBan.push_back(&dnpair.second);
                }
            }
        }
        // no real Dynode found?...
        if (!prealDynode) {
            // this should never be the case normally,
            // only if someone is trying to game the system in some way or smth like that
            LogPrintf("CDynodeMan::ProcessVerifyReply -- ERROR: no real Dynode found for addr %s\n", pnode->addr.ToString());
            Misbehaving(pnode->id, 20);
            return;
        }
        LogPrintf("CDynodeMan::ProcessVerifyReply -- verified real Dynode %s for addr %s\n",
            prealDynode->outpoint.ToStringShort(), pnode->addr.ToString());
        // increase ban score for everyone else
        for (const auto& pdn : vpDynodesToBan) {
            pdn->IncreasePoSeBanScore();
            LogPrint("dynode", "CDynodeMan::ProcessVerifyReply -- increased PoSe ban score for %s addr %s, new score %d\n",
                prealDynode->outpoint.ToStringShort(), pnode->addr.ToString(), pdn->nPoSeBanScore);
        }
        if (!vpDynodesToBan.empty())
            LogPrintf("CDynodeMan::ProcessVerifyReply -- PoSe score increased for %d fake Dynodes, addr %s\n",
                (int)vpDynodesToBan.size(), pnode->addr.ToString());
    }
}

void CDynodeMan::ProcessVerifyBroadcast(CNode* pnode, const CDynodeVerification& dnv)
{
    AssertLockHeld(cs_main);

    std::string strError;

    if (mapSeenDynodeVerification.find(dnv.GetHash()) != mapSeenDynodeVerification.end()) {
        // we already have one
        return;
    }
    mapSeenDynodeVerification[dnv.GetHash()] = dnv;

    // we don't care about history
    if (dnv.nBlockHeight < nCachedBlockHeight - MAX_POSE_BLOCKS) {
        LogPrint("dynode", "CDynodeMan::ProcessVerifyBroadcast -- Outdated: current block %d, verification block %d, peer=%d\n",
            nCachedBlockHeight, dnv.nBlockHeight, pnode->id);
        return;
    }

    if (dnv.dynodeOutpoint1 == dnv.dynodeOutpoint2) {
        LogPrint("dynode", "CDynodeMan::ProcessVerifyBroadcast -- ERROR: same outpoints %s, peer=%d\n",
            dnv.dynodeOutpoint1.ToStringShort(), pnode->id);
        // that was NOT a good idea to cheat and verify itself,
        // ban the node we received such message from
        Misbehaving(pnode->id, 100);
        return;
    }

    uint256 blockHash;
    if (!GetBlockHash(blockHash, dnv.nBlockHeight)) {
        // this shouldn't happen...
        LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- Can't get block hash for unknown block height %d, peer=%d\n", dnv.nBlockHeight, pnode->id);
        return;
    }

    int nRank;

    if (!GetDynodeRank(dnv.dynodeOutpoint2, nRank, dnv.nBlockHeight, MIN_POSE_PROTO_VERSION)) {
        LogPrint("dynode", "CDynodeMan::ProcessVerifyBroadcast -- Can't calculate rank for Dynode %s\n",
            dnv.dynodeOutpoint2.ToStringShort());
        return;
    }

    if (nRank > MAX_POSE_RANK) {
        LogPrint("dynode", "CDynodeMan::ProcessVerifyBroadcast -- Dynode %s is not in top %d, current rank %d, peer=%d\n",
            dnv.dynodeOutpoint2.ToStringShort(), (int)MAX_POSE_RANK, nRank, pnode->id);
        return;
    }

    {
        LOCK(cs);

        CDynode* pdn1 = Find(dnv.dynodeOutpoint1);
        if (!pdn1) {
            LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- can't find Dynode1 %s\n", dnv.dynodeOutpoint1.ToStringShort());
            return;
        }

        CDynode* pdn2 = Find(dnv.dynodeOutpoint2);
        if (!pdn2) {
            LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- can't find Dynode %s\n", dnv.dynodeOutpoint2.ToStringShort());
            return;
        }

        if (pdn1->addr != dnv.addr) {
            LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- addr %s does not match %s\n", dnv.addr.ToString(), pdn1->addr.ToString());
            return;
        }

        if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
            uint256 hash1 = dnv.GetSignatureHash1(blockHash);
            uint256 hash2 = dnv.GetSignatureHash2(blockHash);

            if (!CHashSigner::VerifyHash(hash1, pdn1->pubKeyDynode, dnv.vchSig1, strError)) {
                LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- VerifyHash() failed, error: %s\n", strError);
                return;
            }

            if (!CHashSigner::VerifyHash(hash2, pdn2->pubKeyDynode, dnv.vchSig2, strError)) {
                LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- VerifyHash() failed, error: %s\n", strError);
                return;
            }
        } else {
            std::string strMessage1 = strprintf("%s%d%s", dnv.addr.ToString(false), dnv.nonce, blockHash.ToString());
            std::string strMessage2 = strprintf("%s%d%s%s%s", dnv.addr.ToString(false), dnv.nonce, blockHash.ToString(),
                dnv.dynodeOutpoint1.ToStringShort(), dnv.dynodeOutpoint2.ToStringShort());

            if (!CMessageSigner::VerifyMessage(pdn1->pubKeyDynode, dnv.vchSig1, strMessage1, strError)) {
                LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- VerifyMessage() for dynode1 failed, error: %s\n", strError);
                return;
            }

            if (!CMessageSigner::VerifyMessage(pdn2->pubKeyDynode, dnv.vchSig2, strMessage2, strError)) {
                LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- VerifyMessage() for dynode2 failed, error: %s\n", strError);
                return;
            }
        }

        if (!pdn1->IsPoSeVerified()) {
            pdn1->DecreasePoSeBanScore();
        }
        dnv.Relay();

        LogPrintf("CDynodeMan::ProcessVerifyBroadcast -- verified Dynode %s for addr %s\n",
            pdn1->outpoint.ToStringShort(), pdn1->addr.ToString());

        // increase ban score for everyone else with the same addr
        int nCount = 0;
        for (auto& dnpair : mapDynodes) {
            if (dnpair.second.addr != dnv.addr || dnpair.first == dnv.dynodeOutpoint1)
                continue;
            dnpair.second.IncreasePoSeBanScore();
            nCount++;
            LogPrint("dynode", "CDynodeMan::ProcessVerifyBroadcast -- increased PoSe ban score for %s addr %s, new score %d\n",
                dnpair.first.ToStringShort(), dnpair.second.addr.ToString(), dnpair.second.nPoSeBanScore);
        }
        if (nCount)
            LogPrint("dynode", "CDynodeMan::ProcessVerifyBroadcast -- PoSe score increased for %d fake Dynodes, addr %s\n",
                nCount, pdn1->addr.ToString());
    }
}

std::string CDynodeMan::ToString() const
{
    std::ostringstream info;

    info << "Dynodes: " << (int)mapDynodes.size() << ", peers who asked us for Dynode list: " << (int)mAskedUsForDynodeList.size() << ", peers we asked for Dynode list: " << (int)mWeAskedForDynodeList.size() << ", entries in Dynode list we asked for: " << (int)mWeAskedForDynodeListEntry.size() << ", nPsqCount: " << (int)nPsqCount;

    return info.str();
}

bool CDynodeMan::CheckDnbAndUpdateDynodeList(CNode* pfrom, CDynodeBroadcast dnb, int& nDos, CConnman& connman)
{
    // Need to lock cs_main here to ensure consistent locking order because the SimpleCheck call below locks cs_main
    LOCK(cs_main);

    {
        LOCK(cs);
        nDos = 0;
        LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dynode=%s\n", dnb.outpoint.ToStringShort());

        uint256 hash = dnb.GetHash();
        if (mapSeenDynodeBroadcast.count(hash) && !dnb.fRecovery) { //seen
            LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dynode=%s seen\n", dnb.outpoint.ToStringShort());
            // less then 2 pings left before this DN goes into non-recoverable state, bump sync timeout
            if (GetTime() - mapSeenDynodeBroadcast[hash].first > DYNODE_NEW_START_REQUIRED_SECONDS - DYNODE_MIN_DNP_SECONDS * 2) {
                LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dynode=%s seen update\n", dnb.outpoint.ToStringShort());
                mapSeenDynodeBroadcast[hash].first = GetTime();
                dynodeSync.BumpAssetLastTime("CDynodeMan::CheckDnbAndUpdateDynodeList - seen");
            }
            // did we ask this node for it?
            if (pfrom && IsDnbRecoveryRequested(hash) && GetTime() < mDnbRecoveryRequests[hash].first) {
                LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dnb=%s seen request\n", hash.ToString());
                if (mDnbRecoveryRequests[hash].second.count(pfrom->addr)) {
                    LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dnb=%s seen request, addr=%s\n", hash.ToString(), pfrom->addr.ToString());
                    // do not allow node to send same dnb multiple times in recovery mode
                    mDnbRecoveryRequests[hash].second.erase(pfrom->addr);
                    // does it have newer lastPing?
                    if (dnb.lastPing.sigTime > mapSeenDynodeBroadcast[hash].second.lastPing.sigTime) {
                        // simulate Check
                        CDynode dnTemp = CDynode(dnb);
                        dnTemp.Check();
                        LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dnb=%s seen request, addr=%s, better lastPing: %d min ago, projected dn state: %s\n", hash.ToString(), pfrom->addr.ToString(), (GetAdjustedTime() - dnb.lastPing.sigTime) / 60, dnTemp.GetStateString());
                        if (dnTemp.IsValidStateForAutoStart(dnTemp.nActiveState)) {
                            // this node thinks it's a good one
                            LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dynode=%s seen good\n", dnb.outpoint.ToStringShort());
                            mDnbRecoveryGoodReplies[hash].push_back(dnb);
                        }
                    }
                }
            }
            return true;
        }
        mapSeenDynodeBroadcast.insert(std::make_pair(hash, std::make_pair(GetTime(), dnb)));

        LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- dynode=%s new\n", dnb.outpoint.ToStringShort());

        if (!dnb.SimpleCheck(nDos)) {
            LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- SimpleCheck() failed, dynode=%s\n", dnb.outpoint.ToStringShort());
            return false;
        }

        // search Dynode list
        CDynode* pdn = Find(dnb.outpoint);
        if (pdn) {
            CDynodeBroadcast dnbOld = mapSeenDynodeBroadcast[CDynodeBroadcast(*pdn).GetHash()].second;
            if (!dnb.Update(pdn, nDos, connman)) {
                LogPrint("dynode", "CDynodeMan::CheckDnbAndUpdateDynodeList -- Update() failed, dynode=%s\n", dnb.outpoint.ToStringShort());
                return false;
            }
            if (hash != dnbOld.GetHash()) {
                mapSeenDynodeBroadcast.erase(dnbOld.GetHash());
            }
            return true;
        }
    }

    if (dnb.CheckOutpoint(nDos)) {
        Add(dnb);
        dynodeSync.BumpAssetLastTime("CDynodeMan::CheckDnbAndUpdateDynodeList - new");
        // if it matches our Dynode privkey...
        if (fDynodeMode && dnb.pubKeyDynode == activeDynode.pubKeyDynode) {
            dnb.nPoSeBanScore = -DYNODE_POSE_BAN_MAX_SCORE;
            if (dnb.nProtocolVersion == PROTOCOL_VERSION) {
                // ... and PROTOCOL_VERSION, then we've been remotely activated ...
                LogPrintf("CDynodeMan::CheckDnbAndUpdateDynodeList -- Got NEW Dynode entry: dynode=%s  sigTime=%lld  addr=%s\n",
                    dnb.outpoint.ToStringShort(), dnb.sigTime, dnb.addr.ToString());
                activeDynode.ManageState(connman);
            } else {
                // ... otherwise we need to reactivate our node, do not add it to the list and do not relay
                // but also do not ban the node we get this message from
                LogPrintf("CDynodeMan::CheckDnbAndUpdateDynodeList -- wrong PROTOCOL_VERSION, re-activate your DN: message nProtocolVersion=%d  PROTOCOL_VERSION=%d\n", dnb.nProtocolVersion, PROTOCOL_VERSION);
                return false;
            }
        }
        dnb.Relay(connman);
    } else {
        LogPrintf("CDynodeMan::CheckDnbAndUpdateDynodeList -- Rejected Dynode entry: %s  addr=%s\n", dnb.outpoint.ToStringShort(), dnb.addr.ToString());
        return false;
    }

    return true;
}

void CDynodeMan::UpdateLastPaid(const CBlockIndex* pindex)
{
    LOCK(cs);

    if (fLiteMode || !dynodeSync.IsWinnersListSynced() || mapDynodes.empty())
        return;

    static int nLastRunBlockHeight = 0;
    // Scan at least LAST_PAID_SCAN_BLOCKS but no more than dnpayments.GetStorageLimit()
    int nMaxBlocksToScanBack = std::max(LAST_PAID_SCAN_BLOCKS, nCachedBlockHeight - nLastRunBlockHeight);
    nMaxBlocksToScanBack = std::min(nMaxBlocksToScanBack, dnpayments.GetStorageLimit());

    LogPrint("dynode", "CDynodeMan::UpdateLastPaid -- nCachedBlockHeight=%d, nLastRunBlockHeight=%d, nMaxBlocksToScanBack=%d\n",
        nCachedBlockHeight, nLastRunBlockHeight, nMaxBlocksToScanBack);

    for (auto& dnpair : mapDynodes) {
        dnpair.second.UpdateLastPaid(pindex, nMaxBlocksToScanBack);
    }

    nLastRunBlockHeight = nCachedBlockHeight;
}

void CDynodeMan::UpdateLastSentinelPingTime()
{
    LOCK(cs);
    nLastSentinelPingTime = GetTime();
}

bool CDynodeMan::IsSentinelPingActive()
{
    LOCK(cs);
    // Check if any Dynodes have voted recently, otherwise return false
    return (GetTime() - nLastSentinelPingTime) <= DYNODE_SENTINEL_PING_MAX_SECONDS;
}

bool CDynodeMan::AddGovernanceVote(const COutPoint& outpoint, uint256 nGovernanceObjectHash)
{
    LOCK(cs);
    CDynode* pdn = Find(outpoint);
    if (!pdn) {
        return false;
    }
    pdn->AddGovernanceVote(nGovernanceObjectHash);
    return true;
}

void CDynodeMan::RemoveGovernanceObject(uint256 nGovernanceObjectHash)
{
    LOCK(cs);
    for (auto& dnpair : mapDynodes) {
        dnpair.second.RemoveGovernanceObject(nGovernanceObjectHash);
    }
}

void CDynodeMan::CheckDynode(const CPubKey& pubKeyDynode, bool fForce)
{
    LOCK(cs);
    for (auto& dnpair : mapDynodes) {
        if (dnpair.second.pubKeyDynode == pubKeyDynode) {
            dnpair.second.Check(fForce);
            return;
        }
    }
}

bool CDynodeMan::IsDynodePingedWithin(const COutPoint& outpoint, int nSeconds, int64_t nTimeToCheckAt)
{
    LOCK(cs);
    CDynode* pdn = Find(outpoint);
    return pdn ? pdn->IsPingedWithin(nSeconds, nTimeToCheckAt) : false;
}

void CDynodeMan::SetDynodeLastPing(const COutPoint& outpoint, const CDynodePing& dnp)
{
    LOCK(cs);
    CDynode* pdn = Find(outpoint);
    if (!pdn) {
        return;
    }
    pdn->lastPing = dnp;
    if (dnp.fSentinelIsCurrent) {
        UpdateLastSentinelPingTime();
    }
    mapSeenDynodePing.insert(std::make_pair(dnp.GetHash(), dnp));

    CDynodeBroadcast dnb(*pdn);
    uint256 hash = dnb.GetHash();
    if (mapSeenDynodeBroadcast.count(hash)) {
        mapSeenDynodeBroadcast[hash].second.lastPing = dnp;
    }
}

void CDynodeMan::UpdatedBlockTip(const CBlockIndex* pindex)
{
    nCachedBlockHeight = pindex->nHeight;
    LogPrint("dynode", "CDynodeMan::UpdatedBlockTip -- nCachedBlockHeight=%d\n", nCachedBlockHeight);

    CheckSameAddr();

    if (fDynodeMode) {
        // normal wallet does not need to update this every block, doing update on rpc call should be enough
        UpdateLastPaid(pindex);
    }
}

void CDynodeMan::WarnDynodeDaemonUpdates()
{
    LOCK(cs);

    static bool fWarned = false;

    if (fWarned || !size() || !dynodeSync.IsDynodeListSynced())
        return;

    int nUpdatedDynodes{0};

    for (const auto& dnpair : mapDynodes) {
        if (dnpair.second.lastPing.nDaemonVersion > CLIENT_VERSION) {
            ++nUpdatedDynodes;
        }
    }

    // Warn only when at least half of known dynodes already updated
    if (nUpdatedDynodes < size() / 2)
        return;

    std::string strWarning;
    if (nUpdatedDynodes != size()) {
        strWarning = strprintf(_("Warning: At least %d of %d dynodes are running on a newer software version. Please check latest releases, you might need to update too."),
            nUpdatedDynodes, size());
    } else {
        // someone was postponing this update for way too long probably
        strWarning = strprintf(_("Warning: Every dynode (out of %d known ones) is running on a newer software version. Please check latest releases, it's very likely that you missed a major/critical update."),
            size());
    }

    // notify GetWarnings(), called by Qt and the JSON-RPC code to warn the user
    SetMiscWarning(strWarning);
    // trigger GUI update
    uiInterface.NotifyAlertChanged(SerializeHash(strWarning), CT_NEW);
    // trigger cmd-line notification
    CAlert::Notify(strWarning);

    fWarned = true;
}

void CDynodeMan::NotifyDynodeUpdates(CConnman& connman)
{
    // Avoid double locking
    bool fDynodesAddedLocal = false;
    bool fDynodesRemovedLocal = false;
    {
        LOCK(cs);
        fDynodesAddedLocal = fDynodesAdded;
        fDynodesRemovedLocal = fDynodesRemoved;
    }

    if (fDynodesAddedLocal) {
        governance.CheckDynodeOrphanObjects(connman);
        governance.CheckDynodeOrphanVotes(connman);
    }
    if (fDynodesRemovedLocal) {
        governance.UpdateCachesAndClean();
    }

    LOCK(cs);
    fDynodesAdded = false;
    fDynodesRemoved = false;
}

void CDynodeMan::DoMaintenance(CConnman& connman)
{
    if (fLiteMode)
        return; // disable all Dynamic specific functionality

    if (!dynodeSync.IsBlockchainSynced() || ShutdownRequested())
        return;

    static unsigned int nTick = 0;

    nTick++;

    // make sure to check all dynodes first
    dnodeman.Check();

    dnodeman.ProcessPendingDnbRequests(connman);
    dnodeman.ProcessPendingDnvRequests(connman);

    if (nTick % 60 == 0) {
        dnodeman.ProcessDynodeConnections(connman);
        dnodeman.CheckAndRemove(connman);
        dnodeman.WarnDynodeDaemonUpdates();
    }

    if (fDynodeMode && (nTick % (60 * 5) == 0)) {
        dnodeman.DoFullVerificationStep(connman);
    }
}
