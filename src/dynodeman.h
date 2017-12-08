// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DYNODEMAN_H
#define DYNAMIC_DYNODEMAN_H

#include "dynode.h"
#include "sync.h"

class CDynodeMan;

extern CDynodeMan dnodeman;

class CDynodeMan
{
public:
    typedef std::pair<arith_uint256, CDynode*> score_pair_t;
    typedef std::vector<score_pair_t> score_pair_vec_t;
    typedef std::pair<int, CDynode> rank_pair_t;
    typedef std::vector<rank_pair_t> rank_pair_vec_t;

private:
    static const std::string SERIALIZATION_VERSION_STRING;

    static const int PSEG_UPDATE_SECONDS        = 3 * 60 * 60;

    static const int LAST_PAID_SCAN_BLOCKS      = 100;

    static const int MIN_POSE_PROTO_VERSION     = 70500;
    static const int MAX_POSE_CONNECTIONS       = 10;
    static const int MAX_POSE_RANK              = 10;
    static const int MAX_POSE_BLOCKS            = 10;

    static const int DNB_RECOVERY_QUORUM_TOTAL      = 10;
    static const int DNB_RECOVERY_QUORUM_REQUIRED   = 10;
    static const int DNB_RECOVERY_MAX_ASK_ENTRIES   = 10;
    static const int DNB_RECOVERY_WAIT_SECONDS      = 60;
    static const int DNB_RECOVERY_RETRY_SECONDS     = 3 * 60 * 60;

    // critical section to protect the inner data structures
    mutable CCriticalSection cs;

    // Keep track of current block height
    int nCachedBlockHeight;

    // map to hold all DNs
    std::map<COutPoint, CDynode> mapDynodes;
    // who's asked for the Dynode list and the last time
    std::map<CNetAddr, int64_t> mAskedUsForDynodeList;
    // who we asked for the Dynode list and the last time
    std::map<CNetAddr, int64_t> mWeAskedForDynodeList;
    // which Dynodes we've asked for
    std::map<COutPoint, std::map<CNetAddr, int64_t> > mWeAskedForDynodeListEntry;
    // who we asked for the Dynode verification
    std::map<CNetAddr, CDynodeVerification> mWeAskedForVerification;

    // these maps are used for Dynode recovery from DYNODE_NEW_START_REQUIRED state
    std::map<uint256, std::pair< int64_t, std::set<CNetAddr> > > mDnbRecoveryRequests;
    std::map<uint256, std::vector<CDynodeBroadcast> > mDnbRecoveryGoodReplies;
    std::list< std::pair<CService, uint256> > listScheduledDnbRequestConnections;

    /// Set when Dynodes are added, cleared when CGovernanceManager is notified
    bool fDynodesAdded;

    /// Set when Dynodes are removed, cleared when CGovernanceManager is notified
    bool fDynodesRemoved;

    std::vector<uint256> vecDirtyGovernanceObjectHashes;

    int64_t nLastWatchdogVoteTime;

    friend class CDynodeSync;
    /// Find an entry
    CDynode* Find(const COutPoint& outpoint);

    bool GetDynodeScores(const uint256& nBlockHash, score_pair_vec_t& vecDynodeScoresRet, int nMinProtocol = 0);

public:
    // Keep track of all broadcasts I've seen
    std::map<uint256, std::pair<int64_t, CDynodeBroadcast> > mapSeenDynodeBroadcast;
    // Keep track of all pings I've seen
    std::map<uint256, CDynodePing> mapSeenDynodePing;
    // Keep track of all verifications I've seen
    std::map<uint256, CDynodeVerification> mapSeenDynodeVerification;
    // keep track of psq count to prevent Dynodes from gaming privatesend queue
    int64_t nPsqCount;


    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        LOCK(cs);
        std::string strVersion;
        if(ser_action.ForRead()) {
            READWRITE(strVersion);
        }
        else {
            strVersion = SERIALIZATION_VERSION_STRING; 
            READWRITE(strVersion);
        }

        READWRITE(mapDynodes);
        READWRITE(mAskedUsForDynodeList);
        READWRITE(mWeAskedForDynodeList);
        READWRITE(mWeAskedForDynodeListEntry);
        READWRITE(mDnbRecoveryRequests);
        READWRITE(mDnbRecoveryGoodReplies);
        READWRITE(nLastWatchdogVoteTime);
        READWRITE(nPsqCount);

        READWRITE(mapSeenDynodeBroadcast);
        READWRITE(mapSeenDynodePing);
        if(ser_action.ForRead() && (strVersion != SERIALIZATION_VERSION_STRING)) {
            Clear();
        }
    }

    CDynodeMan();

    /// Add an entry
    bool Add(CDynode &dn);

    /// Ask (source) node for dnb
    void AskForDN(CNode *pnode, const COutPoint& outpoint, CConnman& connman);
    void AskForDnb(CNode *pnode, const uint256 &hash);

    bool PoSeBan(const COutPoint &outpoint);
    bool AllowMixing(const COutPoint &outpoint);
    bool DisallowMixing(const COutPoint &outpoint);

    /// Check all Dynodes
    void Check();

    /// Check all Dynode and remove inactive
    void CheckAndRemove(CConnman& connman);
    /// This is dummy overload to be used for dumping/loading dncache.dat
    void CheckAndRemove() {}

    /// Clear Dynode vector
    void Clear();

    /// Count Dynodes filtered by nProtocolVersion.
    /// Dynode nProtocolVersion should match or be above the one specified in param here.
    int CountDynodes(int nProtocolVersion = -1);
    /// Count enabled Dynodes filtered by nProtocolVersion.
    /// Dynode nProtocolVersion should match or be above the one specified in param here.
    int CountEnabled(int nProtocolVersion = -1);

    /// Count Dynodes by network type - NET_IPV4, NET_IPV6, NET_TOR
    // int CountByIP(int nNetworkType);

    void PsegUpdate(CNode* pnode, CConnman& connman);

    /// Versions of Find that are safe to use from outside the class
    bool Get(const COutPoint& outpoint, CDynode& dynodeRet);
    bool Has(const COutPoint& outpoint);

    bool GetDynodeInfo(const COutPoint& outpoint, dynode_info_t& dnInfoRet);

    bool GetDynodeInfo(const CPubKey& pubKeyDynode, dynode_info_t& dnInfoRet);

    /// Find an entry in the Dynode list that is next to be paid
    bool GetNextDynodeInQueueForPayment(int nBlockHeight, bool fFilterSigTime, int& nCountRet, dynode_info_t& dnInfoRet);
    /// Same as above but use current block height
    bool GetNextDynodeInQueueForPayment(bool fFilterSigTime, int& nCountRet, dynode_info_t& dnInfoRet);

    /// Find a random entry
    dynode_info_t FindRandomNotInVec(const std::vector<COutPoint> &vecToExclude, int nProtocolVersion = -1);

    std::map<COutPoint, CDynode> GetFullDynodeMap() { return mapDynodes; }

    bool GetDynodeRanks(rank_pair_vec_t& vecDynodeRanksRet, int nBlockHeight = -1, int nMinProtocol = 0);
    bool GetDynodeRank(const COutPoint &outpoint, int& nRankRet, int nBlockHeight = -1, int nMinProtocol = 0);

    void ProcessDynodeConnections(CConnman& connman);
    std::pair<CService, std::set<uint256> > PopScheduledDnbRequestConnection();

    void ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv, CConnman& connman);

    void DoFullVerificationStep(CConnman& connman);
    void CheckSameAddr();
    bool SendVerifyRequest(const CAddress& addr, const std::vector<CDynode*>& vSortedByAddr, CConnman& connman);
    void SendVerifyReply(CNode* pnode, CDynodeVerification& dnv, CConnman& connman);
    void ProcessVerifyReply(CNode* pnode, CDynodeVerification& dnv);
    void ProcessVerifyBroadcast(CNode* pnode, const CDynodeVerification& dnv);

    /// Return the number of (unique) Dynodes
    int size() { return mapDynodes.size(); }

    std::string ToString() const;

    /// Update Dynode list and maps using provided CDynodeBroadcast
    void UpdateDynodeList(CDynodeBroadcast dnb, CConnman& connman);
    /// Perform complete check and only then update list and maps
    bool CheckDnbAndUpdateDynodeList(CNode* pfrom, CDynodeBroadcast dnb, int& nDos, CConnman& connman);
    bool IsDnbRecoveryRequested(const uint256& hash) { return mDnbRecoveryRequests.count(hash); }

    void UpdateLastPaid(const CBlockIndex* pindex);
    bool UpdateLastPsq(const CTxIn& vin);

    void AddDirtyGovernanceObjectHash(const uint256& nHash)
    {
        LOCK(cs);
        vecDirtyGovernanceObjectHashes.push_back(nHash);
    }

    std::vector<uint256> GetAndClearDirtyGovernanceObjectHashes()
    {
        LOCK(cs);
        std::vector<uint256> vecTmp = vecDirtyGovernanceObjectHashes;
        vecDirtyGovernanceObjectHashes.clear();
        return vecTmp;
    }

    bool IsWatchdogActive();
    void UpdateWatchdogVoteTime(const COutPoint& outpoint, uint64_t nVoteTime = 0);
    bool AddGovernanceVote(const COutPoint& outpoint, uint256 nGovernanceObjectHash);
    void RemoveGovernanceObject(uint256 nGovernanceObjectHash);

    void CheckDynode(const CPubKey& pubKeyDynode, bool fForce);

    bool IsDynodePingedWithin(const COutPoint& outpoint, int nSeconds, int64_t nTimeToCheckAt = -1);
    void SetDynodeLastPing(const COutPoint& outpoint, const CDynodePing& dnp);

    void UpdatedBlockTip(const CBlockIndex *pindex);

    /**
     * Called to notify CGovernanceManager that the Dynode index has been updated.
     * Must be called while not holding the CDynodeMan::cs mutex
     */
    void NotifyDynodeUpdates(CConnman& connman);

};

#endif // DYNAMIC_DYNODEMAN_H
