// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_STORMNODEMAN_H
#define DARKSILK_STORMNODEMAN_H

#include "stormnode.h"
#include "sync.h"

using namespace std;

class CStormnodeMan;

extern CStormnodeMan snodeman;

/**
 * Provides a forward and reverse index between SN vin's and integers.
 *
 * This mapping is normally add-only and is expected to be permanent
 * It is only rebuilt if the size of the index exceeds the expected maximum number
 * of SN's and the current number of known SN's.
 *
 * The external interface to this index is provided via delegation by CStormnodeMan
 */
class CStormnodeIndex
{
public: // Types
    typedef std::map<CTxIn,int> index_m_t;

    typedef index_m_t::iterator index_m_it;

    typedef index_m_t::const_iterator index_m_cit;

    typedef std::map<int,CTxIn> rindex_m_t;

    typedef rindex_m_t::iterator rindex_m_it;

    typedef rindex_m_t::const_iterator rindex_m_cit;

private:
    int                  nSize;

    index_m_t            mapIndex;

    rindex_m_t           mapReverseIndex;

public:
    CStormnodeIndex();

    int GetSize() const {
        return nSize;
    }

    /// Retrieve stormnode vin by index
    bool Get(int nIndex, CTxIn& vinStormnode) const;

    /// Get index of a stormnode vin
    int GetStormnodeIndex(const CTxIn& vinStormnode) const;

    void AddStormnodeVIN(const CTxIn& vinStormnode);

    void Clear();

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion)
    {
        READWRITE(mapIndex);
        if(ser_action.ForRead()) {
            RebuildIndex();
        }
    }

private:
    void RebuildIndex();

};

class CStormnodeMan
{
public:
    typedef std::map<CTxIn,int> index_m_t;

    typedef index_m_t::iterator index_m_it;

    typedef index_m_t::const_iterator index_m_cit;

private:
    static const int MAX_EXPECTED_INDEX_SIZE = 30000;

    /// Only allow 1 index rebuild per hour
    static const int64_t MIN_INDEX_REBUILD_TIME = 3600;

    static const std::string SERIALIZATION_VERSION_STRING;

    static const int SSEG_UPDATE_SECONDS        = 3 * 60 * 60;

    static const int LAST_PAID_SCAN_BLOCKS      = 100;

    static const int MIN_POSE_PROTO_VERSION     = 60800;
    static const int MAX_POSE_RANK              = 10;
    static const int MAX_POSE_BLOCKS            = 10;


    // critical section to protect the inner data structures
    mutable CCriticalSection cs;

    // Keep track of current block index
    const CBlockIndex *pCurrentBlockIndex;

    // map to hold all SNs
    std::vector<CStormnode> vStormnodes;
    // who's asked for the Stormnode list and the last time
    std::map<CNetAddr, int64_t> mAskedUsForStormnodeList;
    // who we asked for the Stormnode list and the last time
    std::map<CNetAddr, int64_t> mWeAskedForStormnodeList;
    // which Stormnodes we've asked for
    std::map<COutPoint, std::map<CNetAddr, int64_t> > mWeAskedForStormnodeListEntry;
    // who we asked for the Stormnode verification
    std::map<CNetAddr, CStormnodeVerification> mWeAskedForVerification;

    int64_t nLastIndexRebuildTime;

    CStormnodeIndex indexStormnodes;

    CStormnodeIndex indexStormnodesOld;

    /// Set when index has been rebuilt, clear when read
    bool fIndexRebuilt;

    /// Set when Stormnodes are added, cleared when CGovernanceManager is notified
    bool fStormnodesAdded;

    /// Set when Stormnodes are removed, cleared when CGovernanceManager is notified
    bool fStormnodesRemoved;

    std::vector<uint256> vecDirtyGovernanceObjectHashes;

    int64_t nLastWatchdogVoteTime;

    friend class CStormnodeSync;

public:
    // Keep track of all broadcasts I've seen
    std::map<uint256, std::pair<int64_t, CStormnodeBroadcast> > mapSeenStormnodeBroadcast;
    // Keep track of all pings I've seen
    std::map<uint256, CStormnodePing> mapSeenStormnodePing;
    // Keep track of all verifications I've seen
    std::map<uint256, CStormnodeVerification> mapSeenStormnodeVerification;
    // keep track of ssq count to prevent Stormnodes from gaming sandstorm queue
    int64_t nSsqCount;


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

        READWRITE(vStormnodes);
        READWRITE(mAskedUsForStormnodeList);
        READWRITE(mWeAskedForStormnodeList);
        READWRITE(mWeAskedForStormnodeListEntry);
        READWRITE(nLastWatchdogVoteTime);
        READWRITE(nSsqCount);

        READWRITE(mapSeenStormnodeBroadcast);
        READWRITE(mapSeenStormnodePing);
        READWRITE(indexStormnodes);
        if(ser_action.ForRead() && (strVersion != SERIALIZATION_VERSION_STRING)) {
            Clear();
        }
    }

    CStormnodeMan();

    /// Add an entry
    bool Add(CStormnode &sn);

    /// Ask (source) node for snb
    void AskForSN(CNode *pnode, const CTxIn &vin);

    /// Check all Stormnodes
    void Check();

    /// Check all Stormnodes and remove inactive
    void CheckAndRemove();

    /// Clear Stormnode vector
    void Clear();

    /// Count Stormnodes filtered by nProtocolVersion.
    /// Stormnode nProtocolVersion should match or be above the one specified in param here.
    int CountStormnodes(int nProtocolVersion = -1);
    /// Count enabled Stormnodes filtered by nProtocolVersion.
    /// Stormnode nProtocolVersion should match or be above the one specified in param here.
    int CountEnabled(int nProtocolVersion = -1);

    /// Count Stormnodes by network type - NET_IPV4, NET_IPV6, NET_TOR
    // int CountByIP(int nNetworkType);

    void SsegUpdate(CNode* pnode);

    /// Find an entry
    CStormnode* Find(const CScript &payee);
    CStormnode* Find(const CTxIn& vin);
    CStormnode* Find(const CPubKey& pubKeyStormnode);

    /// Versions of Find that are safe to use from outside the class
    bool Get(const CPubKey& pubKeyStormnode, CStormnode& stormnode);
    bool Get(const CTxIn& vin, CStormnode& stormnode);

    /// Retrieve stormnode vin by index
    bool Get(int nIndex, CTxIn& vinStormnode, bool& fIndexRebuiltOut) {
        LOCK(cs);
        fIndexRebuiltOut = fIndexRebuilt;
        return indexStormnodes.Get(nIndex, vinStormnode);
    }

    bool GetIndexRebuiltFlag() {
        LOCK(cs);
        return fIndexRebuilt;
    }

    /// Get index of a stormnode vin
    int GetStormnodeIndex(const CTxIn& vinStormnode) {
        LOCK(cs);
        return indexStormnodes.GetStormnodeIndex(vinStormnode);
    }

    /// Get old index of a stormnode vin
    int GetStormnodeIndexOld(const CTxIn& vinStormnode) {
        LOCK(cs);
        return indexStormnodesOld.GetStormnodeIndex(vinStormnode);
    }

    /// Get stormnode VIN for an old index value
    bool GetStormnodeVinForIndexOld(int nStormnodeIndex, CTxIn& vinStormnodeOut) {
        LOCK(cs);
        return indexStormnodesOld.Get(nStormnodeIndex, vinStormnodeOut);
    }

    /// Get index of a stormnode vin, returning rebuild flag
    int GetStormnodeIndex(const CTxIn& vinStormnode, bool& fIndexRebuiltOut) {
        LOCK(cs);
        fIndexRebuiltOut = fIndexRebuilt;
        return indexStormnodes.GetStormnodeIndex(vinStormnode);
    }

    void ClearOldStormnodeIndex() {
        LOCK(cs);
        indexStormnodesOld.Clear();
        fIndexRebuilt = false;
    }

    bool Has(const CTxIn& vin);

    stormnode_info_t GetStormnodeInfo(const CTxIn& vin);

    stormnode_info_t GetStormnodeInfo(const CPubKey& pubKeyStormnode);

    /// Find an entry in the stormnode list that is next to be paid
    CStormnode* GetNextStormnodeInQueueForPayment(int nBlockHeight, bool fFilterSigTime, int& nCount);
    /// Same as above but use current block height
    CStormnode* GetNextStormnodeInQueueForPayment(bool fFilterSigTime, int& nCount);

    /// Find a random entry
    CStormnode* FindRandomNotInVec(const std::vector<CTxIn> &vecToExclude, int nProtocolVersion = -1);

    std::vector<CStormnode> GetFullStormnodeVector() { return vStormnodes; }

    std::vector<std::pair<int, CStormnode> > GetStormnodeRanks(int nBlockHeight = -1, int nMinProtocol=0);
    int GetStormnodeRank(const CTxIn &vin, int nBlockHeight, int nMinProtocol=0, bool fOnlyActive=true);
    CStormnode* GetStormnodeByRank(int nRank, int nBlockHeight, int nMinProtocol=0, bool fOnlyActive=true);

    void ProcessStormnodeConnections();

    void ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv);

    void DoFullVerificationStep();
    void CheckSameAddr();
    bool SendVerifyRequest(const CAddress& addr, const std::vector<CStormnode*>& vSortedByAddr);
    void SendVerifyReply(CNode* pnode, CStormnodeVerification& snv);
    void ProcessVerifyReply(CNode* pnode, CStormnodeVerification& snv);
    void ProcessVerifyBroadcast(CNode* pnode, const CStormnodeVerification& snv);

    /// Return the number of (unique) Stormnodes
    int size() { return vStormnodes.size(); }

    std::string ToString() const;

    int GetEstimatedStormnodes(int nBlock);

    /// Update stormnode list and maps using provided CStormnodeBroadcast
    void UpdateStormnodeList(CStormnodeBroadcast snb);
    /// Perform complete check and only then update list and maps
    bool CheckSnbAndUpdateStormnodeList(CStormnodeBroadcast snb, int& nDos);

    void UpdateLastPaid();

    void CheckAndRebuildStormnodeIndex();

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
        return vecTmp;;
    }

    bool IsWatchdogActive();
    void UpdateWatchdogVoteTime(const CTxIn& vin);
    void AddGovernanceVote(const CTxIn& vin, uint256 nGovernanceObjectHash);
    void RemoveGovernanceObject(uint256 nGovernanceObjectHash);

    void CheckStormnode(const CTxIn& vin, bool fForce = false);
    void CheckStormnode(const CPubKey& pubKeyStormnode, bool fForce = false);

    int GetStormnodeState(const CTxIn& vin);
    int GetStormnodeState(const CPubKey& pubKeyStormnode);

    bool IsStormnodePingedWithin(const CTxIn& vin, int nSeconds, int64_t nTimeToCheckAt = -1);
    void SetStormnodeLastPing(const CTxIn& vin, const CStormnodePing& snp);

    void UpdatedBlockTip(const CBlockIndex *pindex);

    /**
     * Called to notify CGovernanceManager that the Stormnode index has been updated.
     * Must be called while not holding the CStormnodeMan::cs mutex
     */
    void NotifyStormnodeUpdates();

};

#endif // DARKSILK_STORMNODEMAN_H
