// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DYNODE_PAYMENTS_H
#define DYNAMIC_DYNODE_PAYMENTS_H

#include "core_io.h"
#include "dynode.h"
#include "key.h"
#include "main.h"
#include "util.h"
#include "utilstrencodings.h"

class CDynodeBlockPayees;
class CDynodePayments;
class CDynodePaymentVote;

static const int DNPAYMENTS_SIGNATURES_REQUIRED         = 10;
static const int DNPAYMENTS_SIGNATURES_TOTAL            = 20;

//! minimum peer version that can receive and send Dynode payment messages,
//  vote for Dynode and be elected as a payment winner
static const int MIN_DYNODE_PAYMENT_PROTO_VERSION = 70200;

extern CCriticalSection cs_vecPayees;
extern CCriticalSection cs_mapDynodeBlocks;
extern CCriticalSection cs_mapDynodePayeeVotes;

extern CDynodePayments dnpayments;

/// TODO: all 4 functions do not belong here really, they should be refactored/moved somewhere (main.cpp ?)
bool IsBlockValueValid(const CBlock& block, int nBlockHeight, CAmount blockReward, std::string &strErrorRet);
bool IsBlockPayeeValid(const CTransaction& txNew, int nBlockHeight, CAmount blockReward);
void FillBlockPayments(CMutableTransaction& txNew, int nBlockHeight, CAmount blockReward, CTxOut& txoutDynodeRet, std::vector<CTxOut>& voutSuperblockRet);
std::string GetRequiredPaymentsString(int nBlockHeight);

class CDynodePayee
{
private:
    CScript scriptPubKey;
    std::vector<uint256> vecVoteHashes;

public:
    CDynodePayee() :
        scriptPubKey(),
        vecVoteHashes()
        {}

    CDynodePayee(CScript payee, uint256 hashIn) :
        scriptPubKey(payee),
        vecVoteHashes()
    {
        vecVoteHashes.push_back(hashIn);
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(*(CScriptBase*)(&scriptPubKey));
        READWRITE(vecVoteHashes);
    }

    CScript GetPayee() { return scriptPubKey; }

    void AddVoteHash(uint256 hashIn) { vecVoteHashes.push_back(hashIn); }
    std::vector<uint256> GetVoteHashes() { return vecVoteHashes; }
    int GetVoteCount() { return vecVoteHashes.size(); }
};

// Keep track of votes for payees from Dynodes
class CDynodeBlockPayees
{
public:
    int nBlockHeight;
    std::vector<CDynodePayee> vecPayees;

    CDynodeBlockPayees() :
        nBlockHeight(0),
        vecPayees()
        {}
    CDynodeBlockPayees(int nBlockHeightIn) :
        nBlockHeight(nBlockHeightIn),
        vecPayees()
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(nBlockHeight);
        READWRITE(vecPayees);
    }

    void AddPayee(const CDynodePaymentVote& vote);
    bool GetBestPayee(CScript& payeeRet);
    bool HasPayeeWithVotes(CScript payeeIn, int nVotesReq);

    bool IsTransactionValid(const CTransaction& txNew);

    std::string GetRequiredPaymentsString();
};

// vote for the winning payment
class CDynodePaymentVote
{
public:
    CTxIn vinDynode;

    int nBlockHeight;
    CScript payee;
    std::vector<unsigned char> vchSig;

    CDynodePaymentVote() :
        vinDynode(),
        nBlockHeight(0),
        payee(),
        vchSig()
        {}

    CDynodePaymentVote(CTxIn vinDynode, int nBlockHeight, CScript payee) :
        vinDynode(vinDynode),
        nBlockHeight(nBlockHeight),
        payee(payee),
        vchSig()
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vinDynode);
        READWRITE(nBlockHeight);
        READWRITE(*(CScriptBase*)(&payee));
        READWRITE(vchSig);
    }

    uint256 GetHash() const {
        CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
        ss << *(CScriptBase*)(&payee);
        ss << nBlockHeight;
        ss << vinDynode.prevout;
        return ss.GetHash();
    }

    bool Sign();
    bool CheckSignature(const CPubKey& pubKeyDynode, int nValidationHeight, int &nDos);

    bool IsValid(CNode* pnode, int nValidationHeight, std::string& strError);
    void Relay();

    bool IsVerified() { return !vchSig.empty(); }
    void MarkAsNotVerified() { vchSig.clear(); }

    std::string ToString() const;
};

//
// Dynode Payments Class
// Keeps track of who should get paid for which blocks
//

class CDynodePayments
{
private:
    // Dynode count times nStorageCoeff payments blocks should be stored ...
    const float nStorageCoeff;
    // ... but at least nMinBlocksToStore (payments blocks)
    const int nMinBlocksToStore;

    // Keep track of current block index
    const CBlockIndex *pCurrentBlockIndex;

public:
    std::map<uint256, CDynodePaymentVote> mapDynodePaymentVotes;
    std::map<int, CDynodeBlockPayees> mapDynodeBlocks;
    std::map<COutPoint, int> mapDynodesLastVote;

    CDynodePayments() : nStorageCoeff(1.25), nMinBlocksToStore(5000) {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(mapDynodePaymentVotes);
        READWRITE(mapDynodeBlocks);
    }

    void Clear();

    bool AddPaymentVote(const CDynodePaymentVote& vote);
    bool HasVerifiedPaymentVote(uint256 hashIn);
    bool ProcessBlock(int nBlockHeight);

    void Sync(CNode* node);
    void RequestLowDataPaymentBlocks(CNode* pnode);
    void CheckAndRemove();

    bool GetBlockPayee(int nBlockHeight, CScript& payee);
    bool IsTransactionValid(const CTransaction& txNew, int nBlockHeight);
    bool IsScheduled(CDynode& dn, int nNotBlockHeight);

    bool CanVote(COutPoint outDynode, int nBlockHeight);

    int GetMinDynodePaymentsProto();
    void ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv);
    std::string GetRequiredPaymentsString(int nBlockHeight);
    void FillBlockPayee(CMutableTransaction& txNew, CAmount nFees);
    std::string ToString() const;

    int GetBlockCount() { return mapDynodeBlocks.size(); }
    int GetVoteCount() { return mapDynodePaymentVotes.size(); }

    bool IsEnoughData();
    int GetStorageLimit();

    void UpdatedBlockTip(const CBlockIndex *pindex);
};

#endif // DYNAMIC_DYNODE_PAYMENTS_H
