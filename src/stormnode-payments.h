// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_STORMNODE_PAYMENTS_H
#define DARKSILK_STORMNODE_PAYMENTS_H

#include "core_io.h"
#include "key.h"
#include "main.h"
#include "stormnode.h"
#include "util.h"
#include "utilstrencodings.h"

class CStormnodePayments;
class CStormnodePaymentVote;
class CStormnodeBlockPayees;

static const int SNPAYMENTS_SIGNATURES_REQUIRED         = 10;
static const int SNPAYMENTS_SIGNATURES_TOTAL            = 20;

//! minimum peer version that can receive and send Stormnode payment messages,
//  vote for Stormnode and be elected as a payment winner
static const int MIN_STORMNODE_PAYMENT_PROTO_VERSION = 70300;

extern CCriticalSection cs_vecPayees;
extern CCriticalSection cs_mapStormnodeBlocks;
extern CCriticalSection cs_mapStormnodePayeeVotes;

extern CStormnodePayments snpayments;

/// TODO: all 4 functions do not belong here really, they should be refactored/moved somewhere (main.cpp ?)
bool IsBlockValueValid(const CBlock& block, int nBlockHeight, CAmount blockReward, std::string &strErrorRet);
bool IsBlockPayeeValid(const CTransaction& txNew, int nBlockHeight, CAmount blockReward);
void FillBlockPayments(CMutableTransaction& txNew, int nBlockHeight, CAmount blockReward, CTxOut& txoutStormnodeRet, std::vector<CTxOut>& voutSuperblockRet);
std::string GetRequiredPaymentsString(int nBlockHeight);

class CStormnodePayee
{
private:
    CScript scriptPubKey;
    std::vector<uint256> vecVoteHashes;

public:
    CStormnodePayee() :
        scriptPubKey(),
        vecVoteHashes()
        {}

    CStormnodePayee(CScript payee, uint256 hashIn) :
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

// Keep track of votes for payees from Stormnodes
class CStormnodeBlockPayees
{
public:
    int nBlockHeight;
    std::vector<CStormnodePayee> vecPayees;

    CStormnodeBlockPayees() :
        nBlockHeight(0),
        vecPayees()
        {}
    CStormnodeBlockPayees(int nBlockHeightIn) :
        nBlockHeight(nBlockHeightIn),
        vecPayees()
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(nBlockHeight);
        READWRITE(vecPayees);
    }

    void AddPayee(const CStormnodePaymentVote& vote);
    bool GetBestPayee(CScript& payeeRet);
    bool HasPayeeWithVotes(CScript payeeIn, int nVotesReq);

    bool IsTransactionValid(const CTransaction& txNew);

    std::string GetRequiredPaymentsString();
};

// vote for the winning payment
class CStormnodePaymentVote
{
public:
    CTxIn vinStormnode;

    int nBlockHeight;
    CScript payee;
    std::vector<unsigned char> vchSig;

    CStormnodePaymentVote() :
        vinStormnode(),
        nBlockHeight(0),
        payee(),
        vchSig()
        {}

    CStormnodePaymentVote(CTxIn vinStormnode, int nBlockHeight, CScript payee) :
        vinStormnode(vinStormnode),
        nBlockHeight(nBlockHeight),
        payee(payee),
        vchSig()
        {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(vinStormnode);
        READWRITE(nBlockHeight);
        READWRITE(*(CScriptBase*)(&payee));
        READWRITE(vchSig);
    }

    uint256 GetHash() const {
        CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
        ss << *(CScriptBase*)(&payee);
        ss << nBlockHeight;
        ss << vinStormnode.prevout;
        return ss.GetHash();
    }

    bool Sign();
    bool CheckSignature(const CPubKey& pubKeyStormnode, int nValidationHeight, int &nDos);

    bool IsValid(CNode* pnode, int nValidationHeight, std::string& strError);
    void Relay();

    bool IsVerified() { return !vchSig.empty(); }
    void MarkAsNotVerified() { vchSig.clear(); }

    std::string ToString() const;
};

//
// Stormnode Payments Class
// Keeps track of who should get paid for which blocks
//

class CStormnodePayments
{
private:
    // Stormnode count times nStorageCoeff payments blocks should be stored ...
    const float nStorageCoeff;
    // ... but at least nMinBlocksToStore (payments blocks)
    const int nMinBlocksToStore;

    // Keep track of current block index
    const CBlockIndex *pCurrentBlockIndex;

public:
    std::map<uint256, CStormnodePaymentVote> mapStormnodePaymentVotes;
    std::map<int, CStormnodeBlockPayees> mapStormnodeBlocks;
    std::map<COutPoint, int> mapStormnodesLastVote;

    CStormnodePayments() : nStorageCoeff(1.25), nMinBlocksToStore(5000) {}

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action, int nType, int nVersion) {
        READWRITE(mapStormnodePaymentVotes);
        READWRITE(mapStormnodeBlocks);
    }

    void Clear();

    bool AddPaymentVote(const CStormnodePaymentVote& vote);
    bool HasVerifiedPaymentVote(uint256 hashIn);
    bool ProcessBlock(int nBlockHeight);

    void Sync(CNode* node, int nCountNeeded);
    void RequestLowDataPaymentBlocks(CNode* pnode);
    void CheckAndRemove();

    bool GetBlockPayee(int nBlockHeight, CScript& payee);
    bool IsTransactionValid(const CTransaction& txNew, int nBlockHeight);
    bool IsScheduled(CStormnode& sn, int nNotBlockHeight);

    bool CanVote(COutPoint outStormnode, int nBlockHeight);

    int GetMinStormnodePaymentsProto();
    void ProcessMessage(CNode* pfrom, std::string& strCommand, CDataStream& vRecv);
    std::string GetRequiredPaymentsString(int nBlockHeight);
    void FillBlockPayee(CMutableTransaction& txNew);
    std::string ToString() const;

    int GetBlockCount() { return mapStormnodeBlocks.size(); }
    int GetVoteCount() { return mapStormnodePaymentVotes.size(); }

    bool IsEnoughData();
    int GetStorageLimit();

    void UpdatedBlockTip(const CBlockIndex *pindex);
};

#endif // DARKSILK_STORMNODE_PAYMENTS_H
