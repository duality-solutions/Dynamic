// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_LINKINGDB_H
#define DYNAMIC_BDAP_LINKINGDB_H

#include "bdap/linking.h"
#include "dbwrapper.h"
#include "sync.h"

class uint256;

static CCriticalSection cs_link_request;
static CCriticalSection cs_link_accept;

class CLinkRequestDB : public CDBWrapper {
public:
    CLinkRequestDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "linkrequest", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    bool AddLinkRequestIndex(const vchCharString& vvchOpParameters, const uint256& txid);
    bool ReadLinkRequestIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid);
    bool EraseLinkRequestIndex(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey);
    bool LinkRequestExists(const std::vector<unsigned char>& vchPubKey);
    //bool CleanupIndexLinkRequestDB(int& nRemoved);
};

class CLinkAcceptDB : public CDBWrapper {
public:
    CLinkAcceptDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "linkaccept", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    bool AddLinkAcceptIndex(const vchCharString& vvchOpParameters, const uint256& txid);
    bool ReadLinkAcceptIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid);
    bool EraseLinkAcceptIndex(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey);
    bool LinkAcceptExists(const std::vector<unsigned char>& vchPubKey);
    //bool CleanupIndexLinkAcceptDB(int& nRemoved);
};

bool GetLinkRequestIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid);
bool GetLinkAcceptIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid);
bool CheckLinkRequestDB();
bool CheckLinkAcceptDB();
bool CheckLinkDBs();
bool FlushLinkRequestDB();
bool FlushLinkAcceptDB();
bool CheckLinkTx(const CTransactionRef& tx, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage);

bool CheckPreviousLinkInputs(const std::string& strOpType, const CScript& scriptOp, const std::vector<std::vector<unsigned char>>& vvchOpParameters, std::string& errorMessage, bool fJustCheck);

extern CLinkRequestDB *pLinkRequestDB;
extern CLinkAcceptDB *pLinkAcceptDB;

#endif // DYNAMIC_BDAP_LINKINGDB_H