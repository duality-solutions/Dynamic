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

    bool AddMyLinkRequest(const CLinkRequest& link);
    bool ReadMyLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link);
    bool ListMyLinkRequests(std::vector<CLinkRequest>& vchLinkRequests);
    bool EraseMyLinkRequest(const std::vector<unsigned char>& vchPubKey);
    bool MyLinkRequestExists(const std::vector<unsigned char>& vchPubKey);
    bool MyLinkageExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN);
    bool GetMyLinkRequest(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN, CLinkRequest& link);
    bool CleanupMyLinkRequestDB(int& nRemoved);

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

    bool AddMyLinkAccept(const CLinkAccept& link);
    bool ReadMyLinkAccept(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link);
    bool ListMyLinkAccepts(std::vector<CLinkAccept>& vchLinkAccepts);
    bool EraseMyLinkAccept(const std::vector<unsigned char>& vchPubKey);
    bool MyLinkAcceptExists(const std::vector<unsigned char>& vchPubKey);
    bool MyLinkageExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN);
    bool GetMyLinkAccept(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN, CLinkAccept& link);
    bool CleanupMyLinkAcceptDB(int& nRemoved);

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
void CleanupLinkRequestDB(int& nRemoved);
void CleanupLinkAcceptDB(int& nRemoved);
bool CheckLinkTx(const CTransactionRef& tx, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck);

bool CheckLinkageRequestExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN);
bool CheckLinkageAcceptExists(const std::string& strRequestorFQDN, const std::string& strRecipientFQDN);
bool CheckPreviousLinkInputs(const std::string& strOpType, const CScript& scriptOp, const std::vector<std::vector<unsigned char>>& vvchOpParameters, std::string& errorMessage, bool fJustCheck);

std::vector<unsigned char> AddVersionToLinkData(const std::vector<unsigned char>& vchData, const int& nVersion);
std::vector<unsigned char> RemoveVersionFromLinkData(const std::vector<unsigned char>& vchData, int& nVersion);

extern CLinkRequestDB *pLinkRequestDB;
extern CLinkAcceptDB *pLinkAcceptDB;

#endif // DYNAMIC_BDAP_LINKINGDB_H