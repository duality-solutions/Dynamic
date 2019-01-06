// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_LINKINGDB_H
#define DYNAMIC_BDAP_LINKINGDB_H

#include "bdap/linking.h"
#include "dbwrapper.h"
#include "sync.h"

static CCriticalSection cs_link_request;
static CCriticalSection cs_link_accept;

class CLinkRequestDB : public CDBWrapper {
public:
    CLinkRequestDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "linkreqs", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    bool AddLinkRequest(const CLinkRequest& link, const int op);
    bool ReadLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link);
    bool EraseLinkRequest(const std::vector<unsigned char>& vchPubKey);
    bool LinkRequestExists(const std::vector<unsigned char>& vchPubKey);
    bool UpdateLinkRequest(const std::vector<unsigned char>& vchPubKey, const CLinkRequest& link);
    bool CleanupLinkRequestDB(int& nRemoved);
    bool GetLinkRequestInfo(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link);
};

class CLinkAcceptDB : public CDBWrapper {
public:
    CLinkAcceptDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "linkaccs", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    bool AddLinkAccept(const CLinkAccept& link, const int op);
    bool ReadLinkAccept(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link);
    bool EraseLinkAccept(const std::vector<unsigned char>& vchAcceptPubKey, const std::vector<unsigned char>& vchSharedPubKey) ;
    bool LinkAcceptExists(const std::vector<unsigned char>& vchPubKey);
    bool UpdateLinkAccept(const CLinkAccept& link);
    bool CleanupLinkAcceptDB(int& nRemoved);
    bool GetLinkAcceptInfo(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link);
};

bool GetLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link);
bool GetLinkAccept(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link);
bool CheckLinkRequestDB();
bool CheckLinkAcceptDB();
bool CheckLinkDBs();
bool FlushLinkRequestDB();
bool FlushLinkAcceptDB();
void CleanupLinkRequestDB(int& nRemoved);
void CleanupLinkAcceptDB(int& nRemoved);
bool CheckLinkTx(const CTransactionRef& tx, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                bool fJustCheck, int nHeight, std::string& errorMessage, bool bSanityCheck);


extern CLinkRequestDB *pLinkRequestDB;
extern CLinkAcceptDB *pLinkAcceptDB;

#endif // DYNAMIC_BDAP_LINKINGDB_H