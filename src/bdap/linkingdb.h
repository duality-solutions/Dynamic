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

    bool AddAcceptLink(const CLinkAccept& link, const int op);
    bool ReadAcceptLink(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link);
    bool EraseAcceptLink(const std::vector<unsigned char>& vchAcceptPubKey, const std::vector<unsigned char>& vchSharedPubKey) ;
    bool AcceptLinkExists(const std::vector<unsigned char>& vchPubKey);
    bool UpdateAcceptLink(const CLinkAccept& link);
    bool CleanupAcceptLinkDB(int& nRemoved);
    bool GetAcceptLinkInfo(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link);
};

bool GetLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link);

extern CLinkRequestDB *pLinkRequestDB;
extern CLinkAcceptDB *pLinkAcceptDB;

#endif // DYNAMIC_BDAP_LINKINGDB_H