// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_LINKINGDB_H
#define DYNAMIC_BDAP_LINKINGDB_H

#include "bdap/linking.h"
#include "dbwrapper.h"
#include "sync.h"

class uint256;

static CCriticalSection cs_link;

class CLinkDB : public CDBWrapper {
public:
    CLinkDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "links", nCacheSize, fMemory, fWipe, obfuscate) {
    }

    bool AddLinkIndex(const vchCharString& vvchOpParameters, const uint256& txid);
    bool ReadLinkIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid);
    bool EraseLinkIndex(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey);
    bool LinkExists(const std::vector<unsigned char>& vchPubKey);
    //bool CleanupIndexLinkDB(int& nRemoved);
};

bool UndoLinkData(const std::vector<unsigned char>& vchPubKey, const std::vector<unsigned char>& vchSharedPubKey);
bool GetLinkIndex(const std::vector<unsigned char>& vchPubKey, uint256& txid);
bool CheckLinkDB();
bool FlushLinkDB();
bool CheckLinkTx(const CTransactionRef& tx, const int& op1, const int& op2, const std::vector<std::vector<unsigned char> >& vvchArgs, 
                                const bool fJustCheck, const int& nHeight, const uint32_t& nBlockTime, const bool bSanityCheck, std::string& errorMessage);

bool CheckPreviousLinkInputs(const std::string& strOpType, const CScript& scriptOp, const std::vector<std::vector<unsigned char>>& vvchOpParameters, std::string& errorMessage, bool fJustCheck);

bool LinkPubKeyExists(const std::vector<unsigned char>& vchPubKey);

extern CLinkDB *pLinkDB;

#endif // DYNAMIC_BDAP_LINKINGDB_H