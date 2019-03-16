// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_LINKMANAGER_H
#define DYNAMIC_BDAP_LINKMANAGER_H

#include "bdap/linkstorage.h"
#include "uint256.h"

#include <array>
#include <forward_list>
#include <map>
#include <string>
#include <vector>

class CLinkManager {
public:
    std::forward_list<std::pair<uint256, CLinkStorage>> linkQueue;
    //std::map<uint256, CLinkStorage> m_Links;

    CLinkManager() {
        SetNull();
    }

    inline void SetNull()
    {
        linkQueue.clear();
    }


private:
    bool IsLinkFromMe(const std::vector<unsigned char>& vchLinkPubKey);
    bool IsLinkForMe(const std::vector<unsigned char>& vchLinkPubKey, const std::vector<unsigned char>& vchSharedPubKey);
    bool GetLinkPrivateKey(const std::vector<unsigned char>& vchSenderPubKey, const std::vector<unsigned char>& vchSharedPubKey, std::array<char, 32>& sharedSeed, std::string& strErrorMessage);
};

extern CLinkManager* pLinkManager;

#endif // DYNAMIC_BDAP_LINKMANAGER_H