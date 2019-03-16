// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/linkstorage.h"

#include "hash.h"
#include "serialize.h"
#include "streams.h"
#include "version.h"

void CLinkStorage::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsLinkStorage(SER_NETWORK, PROTOCOL_VERSION);
    dsLinkStorage << *this;
    vchData = std::vector<unsigned char>(dsLinkStorage.begin(), dsLinkStorage.end());
}

bool CLinkStorage::UnserializeFromData(const std::vector<unsigned char>& vchData) 
{
    try {
        CDataStream dsLinkStorage(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsLinkStorage >> *this;
        std::vector<unsigned char> vchLinkData;
        Serialize(vchLinkData);
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}