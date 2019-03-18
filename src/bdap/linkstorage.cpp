// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/linkstorage.h"

#include "bdap/linkmanager.h"
#include "bdap/utils.h"
#include "hash.h"
#include "serialize.h"
#include "streams.h"
#include "version.h"

void ProcessLink(const CLinkStorage& storage)
{
    if (!pLinkManager)
        throw std::runtime_error("pLinkManager is null.\n");

    pLinkManager->ProcessLink(storage);
}

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

int CLinkStorage::DataVersion() const
{
    if (vchRawData.size() == 0)
        return -1;

    return GetLinkVersionFromData(vchRawData);
}

bool CLinkStorage::Encrypted() const
{
    if (DataVersion() == 0 || vchRawData.size() == 0)
        return false;

    return true;
}
