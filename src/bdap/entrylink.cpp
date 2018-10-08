
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/entrylink.h"

#include "bdap/domainentry.h"
#include "hash.h"
#include "script/script.h"
#include "streams.h"

void CEntryLink::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsEntryLink(SER_NETWORK, PROTOCOL_VERSION);
    dsEntryLink << *this;
    vchData = std::vector<unsigned char>(dsEntryLink.begin(), dsEntryLink.end());
}

bool CEntryLink::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsEntryLink(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsEntryLink >> *this;

        std::vector<unsigned char> vchEntryLinkData;
        Serialize(vchEntryLinkData);
        const uint256 &calculatedHash = Hash(vchEntryLinkData.begin(), vchEntryLinkData.end());
        const std::vector<unsigned char> &vchRandEntryLink = vchFromValue(calculatedHash.GetHex());
        if(vchRandEntryLink != vchHash)
        {
            SetNull();
            return false;
        }
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

bool CEntryLink::UnserializeFromTx(const CTransactionRef& tx) 
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if(!GetBDAPData(tx, vchData, vchHash, nOut))
    {
        SetNull();
        return false;
    }
    if(!UnserializeFromData(vchData, vchHash))
    {
        return false;
    }
    return true;
}