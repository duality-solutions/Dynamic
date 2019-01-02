// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/linkingdb.h"

#include "base58.h"
#include "validation.h"
#include "validationinterface.h"

#include <boost/thread.hpp>

CLinkRequestDB *pLinkRequestDB = NULL;
CLinkAcceptDB *pLinkAcceptDB = NULL;

bool CLinkRequestDB::AddLinkRequest(const CLinkRequest& link, const int op) 
{ 
    bool writeState = false;
    {
        LOCK(cs_link_request);
        writeState = Write(make_pair(std::string("pk"), link.RequestorPubKey), link);
    }

    return writeState;
}

bool CLinkRequestDB::ReadLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link) 
{
    LOCK(cs_link_request);
    return CDBWrapper::Read(make_pair(std::string("pk"), vchPubKey), link);
}

bool CLinkRequestDB::EraseLinkRequest(const std::vector<unsigned char>& vchPubKey) 
{
    LOCK(cs_link_request);
    if (!LinkRequestExists(vchPubKey)) 
        return false;

    return CDBWrapper::Erase(make_pair(std::string("pk"), vchPubKey));
}

bool CLinkRequestDB::LinkRequestExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_request);
    return CDBWrapper::Exists(make_pair(std::string("pk"), vchPubKey));
}

bool CLinkRequestDB::UpdateLinkRequest(const std::vector<unsigned char>& vchPubKey, const CLinkRequest& link)
{
    LOCK(cs_link_request);

    if (!EraseLinkRequest(vchPubKey))
        return false;

    bool writeState = false;
    writeState = Update(make_pair(std::string("pk"), link.RequestorPubKey), link);

    return writeState;
}

// Removes expired records from databases.
bool CLinkRequestDB::CleanupLinkRequestDB(int& nRemoved)
{
    LOCK(cs_link_request);
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CLinkRequest link;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "pk") {
                pcursor->GetValue(link);
                if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= link.nExpireTime)
                {
                    nRemoved++;
                    EraseLinkRequest(link.RequestorPubKey);
                }
            }
            pcursor->Next();
        } catch (std::exception &e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CLinkRequestDB::GetLinkRequestInfo(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link)
{
    LOCK(cs_link_request);
    if (!ReadLinkRequest(vchPubKey, link)) {
        return false;
    }
    
    return true;
}

bool CLinkAcceptDB::AddAcceptLink(const CLinkAccept& link, const int op) 
{ 
    bool writeState = false;
    {
        LOCK(cs_link_accept);
        writeState = Write(make_pair(std::string("rpk"), link.RecipientPubKey), link) 
            && Write(make_pair(std::string("spk"), link.SharedPubKey), link);
    }

    return writeState;
}

bool CLinkAcceptDB::ReadAcceptLink(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link) 
{
    LOCK(cs_link_accept);
    bool ret = CDBWrapper::Read(make_pair(std::string("rpk"), vchPubKey), link);
    if (!ret) {
        ret = CDBWrapper::Read(make_pair(std::string("spk"), vchPubKey), link);
    }
    return ret;
}

bool CLinkAcceptDB::EraseAcceptLink(const std::vector<unsigned char>& vchAcceptPubKey, const std::vector<unsigned char>& vchSharedPubKey) 
{
    LOCK(cs_link_accept);
    if (!AcceptLinkExists(vchAcceptPubKey)) 
        return false;

    return CDBWrapper::Erase(make_pair(std::string("rpk"), vchAcceptPubKey)) 
            && CDBWrapper::Erase(make_pair(std::string("spk"), vchSharedPubKey));
}

bool CLinkAcceptDB::AcceptLinkExists(const std::vector<unsigned char>& vchPubKey)
{
    LOCK(cs_link_accept);
    bool ret = CDBWrapper::Exists(make_pair(std::string("rpk"), vchPubKey));
    if (!ret) {
        ret = CDBWrapper::Exists(make_pair(std::string("spk"), vchPubKey));
    }
    return ret;
}

bool CLinkAcceptDB::UpdateAcceptLink(const CLinkAccept& link)
{
    LOCK(cs_link_accept);

    if (!EraseAcceptLink(link.RecipientPubKey, link.RecipientPubKey))
        return false;

    bool writeState = false;
    writeState = Update(make_pair(std::string("rpk"), link.RecipientPubKey), link)
        & Update(make_pair(std::string("spk"), link.SharedPubKey), link);

    return writeState;
}

// Removes expired records from databases.
bool CLinkAcceptDB::CleanupAcceptLinkDB(int& nRemoved)
{
    LOCK(cs_link_accept);
    boost::scoped_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    CLinkAccept link;
    std::pair<std::string, std::vector<unsigned char> > key;
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        try {
            if (pcursor->GetKey(key) && key.first == "rpk") {
                pcursor->GetValue(link);
                if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= link.nExpireTime)
                {
                    nRemoved++;
                    EraseAcceptLink(link.RecipientPubKey, link.SharedPubKey);
                }
            }
            pcursor->Next();
        } catch (std::exception &e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CLinkAcceptDB::GetAcceptLinkInfo(const std::vector<unsigned char>& vchPubKey, CLinkAccept& link)
{
    LOCK(cs_link_accept);
    if (!ReadAcceptLink(vchPubKey, link)) {
        return false;
    }
    
    return true;
}

bool GetLinkRequest(const std::vector<unsigned char>& vchPubKey, CLinkRequest& link)
{
    if (!pLinkRequestDB || !pLinkRequestDB->ReadLinkRequest(vchPubKey, link)) {
        return false;
    }
    
    if ((unsigned int)chainActive.Tip()->GetMedianTimePast() >= link.nExpireTime) {
        link.SetNull();
        return false;
    }
    return !link.IsNull();
}