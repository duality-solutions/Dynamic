// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"

#include "base58.h"
#include "bdap/utils.h"
#include "rpcclient.h"
#include "rpcserver.h"
#include "primitives/block.h"
#include "txmempool.h"
#include "serialize.h"
#include "streams.h"
#include "validation.h"
#include "wallet/wallet.h"

#include <univalue.h>

#include <boost/algorithm/string/find.hpp>
#include <boost/xpressive/xpressive_dynamic.hpp>

using namespace boost::xpressive;

bool CDomainEntry::UnserializeFromTx(const CTransactionRef& tx) {
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

void CDomainEntry::Serialize(std::vector<unsigned char>& vchData) {
    CDataStream dsBDAP(SER_NETWORK, PROTOCOL_VERSION);
    dsBDAP << *this;
    vchData = std::vector<unsigned char>(dsBDAP.begin(), dsBDAP.end());
}

bool CDomainEntry::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) {
    try {
        CDataStream dsBDAP(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsBDAP >> *this;

        std::vector<unsigned char> vchBDAPData;
        Serialize(vchBDAPData);
        const uint256 &calculatedHash = Hash(vchBDAPData.begin(), vchBDAPData.end());
        const std::vector<unsigned char> &vchRandBDAP = vchFromValue(calculatedHash.GetHex());
        if(vchRandBDAP != vchHash)
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

CDynamicAddress CDomainEntry::GetWalletAddress() const {
    return CDynamicAddress(stringFromVch(WalletAddress));
}

CDynamicAddress CDomainEntry::GetLinkAddress() const {
    return CDynamicAddress(stringFromVch(LinkAddress));
}

std::string CDomainEntry::DHTPubKeyString() const {
    return stringFromVch(DHTPublicKey);
}

std::string CDomainEntry::GetFullObjectPath() const {
    return stringFromVch(ObjectID) + "@" + stringFromVch(OrganizationalUnit) + "." + stringFromVch(DomainComponent);
}

std::string CDomainEntry::GetObjectLocation() const {
    return stringFromVch(OrganizationalUnit) + "." + stringFromVch(DomainComponent);
}

std::vector<unsigned char> CDomainEntry::vchFullObjectPath() const {
    std::string strFullObjectPath = GetFullObjectPath();
    std::vector<unsigned char> vchReturnValue(strFullObjectPath.begin(), strFullObjectPath.end());
    return vchReturnValue;
}

std::vector<unsigned char> CDomainEntry::vchObjectLocation() const {
    std::string strObjectLocation = GetObjectLocation();
    std::vector<unsigned char> vchReturnValue(strObjectLocation.begin(), strObjectLocation.end());
    return vchReturnValue;
}

bool CDomainEntry::ValidateValues(std::string& errorMessage)
{
    smatch sMatch;
    std::string regExWithDot = "^((?!-)[a-z0-9-]{2," + std::to_string(MAX_OBJECT_NAME_LENGTH) + "}(?<!-)\\.)+[a-z]{2,6}$";
    std::string regExWithOutDot = "^((?!-)[a-z0-9-]{2," + std::to_string(MAX_OBJECT_NAME_LENGTH) + "}(?<!-))";

    // check domain name component
    std::string strDomainComponent = stringFromVch(DomainComponent);
    if (boost::find_first(strDomainComponent, "."))
    {
        sregex regexValidName = sregex::compile(regExWithDot);
        if (!regex_search(strDomainComponent, sMatch, regexValidName) || std::string(sMatch[0]) != strDomainComponent) {
            errorMessage = "Invalid BDAP domain name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }  
    }
    else
    {
        sregex regexValidName = sregex::compile(regExWithOutDot);
        if (!regex_search(strDomainComponent, sMatch, regexValidName) || std::string(sMatch[0]) != strDomainComponent) {
            errorMessage = "Invalid BDAP domain name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }
    }

    // check organizational unit component
    std::string strOrganizationalUnit = stringFromVch(OrganizationalUnit);
    if (boost::find_first(strOrganizationalUnit, "."))
    {
        sregex regexValidName = sregex::compile(regExWithDot);
        if (!regex_search(strOrganizationalUnit, sMatch, regexValidName) || std::string(sMatch[0]) != strOrganizationalUnit) {
            errorMessage = "Invalid BDAP organizational unit. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }  
    }
    else
    {
        sregex regexValidName = sregex::compile(regExWithOutDot);
        if (!regex_search(strOrganizationalUnit, sMatch, regexValidName) || std::string(sMatch[0]) != strOrganizationalUnit) {
            errorMessage = "Invalid BDAP organizational unit. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }
    }

    // check object name component
    std::string strObjectID = stringFromVch(ObjectID);
    if (boost::find_first(strObjectID, "."))
    {
        sregex regexValidName = sregex::compile(regExWithDot);
        if (!regex_search(strObjectID, sMatch, regexValidName) || std::string(sMatch[0]) != strObjectID) {
            errorMessage = "Invalid BDAP object name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }  
    }
    else
    {
        sregex regexValidName = sregex::compile(regExWithOutDot);
        if (!regex_search(strObjectID, sMatch, regexValidName) || std::string(sMatch[0]) != strObjectID) {
            errorMessage = "Invalid BDAP object name. Must follow the domain name spec of 2 to " + std::to_string(MAX_OBJECT_NAME_LENGTH) + " characters with no preceding or trailing dashes.";
            return false;
        }
    }

    // check object common name component
    if (CommonName.size() > MAX_COMMON_NAME_LENGTH) 
    {
        errorMessage = "Invalid BDAP common name. Can not have more than " + std::to_string(MAX_COMMON_NAME_LENGTH) + " characters.";
        return false;
    }

    // check object organization name component
    if (OrganizationName.size() > MAX_ORG_NAME_LENGTH) 
    {
        errorMessage = "Invalid BDAP organization name. Can not have more than " + std::to_string(MAX_ORG_NAME_LENGTH) + " characters.";
        return false;
    }

    if (WalletAddress.size() > MAX_WALLET_ADDRESS_LENGTH) 
    {
        errorMessage = "Invalid BDAP wallet address. Can not have more than " + std::to_string(MAX_WALLET_ADDRESS_LENGTH) + " characters.";
        return false;
    }
    else {
        std::string strWalletAddress = stringFromVch(WalletAddress);
        CDynamicAddress entryAddress(strWalletAddress);
        if (!entryAddress.IsValid()) {
            errorMessage = "Invalid BDAP wallet address. Wallet address failed IsValid check.";
            return false;
        }
    }
    
    if (LinkAddress.size() > MAX_WALLET_ADDRESS_LENGTH) 
    {
        errorMessage = "Invalid BDAP link address. Can not have more than " + std::to_string(MAX_WALLET_ADDRESS_LENGTH) + " characters.";
        return false;
    }
    else {
        if (LinkAddress.size() > 0) {
            std::string strLinkAddress = stringFromVch(LinkAddress);
            CTxDestination destLink = DecodeDestination(strLinkAddress);
            if (destLink.type() == typeid(CKeyID)) {
                CDynamicAddress entryLinkAddress(strLinkAddress);
                if (!entryLinkAddress.IsValid()) {
                    errorMessage = "Invalid BDAP public link address. Link wallet address failed IsValid check.";
                    return false;
                }
            }
            else if (destLink.type() == typeid(CStealthAddress)) {
                CStealthAddress sxAddr;
                if (!sxAddr.SetEncoded(strLinkAddress)) {
                    errorMessage = "Invalid BDAP stealth link address. Link wallet address failed SetEncoded check.";
                    return false;
                }
            }
        }
    }

    if (DHTPublicKey.size() > MAX_KEY_LENGTH) 
    {
        errorMessage = "Invalid BDAP encryption public key. Can not have more than " + std::to_string(MAX_KEY_LENGTH) + " characters.";
        return false;
    }
    //TODO (BDAP): validate Ed25519 public key if possible.
    /*
    else {
        if (DHTPublicKey.size() > 0) {
            CPubKey entryDHTPublicKey(DHTPublicKey);
            if (!entryDHTPublicKey.IsFullyValid()) {
                errorMessage = "Invalid BDAP encryption public key. Encryption public key failed IsFullyValid check.";
                return false;
            }
        }
    }
    */
    return true;
}

/** Checks if BDAP transaction exists in the memory pool */
bool CDomainEntry::CheckIfExistsInMemPool(const CTxMemPool& pool, std::string& errorMessage)
{
    for (const CTxMemPoolEntry& e : pool.mapTx) {
        const CTransactionRef& tx = e.GetSharedTx();
        for (const CTxOut& txOut : tx->vout) {
            if (IsBDAPDataOutput(txOut)) {
                CDomainEntry domainEntry(tx);
                if (this->GetFullObjectPath() == domainEntry.GetFullObjectPath()) {
                    errorMessage = "CheckIfExistsInMemPool: A BDAP domain entry transaction for " + GetFullObjectPath() + " is already in the memory pool!";
                    return true;
                }
            }
        }
    }
    return false;
}

/** Checks if the domain entry transaction uses the entry's UTXO */
bool CDomainEntry::TxUsesPreviousUTXO(const CTransactionRef& tx)
{
    int nIn = GetBDAPOperationOutIndex(tx);
    COutPoint entryOutpoint = COutPoint(txHash, nIn);
    for (const CTxIn& txIn : tx->vin) {
        if (txIn.prevout == entryOutpoint)
            return true;
    }
    return false;
}

/** Generate unique OID when using the root BDAP OID prefix */
std::string CDomainEntry::GenerateOID() const
{
    std::string strOID;
    if (RootOID == vchDefaultOIDPrefix)
    {
        // BDAP generated OID: 2.16.840.1.114564.block-height.tx-ordinal
        std::string strTxOrdinal = "?";
        std::string strHeight = "?";
        if (nHeight > 0)
        {
            strHeight = std::to_string(nHeight);
        }
        if (!txHash.IsNull() && !IsInitialBlockDownload())
        {
            CTransactionRef txRef;
            uint256 hashBlock;
            const Consensus::Params& consensusParams = Params().GetConsensus();
            if (GetTransaction(txHash, txRef, consensusParams, hashBlock, true))
            {
                if (!hashBlock.IsNull())
                {
                    CBlock block;
                    CBlockIndex* pblockindex = mapBlockIndex[hashBlock];
                    if (pblockindex && ReadBlockFromDisk(block, pblockindex, consensusParams))
                    {
                        int nTxOrdinal = 0;
                        for (const auto& tx : block.vtx) {
                            if (tx->GetHash() == txHash)
                            {
                                strTxOrdinal = std::to_string(nTxOrdinal);
                                break;
                            }
                            nTxOrdinal++;
                        }
                    } 
                }
            }
        }
        strOID = strprintf("%s.%s.%s", stringFromVch(RootOID), strHeight, strTxOrdinal);
    }
    else
    {
        strOID = stringFromVch(RootOID);
    }
    return strOID;
}

bool BuildBDAPJson(const CDomainEntry& entry, UniValue& oName, bool fAbridged)
{
    bool expired = false;
    int64_t expired_time = 0;
    int64_t nTime = 0;
    if (!fAbridged) {
        oName.push_back(Pair("oid", entry.GenerateOID()));
        oName.push_back(Pair("version", entry.nVersion));
        oName.push_back(Pair("domain_component", stringFromVch(entry.DomainComponent)));
        oName.push_back(Pair("common_name", stringFromVch(entry.CommonName)));
        oName.push_back(Pair("organizational_unit", stringFromVch(entry.OrganizationalUnit)));
        oName.push_back(Pair("organization_name", stringFromVch(entry.DomainComponent)));
        oName.push_back(Pair("object_id", stringFromVch(entry.ObjectID)));
        oName.push_back(Pair("object_full_path", entry.GetFullObjectPath()));
        oName.push_back(Pair("object_type", entry.ObjectTypeString()));
        oName.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
        oName.push_back(Pair("public", (int)entry.fPublicObject));
        oName.push_back(Pair("dht_publickey", stringFromVch(entry.DHTPublicKey)));
        oName.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
        oName.push_back(Pair("txid", entry.txHash.GetHex()));
        if ((unsigned int)chainActive.Height() >= entry.nHeight-1) {
            CBlockIndex *pindex = chainActive[entry.nHeight-1];
            if (pindex) {
                nTime = pindex->GetMedianTimePast();
            }
        }
        oName.push_back(Pair("time", nTime));
        //oName.push_back(Pair("height", entry.nHeight));
        expired_time = entry.nExpireTime;
        if(expired_time <= (unsigned int)chainActive.Tip()->GetMedianTimePast())
        {
            expired = true;
        }
        oName.push_back(Pair("expires_on", expired_time));
        oName.push_back(Pair("expired", expired));
    }
    else {
        oName.push_back(Pair("common_name", stringFromVch(entry.CommonName)));
        oName.push_back(Pair("object_full_path", stringFromVch(entry.vchFullObjectPath())));
        oName.push_back(Pair("wallet_address", stringFromVch(entry.WalletAddress)));
        oName.push_back(Pair("dht_publickey", stringFromVch(entry.DHTPublicKey)));
        oName.push_back(Pair("link_address", stringFromVch(entry.LinkAddress)));
        oName.push_back(Pair("object_type", entry.ObjectTypeString()));
    }
    return true;
}