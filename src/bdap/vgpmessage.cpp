// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/vgpmessage.h"

#include "base58.h"
#include "bdap/linkmanager.h"
#include "bdap/utils.h"
#include "bdap/vgp/include/encryption.h" // for VGP DecryptBDAPData
#include "clientversion.h"
#include "dht/ed25519.h"
#include "hash.h"
#include "key.h"
#include "netmessagemaker.h"
#include "script/script.h"
#include "streams.h"
#include "timedata.h"
#include "util.h"

class CMessage
{
public:
    static const int CURRENT_VERSION = 1;
    int nMessageVersion;
    std::vector<unsigned char> vchMessageType;
    std::vector<unsigned char> vchMessage;

    CMessage()
    {
        SetNull();
    }

    CMessage(const int& version, const std::vector<unsigned char>& type, const std::vector<unsigned char>& message) 
                : nMessageVersion(version), vchMessageType(type), vchMessage(message) {}

    CMessage(const std::vector<unsigned char>& vchData)
    {
        UnserializeFromData(vchData);
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(nMessageVersion);
        READWRITE(vchMessageType);
        READWRITE(vchMessage);
    }

    void SetNull()
    {
        nMessageVersion = -1;
        vchMessageType.clear();
        vchMessage.clear();
    }

    inline bool IsNull() const { return (nMessageVersion == -1); }

    inline CMessage operator=(const CMessage& b)
    {
        nMessageVersion = b.nMessageVersion;
        vchMessageType = b.vchMessageType;
        vchMessage = b.vchMessage;
        return *this;
    }

    void Serialize(std::vector<unsigned char>& vchData);
    bool UnserializeFromData(const std::vector<unsigned char>& vchData);
};

void CMessage::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsMessageData(SER_NETWORK, PROTOCOL_VERSION);
    dsMessageData << *this;
    vchData = std::vector<unsigned char>(dsMessageData.begin(), dsMessageData.end());
}

bool CMessage::UnserializeFromData(const std::vector<unsigned char>& vchData)
{
    try {
        CDataStream dsMessageData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsMessageData >> *this;
    } catch (std::exception &e) {
        SetNull();
        return false;
    }
    return true;
}

void CUnsignedVGPMessage::SetNull()
{
    nVersion = CURRENT_VERSION;
    SubjectID.SetNull();
    MessageID.SetNull();
    fEncrypted = false;
    vchRelayWallet.clear();
    nTimeStamp = 0;
    nRelayUntil = 0;
    vchMessageData.clear();
}

std::string CUnsignedVGPMessage::ToString() const
{
    return strprintf(
        "CUnsignedVGPMessage(\n"
        "    nVersion             = %d\n"
        "    SubjectID            = %s\n"
        "    MessageID            = %s\n"
        "    Encrypted            = %s\n"
        "    RelayWallet          = %s\n"
        "    nTimeStamp           = %d\n"
        "    nRelayUntil          = %d\n"
        "    Message Size         = %d\n"
        ")\n",
        nVersion,
        SubjectID.ToString(),
        MessageID.ToString(),
        fEncrypted ? "true" : "false",
        stringFromVch(vchRelayWallet),
        nTimeStamp,
        nRelayUntil,
        vchMessageData.size());
}

bool CUnsignedVGPMessage::EncryptMessage(const std::vector<unsigned char>& vchType, const std::vector<unsigned char>& vchMessage, const std::vector<std::vector<unsigned char>>& vvchPubKeys, std::string& strErrorMessage)
{
    CMessage message(CMessage::CURRENT_VERSION, vchType, vchMessage);
    std::vector<unsigned char> vchData;
    message.Serialize(vchData);
    if (!EncryptBDAPData(vvchPubKeys, vchMessage, vchData, strErrorMessage))
    {
        return false;
    }
    fEncrypted = true;
    return true;
}

bool CUnsignedVGPMessage::DecryptMessage(const std::array<char, 32>& arrPrivateSeed, std::vector<unsigned char>& vchType, std::vector<unsigned char>& vchMessage, std::string& strErrorMessage)
{
    if (!fEncrypted)
        return false;

    std::vector<unsigned char> vchRawPrivSeed;
    for (unsigned int i = 0; i < sizeof(arrPrivateSeed); i++)
    {
        vchRawPrivSeed.push_back(arrPrivateSeed[i]);
    }
    std::vector<unsigned char> vchDecryptMessage;
    if (!DecryptBDAPData(vchRawPrivSeed, vchMessageData, vchDecryptMessage, strErrorMessage))
    {
        return false;
    }
    CMessage message(vchDecryptMessage);
    vchType = message.vchMessageType;
    vchMessage = message.vchMessage;
    return true;
}

void CVGPMessage::SetNull()
{
    CUnsignedVGPMessage::SetNull();
    vchMsg.clear();
    vchSig.clear();
}

bool CVGPMessage::IsNull() const
{
    return (nTimeStamp == 0);
}

uint256 CVGPMessage::GetHash() const
{
    return Hash(this->vchMsg.begin(), this->vchMsg.end());
}

bool CVGPMessage::IsInEffect() const
{
	// only keep for 1 minute
    return (nTimeStamp + 60 >= GetAdjustedTime());
}

bool CVGPMessage::AppliesTo(const int nVersion, const std::string& strSubVerIn) const
{
    return false;
    // convert nVersion and subversion to client version format 2041400
    //return (IsInEffect() && nClientVersion >= MIN_CLIENT_VERSION && nProtoVersion >= MIN_PROTOCOL_VERSION);
}

bool CVGPMessage::RelayMessage(CNode* pnode, CConnman& connman) const
{
    if (!IsInEffect())
        return false;

    // don't relay to nodes which haven't sent their version message
    if (pnode->nVersion == 0)
        return false;
    // returns true if wasn't already contained in the set
    if (pnode->setKnown.insert(GetHash()).second) {
        if (AppliesTo(pnode->nVersion, pnode->strSubVer) || GetAdjustedTime() < nRelayUntil) {
            connman.PushMessage(pnode, CNetMsgMaker(pnode->GetSendVersion()).Make(NetMsgType::VGPMESSAGE, *this, GetHash(), SubjectID));
            return true;
        }
    }
    return false;
}

bool CVGPMessage::Sign()
{
    CDataStream sMsg(SER_NETWORK, CLIENT_VERSION);
    sMsg << *(CUnsignedVGPMessage*)this;
    vchMsg = std::vector<unsigned char>(sMsg.begin(), sMsg.end());
    CDynamicSecret vchSecret;
    if (!vchSecret.SetString(stringFromVch(vchRelayWallet))) {
        LogPrintf("CVGPMessage::%s -- vchSecret.SetString failed\n", __func__);
        return false;
    }
    CKey key = vchSecret.GetKey();
    if (!key.Sign(Hash(vchMsg.begin(), vchMsg.end()), vchSig)) {
        LogPrintf("CVGPMessage::%s -- Failed to sign relay message.\n", __func__);
        return false;
    }

    return true;
}

bool CVGPMessage::CheckSignature(const std::vector<unsigned char>& vchPubKey) const
{
    CPubKey key(vchPubKey);
    if (!key.Verify(Hash(vchMsg.begin(), vchMsg.end()), vchSig))
        return error("CVGPMessage::%s(): verify signature failed", __func__);

    // Now unserialize the data
    CDataStream sRelayMsg(vchMsg, SER_NETWORK, PROTOCOL_VERSION);
    sRelayMsg >> *(CUnsignedVGPMessage*)this;
    return true;
}
/*
CVGPMessage CVGPMessage::getAlertByHash(const uint256& hash)
{
    CVGPMessage retval;
    {
        LOCK(cs_mapVGPMessages);
        std::map<uint256, CVGPMessage>::iterator mi = mapRelayMessages.find(hash);
        if (mi != mapRelayMessages.end())
            retval = mi->second;
    }
    return retval;
}
*/
bool CVGPMessage::ProcessRelayMessage(const std::vector<unsigned char>& vchPubKey, bool fThread) const
{
    if (!CheckSignature(vchPubKey))
        return false;
    if (!IsInEffect())
        return false;

    // alert.nID=max is reserved for if the alert key is
    // compromised. It must have a pre-defined message,
    // must never expire, must apply to all versions,
    // and must cancel all previous
    // alerts or it will be ignored (so an attacker can't
    // send an "everything is OK, don't panic" version that
    // cannot be overridden):
    /*
    int maxInt = std::numeric_limits<int>::max();
    if (nID == maxInt) {
        if (!(
                nExpiration == maxInt &&
                nCancel == (maxInt - 1) &&
                nMinVer == 0 &&
                nMaxVer == maxInt &&
                setSubVer.empty() &&
                nPriority == maxInt &&
                strStatusBar == "URGENT: Alert key compromised, upgrade required"))
            return false;
    }

    {
        LOCK(cs_mapVGPMessages);
        // Cancel previous alerts
        for (std::map<uint256, CVGPMessage>::iterator mi = mapRelayMessages.begin(); mi != mapRelayMessages.end();) {
            const CVGPMessage& alert = (*mi).second;
            if (Cancels(alert)) {
                LogPrint("alert", "cancelling alert %d\n", alert.nID);
                uiInterface.NotifyAlertChanged((*mi).first, CT_DELETED);
                mapRelayMessages.erase(mi++);
            } else if (!alert.IsInEffect()) {
                LogPrint("alert", "expiring alert %d\n", alert.nID);
                uiInterface.NotifyAlertChanged((*mi).first, CT_DELETED);
                mapRelayMessages.erase(mi++);
            } else
                mi++;
        }


        // Check if this alert has been cancelled
        BOOST_FOREACH (PAIRTYPE(const uint256, CVGPMessage) & item, mapRelayMessages) {
            const CVGPMessage& alert = item.second;
            if (alert.Cancels(*this)) {
                LogPrintf("relay message already cancelled by %d\n", alert.nID);
                return false;
            }
        }

        // Add to mapRelayMessages
        mapRelayMessages.insert(std::make_pair(GetHash(), *this));
        // Notify UI and -alertnotify if it applies to me
        if (AppliesToMe()) {
            uiInterface.NotifyAlertChanged(GetHash(), CT_NEW);
            Notify(strStatusBar, fThread);
        }
    }
	*/
    //LogPrintf("CVGPMessage::%s() -- accepted alert %d, AppliesToMe()=%d\n", nID, AppliesToMe());
    return true;
}

void CVGPMessage::Notify(const std::string& strMessage, bool fThread)
{
	/*
    std::string strCmd = GetArg("-alertnotify", "");
    if (strCmd.empty())
        return;

    // Alert text should be plain ascii coming from a trusted source, but to
    // be safe we first strip anything not in safeChars, then add single quotes around
    // the whole string before passing it to the shell:
    std::string singleQuote("'");
    std::string safeStatus = SanitizeString(strMessage);
    safeStatus = singleQuote + safeStatus + singleQuote;
    boost::replace_all(strCmd, "%s", safeStatus);

    if (fThread)
        boost::thread t(runCommand, strCmd); // thread runs free
    else
        runCommand(strCmd);
    */
}

bool GetSecretSharedKey(const std::string& strSenderFQDN, const std::string& strRecipientFQDN, CKeyEd25519& key, std::string& strErrorMessage)
{
    if (!pLinkManager)
    {
        strErrorMessage = "Link manager is null.";
        return false;
    }

    uint256 id = GetLinkID(strSenderFQDN, strRecipientFQDN);
    CLink link;
    if (!pLinkManager->FindLink(id, link))
    {
        strErrorMessage = "Link not found.";
        return false;
    }

    std::array<char, 32> seed;
    if (!GetSharedPrivateSeed(link, seed, strErrorMessage))
    {
        strErrorMessage = "Failed to get shared secret link key.";
        return false;
    }

    return true;
}

uint256 GetSubjectIDFromKey(const CKeyEd25519& key)
{
    std::vector<unsigned char> vchPubKey = key.GetPubKeyBytes();
    return Hash(vchPubKey.begin(), vchPubKey.end());
}

uint256 GetMessageID(const CKeyEd25519& key, const int64_t& timestamp)
{
    CScript scriptMessage;
    scriptMessage << key.GetPubKeyBytes() << timestamp;
    return Hash(scriptMessage.begin(), scriptMessage.end());
}