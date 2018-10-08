// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "governance-vote.h"
#include "governance-object.h"
#include "dynode-sync.h"
#include "dynodeman.h"
#include "messagesigner.h"
#include "util.h"

std::string CGovernanceVoting::ConvertOutcomeToString(vote_outcome_enum_t nOutcome)
{
    switch(nOutcome)
    {
        case VOTE_OUTCOME_NONE:
            return "NONE"; break;
        case VOTE_OUTCOME_YES:
            return "YES"; break;
        case VOTE_OUTCOME_NO:
            return "NO"; break;
        case VOTE_OUTCOME_ABSTAIN:
            return "ABSTAIN"; break;
    }
    return "error";
}

std::string CGovernanceVoting::ConvertSignalToString(vote_signal_enum_t nSignal)
{
    std::string strReturn = "NONE";
    switch(nSignal)
    {
        case VOTE_SIGNAL_NONE:
            strReturn = "NONE";
            break;
        case VOTE_SIGNAL_FUNDING:
            strReturn = "FUNDING";
            break;
        case VOTE_SIGNAL_VALID:
            strReturn = "VALID";
            break;
        case VOTE_SIGNAL_DELETE:
            strReturn = "DELETE";
            break;
        case VOTE_SIGNAL_ENDORSED:
            strReturn = "ENDORSED";
            break;
    }

    return strReturn;
}


vote_outcome_enum_t CGovernanceVoting::ConvertVoteOutcome(const std::string& strVoteOutcome)
{
    vote_outcome_enum_t eVote = VOTE_OUTCOME_NONE;
    if(strVoteOutcome == "yes") {
        eVote = VOTE_OUTCOME_YES;
    }
    else if(strVoteOutcome == "no") {
        eVote = VOTE_OUTCOME_NO;
    }
    else if(strVoteOutcome == "abstain") {
        eVote = VOTE_OUTCOME_ABSTAIN;
    }
    return eVote;
}

vote_signal_enum_t CGovernanceVoting::ConvertVoteSignal(const std::string& strVoteSignal)
{
    static const std::map <std::string, vote_signal_enum_t> mapStrVoteSignals = {
        {"funding",  VOTE_SIGNAL_FUNDING},
        {"valid",    VOTE_SIGNAL_VALID},
        {"delete",   VOTE_SIGNAL_DELETE},
        {"endorsed", VOTE_SIGNAL_ENDORSED}
    };

    const auto& it = mapStrVoteSignals.find(strVoteSignal);
    if (it == mapStrVoteSignals.end()) {
        LogPrintf("CGovernanceVoting::%s -- ERROR: Unknown signal %s\n", __func__, strVoteSignal);
        return VOTE_SIGNAL_NONE;
    }
    return it->second;
}

CGovernanceVote::CGovernanceVote()
    : fValid(true),
      fSynced(false),
      nVoteSignal(int(VOTE_SIGNAL_NONE)),
      dynodeOutpoint(),
      nParentHash(),
      nVoteOutcome(int(VOTE_OUTCOME_NONE)),
      nTime(0),
      vchSig()
{}

CGovernanceVote::CGovernanceVote(const COutPoint& outpointDynodeIn, const uint256& nParentHashIn, vote_signal_enum_t eVoteSignalIn, vote_outcome_enum_t eVoteOutcomeIn)
    : fValid(true),
      fSynced(false),
      nVoteSignal(eVoteSignalIn),
      dynodeOutpoint(outpointDynodeIn),
      nParentHash(nParentHashIn),
      nVoteOutcome(eVoteOutcomeIn),
      nTime(GetAdjustedTime()),
      vchSig()
{
    UpdateHash();
}

std::string CGovernanceVote::ToString() const
{
    std::ostringstream ostr;
    ostr << dynodeOutpoint.ToStringShort() << ":"
         << nTime << ":"
         << CGovernanceVoting::ConvertOutcomeToString(GetOutcome()) << ":"
         << CGovernanceVoting::ConvertSignalToString(GetSignal());
    return ostr.str();
}

void CGovernanceVote::Relay(CConnman& connman) const
{
    // Do not relay until fully synced
    if(!dynodeSync.IsSynced()) {
        LogPrint("gobject", "CGovernanceVote::Relay -- won't relay until fully synced\n");
        return;
    }

    CInv inv(MSG_GOVERNANCE_OBJECT_VOTE, GetHash());
    connman.RelayInv(inv, MIN_GOVERNANCE_PEER_PROTO_VERSION);
}

void CGovernanceVote::UpdateHash() const
{
    // Note: doesn't match serialization

    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    ss << dynodeOutpoint << uint8_t{} << 0xffffffff; // adding dummy values here to match old hashing format
    ss << nParentHash;
    ss << nVoteSignal;
    ss << nVoteOutcome;
    ss << nTime;
    *const_cast<uint256*>(&hash) = ss.GetHash();
}

uint256 CGovernanceVote::GetHash() const
{
    return hash;
}

uint256 CGovernanceVote::GetSignatureHash() const
{
    return SerializeHash(*this);
}

bool CGovernanceVote::Sign(const CKey& keyDynode, const CPubKey& pubKeyDynode)
{
    std::string strError;

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if(!CHashSigner::SignHash(hash, keyDynode, vchSig)) {
            LogPrintf("CGovernanceVote::Sign -- SignHash() failed\n");
            return false;
        }

        if (!CHashSigner::VerifyHash(hash, pubKeyDynode, vchSig, strError)) {
            LogPrintf("CGovernanceVote::Sign -- VerifyHash() failed, error: %s\n", strError);
            return false;
        }
    } else {

        std::string strMessage = dynodeOutpoint.ToStringShort() + "|" + nParentHash.ToString() + "|" +
            std::to_string(nVoteSignal) + "|" + std::to_string(nVoteOutcome) + "|" + std::to_string(nTime);

        if(!CMessageSigner::SignMessage(strMessage, vchSig, keyDynode)) {
            LogPrintf("CGovernanceVote::Sign -- SignMessage() failed\n");
            return false;
        }

        if(!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
            LogPrintf("CGovernanceVote::Sign -- VerifyMessage() failed, error: %s\n", strError);
            return false;
        }
    }

    return true;
}

bool CGovernanceVote::CheckSignature(const CPubKey& pubKeyDynode) const
{
    std::string strError;

    if (sporkManager.IsSporkActive(SPORK_6_NEW_SIGS)) {
        uint256 hash = GetSignatureHash();

        if (!CHashSigner::VerifyHash(hash, pubKeyDynode, vchSig, strError)) {
            // could be a signature in old format
            std::string strMessage = dynodeOutpoint.ToStringShort() + "|" + nParentHash.ToString() + "|" +
                std::to_string(nVoteSignal) + "|" +
                std::to_string(nVoteOutcome) + "|" +
                std::to_string(nTime);

            if(!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
                // nope, not in old format either
                LogPrint("gobject", "CGovernanceVote::IsValid -- VerifyMessage() failed, error: %s\n", strError);
                return false;
            }
        }
    } else {
        std::string strMessage = dynodeOutpoint.ToStringShort() + "|" + nParentHash.ToString() + "|" +
            std::to_string(nVoteSignal) + "|" +
            std::to_string(nVoteOutcome) + "|" +
            std::to_string(nTime);

        if(!CMessageSigner::VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
            LogPrint("gobject", "CGovernanceVote::IsValid -- VerifyMessage() failed, error: %s\n", strError);
            return false;
        }
    }

    return true;
}

bool CGovernanceVote::IsValid(bool fSignatureCheck) const
{
    if(nTime > GetAdjustedTime() + (60*60)) {
        LogPrint("gobject", "CGovernanceVote::IsValid -- vote is too far ahead of current time - %s - nTime %lli - Max Time %lli\n", GetHash().ToString(), nTime, GetAdjustedTime() + (60*60));
        return false;
    }

    // support up to MAX_SUPPORTED_VOTE_SIGNAL, can be extended
    if(nVoteSignal > MAX_SUPPORTED_VOTE_SIGNAL)
    {
        LogPrint("gobject", "CGovernanceVote::IsValid -- Client attempted to vote on invalid signal(%d) - %s\n", nVoteSignal, GetHash().ToString());
        return false;
    }

    // 0=none, 1=yes, 2=no, 3=abstain. Beyond that reject votes
    if(nVoteOutcome > 3)
    {
        LogPrint("gobject", "CGovernanceVote::IsValid -- Client attempted to vote on invalid outcome(%d) - %s\n", nVoteSignal, GetHash().ToString());
        return false;
    }

    dynode_info_t infoDn;
    if(!dnodeman.GetDynodeInfo(dynodeOutpoint, infoDn)) {
        LogPrint("gobject", "CGovernanceVote::IsValid -- Unknown Dynode - %s\n", dynodeOutpoint.ToStringShort());
        return false;
    }

    if(!fSignatureCheck) return true;

    return CheckSignature(infoDn.pubKeyDynode);
}

bool operator==(const CGovernanceVote& vote1, const CGovernanceVote& vote2)
{
    bool fResult = ((vote1.dynodeOutpoint == vote2.dynodeOutpoint) &&
                    (vote1.nParentHash == vote2.nParentHash) &&
                    (vote1.nVoteOutcome == vote2.nVoteOutcome) &&
                    (vote1.nVoteSignal == vote2.nVoteSignal) &&
                    (vote1.nTime == vote2.nTime));
    return fResult;
}

bool operator<(const CGovernanceVote& vote1, const CGovernanceVote& vote2)
{
    bool fResult = (vote1.dynodeOutpoint < vote2.dynodeOutpoint);
    if(!fResult) {
        return false;
    }
    fResult = (vote1.dynodeOutpoint == vote2.dynodeOutpoint);

    fResult = fResult && (vote1.nParentHash < vote2.nParentHash);
    if(!fResult) {
        return false;
    }
    fResult = fResult && (vote1.nParentHash == vote2.nParentHash);

    fResult = fResult && (vote1.nVoteOutcome < vote2.nVoteOutcome);
    if(!fResult) {
        return false;
    }
    fResult = fResult && (vote1.nVoteOutcome == vote2.nVoteOutcome);

    fResult = fResult && (vote1.nVoteSignal == vote2.nVoteSignal);
    if(!fResult) {
        return false;
    }
    fResult = fResult && (vote1.nVoteSignal == vote2.nVoteSignal);

    fResult = fResult && (vote1.nTime < vote2.nTime);

    return fResult;
}