// Copyright (c) 2014-2016 The Dash Core developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "governance-object.h"

#include "core_io.h"
#include "util.h"

#include "dynodeman.h"
#include "governance.h"
#include "governance-classes.h"
#include "governance-vote.h"
#include "privatesend.h"

#include <univalue.h>

CGovernanceObject::CGovernanceObject()
: cs(),
  nObjectType(GOVERNANCE_OBJECT_UNKNOWN),
  nHashParent(),
  nRevision(0),
  nTime(0),
  nDeletionTime(0),
  nCollateralHash(),
  strData(),
  vinDynode(),
  vchSig(),
  fCachedLocalValidity(false),
  strLocalValidityError(),
  fCachedFunding(false),
  fCachedValid(true),
  fCachedDelete(false),
  fCachedEndorsed(false),
  fDirtyCache(true),
  fExpired(false),
  fUnparsable(false),
  mapCurrentDNVotes(),
  mapOrphanVotes(),
  fileVotes()
{
    // PARSE JSON DATA STORAGE (STRDATA)
    LoadData();
}

CGovernanceObject::CGovernanceObject(uint256 nHashParentIn, int nRevisionIn, int64_t nTimeIn, uint256 nCollateralHashIn, std::string strDataIn)
: cs(),
  nObjectType(GOVERNANCE_OBJECT_UNKNOWN),
  nHashParent(nHashParentIn),
  nRevision(nRevisionIn),
  nTime(nTimeIn),
  nDeletionTime(0),
  nCollateralHash(nCollateralHashIn),
  strData(strDataIn),
  vinDynode(),
  vchSig(),
  fCachedLocalValidity(false),
  strLocalValidityError(),
  fCachedFunding(false),
  fCachedValid(true),
  fCachedDelete(false),
  fCachedEndorsed(false),
  fDirtyCache(true),
  fExpired(false),
  fUnparsable(false),
  mapCurrentDNVotes(),
  mapOrphanVotes(),
  fileVotes()
{
    // PARSE JSON DATA STORAGE (STRDATA)
    LoadData();
}

CGovernanceObject::CGovernanceObject(const CGovernanceObject& other)
: cs(),
  nObjectType(other.nObjectType),
  nHashParent(other.nHashParent),
  nRevision(other.nRevision),
  nTime(other.nTime),
  nDeletionTime(other.nDeletionTime),
  nCollateralHash(other.nCollateralHash),
  strData(other.strData),
  vinDynode(other.vinDynode),
  vchSig(other.vchSig),
  fCachedLocalValidity(other.fCachedLocalValidity),
  strLocalValidityError(other.strLocalValidityError),
  fCachedFunding(other.fCachedFunding),
  fCachedValid(other.fCachedValid),
  fCachedDelete(other.fCachedDelete),
  fCachedEndorsed(other.fCachedEndorsed),
  fDirtyCache(other.fDirtyCache),
  fExpired(other.fExpired),
  fUnparsable(other.fUnparsable),
  mapCurrentDNVotes(other.mapCurrentDNVotes),
  mapOrphanVotes(other.mapOrphanVotes),
  fileVotes(other.fileVotes)
{}

bool CGovernanceObject::ProcessVote(CNode* pfrom,
                                    const CGovernanceVote& vote,
                                    CGovernanceException& exception)
{
    int nDNIndex = governance.GetDynodeIndex(vote.GetVinDynode());
    if(nDNIndex < 0) {
        std::ostringstream ostr;
        ostr << "CGovernanceObject::ProcessVote -- Dynode index not found\n";
        exception = CGovernanceException(ostr.str(), GOVERNANCE_EXCEPTION_WARNING);
        if(mapOrphanVotes.Insert(vote.GetVinDynode(), vote_time_pair_t(vote, GetAdjustedTime() + GOVERNANCE_ORPHAN_EXPIRATION_TIME))) {
            if(pfrom) {
                dnodeman.AskForDN(pfrom, vote.GetVinDynode());
            }
            LogPrintf(ostr.str().c_str());
        }
        else {
            LogPrint("gobject", ostr.str().c_str());
        }
        return false;
    }

    vote_m_it it = mapCurrentDNVotes.find(nDNIndex);
    if(it == mapCurrentDNVotes.end()) {
        it = mapCurrentDNVotes.insert(vote_m_t::value_type(nDNIndex,vote_rec_t())).first;
    }
    vote_rec_t& recVote = it->second;
    vote_signal_enum_t eSignal = vote.GetSignal();
    if(eSignal == VOTE_SIGNAL_NONE) {
        std::ostringstream ostr;
        ostr << "CGovernanceObject::ProcessVote -- Vote signal: none" << "\n";
        LogPrint("gobject", ostr.str().c_str());
        exception = CGovernanceException(ostr.str(), GOVERNANCE_EXCEPTION_WARNING);
        return false;
    }
    if(eSignal > MAX_SUPPORTED_VOTE_SIGNAL) {
        std::ostringstream ostr;
        ostr << "CGovernanceObject::ProcessVote -- Unsupported vote signal:" << CGovernanceVoting::ConvertSignalToString(vote.GetSignal()) << "\n";
        LogPrintf(ostr.str().c_str());
        exception = CGovernanceException(ostr.str(), GOVERNANCE_EXCEPTION_PERMANENT_ERROR, 20);
        return false;
    }
    vote_instance_m_it it2 = recVote.mapInstances.find(int(eSignal));
    if(it2 == recVote.mapInstances.end()) {
        it2 = recVote.mapInstances.insert(vote_instance_m_t::value_type(int(eSignal), vote_instance_t())).first;
    }
    vote_instance_t& voteInstance = it2->second;

    // Reject obsolete votes
    if(vote.GetTimestamp() < voteInstance.nCreationTime) {
        std::ostringstream ostr;
        ostr << "CGovernanceObject::ProcessVote -- Obsolete vote" << "\n";
        LogPrint("gobject", ostr.str().c_str());
        exception = CGovernanceException(ostr.str(), GOVERNANCE_EXCEPTION_NONE);
        return false;
    }

    int64_t nNow = GetTime();
    int64_t nVoteTimeUpdate = voteInstance.nTime;
    if(governance.AreRateChecksEnabled()) {
        int64_t nTimeDelta = nNow - voteInstance.nTime;
        if(nTimeDelta < GOVERNANCE_UPDATE_MIN) {
            std::ostringstream ostr;
            ostr << "CGovernanceObject::ProcessVote -- Dynode voting too often "
                 << ", DN outpoint = " << vote.GetVinDynode().prevout.ToStringShort()
                 << ", governance object hash = " << GetHash().ToString()
                 << ", time delta = " << nTimeDelta << "\n";
            LogPrint("gobject", ostr.str().c_str());
            exception = CGovernanceException(ostr.str(), GOVERNANCE_EXCEPTION_TEMPORARY_ERROR);
            nVoteTimeUpdate = nNow;
            return false;
        }
    }
    // Finally check that the vote is actually valid (done last because of cost of signature verification)
    if(!vote.IsValid(true)) {
        std::ostringstream ostr;
        ostr << "CGovernanceObject::ProcessVote -- Invalid vote "
                << ", DN outpoint = " << vote.GetVinDynode().prevout.ToStringShort()
                << ", governance object hash = " << GetHash().ToString()
                << ", vote hash = " << vote.GetHash().ToString() << "\n";
        LogPrintf(ostr.str().c_str());
        exception = CGovernanceException(ostr.str(), GOVERNANCE_EXCEPTION_PERMANENT_ERROR, 20);
        governance.AddInvalidVote(vote);
        return false;
    }
    voteInstance = vote_instance_t(vote.GetOutcome(), nVoteTimeUpdate, vote.GetTimestamp());
    fileVotes.AddVote(vote);
    dnodeman.AddGovernanceVote(vote.GetVinDynode(), vote.GetParentHash());
    fDirtyCache = true;
    return true;
}

void CGovernanceObject::RebuildVoteMap()
{
    vote_m_t mapDNVotesNew;
    for(vote_m_it it = mapCurrentDNVotes.begin(); it != mapCurrentDNVotes.end(); ++it) {
        CTxIn vinDynode;
        if(dnodeman.GetDynodeVinForIndexOld(it->first, vinDynode)) {
            int nNewIndex = dnodeman.GetDynodeIndex(vinDynode);
            if((nNewIndex >= 0)) {
                mapDNVotesNew[nNewIndex] = it->second;
            }
        }
    }
    mapCurrentDNVotes = mapDNVotesNew;
}

void CGovernanceObject::ClearDynodeVotes()
{
    vote_m_it it = mapCurrentDNVotes.begin();
    while(it != mapCurrentDNVotes.end()) {
        bool fIndexRebuilt = false;
        CTxIn vinDynode;
        bool fRemove = true;
        if(dnodeman.Get(it->first, vinDynode, fIndexRebuilt)) {
            if(dnodeman.Has(vinDynode)) {
                fRemove = false;
            }
            else {
                fileVotes.RemoveVotesFromDynode(vinDynode);
            }
        }

        if(fRemove) {
            mapCurrentDNVotes.erase(it++);
        }
        else {
            ++it;
        }
    }
}

std::string CGovernanceObject::GetSignatureMessage() const
{
    LOCK(cs);
    std::string strMessage = nHashParent.ToString() + "|" +
        boost::lexical_cast<std::string>(nRevision) + "|" +
        boost::lexical_cast<std::string>(nTime) + "|" +
        strData + "|" +
        vinDynode.prevout.ToStringShort() + "|" +
        nCollateralHash.ToString();

    return strMessage;
}

void CGovernanceObject::SetDynodeInfo(const CTxIn& vin)
{
    vinDynode = vin;
}

bool CGovernanceObject::Sign(CKey& keyDynode, CPubKey& pubKeyDynode)
{
    std::string strError;
    std::string strMessage = GetSignatureMessage();

    LOCK(cs);

    if(!privateSendSigner.SignMessage(strMessage, vchSig, keyDynode)) {
        LogPrintf("CGovernanceObject::Sign -- SignMessage() failed\n");
        return false;
    }

    if(!privateSendSigner.VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CGovernanceObject::Sign -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    LogPrint("gobject", "CGovernanceObject::Sign -- pubkey id = %s, vin = %s\n",
             pubKeyDynode.GetID().ToString(), vinDynode.prevout.ToStringShort());


    return true;
}

bool CGovernanceObject::CheckSignature(CPubKey& pubKeyDynode)
{
    std::string strError;

    std::string strMessage = GetSignatureMessage();

    LOCK(cs);
    if(!privateSendSigner.VerifyMessage(pubKeyDynode, vchSig, strMessage, strError)) {
        LogPrintf("CGovernance::CheckSignature -- VerifyMessage() failed, error: %s\n", strError);
        return false;
    }

    return true;
}

int CGovernanceObject::GetObjectSubtype()
{
    if(nObjectType == GOVERNANCE_OBJECT_TRIGGER) return TRIGGER_SUPERBLOCK;
    return -1;
}

uint256 CGovernanceObject::GetHash() const
{
    // CREATE HASH OF ALL IMPORTANT PIECES OF DATA

    CHashWriter ss(SER_GETHASH, PROTOCOL_VERSION);
    ss << nHashParent;
    ss << nRevision;
    ss << nTime;
    ss << strData;
    // fee_tx is left out on purpose
    uint256 h1 = ss.GetHash();

    DBG( printf("CGovernanceObject::GetHash %i %li %s\n", nRevision, nTime, strData.c_str()); );

    return h1;
}

/**
   Return the actual object from the strData JSON structure.

   Returns an empty object on error.
 */
UniValue CGovernanceObject::GetJSONObject()
{
    UniValue obj(UniValue::VOBJ);
    if(strData.empty()) {
        return obj;
    }

    UniValue objResult(UniValue::VOBJ);
    GetData(objResult);

    std::vector<UniValue> arr1 = objResult.getValues();
    std::vector<UniValue> arr2 = arr1.at( 0 ).getValues();
    obj = arr2.at( 1 );

    return obj;
}

/**
*   LoadData
*   --------------------------------------------------------
*
*   Attempt to load data from strData
*
*/

void CGovernanceObject::LoadData()
{
    if(strData.empty()) {
        return;
    }

    try  {
        // ATTEMPT TO LOAD JSON STRING FROM STRDATA
        UniValue objResult(UniValue::VOBJ);
        GetData(objResult);

        DBG( cout << "CGovernanceObject::LoadData strData = "
             << GetDataAsString()
             << endl; );

        UniValue obj = GetJSONObject();
        nObjectType = obj["type"].get_int();
    }
    catch(std::exception& e) {
        fUnparsable = true;
        std::ostringstream ostr;
        ostr << "CGovernanceObject::LoadData Error parsing JSON"
             << ", e.what() = " << e.what();
        DBG( cout << ostr.str() << endl; );
        LogPrintf( ostr.str().c_str() );
        return;
    }
    catch(...) {
        fUnparsable = true;
        std::ostringstream ostr;
        ostr << "CGovernanceObject::LoadData Unknown Error parsing JSON";
        DBG( cout << ostr.str() << endl; );
        LogPrintf( ostr.str().c_str() );
        return;
    }
}

/**
*   GetData - Example usage:
*   --------------------------------------------------------
*
*   Decode governance object data into UniValue(VOBJ)
*
*/

void CGovernanceObject::GetData(UniValue& objResult)
{
    UniValue o(UniValue::VOBJ);
    std::string s = GetDataAsString();
    o.read(s);
    objResult = o;
}

/**
*   GetData - As
*   --------------------------------------------------------
*
*/

std::string CGovernanceObject::GetDataAsHex()
{
    return strData;
}

std::string CGovernanceObject::GetDataAsString()
{
    std::vector<unsigned char> v = ParseHex(strData);
    std::string s(v.begin(), v.end());

    return s;
}

void CGovernanceObject::UpdateLocalValidity()
{
    // THIS DOES NOT CHECK COLLATERAL, THIS IS CHECKED UPON ORIGINAL ARRIVAL
    fCachedLocalValidity = IsValidLocally(strLocalValidityError, false);
};


bool CGovernanceObject::IsValidLocally(std::string& strError, bool fCheckCollateral)
{
    bool fMissingDynode = false;

    return IsValidLocally(strError, fMissingDynode, fCheckCollateral);
}

bool CGovernanceObject::IsValidLocally(std::string& strError, bool& fMissingDynode, bool fCheckCollateral)
{
    fMissingDynode = false;

    if(fUnparsable) {
        strError = "Object data unparseable";
        return false;
    }

    switch(nObjectType) {
        case GOVERNANCE_OBJECT_PROPOSAL:
        case GOVERNANCE_OBJECT_TRIGGER:
        case GOVERNANCE_OBJECT_WATCHDOG:
            break;
        default:
            strError = strprintf("Invalid object type %d", nObjectType);
            return false;
    }

    // IF ABSOLUTE NO COUNT (NO-YES VALID VOTES) IS MORE THAN 10% OF THE NETWORK DYNODES, OBJ IS INVALID

    // CHECK COLLATERAL IF REQUIRED (HIGH CPU USAGE)

    if(fCheckCollateral) { 
        if((nObjectType == GOVERNANCE_OBJECT_TRIGGER) || (nObjectType == GOVERNANCE_OBJECT_WATCHDOG)) {
            std::string strOutpoint = vinDynode.prevout.ToStringShort();
            dynode_info_t infoDn = dnodeman.GetDynodeInfo(vinDynode);
            if(!infoDn.fInfoValid) {
                fMissingDynode = true;
                strError = "Dynode not found: " + strOutpoint;
                return false;
            }

            // Check that we have a valid DN signature
            if(!CheckSignature(infoDn.pubKeyDynode)) {
                strError = "Invalid dynode signature for: " + strOutpoint + ", pubkey id = " + infoDn.pubKeyDynode.GetID().ToString();
                return false;
            }
            return true;
        }

        if(!IsCollateralValid(strError)) {
            // strError set in IsCollateralValid
            if(strError == "") strError = "Collateral is invalid";
            return false;
        }
    }

    return true;
}

CAmount CGovernanceObject::GetMinCollateralFee()
{
    // Only 1 type has a fee for the moment but switch statement allows for future object types
    switch(nObjectType) {
        case GOVERNANCE_OBJECT_PROPOSAL:    return GOVERNANCE_PROPOSAL_FEE_TX;
        case GOVERNANCE_OBJECT_TRIGGER:     return 0;
        case GOVERNANCE_OBJECT_WATCHDOG:    return 0;
        default:                            return MAX_MONEY;
    }
}

bool CGovernanceObject::IsCollateralValid(std::string& strError)
{
    strError = "";
    CAmount nMinFee = GetMinCollateralFee();
    uint256 nExpectedHash = GetHash();

    CTransaction txCollateral;
    uint256 nBlockHash;

    // RETRIEVE TRANSACTION IN QUESTION

    if(!GetTransaction(nCollateralHash, txCollateral, Params().GetConsensus(), nBlockHash, true)){
        strError = strprintf("Can't find collateral tx %s", txCollateral.ToString());
        LogPrintf("CGovernanceObject::IsCollateralValid -- %s\n", strError);
        return false;
    }

    if(txCollateral.vout.size() < 1) {
        strError = strprintf("tx vout size less than 1 | %d", txCollateral.vout.size());
        LogPrintf("CGovernanceObject::IsCollateralValid -- %s\n", strError);
        return false;
    }

    // LOOK FOR SPECIALIZED GOVERNANCE SCRIPT (PROOF OF BURN)

    CScript findScript;
    findScript << OP_RETURN << ToByteVector(nExpectedHash);

    DBG( cout << "IsCollateralValid txCollateral.vout.size() = " << txCollateral.vout.size() << endl; );

    DBG( cout << "IsCollateralValid: findScript = " << ScriptToAsmStr( findScript, false ) << endl; );

    DBG( cout << "IsCollateralValid: nMinFee = " << nMinFee << endl; );


    bool foundOpReturn = false;
    BOOST_FOREACH(const CTxOut o, txCollateral.vout) {
        DBG( cout << "IsCollateralValid txout : " << o.ToString()
             << ", o.nValue = " << o.nValue
             << ", o.scriptPubKey = " << ScriptToAsmStr( o.scriptPubKey, false )
             << endl; );
        if(!o.scriptPubKey.IsNormalPaymentScript() && !o.scriptPubKey.IsUnspendable()){
            strError = strprintf("Invalid Script %s", txCollateral.ToString());
            LogPrintf ("CGovernanceObject::IsCollateralValid -- %s\n", strError);
            return false;
        }
        if(o.scriptPubKey == findScript && o.nValue >= nMinFee) {
            DBG( cout << "IsCollateralValid foundOpReturn = true" << endl; );
            foundOpReturn = true;
        }
        else  {
            DBG( cout << "IsCollateralValid No match, continuing" << endl; );
        }

    }

    if(!foundOpReturn){
        strError = strprintf("Couldn't find opReturn %s in %s", nExpectedHash.ToString(), txCollateral.ToString());
        LogPrintf ("CGovernanceObject::IsCollateralValid -- %s\n", strError);
        return false;
    }

    // GET CONFIRMATIONS FOR TRANSACTION

    LOCK(cs_main);
    int nConfirmationsIn = GetISConfirmations(nCollateralHash);
    if (nBlockHash != uint256()) {
        BlockMap::iterator mi = mapBlockIndex.find(nBlockHash);
        if (mi != mapBlockIndex.end() && (*mi).second) {
            CBlockIndex* pindex = (*mi).second;
            if (chainActive.Contains(pindex)) {
                nConfirmationsIn += chainActive.Height() - pindex->nHeight + 1;
            }
        }
    }

    if(nConfirmationsIn >= GOVERNANCE_FEE_CONFIRMATIONS) {
        strError = "valid";
    } else {
        strError = strprintf("Collateral requires at least %d confirmations - %d confirmations", GOVERNANCE_FEE_CONFIRMATIONS, nConfirmationsIn);
        LogPrintf ("CGovernanceObject::IsCollateralValid -- %s - %d confirmations\n", strError, nConfirmationsIn);
        return false;
    }

    return true;
}

int CGovernanceObject::CountMatchingVotes(vote_signal_enum_t eVoteSignalIn, vote_outcome_enum_t eVoteOutcomeIn) const
{
    int nCount = 0;
    for(vote_m_cit it = mapCurrentDNVotes.begin(); it != mapCurrentDNVotes.end(); ++it) {
        const vote_rec_t& recVote = it->second;
        vote_instance_m_cit it2 = recVote.mapInstances.find(eVoteSignalIn);
        if(it2 == recVote.mapInstances.end()) {
            continue;
        }
        const vote_instance_t& voteInstance = it2->second;
        if(voteInstance.eOutcome == eVoteOutcomeIn) {
            ++nCount;
        }
    }
    return nCount;
}

/**
*   Get specific vote counts for each outcome (funding, validity, etc)
*/

int CGovernanceObject::GetAbsoluteYesCount(vote_signal_enum_t eVoteSignalIn) const
{
    return GetYesCount(eVoteSignalIn) - GetNoCount(eVoteSignalIn);
}

int CGovernanceObject::GetAbsoluteNoCount(vote_signal_enum_t eVoteSignalIn) const
{
    return GetNoCount(eVoteSignalIn) - GetYesCount(eVoteSignalIn);
}

int CGovernanceObject::GetYesCount(vote_signal_enum_t eVoteSignalIn) const
{
    return CountMatchingVotes(eVoteSignalIn, VOTE_OUTCOME_YES);
}

int CGovernanceObject::GetNoCount(vote_signal_enum_t eVoteSignalIn) const
{
    return CountMatchingVotes(eVoteSignalIn, VOTE_OUTCOME_NO);
}

int CGovernanceObject::GetAbstainCount(vote_signal_enum_t eVoteSignalIn) const
{
    return CountMatchingVotes(eVoteSignalIn, VOTE_OUTCOME_ABSTAIN);
}

bool CGovernanceObject::GetCurrentDNVotes(const CTxIn& dnCollateralOutpoint, vote_rec_t& voteRecord)
{
    int nDNIndex = governance.GetDynodeIndex(dnCollateralOutpoint);
    vote_m_it it = mapCurrentDNVotes.find(nDNIndex);
    if (it == mapCurrentDNVotes.end()) {
        return false;
    }
    voteRecord = it->second;
    return  true;
}

void CGovernanceObject::Relay()
{
    CInv inv(MSG_GOVERNANCE_OBJECT, GetHash());
    RelayInv(inv, PROTOCOL_VERSION);
}

void CGovernanceObject::UpdateSentinelVariables()
{
    // CALCULATE MINIMUM SUPPORT LEVELS REQUIRED

    int nDnCount = dnodeman.CountEnabled();
    if(nDnCount == 0) return;

    // CALCULATE THE MINUMUM VOTE COUNT REQUIRED FOR FULL SIGNAL

    int nAbsVoteReq = std::max(Params().GetConsensus().nGovernanceMinQuorum, nDnCount / 10);
    int nAbsDeleteReq = std::max(Params().GetConsensus().nGovernanceMinQuorum, (2 * nDnCount) / 3);

    // SET SENTINEL FLAGS TO FALSE

    fCachedFunding = false;
    fCachedValid = true; //default to valid

    fCachedEndorsed = false;
    fDirtyCache = false;

    // SET SENTINEL FLAGS TO TRUE IF MIMIMUM SUPPORT LEVELS ARE REACHED
    // ARE ANY OF THESE FLAGS CURRENTLY ACTIVATED?

    if(GetAbsoluteYesCount(VOTE_SIGNAL_FUNDING) >= nAbsVoteReq) fCachedFunding = true;
    if((GetAbsoluteYesCount(VOTE_SIGNAL_DELETE) >= nAbsDeleteReq) && !fCachedDelete) {
        fCachedDelete = true;
        if(nDeletionTime == 0) {
            nDeletionTime = GetAdjustedTime();
        }
    }
    if(GetAbsoluteYesCount(VOTE_SIGNAL_ENDORSED) >= nAbsVoteReq) fCachedEndorsed = true;

    if(GetAbsoluteNoCount(VOTE_SIGNAL_VALID) >= nAbsVoteReq) fCachedValid = false;
}

void CGovernanceObject::swap(CGovernanceObject& first, CGovernanceObject& second) // nothrow
{
    // enable ADL (not necessary in our case, but good practice)
    using std::swap;

    // by swapping the members of two classes,
    // the two classes are effectively swapped
    swap(first.nHashParent, second.nHashParent);
    swap(first.nRevision, second.nRevision);
    swap(first.nTime, second.nTime);
    swap(first.nDeletionTime, second.nDeletionTime);
    swap(first.nCollateralHash, second.nCollateralHash);
    swap(first.strData, second.strData);
    swap(first.nObjectType, second.nObjectType);

    // swap all cached valid flags
    swap(first.fCachedFunding, second.fCachedFunding);
    swap(first.fCachedValid, second.fCachedValid);
    swap(first.fCachedDelete, second.fCachedDelete);
    swap(first.fCachedEndorsed, second.fCachedEndorsed);
    swap(first.fDirtyCache, second.fDirtyCache);
    swap(first.fExpired, second.fExpired);
}

void CGovernanceObject::CheckOrphanVotes()
{
    int64_t nNow = GetAdjustedTime();
    const vote_mcache_t::list_t& listVotes = mapOrphanVotes.GetItemList();
    vote_mcache_t::list_cit it = listVotes.begin();
    while(it != listVotes.end()) {
        bool fRemove = false;
        const CTxIn& key = it->key;
        const vote_time_pair_t& pairVote = it->value;
        const CGovernanceVote& vote = pairVote.first;
        if(pairVote.second < nNow) {
            fRemove = true;
        }
        else if(!dnodeman.Has(vote.GetVinDynode())) {
            ++it;
            continue;
        }
        CGovernanceException exception;
        if(!ProcessVote(NULL, vote, exception)) {
            LogPrintf("CGovernanceObject::CheckOrphanVotes -- Failed to add orphan vote: %s\n", exception.what());
        }
        else {
            vote.Relay();
            fRemove = true;
        }
        ++it;
        if(fRemove) {
            mapOrphanVotes.Erase(key, pairVote);
        }
    }
}