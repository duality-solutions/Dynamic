// Copyright (c) 2019 Duality Blockchain Solutions Developers


#include "fluidsovereign.h"

#include "core_io.h"
#include "fluid.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidSovereignDB* pFluidSovereignDB = NULL;

bool GetFluidSovereignData(const CScript& scriptPubKey, CFluidSovereign& entry)
{
    std::string fluidOperationString = ScriptToAsmStr(scriptPubKey);
    std::string strOperationCode = GetRidOfScriptStatement(fluidOperationString, 0);
    std::string verificationWithoutOpCode = GetRidOfScriptStatement(fluidOperationString);
    std::vector<std::string> splitString;
    HexFunctions hexConvert;
    hexConvert.ConvertToString(verificationWithoutOpCode);
    SeparateString(verificationWithoutOpCode, splitString, false);
    std::string messageTokenKey = splitString.at(0);
    std::vector<std::string> vecSplitScript;
    SeparateFluidOpString(verificationWithoutOpCode, vecSplitScript);

    if (vecSplitScript.size() == 5 && strOperationCode == "OP_SWAP_SOVEREIGN_ADDRESS") {
        std::vector<unsigned char> vchFluidOperation = CharVectorFromString(fluidOperationString);
        entry.FluidScript.insert(entry.FluidScript.end(), vchFluidOperation.begin(), vchFluidOperation.end());
        entry.SovereignAddresses.clear();
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[0], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[1], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[2], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[3], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[4], messageTokenKey).ToString()));
        std::string strTimeStamp = vecSplitScript[5];
        int64_t tokenTimeStamp;
        if (ParseInt64(strTimeStamp, &tokenTimeStamp)) {
            entry.nTimeStamp = tokenTimeStamp;
        }
        return true;
    }
    return false;
}

bool GetFluidSovereignData(const CTransaction& tx, CFluidSovereign& entry, int& nOut)
{
    int n = 0;
    for (const CTxOut& txout : tx.vout) {
        CScript txOut = txout.scriptPubKey;
        if (IsTransactionFluid(txOut)) {
            nOut = n;
            return GetFluidSovereignData(txOut, entry);
        }
        n++;
    }
    return false;
}

bool CFluidSovereign::UnserializeFromTx(const CTransaction& tx)
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if (!GetFluidSovereignData(tx, *this, nOut)) {
        SetNull();
        return false;
    }
    return true;
}

bool CFluidSovereign::UnserializeFromScript(const CScript& fluidScript)
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    if (!GetFluidSovereignData(fluidScript, *this)) {
        SetNull();
        return false;
    }
    return true;
}

void CFluidSovereign::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsFluidOp(SER_NETWORK, PROTOCOL_VERSION);
    dsFluidOp << *this;
    vchData = std::vector<unsigned char>(dsFluidOp.begin(), dsFluidOp.end());
}

std::vector<std::string> CFluidSovereign::SovereignAddressesStrings()
{
    std::vector<std::string> vchAddressStrings;
    for (const std::vector<unsigned char>& vchAddress : SovereignAddresses) {
        vchAddressStrings.push_back(StringFromCharVector(vchAddress));
    }
    return vchAddressStrings;
}

CFluidSovereignDB::CFluidSovereignDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "fluid-sovereign", nCacheSize, fMemory, fWipe, obfuscate)
{
    InitEmpty();
}

void CFluidSovereignDB::InitEmpty()
{
    if (IsEmpty()) {
        LOCK(cs_fluid_sovereign);
        CFluidParameters initSovereign;
        std::vector<std::vector<unsigned char> > vchAddresses = initSovereign.InitialiseAddressCharVector();
        CFluidSovereign fluidSovereign;
        for (const std::vector<unsigned char>& sovereignId : vchAddresses) {
            fluidSovereign.SovereignAddresses.push_back(sovereignId);
        }
        fluidSovereign.FluidScript = CharVectorFromString("init sovereign");
        fluidSovereign.nTimeStamp = 1;
        fluidSovereign.nHeight = 1;
        if (!AddFluidSovereignEntry(fluidSovereign)) {
            LogPrintf("CFluidSovereignDB::InitEmpty add failed.\n");
        }
    }
}

bool CFluidSovereignDB::AddFluidSovereignEntry(const CFluidSovereign& entry)
{
    bool writeState = false;
    {
        LOCK(cs_fluid_sovereign);
        writeState = Write(make_pair(std::string("script"), entry.FluidScript), entry) && Write(make_pair(std::string("txid"), entry.txHash), entry.FluidScript);
    }
    return writeState;
}

bool CFluidSovereignDB::GetLastFluidSovereignRecord(CFluidSovereign& returnEntry)
{
    LOCK(cs_fluid_sovereign);
    returnEntry.SetNull();
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidSovereign entry;
        try {
            if (pcursor->GetKey(key) && key.first == "script") {
                pcursor->GetValue(entry);
                if (entry.nHeight > returnEntry.nHeight) {
                    returnEntry = entry;
                }
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CFluidSovereignDB::GetAllFluidSovereignRecords(std::vector<CFluidSovereign>& entries)
{
    LOCK(cs_fluid_sovereign);
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidSovereign entry;
        try {
            if (pcursor->GetKey(key) && key.first == "script") {
                pcursor->GetValue(entry);
                if (!entry.IsNull()) {
                    entries.push_back(entry);
                }
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return error("%s() : deserialize error", __PRETTY_FUNCTION__);
        }
    }
    return true;
}

bool CFluidSovereignDB::IsEmpty()
{
    LOCK(cs_fluid_sovereign);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    if (pcursor->Valid()) {
        CFluidSovereign entry;
        try {
            std::pair<std::string, std::vector<unsigned char> > key;
            if (pcursor->GetKey(key) && key.first == "script") {
                pcursor->GetValue(entry);
            }
            pcursor->Next();
        } catch (std::exception& e) {
            return true;
        }
        return false;
    }
    return true;
}

bool CheckFluidSovereignDB()
{
    if (!pFluidSovereignDB)
        return false;

    return true;
}
