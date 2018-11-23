// Copyright (c) 2019 Duality Blockchain Solutions Developers


#include "fluidmining.h"

#include "core_io.h"
#include "fluid.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidMiningDB* pFluidMiningDB = NULL;

bool GetFluidMiningData(const CScript& scriptPubKey, CFluidMining& entry)
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

    if (vecSplitScript.size() == 5 && strOperationCode == "OP_REWARD_MINING") {
        std::vector<unsigned char> vchFluidOperation = CharVectorFromString(fluidOperationString);
        entry.FluidScript.insert(entry.FluidScript.end(), vchFluidOperation.begin(), vchFluidOperation.end());
        std::string strAmount = vecSplitScript[0];
        CAmount fluidAmount;
        if (ParseFixedPoint(strAmount, 8, &fluidAmount)) {
            entry.MiningReward = fluidAmount;
        }
        std::string strTimeStamp = vecSplitScript[1];
        int64_t tokenTimeStamp;
        if (ParseInt64(strTimeStamp, &tokenTimeStamp)) {
            entry.nTimeStamp = tokenTimeStamp;
        }
        entry.SovereignAddresses.clear();
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[2], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[3], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[4], messageTokenKey).ToString()));

        LogPrintf("GetFluidMiningData: strAmount = %s, strTimeStamp = %d, Addresses1 = %s, Addresses2 = %s, Addresses3 = %s \n",
            strAmount, entry.nTimeStamp, StringFromCharVector(entry.SovereignAddresses[0]),
            StringFromCharVector(entry.SovereignAddresses[1]), StringFromCharVector(entry.SovereignAddresses[2]));

        return true;
    }
    return false;
}

bool GetFluidMiningData(const CTransaction& tx, CFluidMining& entry, int& nOut)
{
    int n = 0;
    for (const CTxOut& txout : tx.vout) {
        CScript txOut = txout.scriptPubKey;
        if (IsTransactionFluid(txOut)) {
            nOut = n;
            return GetFluidMiningData(txOut, entry);
        }
        n++;
    }
    return false;
}

bool CFluidMining::UnserializeFromTx(const CTransaction& tx)
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if (!GetFluidMiningData(tx, *this, nOut)) {
        SetNull();
        return false;
    }
    return true;
}

bool CFluidMining::UnserializeFromScript(const CScript& fluidScript)
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    if (!GetFluidMiningData(fluidScript, *this)) {
        SetNull();
        return false;
    }
    return true;
}

void CFluidMining::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsFluidOp(SER_NETWORK, PROTOCOL_VERSION);
    dsFluidOp << *this;
    vchData = std::vector<unsigned char>(dsFluidOp.begin(), dsFluidOp.end());
}

CFluidMiningDB::CFluidMiningDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "fluid-mining", nCacheSize, fMemory, fWipe, obfuscate)
{
}

bool CFluidMiningDB::AddFluidMiningEntry(const CFluidMining& entry, const int op)
{
    bool writeState = false;
    {
        LOCK(cs_fluid_mining);
        writeState = Write(make_pair(std::string("script"), entry.FluidScript), entry) && Write(make_pair(std::string("txid"), entry.txHash), entry.FluidScript);
    }

    return writeState;
}

bool CFluidMiningDB::GetLastFluidMiningRecord(CFluidMining& returnEntry, const int nHeight)
{
    LOCK(cs_fluid_mining);
    returnEntry.SetNull();
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidMining entry;
        try {
            if (pcursor->GetKey(key) && key.first == "script") {
                pcursor->GetValue(entry);
                if (entry.IsNull()) {
                    return false;
                }
                if (entry.nHeight > returnEntry.nHeight && (int)(entry.nHeight + 1) < nHeight) {
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

bool CFluidMiningDB::GetAllFluidMiningRecords(std::vector<CFluidMining>& entries)
{
    LOCK(cs_fluid_mining);
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidMining entry;
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

bool CFluidMiningDB::IsEmpty()
{
    LOCK(cs_fluid_mining);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    if (pcursor->Valid()) {
        CFluidMining entry;
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

bool CFluidMiningDB::RecordExists(const std::vector<unsigned char>& vchFluidScript)
{
    LOCK(cs_fluid_mining);
    CFluidMining fluidMining;
    return CDBWrapper::Read(make_pair(std::string("script"), vchFluidScript), fluidMining);
}

bool CheckFluidMiningDB()
{
    if (!pFluidMiningDB)
        return false;

    return true;
}
