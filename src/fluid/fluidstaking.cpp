// Copyright (c) 2019 Duality Blockchain Solutions Developers


#include "fluidstaking.h"

#include "core_io.h"
#include "fluid.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidStakingDB* pFluidStakingDB = NULL;

bool GetFluidStakingData(const CScript& scriptPubKey, CFluidStaking& entry)
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

    if (vecSplitScript.size() == 5 && strOperationCode == "OP_REWARD_STAKE") {
        std::vector<unsigned char> vchFluidOperation = CharVectorFromString(fluidOperationString);
        entry.FluidScript.insert(entry.FluidScript.end(), vchFluidOperation.begin(), vchFluidOperation.end());
        std::string strAmount = vecSplitScript[0];
        CAmount fluidAmount;
        if (ParseFixedPoint(strAmount, 8, &fluidAmount)) {
            entry.StakeReward = fluidAmount;
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

        LogPrintf("GetFluidStakingData: strAmount = %s, strTimeStamp = %d, Addresses1 = %s, Addresses2 = %s, Addresses3 = %s \n",
            strAmount, entry.nTimeStamp, StringFromCharVector(entry.SovereignAddresses[0]),
            StringFromCharVector(entry.SovereignAddresses[1]), StringFromCharVector(entry.SovereignAddresses[2]));

        return true;
    }
    return false;
}

bool GetFluidStakingData(const CTransaction& tx, CFluidStaking& entry, int& nOut)
{
    int n = 0;
    for (const CTxOut& txout : tx.vout) {
        CScript txOut = txout.scriptPubKey;
        if (IsTransactionFluid(txOut)) {
            nOut = n;
            return GetFluidStakingData(txOut, entry);
        }
        n++;
    }
    return false;
}

bool CFluidStaking::UnserializeFromTx(const CTransaction& tx)
{
    int nOut;
    if (!GetFluidStakingData(tx, *this, nOut)) {
        SetNull();
        return false;
    }
    return true;
}

bool CFluidStaking::UnserializeFromScript(const CScript& fluidScript)
{
    if (!GetFluidStakingData(fluidScript, *this)) {
        SetNull();
        return false;
    }
    return true;
}

void CFluidStaking::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsFluidOp(SER_NETWORK, PROTOCOL_VERSION);
    dsFluidOp << *this;
    vchData = std::vector<unsigned char>(dsFluidOp.begin(), dsFluidOp.end());
}

CFluidStakingDB::CFluidStakingDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "fluid-staking", nCacheSize, fMemory, fWipe, obfuscate)
{
}

bool CFluidStakingDB::AddFluidStakingEntry(const CFluidStaking& entry, const int op)
{
    bool writeState = false;
    {
        LOCK(cs_fluid_staking);
        writeState = Write(make_pair(std::string("script"), entry.FluidScript), entry) && Write(make_pair(std::string("txid"), entry.txHash), entry.FluidScript);
    }

    return writeState;
}

bool CFluidStakingDB::GetLastFluidStakingRecord(CFluidStaking& returnEntry, const int nHeight)
{
    LOCK(cs_fluid_staking);
    returnEntry.SetNull();
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidStaking entry;
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

bool CFluidStakingDB::GetAllFluidStakingRecords(std::vector<CFluidStaking>& entries)
{
    LOCK(cs_fluid_staking);
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidStaking entry;
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

bool CFluidStakingDB::IsEmpty()
{
    LOCK(cs_fluid_staking);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    if (pcursor->Valid()) {
        CFluidStaking entry;
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

bool CFluidStakingDB::RecordExists(const std::vector<unsigned char>& vchFluidScript)
{
    LOCK(cs_fluid_staking);
    CFluidStaking fluidStaking;
    return CDBWrapper::Read(make_pair(std::string("script"), vchFluidScript), fluidStaking);
}

bool CheckFluidStakingDB()
{
    if (!pFluidStakingDB)
        return false;

    return true;
}
