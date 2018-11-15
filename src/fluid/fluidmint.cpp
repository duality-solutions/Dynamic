// Copyright (c) 2017 Duality Blockchain Solutions Developers


#include "fluidmint.h"

#include "base58.h"
#include "core_io.h"
#include "fluid.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidMintDB* pFluidMintDB = NULL;

bool GetFluidMintData(const CScript& scriptPubKey, CFluidMint& entry)
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

    if (vecSplitScript.size() >= 6 && strOperationCode == "OP_MINT") {
        std::vector<unsigned char> vchFluidOperation = CharVectorFromString(fluidOperationString);
        entry.FluidScript.insert(entry.FluidScript.end(), vchFluidOperation.begin(), vchFluidOperation.end());
        std::string strAmount = vecSplitScript[0];
        CAmount fluidAmount;
        if (ParseFixedPoint(strAmount, 8, &fluidAmount)) {
            entry.MintAmount = fluidAmount;
        }
        std::string strTimeStamp = vecSplitScript[1];
        int64_t tokenTimeStamp;
        if (ParseInt64(strTimeStamp, &tokenTimeStamp)) {
            entry.nTimeStamp = tokenTimeStamp;
        }
        std::vector<unsigned char> vchDestinationAddress = CharVectorFromString(vecSplitScript[2]);
        entry.DestinationAddress.insert(entry.DestinationAddress.end(), vchDestinationAddress.begin(), vchDestinationAddress.end());
        entry.SovereignAddresses.clear();
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[3], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[4], messageTokenKey).ToString()));
        entry.SovereignAddresses.push_back(CharVectorFromString(fluid.GetAddressFromDigestSignature(vecSplitScript[5], messageTokenKey).ToString()));

        LogPrintf("GetFluidMintData: strAmount = %s, strTimeStamp = %d, DestinationAddress = %s, Addresses1 = %s, Addresses2 = %s, Addresses3 = %s \n",
            strAmount, entry.nTimeStamp,
            StringFromCharVector(entry.DestinationAddress), StringFromCharVector(entry.SovereignAddresses[0]),
            StringFromCharVector(entry.SovereignAddresses[1]), StringFromCharVector(entry.SovereignAddresses[2]));

        return true;
    }
    return false;
}

bool GetFluidMintData(const CTransaction& tx, CFluidMint& entry, int& nOut)
{
    int n = 0;
    for (const CTxOut& txout : tx.vout) {
        CScript txOut = txout.scriptPubKey;
        if (IsTransactionFluid(txOut)) {
            nOut = n;
            return GetFluidMintData(txOut, entry);
        }
        n++;
    }
    return false;
}

bool CFluidMint::UnserializeFromTx(const CTransaction& tx)
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    int nOut;
    if (!GetFluidMintData(tx, *this, nOut)) {
        SetNull();
        return false;
    }
    return true;
}

bool CFluidMint::UnserializeFromScript(const CScript& fluidScript)
{
    std::vector<unsigned char> vchData;
    std::vector<unsigned char> vchHash;
    if (!GetFluidMintData(fluidScript, *this)) {
        SetNull();
        return false;
    }
    return true;
}

void CFluidMint::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsFluidOp(SER_NETWORK, PROTOCOL_VERSION);
    dsFluidOp << *this;
    vchData = std::vector<unsigned char>(dsFluidOp.begin(), dsFluidOp.end());
}

CDynamicAddress CFluidMint::GetDestinationAddress() const
{
    return CDynamicAddress(StringFromCharVector(DestinationAddress));
}

CFluidMintDB::CFluidMintDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "fluid-mint", nCacheSize, fMemory, fWipe, obfuscate)
{
}

bool CFluidMintDB::AddFluidMintEntry(const CFluidMint& entry, const int op)
{
    bool writeState = false;
    {
        LOCK(cs_fluid_mint);
        writeState = Write(make_pair(std::string("script"), entry.FluidScript), entry) && Write(make_pair(std::string("txid"), entry.txHash), entry.FluidScript);
    }

    return writeState;
}

bool CFluidMintDB::GetLastFluidMintRecord(CFluidMint& returnEntry)
{
    LOCK(cs_fluid_mint);
    returnEntry.SetNull();
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidMint entry;
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

bool CFluidMintDB::GetAllFluidMintRecords(std::vector<CFluidMint>& entries)
{
    LOCK(cs_fluid_mint);
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidMint entry;
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

bool CFluidMintDB::IsEmpty()
{
    LOCK(cs_fluid_mint);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    if (pcursor->Valid()) {
        CFluidMint entry;
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

bool CFluidMintDB::RecordExists(const std::vector<unsigned char>& vchFluidScript)
{
    LOCK(cs_fluid_mint);
    CFluidMint fluidMint;
    return CDBWrapper::Read(make_pair(std::string("script"), vchFluidScript), fluidMint);
}

bool CheckFluidMintDB()
{
    if (!pFluidMintDB)
        return false;

    return true;
}
