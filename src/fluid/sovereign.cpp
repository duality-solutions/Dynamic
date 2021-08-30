// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers


#include "fluid/sovereign.h"

#include "core_io.h"
#include "fluid/fluid.h"
#include "fluid/script.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidSovereignDB* pFluidSovereignDB = NULL;

bool CFluidSovereign::UnserializeFromTx(const CTransaction& tx)
{
    int nOut;
    return ParseData(tx, *this, nOut);
}

bool CFluidSovereign::UnserializeFromScript(const CScript& fluidScript)
{
    return ParseScript(fluidScript, *this);
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
    for (const std::vector<unsigned char>& vchAddress : obj_sigs) {
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
        CFluidSovereign fluidSovereign;
        for (const auto& pk : Params().FluidSignatureKeys()) {
            fluidSovereign.obj_sigs.insert(
              CharVectorFromString(CDynamicAddress(pk).ToString())
            );
        }
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
        writeState = Write(make_pair(std::string("script"), entry.GetTransactionScript()), entry) && Write(make_pair(std::string("txid"), entry.GetTransactionHash()), entry.GetTransactionScript());
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
                if (entry.GetHeight() > returnEntry.GetHeight()) {
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
