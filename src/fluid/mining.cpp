// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers


#include "fluid/mining.h"

#include "core_io.h"
#include "fluid/fluid.h"
#include "fluid/script.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidMiningDB* pFluidMiningDB = NULL;

bool GetFluidMiningData(const CTransaction& tx, CFluidMining& entry, int& nOut)
{
    int n = 0;
    for (const CTxOut& txout : tx.vout) {
        CScript txOut = txout.scriptPubKey;
        if (WithinFluidRange(txOut.GetFlag())) {
            nOut = n;
            return ParseScript(txOut, entry);
        }
        n++;
    }
    return false;
}

bool CFluidMining::UnserializeFromTx(const CTransaction& tx)
{
    int nOut;
    if (!GetFluidMiningData(tx, *this, nOut)) {
        return false;
    }
    return true;
}

bool CFluidMining::UnserializeFromScript(const CScript& fluidScript)
{
    if (!ParseScript(fluidScript, *this)) {
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
        writeState = Write(make_pair(std::string("script"), entry.GetTransactionScript()), entry) && Write(make_pair(std::string("txid"), entry.GetTransactionHash()), entry.GetTransactionScript());
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
                if (entry.GetHeight() > returnEntry.GetHeight() && (int)(entry.GetHeight() + 1) < nHeight) {
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
