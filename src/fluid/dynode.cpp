// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers


#include "fluid/dynode.h"

#include "core_io.h"
#include "fluid/fluid.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidDynodeDB* pFluidDynodeDB = NULL;

bool GetFluidDynodeData(const CScript& scriptPubKey, CFluidDynode& entry)
{
    return ParseScript(scriptPubKey, entry);
}

bool GetFluidDynodeData(const CTransaction& tx, CFluidDynode& entry, int& nOut)
{
    int n = 0;
    for (const CTxOut& txout : tx.vout) {
        CScript txOut = txout.scriptPubKey;
        if (IsTransactionFluid(txOut)) {
            nOut = n;
            return GetFluidDynodeData(txOut, entry);
        }
        n++;
    }
    return false;
}

bool CFluidDynode::UnserializeFromTx(const CTransaction& tx)
{
    int nOut;
    if (!GetFluidDynodeData(tx, *this, nOut)) {
        return false;
    }
    return true;
}

bool CFluidDynode::UnserializeFromScript(const CScript& fluidScript)
{
    if (!GetFluidDynodeData(fluidScript, *this)) {
        return false;
    }
    return true;
}

void CFluidDynode::Serialize(std::vector<unsigned char>& vchData)
{
    CDataStream dsFluidOp(SER_NETWORK, PROTOCOL_VERSION);
    dsFluidOp << *this;
    vchData = std::vector<unsigned char>(dsFluidOp.begin(), dsFluidOp.end());
}

CFluidDynodeDB::CFluidDynodeDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "fluid-dynode", nCacheSize, fMemory, fWipe, obfuscate)
{
}

bool CFluidDynodeDB::AddFluidDynodeEntry(const CFluidDynode& entry, const int op)
{
    bool writeState = false;
    {
        LOCK(cs_fluid_dynode);
        writeState = Write(make_pair(std::string("script"), entry.GetTransactionScript()), entry) && Write(make_pair(std::string("txid"), entry.GetTransactionHash()), entry.GetTransactionScript());
    }

    return writeState;
}

bool CFluidDynodeDB::GetLastFluidDynodeRecord(CFluidDynode& returnEntry, const int nHeight)
{
    LOCK(cs_fluid_dynode);
    returnEntry.SetNull();
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidDynode entry;
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

bool CFluidDynodeDB::GetAllFluidDynodeRecords(std::vector<CFluidDynode>& entries)
{
    LOCK(cs_fluid_dynode);
    std::pair<std::string, std::vector<unsigned char> > key;
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    while (pcursor->Valid()) {
        boost::this_thread::interruption_point();
        CFluidDynode entry;
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

bool CFluidDynodeDB::IsEmpty()
{
    LOCK(cs_fluid_dynode);
    std::unique_ptr<CDBIterator> pcursor(NewIterator());
    pcursor->SeekToFirst();
    if (pcursor->Valid()) {
        CFluidDynode entry;
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

bool CFluidDynodeDB::RecordExists(const std::vector<unsigned char>& vchFluidScript)
{
    LOCK(cs_fluid_dynode);
    CFluidDynode fluidDynode;
    return CDBWrapper::Read(make_pair(std::string("script"), vchFluidScript), fluidDynode);
}
