// Copyright (c) 2017 Duality Blockchain Solutions Developers


#include "fluid/mint.h"

#include "base58.h"
#include "core_io.h"
#include "fluid/fluid.h"
#include "operations.h"
#include "script/script.h"

#include <boost/thread.hpp>

CFluidMintDB* pFluidMintDB = NULL;

bool GetFluidMintData(const CScript& scriptPubKey, CFluidMint& entry)
{
    return ParseScript(scriptPubKey, entry);
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
    int nOut;
    if (!GetFluidMintData(tx, *this, nOut)) {
        return false;
    }
    return true;
}

bool CFluidMint::UnserializeFromScript(const CScript& fluidScript)
{
    if (!GetFluidMintData(fluidScript, *this)) {
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
    return CDynamicAddress(StringFromCharVector(obj_address));
}

CFluidMintDB::CFluidMintDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate) : CDBWrapper(GetDataDir() / "blocks" / "fluid-mint", nCacheSize, fMemory, fWipe, obfuscate)
{
}

bool CFluidMintDB::AddFluidMintEntry(const CFluidMint& entry, const int op)
{
    bool writeState = false;
    {
        LOCK(cs_fluid_mint);
        writeState = Write(make_pair(std::string("script"), entry.GetTransactionScript()), entry) && Write(make_pair(std::string("txid"), entry.GetTransactionHash()), entry.GetTransactionScript());
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
