// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers

#ifndef FLUID_SOVEREIGN_H
#define FLUID_SOVEREIGN_H

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"
#include "fluid/fluid.h"

#include "sync.h"
#include "uint256.h"

class CScript;
class CTransaction;

class CFluidSovereign : public DSFluidObject
{
public:
    CFluidSovereign() = default;
    CFluidSovereign(const CTransaction& tx) { UnserializeFromTx(tx); }
    CFluidSovereign(const CScript& fluidScript) { UnserializeFromScript(fluidScript); }
    ~CFluidSovereign() = default;

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->version);
        READWRITE(tx_script);
        READWRITE(VARINT(obj_time));
        READWRITE(obj_sigs);
        READWRITE(tx_hash);
        READWRITE(VARINT(tx_height));
    }

    bool UnserializeFromTx(const CTransaction& tx);
    bool UnserializeFromScript(const CScript& fluidScript);
    void Serialize(std::vector<unsigned char>& vchData);

    std::vector<std::string> SovereignAddressesStrings();
};

static CCriticalSection cs_fluid_sovereign;

class CFluidSovereignDB : public CDBWrapper
{
public:
    CFluidSovereignDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate);
    bool AddFluidSovereignEntry(const CFluidSovereign& entry);
    bool GetLastFluidSovereignRecord(CFluidSovereign& returnEntry);
    bool GetAllFluidSovereignRecords(std::vector<CFluidSovereign>& entries);
    bool IsEmpty();

private:
    void InitEmpty();
};

extern CFluidSovereignDB* pFluidSovereignDB;

#endif // FLUID_SOVEREIGN_H
