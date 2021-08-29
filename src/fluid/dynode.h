// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers

#ifndef FLUID_DYNODE_H
#define FLUID_DYNODE_H

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"
#include "fluid/fluid.h"

#include "sync.h"
#include "uint256.h"

class CScript;
class CTransaction;

class CFluidDynode : public DSFluidObject
{
public:
    CFluidDynode() = default;
    CFluidDynode(const CTransaction& tx) { UnserializeFromTx(tx); }
    CFluidDynode(const CScript& fluidScript) { UnserializeFromScript(fluidScript); }
    ~CFluidDynode() = default;

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->version);
        READWRITE(tx_script);
        READWRITE(obj_reward);
        READWRITE(VARINT(obj_time));
        READWRITE(obj_sigs);
        READWRITE(tx_hash);
        READWRITE(VARINT(tx_height));
    }

    bool UnserializeFromTx(const CTransaction& tx);
    bool UnserializeFromScript(const CScript& fluidScript);
    void Serialize(std::vector<unsigned char>& vchData);
};

static CCriticalSection cs_fluid_dynode;

class CFluidDynodeDB : public CDBWrapper
{
public:
    CFluidDynodeDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate);
    bool AddFluidDynodeEntry(const CFluidDynode& entry, const int op);
    bool GetLastFluidDynodeRecord(CFluidDynode& returnEntry, const int nHeight);
    bool GetAllFluidDynodeRecords(std::vector<CFluidDynode>& entries);
    bool IsEmpty();
    bool RecordExists(const std::vector<unsigned char>& vchFluidScript);
};

bool GetFluidDynodeData(const CScript& scriptPubKey, CFluidDynode& entry);
bool GetFluidDynodeData(const CTransaction& tx, CFluidDynode& entry, int& nOut);

extern CFluidDynodeDB* pFluidDynodeDB;

#endif // FLUID_DYNODE_H
