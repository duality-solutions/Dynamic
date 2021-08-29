// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers

#ifndef FLUID_MINING
#define FLUID_MINING

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"
#include "fluid/fluid.h"

#include "sync.h"
#include "uint256.h"

class CScript;
class CTransaction;

class CFluidMining : public DSFluidObject
{
public:
    CFluidMining() = default;
    CFluidMining(const CTransaction& tx) { UnserializeFromTx(tx); }
    CFluidMining(const CScript& fluidScript) { UnserializeFromScript(fluidScript); }
    ~CFluidMining() = default;

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

static CCriticalSection cs_fluid_mining;

class CFluidMiningDB : public CDBWrapper
{
public:
    CFluidMiningDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate);
    bool AddFluidMiningEntry(const CFluidMining& entry, const int op);
    bool GetLastFluidMiningRecord(CFluidMining& returnEntry, const int nHeight);
    bool GetAllFluidMiningRecords(std::vector<CFluidMining>& entries);
    bool IsEmpty();
    bool RecordExists(const std::vector<unsigned char>& vchFluidScript);
};

bool GetFluidMiningData(const CScript& scriptPubKey, CFluidMining& entry);
bool GetFluidMiningData(const CTransaction& tx, CFluidMining& entry, int& nOut);

extern CFluidMiningDB* pFluidMiningDB;

#endif // FLUID_MINING
