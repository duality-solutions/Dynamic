// Copyright (c) 2019 Duality Blockchain Solutions Developers

#ifndef FLUID_DYNODE_H
#define FLUID_DYNODE_H

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"

#include "sync.h"
#include "uint256.h"

class CScript;
class CTransaction;

class CFluidDynode
{
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<unsigned char> FluidScript;
    CAmount DynodeReward;
    int64_t nTimeStamp;
    std::vector<std::vector<unsigned char> > SovereignAddresses;
    uint256 txHash;
    unsigned int nHeight;

    CFluidDynode()
    {
        SetNull();
    }

    CFluidDynode(const CTransaction& tx)
    {
        SetNull();
        UnserializeFromTx(tx);
    }

    CFluidDynode(const CScript& fluidScript)
    {
        SetNull();
        UnserializeFromScript(fluidScript);
    }

    inline void SetNull()
    {
        nVersion = CFluidDynode::CURRENT_VERSION;
        FluidScript.clear();
        DynodeReward = -1;
        nTimeStamp = 0;
        SovereignAddresses.clear();
        txHash.SetNull();
        nHeight = 0;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->nVersion);
        READWRITE(FluidScript);
        READWRITE(DynodeReward);
        READWRITE(VARINT(nTimeStamp));
        READWRITE(SovereignAddresses);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
    }

    inline friend bool operator==(const CFluidDynode& a, const CFluidDynode& b)
    {
        return (a.FluidScript == b.FluidScript && a.DynodeReward == b.DynodeReward && a.nTimeStamp == b.nTimeStamp);
    }

    inline friend bool operator!=(const CFluidDynode& a, const CFluidDynode& b)
    {
        return !(a == b);
    }

    inline CFluidDynode operator=(const CFluidDynode& b)
    {
        FluidScript = b.FluidScript;
        DynodeReward = b.DynodeReward;
        nTimeStamp = b.nTimeStamp;
        for (const std::vector<unsigned char>& vchAddress : b.SovereignAddresses) {
            SovereignAddresses.push_back(vchAddress);
        }
        txHash = b.txHash;
        nHeight = b.nHeight;
        return *this;
    }

    inline bool IsNull() const { return (nTimeStamp == 0); }
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
bool CheckFluidDynodeDB();

extern CFluidDynodeDB* pFluidDynodeDB;

#endif // FLUID_DYNODE_H
