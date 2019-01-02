// Copyright (c) 2019 Duality Blockchain Solutions Developers

#ifndef FLUID_MINER
#define FLUID_MINER

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"

#include "sync.h"
#include "uint256.h"

class CScript;
class CTransaction;

class CFluidMining
{
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<unsigned char> FluidScript;
    CAmount MiningReward;
    int64_t nTimeStamp;
    std::vector<std::vector<unsigned char> > SovereignAddresses;
    uint256 txHash;
    unsigned int nHeight;

    CFluidMining()
    {
        SetNull();
    }

    CFluidMining(const CTransaction& tx)
    {
        SetNull();
        UnserializeFromTx(tx);
    }

    CFluidMining(const CScript& fluidScript)
    {
        SetNull();
        UnserializeFromScript(fluidScript);
    }

    inline void SetNull()
    {
        nVersion = CFluidMining::CURRENT_VERSION;
        FluidScript.clear();
        MiningReward = -1;
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
        READWRITE(MiningReward);
        READWRITE(VARINT(nTimeStamp));
        READWRITE(SovereignAddresses);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
    }

    inline friend bool operator==(const CFluidMining& a, const CFluidMining& b)
    {
        return (a.FluidScript == b.FluidScript && a.MiningReward == b.MiningReward && a.nTimeStamp == b.nTimeStamp);
    }

    inline friend bool operator!=(const CFluidMining& a, const CFluidMining& b)
    {
        return !(a == b);
    }

    inline CFluidMining operator=(const CFluidMining& b)
    {
        FluidScript = b.FluidScript;
        MiningReward = b.MiningReward;
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
bool CheckFluidMiningDB();
CAmount GetFluidMiningReward();

extern CFluidMiningDB* pFluidMiningDB;

#endif // FLUID_MINER
