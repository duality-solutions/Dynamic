// Copyright (c) 2019 Duality Blockchain Solutions Developers

#ifndef FLUID_STAKING
#define FLUID_STAKING

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"

#include "sync.h"
#include "uint256.h"

class CScript;
class CTransaction;

class CFluidStaking
{
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<unsigned char> FluidScript;
    CAmount StakeReward;
    int64_t nTimeStamp;
    std::vector<std::vector<unsigned char> > SovereignAddresses;
    uint256 txHash;
    unsigned int nHeight;

    CFluidStaking()
    {
        SetNull();
    }

    CFluidStaking(const CTransaction& tx)
    {
        SetNull();
        UnserializeFromTx(tx);
    }

    CFluidStaking(const CScript& fluidScript)
    {
        SetNull();
        UnserializeFromScript(fluidScript);
    }

    inline void SetNull()
    {
        nVersion = CFluidStaking::CURRENT_VERSION;
        FluidScript.clear();
        StakeReward = -1;
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
        READWRITE(StakeReward);
        READWRITE(VARINT(nTimeStamp));
        READWRITE(SovereignAddresses);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
    }

    inline friend bool operator==(const CFluidStaking& a, const CFluidStaking& b)
    {
        return (a.FluidScript == b.FluidScript && a.StakeReward == b.StakeReward && a.nTimeStamp == b.nTimeStamp);
    }

    inline friend bool operator!=(const CFluidStaking& a, const CFluidStaking& b)
    {
        return !(a == b);
    }

    friend bool operator<(const CFluidStaking& a, const CFluidStaking& b)
    {
        return (a.nTimeStamp < b.nTimeStamp);
    }

    friend bool operator>(const CFluidStaking& a, const CFluidStaking& b)
    {
        return (a.nTimeStamp > b.nTimeStamp);
    }

    inline CFluidStaking operator=(const CFluidStaking& b)
    {
        FluidScript = b.FluidScript;
        StakeReward = b.StakeReward;
        nTimeStamp = b.nTimeStamp;
        SovereignAddresses.clear(); //clear out previous entries
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

static CCriticalSection cs_fluid_staking;

class CFluidStakingDB : public CDBWrapper
{
public:
    CFluidStakingDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate);
    bool AddFluidStakingEntry(const CFluidStaking& entry, const int op);
    bool GetLastFluidStakingRecord(CFluidStaking& returnEntry, const int nHeight);
    bool GetAllFluidStakingRecords(std::vector<CFluidStaking>& entries);
    bool IsEmpty();
    bool RecordExists(const std::vector<unsigned char>& vchFluidScript);
};

bool GetFluidStakingData(const CScript& scriptPubKey, CFluidStaking& entry);
bool GetFluidStakingData(const CTransaction& tx, CFluidStaking& entry, int& nOut);
bool CheckFluidStakingDB();

extern CFluidStakingDB* pFluidStakingDB;

#endif // FLUID_STAKING