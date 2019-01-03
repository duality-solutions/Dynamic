// Copyright (c) 2019 Duality Blockchain Solutions Developers

#ifndef FLUID_SOVEREIGN_H
#define FLUID_SOVEREIGN_H

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"

#include "sync.h"
#include "uint256.h"

class CScript;
class CTransaction;

class CFluidSovereign
{
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<unsigned char> FluidScript;
    int64_t nTimeStamp;
    std::vector<std::vector<unsigned char> > SovereignAddresses;
    uint256 txHash;
    unsigned int nHeight;

    CFluidSovereign()
    {
        SetNull();
    }

    CFluidSovereign(const CTransaction& tx)
    {
        SetNull();
        UnserializeFromTx(tx);
    }

    CFluidSovereign(const CScript& fluidScript)
    {
        SetNull();
        UnserializeFromScript(fluidScript);
    }

    inline void SetNull()
    {
        nVersion = CFluidSovereign::CURRENT_VERSION;
        FluidScript.clear();
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
        READWRITE(VARINT(nTimeStamp));
        READWRITE(SovereignAddresses);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
    }

    inline friend bool operator==(const CFluidSovereign& a, const CFluidSovereign& b)
    {
        return (a.FluidScript == b.FluidScript && a.SovereignAddresses == b.SovereignAddresses && a.nTimeStamp == b.nTimeStamp);
    }

    inline friend bool operator!=(const CFluidSovereign& a, const CFluidSovereign& b)
    {
        return !(a == b);
    }

    inline CFluidSovereign operator=(const CFluidSovereign& b)
    {
        FluidScript = b.FluidScript;
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
bool GetFluidSovereignData(const CScript& scriptPubKey, CFluidSovereign& entry);
bool GetFluidSovereignData(const CTransaction& tx, CFluidSovereign& entry, int& nOut);
bool CheckFluidSovereignDB();

extern CFluidSovereignDB* pFluidSovereignDB;

#endif // FLUID_SOVEREIGN_H
