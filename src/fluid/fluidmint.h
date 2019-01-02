// Copyright (c) 2019 Duality Blockchain Solutions Developers

#ifndef FLUID_MINT_H
#define FLUID_MINT_H

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"

#include "sync.h"
#include "uint256.h"

class CDynamicAddress;
class CScript;
class CTransaction;

class CFluidMint
{
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<unsigned char> FluidScript;
    CAmount MintAmount;
    std::vector<unsigned char> DestinationAddress;
    int64_t nTimeStamp;
    std::vector<std::vector<unsigned char> > SovereignAddresses;
    uint256 txHash;
    unsigned int nHeight;

    CFluidMint()
    {
        SetNull();
    }

    CFluidMint(const CTransaction& tx)
    {
        SetNull();
        UnserializeFromTx(tx);
    }

    CFluidMint(const CScript& fluidScript)
    {
        SetNull();
        UnserializeFromScript(fluidScript);
    }

    inline void SetNull()
    {
        nVersion = CFluidMint::CURRENT_VERSION;
        FluidScript.clear();
        MintAmount = -1;
        DestinationAddress.clear();
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
        READWRITE(MintAmount);
        READWRITE(DestinationAddress);
        READWRITE(VARINT(nTimeStamp));
        READWRITE(SovereignAddresses);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
    }

    inline friend bool operator==(const CFluidMint& a, const CFluidMint& b)
    {
        return (a.FluidScript == b.FluidScript && a.MintAmount == b.MintAmount && a.DestinationAddress == b.DestinationAddress && a.nTimeStamp == b.nTimeStamp);
    }

    inline friend bool operator!=(const CFluidMint& a, const CFluidMint& b)
    {
        return !(a == b);
    }

    inline CFluidMint operator=(const CFluidMint& b)
    {
        FluidScript = b.FluidScript;
        MintAmount = b.MintAmount;
        DestinationAddress = b.DestinationAddress;
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
    CDynamicAddress GetDestinationAddress() const;
};

static CCriticalSection cs_fluid_mint;

class CFluidMintDB : public CDBWrapper
{
public:
    CFluidMintDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate);
    bool AddFluidMintEntry(const CFluidMint& entry, const int op);
    bool GetLastFluidMintRecord(CFluidMint& returnEntry);
    bool GetAllFluidMintRecords(std::vector<CFluidMint>& entries);
    bool IsEmpty();
    bool RecordExists(const std::vector<unsigned char>& vchFluidScript);
};

bool GetFluidMintData(const CScript& scriptPubKey, CFluidMint& entry);
bool GetFluidMintData(const CTransaction& tx, CFluidMint& entry, int& nOut);
bool CheckFluidMintDB();

extern CFluidMintDB* pFluidMintDB;

#endif // FLUID_MINT_H