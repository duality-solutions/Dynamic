// Copyright (c) 2019-2021 Duality Blockchain Solutions Developers

#ifndef FLUID_MINT_H
#define FLUID_MINT_H

#include "amount.h"
#include "dbwrapper.h"
#include "serialize.h"
#include "fluid/fluid.h"

#include "sync.h"
#include "uint256.h"

class CDynamicAddress;
class CScript;
class CTransaction;

class CFluidMint : public DSFluidObject
{
public:
    CFluidMint() = default;
    CFluidMint(const CTransaction& tx) { UnserializeFromTx(tx); }
    CFluidMint(const CScript& fluidScript) { UnserializeFromScript(fluidScript); }
    ~CFluidMint() = default;

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->version);
        READWRITE(tx_script);
        READWRITE(obj_reward);
        READWRITE(obj_address);
        READWRITE(VARINT(obj_time));
        READWRITE(obj_sigs);
        READWRITE(tx_hash);
        READWRITE(VARINT(tx_height));
    }

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

extern CFluidMintDB* pFluidMintDB;

#endif // FLUID_MINT_H