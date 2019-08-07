// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef FLUID_BANACCOUNT_H
#define FLUID_BANACCOUNT_H

#include "dbwrapper.h"
#include "primitives/transaction.h"
#include "serialize.h"
#include "sync.h"
#include "uint256.h"

class CDynamicAddress;
class CScript;

class CBanAccount
{
public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    std::vector<unsigned char> FluidScript; // We don't need this variable. Remove in version 2.
    std::vector<unsigned char> vchAccountAddress;
    int64_t nTimeStamp;
    std::vector<std::vector<unsigned char>> vSovereignAddresses;
    uint256 txHash;
    unsigned int nHeight;
    bool fIsWalletAddress;

    CBanAccount()
    {
        SetNull();
    }

    CBanAccount(const std::vector<unsigned char>& account, const int64_t& timestamp, 
                const std::vector<std::vector<unsigned char> >& addresses, const uint256& txid, const unsigned int& height);

    inline void SetNull()
    {
        nVersion = CBanAccount::CURRENT_VERSION;
        FluidScript.clear();
        vchAccountAddress.clear();
        nTimeStamp = 0;
        vSovereignAddresses.clear();
        txHash.SetNull();
        nHeight = 0;
        fIsWalletAddress = false;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        READWRITE(this->nVersion);
        READWRITE(FluidScript);
        READWRITE(vchAccountAddress);
        READWRITE(nTimeStamp);
        READWRITE(vSovereignAddresses);
        READWRITE(txHash);
        READWRITE(VARINT(nHeight));
        READWRITE(fIsWalletAddress);
    }

    inline friend bool operator==(const CBanAccount& a, const CBanAccount& b)
    {
        return (a.vchAccountAddress == b.vchAccountAddress && a.txHash == b.txHash && a.nTimeStamp == b.nTimeStamp);
    }

    inline friend bool operator!=(const CBanAccount& a, const CBanAccount& b)
    {
        return !(a == b);
    }

    friend bool operator<(const CBanAccount& a, const CBanAccount& b)
    {
        return (a.nTimeStamp < b.nTimeStamp);
    }

    friend bool operator>(const CBanAccount& a, const CBanAccount& b)
    {
        return (a.nTimeStamp > b.nTimeStamp);
    }

    inline CBanAccount operator=(const CBanAccount& b)
    {
        nVersion = b.nVersion;
        FluidScript = b.FluidScript;
        vchAccountAddress = b.vchAccountAddress;
        nTimeStamp = b.nTimeStamp;
        vSovereignAddresses.clear(); //clear out previous entries
        for (const std::vector<unsigned char>& vchAddress : b.vSovereignAddresses) {
            vSovereignAddresses.push_back(vchAddress);
        }
        txHash = b.txHash;
        nHeight = b.nHeight;
        fIsWalletAddress = b.fIsWalletAddress;
        return *this;
    }

    inline bool IsNull() const { return (nTimeStamp == 0); }

};

static CCriticalSection cs_ban_account;

class CBanAccountDB : public CDBWrapper
{
public:
    CBanAccountDB(size_t nCacheSize, bool fMemory, bool fWipe, bool obfuscate);
    bool AddBanAccountEntry(const CBanAccount& entry);
    bool AddBanWalletEntry(const CBanAccount& entry);
    bool GetAllBanRecords(std::vector<CBanAccount>& entries);
    bool GetAllBanWalletRecords(std::vector<CBanAccount>& entries);
    bool BanRecordExists(const std::vector<unsigned char>& vchAccountAddress);
    bool BanWalletRecordExists(const std::vector<unsigned char>& vchAccountAddress);
};

bool CheckBanAccountDB();
bool AddBanAccountEntry(const CBanAccount& entry);
bool AddBanWalletEntry(const CBanAccount& entry);
bool GetAllBanRecords(std::vector<CBanAccount>& entries);
bool BanRecordExists(const std::vector<unsigned char>& vchAccountAddress);
bool IsBannedWalletAddress(const CTransactionRef& pTx, CDynamicAddress& address);

extern CBanAccountDB* pBanAccountDB;

#endif // FLUID_BANACCOUNT_H