// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2018 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_KEY_IO_H
#define DYNAMIC_KEY_IO_H

#include <chainparams.h>
#include <key.h>
#include <pubkey.h>
#include <script/standard.h>
#include <key/extkey.h>
#include <key/stealth.h>
#include <script/script.h>
#include <support/allocators/zeroafterfree.h>
#include <base58.h>

#include <string>

CKey DecodeSecret(const std::string& str);
std::string EncodeSecret(const CKey& key);

CExtKey DecodeExtKey(const std::string& str);
std::string EncodeExtKey(const CExtKey& extkey);
CExtPubKey DecodeExtPubKey(const std::string& str);
std::string EncodeExtPubKey(const CExtPubKey& extpubkey);

std::string EncodeDestination(const CTxDestination& dest);
CTxDestination DecodeDestination(const std::string& str);
bool IsValidDestinationString(const std::string& str);
bool IsValidDestinationString(const std::string& str, const CChainParams& params);

/**
 * Base class for all base58-encoded data
 */
class CBase58Data
{
protected:
    //! the version byte(s)
    std::vector<unsigned char> vchVersion;

    //! the actually encoded data
    typedef std::vector<unsigned char, zero_after_free_allocator<unsigned char> > vector_uchar;
    vector_uchar vchData;

    CBase58Data();
    void SetData(const std::vector<unsigned char> &vchVersionIn, const void* pdata, size_t nSize);
    void SetData(const std::vector<unsigned char> &vchVersionIn, const unsigned char *pbegin, const unsigned char *pend);

public:
    bool SetString(const char* psz, unsigned int nVersionBytes = 1);
    bool SetString(const std::string& str);
    std::string ToString() const;
    int CompareTo(const CBase58Data& b58) const;

    bool operator==(const CBase58Data& b58) const { return CompareTo(b58) == 0; }
    bool operator<=(const CBase58Data& b58) const { return CompareTo(b58) <= 0; }
    bool operator>=(const CBase58Data& b58) const { return CompareTo(b58) >= 0; }
    bool operator< (const CBase58Data& b58) const { return CompareTo(b58) <  0; }
    bool operator> (const CBase58Data& b58) const { return CompareTo(b58) >  0; }

};

/** base58-encoded Bitcoin addresses.
 * Public-key-hash-addresses have version 0 (or 111 testnet).
 * The data vector contains RIPEMD160(SHA256(pubkey)), where pubkey is the serialized public key.
 * Script-hash-addresses have version 5 (or 196 testnet).
 * The data vector contains RIPEMD160(SHA256(cscript)), where cscript is the serialized redemption script.
 */
class CDynamicAddress : public CBase58Data {
public:
    bool Set(const CKeyID& id);
    bool Set(const CScriptID& id);
    bool Set(const CKeyID& id, CChainParams::Base58Type prefix);
    bool Set(const CStealthAddress& sx);
    bool Set(const CExtKeyPair& ek);
    bool Set(const CTxDestination& dest);

    bool IsValidStealthAddress() const;
    bool IsValidStealthAddress(const CChainParams &params) const;
    bool IsValid() const;
    bool IsValid(const CChainParams &params) const;
    bool IsValid(CChainParams::Base58Type prefix) const;

    CDynamicAddress() {}
    CDynamicAddress(const CTxDestination &dest) { Set(dest); }
    CDynamicAddress(const std::string& strAddress) { SetString(strAddress); }
    CDynamicAddress(const char* pszAddress) { SetString(pszAddress); }

    CTxDestination Get() const;
    bool GetKeyID(CKeyID &keyID) const;
    bool GetKeyID(CKeyID &keyID, CChainParams::Base58Type prefix) const;    
    bool GetIndexKey(uint160 &hashBytes, int &type) const;
    bool GetIndexKey(uint256 &hashBytes, int &type) const;
    bool IsScript() const;

    uint8_t getVersion()
    {
        // TODO: Fix for multibyte versions
        if (vchVersion.size() > 0)
            return vchVersion[0];
        return 0;
    }
    std::vector<uint8_t> &getVchVersion()
    {
        return vchVersion;
    }
    void setVersion(const std::vector<uint8_t> &version)
    {
        vchVersion = version;
        return;
    }
};

/**
 * A base58-encoded secret key
 */
class CDynamicSecret : public CBase58Data
{
public:
    void SetKey(const CKey& vchSecret);
    CKey GetKey() const;
    bool IsValid() const;
    bool SetString(const char* pszSecret);
    bool SetString(const std::string& strSecret);

    CDynamicSecret(const CKey& vchSecret) { SetKey(vchSecret); }
    CDynamicSecret() {}
};

template<typename K, int Size, CChainParams::Base58Type Type> class CDynamicExtKeyBase : public CBase58Data
{
public:
    void SetKey(const K &key) {
        unsigned char vch[Size];
        key.Encode(vch);
        SetData(Params().Base58Prefix(Type), vch, vch+Size);
    }

    int Set58(const char *base58)
    {
        std::vector<uint8_t> vchBytes;
        if (!DecodeBase58(base58, vchBytes))
            return 1;

        if (vchBytes.size() != BIP32_KEY_LEN)
            return 2;

        if (!VerifyChecksum(vchBytes))
            return 3;

        if (0 != memcmp(&vchBytes[0], &Params().Base58Prefix(Type)[0], 4))
            return 4;

        SetData(Params().Base58Prefix(Type), &vchBytes[4], &vchBytes[4]+Size);
        return 0;
    }

    K GetKey() {
        K ret;
        if (vchData.size() == Size) {
            // If base58 encoded data does not hold an ext key, return a !IsValid() key
            ret.Decode(vchData.data());
        }
        return ret;
    }

    CDynamicExtKeyBase(const K &key) {
        SetKey(key);
    }

    CDynamicExtKeyBase(const std::string& strBase58c) {
        SetString(strBase58c.c_str(), Params().Base58Prefix(Type).size());
    }

    CDynamicExtKeyBase() {}
};

typedef CDynamicExtKeyBase<CExtKey, BIP32_EXTKEY_SIZE, CChainParams::EXT_SECRET_KEY> CDynamicExtKey;
typedef CDynamicExtKeyBase<CExtPubKey, BIP32_EXTKEY_SIZE, CChainParams::EXT_PUBLIC_KEY> CDynamicExtPubKey;


class CExtKey58 : public CBase58Data
{
public:
    CExtKey58() {};

    CExtKey58(const CExtKeyPair &key, CChainParams::Base58Type type)
    {
        SetKey(key, type);
    };

    void SetKeyV(const CExtKeyPair &key)
    {
        SetKey(key, CChainParams::EXT_SECRET_KEY);
    };

    void SetKeyP(const CExtKeyPair &key)
    {
        SetKey(key, CChainParams::EXT_PUBLIC_KEY);
    };

    void SetKey(const CExtKeyPair &key, CChainParams::Base58Type type)
    {
        uint8_t vch[74];

        switch (type)
        {
            case CChainParams::EXT_SECRET_KEY:
                key.EncodeV(vch);
                break;
            //case CChainParams::EXT_PUBLIC_KEY:
            default:
                key.EncodeP(vch);
                break;
        };

        SetData(Params().Base58Prefix(type), vch, vch+74);
    };

    CExtKeyPair GetKey()
    {
        CExtKeyPair rv;
        if (vchVersion == Params().Base58Prefix(CChainParams::EXT_SECRET_KEY))
        {
            rv.DecodeV(&vchData[0]);
            return rv;
        };
        rv.DecodeP(&vchData[0]);
        return rv;
    };

    bool GetPubKey(CExtPubKey &rv, const CChainParams *pparams)
    {
        if (vchVersion == pparams->Base58Prefix(CChainParams::EXT_SECRET_KEY))
        {
            CExtKey ek;
            ek.Decode(&vchData[0]);
            rv = ek.Neutered();
            return true;
        };

        if (vchVersion == pparams->Base58Prefix(CChainParams::EXT_PUBLIC_KEY))
        {
            rv.Decode(&vchData[0]);
            return true;
        }

        return false;
    };

    int Set58(const char *base58);
    int Set58(const char *base58, CChainParams::Base58Type type, const CChainParams *pparams);

    bool IsValid(CChainParams::Base58Type prefix) const;

    std::string ToStringVersion(CChainParams::Base58Type prefix);
};

#endif // DYNAMIC_KEY_IO_H
