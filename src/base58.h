// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/**
 * Why base-58 instead of standard base-64 encoding?
 * - Don't want 0OIl characters that look the same in some fonts and
 *      could be used to create visually identical looking data.
 * - A string with non-alphanumeric characters is not as easily accepted as input.
 * - E-mail usually won't line-break if there's no punctuation to break at.
 * - Double-clicking selects the whole string as one word if it's all alphanumeric.
 */
#ifndef DYNAMIC_BASE58_H
#define DYNAMIC_BASE58_H

#include "bdap/stealth.h"
#include "chainparams.h"
#include "key.h"
#include "pubkey.h"
#include "script/script.h"
#include "script/standard.h"
#include "support/allocators/zeroafterfree.h"

#include <string>
#include <vector>

/**
 * Encode a tx destination as a base58-encoded string.
 */
std::string EncodeDestination(const CTxDestination& dest);
/**
 * Decode a tx destination as a base58-encoded string.
 */
CTxDestination DecodeDestination(const std::string& str);
/**
 * Encode a byte sequence as a base58-encoded string.
 * pbegin and pend cannot be NULL, unless both are.
 */
std::string EncodeBase58(const unsigned char* pbegin, const unsigned char* pend);

/**
 * Encode a byte vector as a base58-encoded string
 */
std::string EncodeBase58(const std::vector<unsigned char>& vch);

/**
 * Decode a base58-encoded string (psz) into a byte vector (vchRet).
 * return true if decoding is successful.
 * psz cannot be NULL.
 */
bool DecodeBase58(const char* psz, std::vector<unsigned char>& vchRet);

/**
 * Decode a base58-encoded string (str) into a byte vector (vchRet).
 * return true if decoding is successful.
 */
bool DecodeBase58(const std::string& str, std::vector<unsigned char>& vchRet);

/**
 * Encode a byte vector into a base58-encoded string, including checksum
 */
std::string EncodeBase58Check(const std::vector<unsigned char>& vchIn);

/**
 * Decode a base58-encoded string (psz) that includes a checksum into a byte
 * vector (vchRet), return true if decoding is successful
 */
inline bool DecodeBase58Check(const char* psz, std::vector<unsigned char>& vchRet);

/**
 * Decode a base58-encoded string (str) that includes a checksum into a byte
 * vector (vchRet), return true if decoding is successful
 */
inline bool DecodeBase58Check(const std::string& str, std::vector<unsigned char>& vchRet);

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
    void SetData(const std::vector<unsigned char>& vchVersionIn, const void* pdata, size_t nSize);
    void SetData(const std::vector<unsigned char>& vchVersionIn, const unsigned char* pbegin, const unsigned char* pend);

public:
    bool SetString(const char* psz, unsigned int nVersionBytes = 1);
    bool SetString(const std::string& str);
    std::string ToString() const;
    int CompareTo(const CBase58Data& b58) const;

    bool operator==(const CBase58Data& b58) const { return CompareTo(b58) == 0; }
    bool operator<=(const CBase58Data& b58) const { return CompareTo(b58) <= 0; }
    bool operator>=(const CBase58Data& b58) const { return CompareTo(b58) >= 0; }
    bool operator<(const CBase58Data& b58) const { return CompareTo(b58) < 0; }
    bool operator>(const CBase58Data& b58) const { return CompareTo(b58) > 0; }
};

/** base58-encoded Dynamic addresses.
 * Public-key-hash-addresses have version 76 (or 140 testnet).
 * The data vector contains RIPEMD160(SHA256(pubkey)), where pubkey is the serialized public key.
 * Script-hash-addresses have version 16 (or 19 testnet).
 * The data vector contains RIPEMD160(SHA256(cscript)), where cscript is the serialized redemption script.
 */
class CDynamicAddress : public CBase58Data
{
public:
    bool Set(const CKeyID& id);
    bool Set(const CScriptID& id);
    bool Set(const CTxDestination& dest);
    bool Set(const CStealthAddress& sxAddr);
    bool IsValid() const;
    bool IsValid(const CChainParams& params) const;
    bool IsValidStealthAddress() const;
    bool IsValidStealthAddress(const CChainParams& params) const;

    CDynamicAddress() {}
    CDynamicAddress(const CTxDestination& dest) { Set(dest); }
    CDynamicAddress(const std::string& strAddress) { SetString(strAddress); }
    CDynamicAddress(const char* pszAddress) { SetString(pszAddress); }

    CTxDestination Get() const;
    bool GetKeyID(CKeyID& keyID) const;
    bool GetIndexKey(uint160& hashBytes, int& type) const;
    bool IsScript() const;
};

/**
 * A base58-encoded secret key
 */
class CDynamicSecret : public CBase58Data
{
public:
    void SetKey(const CKey& vchSecret);
    CKey GetKey();
    bool IsValid() const;
    bool SetString(const char* pszSecret);
    bool SetString(const std::string& strSecret);

    CDynamicSecret(const CKey& vchSecret) { SetKey(vchSecret); }
    CDynamicSecret() {}
};

template <typename K, int Size, CChainParams::Base58Type Type>
class CDynamicExtKeyBase : public CBase58Data
{
public:
    void SetKey(const K& key)
    {
        unsigned char vch[Size];
        key.Encode(vch);
        SetData(Params().Base58Prefix(Type), vch, vch + Size);
    }

    K GetKey()
    {
        K ret;
        if (vchData.size() == Size) {
            //if base58 encouded data not holds a ext key, return a !IsValid() key
            ret.Decode(&vchData[0]);
        }
        return ret;
    }

    CDynamicExtKeyBase(const K& key)
    {
        SetKey(key);
    }

    CDynamicExtKeyBase(const std::string& strBase58c)
    {
        SetString(strBase58c.c_str(), Params().Base58Prefix(Type).size());
    }

    CDynamicExtKeyBase() {}
};

typedef CDynamicExtKeyBase<CExtKey, BIP32_EXTKEY_SIZE, CChainParams::EXT_SECRET_KEY> CDynamicExtKey;
typedef CDynamicExtKeyBase<CExtPubKey, BIP32_EXTKEY_SIZE, CChainParams::EXT_PUBLIC_KEY> CDynamicExtPubKey;

#endif // DYNAMIC_BASE58_H
