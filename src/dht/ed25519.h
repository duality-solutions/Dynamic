// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_ED25519_H
#define DYNAMIC_DHT_ED25519_H

#include "pubkey.h"
#include "support/allocators/secure.h"
#include "uint256.h"

#include <array>
#include <cstring>
#include <memory>

static constexpr unsigned int ED25519_PUBLIC_KEY_BYTE_LENGTH        = 32;
static constexpr unsigned int ED25519_PRIVATE_SEED_BYTE_LENGTH      = 32;
static constexpr unsigned int ED25519_SIGTATURE_BYTE_LENGTH         = 64;
static constexpr unsigned int ED25519_PRIVATE_KEY_BYTE_LENGTH       = 64;
/** 
 * ed25519:
 * unsigned char seed[32];
 * unsigned char signature[64];
 * unsigned char public_key[32];
 * unsigned char private_key[64];
 * unsigned char scalar[32];
 * unsigned char shared_secret[32];
 */

/** An encapsulated ed25519 private key. */
class CKeyEd25519
{
public:
    /**
     * ed25519:
     */
    std::array<char, ED25519_PRIVATE_SEED_BYTE_LENGTH> seed;
    std::array<char, ED25519_PRIVATE_KEY_BYTE_LENGTH> privateKey;
    std::array<char, ED25519_PUBLIC_KEY_BYTE_LENGTH> publicKey;

public:
    //! Construct a new private key.
    CKeyEd25519()
    {
        MakeNewKeyPair();
    }

    CKeyEd25519(const std::array<char, 32>& _seed);
    CKeyEd25519(const std::vector<unsigned char>& _seed);
    CKeyEd25519(const std::vector<unsigned char, secure_allocator<unsigned char> >& keyData);

    // TODO (dht): Make sure private keys are destroyed correctly.
    ~CKeyEd25519()
    {
        // fill private key std arrays with all zeros
        seed.fill(0);
        privateKey.fill(0);
    }

    void Set(const std::array<char, 32>& _seed);

    friend bool operator==(const CKeyEd25519& a, const CKeyEd25519& b)
    {
        return a.seed == b.seed &&
               a.privateKey == b.privateKey &&
               a.publicKey == b.publicKey;
    }

    // TODO (dht): use SecureVector and SecureString below:
    std::vector<unsigned char> GetPrivKey() const; 
    std::vector<unsigned char> GetPubKey() const;
    std::vector<unsigned char> GetPrivSeed() const;
    const unsigned char* begin() const { return GetPrivSeed().data(); }
    const unsigned char* end() const { return GetPrivSeed().data() + ED25519_PRIVATE_SEED_BYTE_LENGTH; }
    std::vector<unsigned char> GetPrivKeyBytes() const; 
    std::vector<unsigned char> GetPubKeyBytes() const;
    std::vector<unsigned char> GetPrivSeedBytes() const;
    std::string GetPrivKeyString() const;
    std::string GetPubKeyString() const;
    std::string GetPrivSeedString() const;

    int PubKeySize() const { return sizeof(GetPubKey()); }

    std::array<char, ED25519_PRIVATE_SEED_BYTE_LENGTH> GetDHTPrivSeed() const { return seed; }
    /**
     * Used for the Torrent DHT.
     */
    std::array<char, ED25519_PRIVATE_KEY_BYTE_LENGTH> GetDHTPrivKey() const { return privateKey; }
    /**
     * Used for the Torrent DHT.
     */
    std::array<char, ED25519_PUBLIC_KEY_BYTE_LENGTH> GetDHTPubKey() const { return publicKey; }

    //! Get the 256-bit hash of this public key.
    uint256 GetHash() const
    {
        std::vector<unsigned char> vch = GetPubKey();
        return Hash(vch.begin(), vch.end());
    }

    //! Get the 256-bit hash of this public key.
    CKeyID GetID() const
    {
        std::vector<unsigned char> vch = GetPubKey();
        return CKeyID(Hash160(vch.begin(), vch.end()));
    }

private:
    //! Generate a new private key using LibTorrent's Ed25519 implementation
    void MakeNewKeyPair();

};

struct CEd25519ExtKey {
    unsigned char nDepth;
    unsigned char vchFingerprint[4];
    unsigned int nChild;
    ChainCode chaincode;
    CKeyEd25519 key;

    friend bool operator==(const CEd25519ExtKey& a, const CEd25519ExtKey& b)
    {
        return a.nDepth == b.nDepth &&
               memcmp(&a.vchFingerprint[0], &b.vchFingerprint[0], sizeof(vchFingerprint)) == 0 &&
               a.nChild == b.nChild &&
               a.chaincode == b.chaincode &&
               a.key == b.key;
    }

    void Set(const CKeyEd25519& setKey);

    void Encode(unsigned char code[74]) const;
    void Decode(const unsigned char code[74]);

    void SetMaster(const unsigned char* seed, unsigned int nSeedLen);
    template <typename Stream>
    void Serialize(Stream& s) const
    {
        unsigned int len = 74;
        ::WriteCompactSize(s, len);
        unsigned char code[74];
        Encode(code);
        s.write((const char*)&code[0], len);
    }
    template <typename Stream>
    void Unserialize(Stream& s)
    {
        unsigned int len = ::ReadCompactSize(s);
        unsigned char code[74];
        s.read((char*)&code[0], len);
        Decode(code);
    }
};

std::vector<unsigned char> GetLinkSharedPubKey(const CKeyEd25519& dhtKey, const std::vector<unsigned char>& vchOtherPubKey);
std::array<char, 32> GetLinkSharedPrivateKey(const CKeyEd25519& dhtKey, const std::vector<unsigned char>& vchOtherPubKey);
std::vector<unsigned char> EncodedPubKeyToBytes(const std::vector<unsigned char>& vchEncodedPubKey);
std::string CharVectorToByteArrayString(const std::vector<unsigned char>& vchData);
std::array<char, 32> ConvertSecureVector32ToArray(const std::vector<unsigned char, secure_allocator<unsigned char>>& vIn);

#endif // DYNAMIC_DHT_ED25519_H