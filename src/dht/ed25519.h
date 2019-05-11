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

struct ed25519_context
{
    ed25519_context() = default;

    explicit ed25519_context(char const* b)
    { std::copy(b, b + len, seed.begin()); }

    bool operator==(ed25519_context const& rhs) const
    { return seed == rhs.seed; }

    bool operator!=(ed25519_context const& rhs) const
    { return seed != rhs.seed; }

    constexpr static int len = ED25519_PRIVATE_SEED_BYTE_LENGTH;

    std::array<char, len> seed;
    
    void SetNull() 
    {
        std::fill(seed.begin(),seed.end(),0);
    }

    bool IsNull()
    {
        for(int i=0;i<len;i++) {
            if (seed[i] != 0) {
                return false;
            }
        }
        return true; 
    }
};

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

    CKeyEd25519(const bool fNull = false)
    {
        if (fNull) {
            SetNull();
        }
        else {
            MakeNewKeyPair();
        }
    }

    CKeyEd25519(const std::array<char, ED25519_PRIVATE_SEED_BYTE_LENGTH>& _seed);
    CKeyEd25519(const std::vector<unsigned char>& _seed);

    // TODO (dht): Make sure private keys are destroyed correctly.
    ~CKeyEd25519()
    {
        // fill private key std arrays with all zeros
        seed.fill(0);
        privateKey.fill(0);
    }

    // TODO (dht): use SecureVector and SecureString below:
    std::vector<unsigned char> GetPrivKey() const; 
    std::vector<unsigned char> GetPubKey() const;
    std::vector<unsigned char> GetPrivSeed() const;
    std::vector<unsigned char> GetPrivKeyBytes() const; 
    std::vector<unsigned char> GetPubKeyBytes() const;
    std::vector<unsigned char> GetPrivSeedBytes() const;
    std::string GetPrivKeyString() const;
    std::string GetPubKeyString() const;
    std::string GetPrivSeedString() const;

    int PubKeySize() const { return sizeof(GetPubKey()); }

    void SetNull()
    {
        seed.fill(0);
        privateKey.fill(0);
        publicKey.fill(0);
    }

    bool IsNull()
    {
        std::array<char, 32> null32;
        null32.fill(0);
        if (seed == null32)
            return true;

        return false;
    }

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
        std::vector<unsigned char> vch = GetPubKey();  // TODO (DHT): change to use GetPubKeyBytes()
        return Hash(vch.begin(), vch.end());
    }

    //! Get the KeyID of this public key (hash of its serialization)
    CKeyID GetID() const
    {
        std::vector<unsigned char> vch = GetPubKey(); // TODO (DHT): change to use GetPubKeyBytes()
        return CKeyID(Hash160(vch.begin(), vch.end()));
    }

private:
    //! Generate a new private key using LibTorrent's Ed25519 implementation
    void MakeNewKeyPair();

};

std::vector<unsigned char> GetLinkSharedPubKey(const CKeyEd25519& dhtKey, const std::vector<unsigned char>& vchOtherPubKey);
std::array<char, 32> GetLinkSharedPrivateKey(const CKeyEd25519& dhtKey, const std::vector<unsigned char>& vchOtherPubKey);
std::vector<unsigned char> EncodedPubKeyToBytes(const std::vector<unsigned char>& vchEncodedPubKey);
std::string CharVectorToByteArrayString(const std::vector<unsigned char>& vchData);

bool ECC_Ed25519_InitSanityCheck();
void ECC_Ed25519_Start();
void ECC_Ed25519_Stop();

CKeyID GetIdFromCharVector(const std::vector<unsigned char>& vchIn);
uint256 GetHashFromCharVector(const std::vector<unsigned char>& vchIn);

#endif // DYNAMIC_DHT_ED25519_H