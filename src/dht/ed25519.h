// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_ED25519_H
#define DYNAMIC_DHT_ED25519_H

#include "pubkey.h"
#include "support/allocators/secure.h"


#include <array>
#include <cstring>

struct ed25519_context
{
    ed25519_context() = default;

    explicit ed25519_context(char const* b)
    { std::copy(b, b + len, seed.begin()); }

    bool operator==(ed25519_context const& rhs) const
    { return seed == rhs.seed; }

    bool operator!=(ed25519_context const& rhs) const
    { return seed != rhs.seed; }

    constexpr static int len = 32;

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

/**
 * secure_allocator is defined in support/allocators/secure.h and uses std::allocator
 * CPrivKeyEd25519 is a serialized private key, with all parameters included
 */
typedef std::vector<unsigned char, secure_allocator<unsigned char> > CPrivKeyEd25519;

/** An encapsulated ed25519 private key. */
class CKeyEd25519
{
public:
    /**
     * ed25519:
     */
    std::array<char, 32> seed;
    //TODO (DHT): store privateKey in a secure allocator:
    std::array<char, 64> privateKey;
    //std::vector<unsigned char, secure_allocator<unsigned char> > keyData;
    std::array<char, 32> publicKey;

public:
    //! Construct an invalid private key.
    CKeyEd25519()
    {
    }

    //! Destructor (necessary because of memlocking). ??
    ~CKeyEd25519()
    {
    }

    CKeyEd25519(const std::array<char, 32>& _seed);

    //! Simple read-only vector-like interface.
    unsigned int size() const { return seed.size(); }
    const unsigned char* begin() const { return GetPrivSeed().data(); }
    const unsigned char* end() const { return GetPrivSeed().data() + size(); }

    //! Generate a new private key using LibTorrent's Ed25519 implementation
    void MakeNewKeyPair();

    std::vector<unsigned char> GetPrivKey() const;
    std::vector<unsigned char> GetPubKey() const;
    std::vector<unsigned char> GetPrivSeed() const;

    CPubKey PubKey() const;

    /**
     * Used for the Torrent DHT.
     */
    std::array<char, 64> GetDHTPrivKey() const { return privateKey; }
    /**
     * Used for the Torrent DHT.
     */
    std::array<char, 32> GetDHTPubKey() const { return publicKey; }

    void SetMaster(const unsigned char* seed, unsigned int nSeedLen);
};

bool ECC_Ed25519_InitSanityCheck();
void ECC_Ed25519_Start();
void ECC_Ed25519_Stop();

#endif // DYNAMIC_DHT_ED25519_H