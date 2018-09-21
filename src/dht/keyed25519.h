// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_KEYED25519_H
#define DYNAMIC_DHT_KEYED25519_H

#include "support/allocators/secure.h"

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
    static const unsigned int PRIVATE_KEY_SIZE            = 279;
    static const unsigned int COMPRESSED_PRIVATE_KEY_SIZE = 214;
    /**
     * see www.keylength.com
     * script supports up to 75 for single byte push
     */
    static_assert(
        PRIVATE_KEY_SIZE >= COMPRESSED_PRIVATE_KEY_SIZE,
            "COMPRESSED_PRIVATE_KEY_SIZE is larger than PRIVATE_KEY_SIZE");

private:
    //! Whether this private key is valid. We check for correctness when modifying the key
    //! data, so fValid should always correspond to the actual state.
    bool fValid;

    //! Whether the public key corresponding to this private key is (to be) compressed.
    bool fCompressed;

    //! The actual private key byte data
    std::vector<unsigned char, secure_allocator<unsigned char> > keyData;
    //! The public key byte data
    std::vector<unsigned char> publicKeyData;

    //! Check whether the 32-byte array pointed to be vch is valid keyData.
    bool static Check(const unsigned char* vch);

    //! Initialize using begin and end iterators to byte data.
    template <typename T>
    void SetPubKey(const T pbegin, const T pend)
    {
        if (size_t(pend - pbegin) != publicKeyData.size())
            return;
        memcpy(publicKeyData.data(), (unsigned char*)&pbegin[0], publicKeyData.size());
    }

public:
    //! Construct an invalid private key.
    CKeyEd25519() : fValid(false), fCompressed(false)
    {
        // Important: vch must be 32 bytes in length to not break serialization
        keyData.resize(32);
    }

    //! Destructor (again necessary because of memlocking).
    ~CKeyEd25519()
    {
    }
    
    friend bool operator==(const CKeyEd25519& a, const CKeyEd25519& b)
    {
        return a.fCompressed == b.fCompressed &&
            a.size() == b.size() &&
            memcmp(a.keyData.data(), b.keyData.data(), a.size()) == 0;
    }

    //! Initialize using begin and end iterators to byte data.
    template <typename T>
    void Set(const T pbegin, const T pend, bool fCompressedIn)
    {
        if (size_t(pend - pbegin) != keyData.size()) {
            fValid = false;
        } else if (Check(&pbegin[0])) {
            memcpy(keyData.data(), (unsigned char*)&pbegin[0], keyData.size());
            fValid = true;
            fCompressed = fCompressedIn;
        } else {
            fValid = false;
        }
    }

    //! Simple read-only vector-like interface.
    unsigned int size() const { return (fValid ? keyData.size() : 0); }
    const unsigned char* begin() const { return keyData.data(); }
    const unsigned char* end() const { return keyData.data() + size(); }
    
    unsigned int PubKeySize() const { return publicKeyData.size(); }
    const unsigned char* PubKeyBegin() const { return publicKeyData.data(); }
    const unsigned char* PubKeyEnd() const { return publicKeyData.data() + PubKeySize(); }

    //! Check whether this private key is valid.
    bool IsValid() const { return fValid; }

    //! Check whether the public key corresponding to this private key is (to be) compressed.
    bool IsCompressed() const { return fCompressed; }

    //! Initialize from a CPrivKeyEd25519 (serialized OpenSSL private key data).
    //bool SetPrivKey(const CPrivKeyEd25519& vchPrivKey, bool fCompressed);

    //! Generate a new private key using LibTorrent's Ed25519 implementation
    void MakeNewKeyPair();

    /**
     * Convert the private key to a CPrivKeyEd25519 (serialized OpenSSL private key data).
     * This is expensive. 
     */
    CPrivKeyEd25519 GetPrivKey() const;

    std::vector<unsigned char> GetDHTPrivKey() const;

    std::vector<unsigned char> GetPubKey() const { return publicKeyData; }
    /**
     * Compute the public key from a private key.
     * This is expensive.
     */
    // CPubKeyEd25519 GetPubKey() const;

    void SetMaster(const unsigned char* seed, unsigned int nSeedLen);
    /**
     * Create a DER-serialized signature.
     * The test_case parameter tweaks the deterministic nonce.
     */
    //bool Sign(const uint256& hash, std::vector<unsigned char>& vchSig, uint32_t test_case = 0) const;

    /**
     * Create a compact signature (65 bytes), which allows reconstructing the used public key.
     * The format is one header byte, followed by two times 32 bytes for the serialized r and s values.
     * The header byte: 0x1B = first key with even y, 0x1C = first key with odd y,
     *                  0x1D = second key with even y, 0x1E = second key with odd y,
     *                  add 0x04 for compressed keys.
     */
    //bool SignCompact(const uint256& hash, std::vector<unsigned char>& vchSig) const;

    //! Derive BIP32 child key.
    //bool Derive(CKey& keyChild, ChainCode &ccChild, unsigned int nChild, const ChainCode& cc) const;

    /**
     * Verify thoroughly whether a private key and a public key match.
     * This is done using a different mechanism than just regenerating it.
     */
    //bool VerifyPubKey(const CPubEd25519Key& vchPubKey) const;

    //! Load private key and check that public key matches.
    //bool Load(CPrivKeyEd25519& privkey, CPubEd25519Key& vchPubKey, bool fSkipCheck);
};

bool ECC_Ed25519_InitSanityCheck();
void ECC_Ed25519_Start();
void ECC_Ed25519_Stop();

#endif // DYNAMIC_DHT_KEYED25519_H