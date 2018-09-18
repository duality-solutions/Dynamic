
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "keyed25519.h"

#include <array>

#include "libtorrent/hasher512.hpp"
#include "libtorrent/kademlia/ed25519.hpp"
#include "libtorrent/kademlia/types.hpp"

#include <assert.h>
#include <tuple>

using namespace libtorrent::dht;

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
};

static ed25519_context* ed25519_context_sign = NULL;

// TODO (BDAP): Implement check Ed25519 keys
bool CKeyEd25519::Check(const unsigned char *vch) 
{
    return true;
    //return secp256k1_ec_seckey_verify(secp256k1_context_sign, vch);
}

//! Generate a new private key using LibTorrent's Ed25519
// TODO (BDAP): Support compressed Ed25519 keys
void CKeyEd25519::MakeNewKeyPair()
{
    std::tuple<libtorrent::dht::public_key, libtorrent::dht::secret_key> newKeyPair;
    newKeyPair = ed25519_create_keypair(ed25519_context_sign->seed);
	// Load the new ed25519 private key
    {
        secret_key privateKey = std::get<1>(newKeyPair);
        std::vector<unsigned char> vchPrivateKey;
        for (std::size_t i{0}; i < sizeof(privateKey); ++i) {
            unsigned char charValue = static_cast<unsigned char>(privateKey.bytes[i]);
            vchPrivateKey.push_back(charValue);
        }
        Set(vchPrivateKey.begin(), vchPrivateKey.end(), false);
    }
	// Load the new ed25519 public key
    {
        public_key publicKey = std::get<0>(newKeyPair);
        std::vector<unsigned char> vchPublicKey;
        for (std::size_t i{0}; i < sizeof(publicKey); ++i) {
            unsigned char charValue = static_cast<unsigned char>(publicKey.bytes[i]);
            vchPublicKey.push_back(charValue);
        }
        SetPubKey(vchPublicKey.begin(), vchPublicKey.end());
    }
    fValid = true;
}

void CKeyEd25519::SetMaster(const unsigned char* seed, unsigned int nSeedLen) 
{
    assert(nSeedLen == 32); // TODO: (BDAP) Allow larger seed size with 64 max
    ed25519_context* ctx = new ed25519_context(reinterpret_cast<const char*>(seed));
    assert(ctx != NULL);
    ed25519_context_sign = ctx;
    return;
}

void ECC_Ed25519_Start() 
{
    assert(ed25519_context_sign == NULL);

    ed25519_context* ctx = new ed25519_context();
    assert(ctx != NULL);
    {
        ctx->seed = ed25519_create_seed();
    }
    ed25519_context_sign = ctx;
}

/*
bool ECC_Ed25519_InitSanityCheck() 
{
    CKeyEd25519 key;
    key.MakeNewKey(true);
    CPubKeyEd25519 pubkey = key.GetPubKey();
    return key.VerifyPubKey(pubkey);
}
*/

void ECC_Ed25519_Stop() 
{
    ed25519_context *ctx = ed25519_context_sign;
    ed25519_context_sign = NULL;
    assert(ctx == NULL);
}