
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "keyed25519.h"

#include "random.h"
#include "bdap/domainentry.h"
#include "util.h"

#include <libtorrent/ed25519.hpp>
#include <libtorrent/hex.hpp>
#include <libtorrent/kademlia/ed25519.hpp>
#include <libtorrent/kademlia/types.hpp>

#include <array>
#include <assert.h>
#include <tuple>

using namespace libtorrent;

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
    // Load seed
    std::array<char, 32> seed;
    if (ed25519_context_sign == NULL || sizeof(ed25519_context_sign->seed) == 0) {
        LogPrintf("CKeyEd25519::MakeNewKeyPair -- created new seed.\n");
        seed = dht::ed25519_create_seed();
    }
    else {
        seed = ed25519_context_sign->seed;
    }
    // Load the new ed25519 private key
    std::tuple<dht::public_key, dht::secret_key> newKeyPair = dht::ed25519_create_keypair(seed); 
    {
        dht::secret_key privateKey = std::get<1>(newKeyPair);
        std::vector<unsigned char> vchPrivateKey;
        std::string strPrivateKey = aux::to_hex(privateKey.bytes);
        vchPrivateKey = vchFromString(strPrivateKey);
        memcpy(keyData.data(), &vchPrivateKey[0], vchPrivateKey.size());
        //LogPrintf("CKeyEd25519::MakeNewKeyPair -- vchPrivateKey = %s, size = %u\n", stringFromVch(vchPrivateKey), vchPrivateKey.size());
    }
    // Load the new ed25519 public key
    {
        dht::public_key publicKey = std::get<0>(newKeyPair);
        std::string strPublicKey = aux::to_hex(publicKey.bytes);
        publicKeyData = vchFromString(strPublicKey);
        //LogPrintf("CKeyEd25519::MakeNewKeyPair -- vchPublicKey = %s, size = %u\n", stringFromVch(publicKeyData), publicKeyData.size());
    }
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
        ctx->seed = dht::ed25519_create_seed();
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
    //ctx->seed = NULL;
    ed25519_context_sign = NULL;
    assert(ctx == NULL);
}