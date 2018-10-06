
// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#include "ed25519.h"

#include "hash.h"
#include "random.h"
#include "util.h"

#include <libtorrent/ed25519.hpp>
#include <libtorrent/hex.hpp>
#include <libtorrent/kademlia/ed25519.hpp>
#include <libtorrent/kademlia/types.hpp>

#include <array>
#include <assert.h>
#include <tuple>

using namespace libtorrent;

static ed25519_context* ed25519_context_sign = NULL;

// TODO (BDAP): Implement check Ed25519 keys


CKeyEd25519::CKeyEd25519(const std::array<char, 32>& _seed)
{
    seed = _seed;
    std::tuple<dht::public_key, dht::secret_key> keyPair = dht::ed25519_create_keypair(seed);
    {
        dht::secret_key sk = std::get<1>(keyPair);
        privateKey = sk.bytes;
    }
    {
        dht::public_key pk = std::get<0>(keyPair);
        publicKey = pk.bytes;
    }
}


//! Generate a new private key using LibTorrent's Ed25519
// TODO (BDAP): Support compressed Ed25519 keys
void CKeyEd25519::MakeNewKeyPair()
{
    // Load seed
    seed = dht::ed25519_create_seed();
    /*
    if (ed25519_context_sign == NULL || sizeof(ed25519_context_sign->seed) == 0 || ed25519_context_sign->IsNull()) {
        //LogPrintf("CKeyEd25519::MakeNewKeyPair -- created new seed.\n");
        seed = dht::ed25519_create_seed();
    }
    else {
        seed = ed25519_context_sign->seed;
        std::string strSeed = aux::to_hex(seed);
        //LogPrintf("CKeyEd25519::MakeNewKeyPair -- used existing seed = %s.\n", strSeed);
    }
    */
    // Load the new ed25519 private key
    std::tuple<dht::public_key, dht::secret_key> newKeyPair = dht::ed25519_create_keypair(seed); 
    {
        dht::secret_key sk = std::get<1>(newKeyPair);
        privateKey = sk.bytes;
        //LogPrintf("CKeyEd25519::MakeNewKeyPair -- vchPrivateKey = %s, size = %u\n", stringFromVch(vchPrivateKey), vchPrivateKey.size());
    }
    // Load the new ed25519 public key
    {
        dht::public_key pk = std::get<0>(newKeyPair);
        publicKey = pk.bytes;
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

std::vector<unsigned char> CKeyEd25519::GetPrivKey() const
{
    std::string strPrivateKey = aux::to_hex(privateKey);
    return std::vector<unsigned char>(strPrivateKey.begin(), strPrivateKey.end());
}

std::vector<unsigned char> CKeyEd25519::GetPubKey() const
{
    std::string strPublicKey = aux::to_hex(publicKey);
    return std::vector<unsigned char>(strPublicKey.begin(), strPublicKey.end());
}

std::vector<unsigned char> CKeyEd25519::GetPrivSeed() const
{
    return std::vector<unsigned char>(seed.begin(), seed.end());
}

CPubKey CKeyEd25519::PubKey() const
{
    CPubKey pubkey(GetPubKey(), false);
    return pubkey;
}

void ECC_Ed25519_Start() 
{
    assert(ed25519_context_sign == NULL);
    ed25519_context* ctx = new ed25519_context();
    assert(ctx != NULL);
    {
        ctx->seed = dht::ed25519_create_seed();
        std::string strSeed = aux::to_hex(ctx->seed);
        //LogPrintf("ECC_Ed25519_Start -- new seed created = %s.\n", strSeed);
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
    std::string strSeed = aux::to_hex(ctx->seed);
    //LogPrintf("ECC_Ed25519_Stop -- before null seed = %s.\n", strSeed);
    ctx->SetNull();
    ed25519_context_sign = NULL;
    strSeed = aux::to_hex(ctx->seed);
    //LogPrintf("ECC_Ed25519_Stop -- after null seed = %s.\n", strSeed);
    assert(ed25519_context_sign == NULL);
}