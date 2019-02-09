
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

static std::string StringFromVch(const std::vector<unsigned char>& vch) {
    std::string res;
    std::vector<unsigned char>::const_iterator vi = vch.begin();
    while (vi != vch.end()) {
        res += (char) (*vi);
        vi++;
    }
    return res;
}

CKeyEd25519::CKeyEd25519(const std::vector<unsigned char>& _seed)
{
    std::string strSeed = StringFromVch(_seed);
    aux::from_hex(strSeed, seed.data());
    if (sizeof(_seed) == 32) {
        for(unsigned int i = 0; i < sizeof(_seed); i++) {
            seed[i] = _seed[i];
        }
    }
    
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
void CKeyEd25519::MakeNewKeyPair()
{
    // Load seed
    seed = dht::ed25519_create_seed();
    // Load the new ed25519 private key
    std::tuple<dht::public_key, dht::secret_key> newKeyPair = dht::ed25519_create_keypair(seed); 
    {
        dht::secret_key sk = std::get<1>(newKeyPair);
        privateKey = sk.bytes;
    }
    // Load the new ed25519 public key
    {
        dht::public_key pk = std::get<0>(newKeyPair);
        publicKey = pk.bytes;
    }
}

std::string CKeyEd25519::GetPrivKeyString() const
{
    return aux::to_hex(privateKey);
}

std::string CKeyEd25519::GetPubKeyString() const
{
    return aux::to_hex(publicKey);
}

std::string CKeyEd25519::GetPrivSeedString() const
{
    return aux::to_hex(seed);
}

std::vector<unsigned char> CKeyEd25519::GetPrivKey() const
{
    std::string strPrivateKey = GetPrivKeyString();
    return std::vector<unsigned char>(strPrivateKey.begin(), strPrivateKey.end());
}

std::vector<unsigned char> CKeyEd25519::GetPubKey() const
{
    std::string strPublicKey = GetPubKeyString();
    return std::vector<unsigned char>(strPublicKey.begin(), strPublicKey.end());
}

std::vector<unsigned char> CKeyEd25519::GetPrivSeed() const
{
    std::string strPrivateSeedKey = GetPrivSeedString();
    return std::vector<unsigned char>(strPrivateSeedKey.begin(), strPrivateSeedKey.end());
}

std::vector<unsigned char> CKeyEd25519::GetPrivKeyBytes() const
{
    std::vector<unsigned char> vchRawPrivKey;
    for(unsigned int i = 0; i < sizeof(privateKey); i++) {
        vchRawPrivKey.push_back(privateKey[i]);
    }
    return vchRawPrivKey;
}

std::vector<unsigned char> CKeyEd25519::GetPubKeyBytes() const
{
    std::vector<unsigned char> vchRawPubKey;
    for(unsigned int i = 0; i < sizeof(publicKey); i++) {
        vchRawPubKey.push_back(publicKey[i]);
    }
    return vchRawPubKey;
}

std::vector<unsigned char> CKeyEd25519::GetPrivSeedBytes() const
{
    std::vector<unsigned char> vchRawPrivSeed;
    for(unsigned int i = 0; i < sizeof(seed); i++) {
        vchRawPrivSeed.push_back(seed[i]);
    }
    return vchRawPrivSeed;
}
void ECC_Ed25519_Start() 
{
    assert(ed25519_context_sign == NULL);
    ed25519_context* ctx = new ed25519_context();
    assert(ctx != NULL);
    {
        ctx->seed = dht::ed25519_create_seed();
        std::string strSeed = aux::to_hex(ctx->seed);
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
    ctx->SetNull();
    ed25519_context_sign = NULL;
    strSeed = aux::to_hex(ctx->seed);
    assert(ed25519_context_sign == NULL);
}

static unsigned char const* StardardArrayToArrayPtr32(const std::array<char, 32>& stdArray32)
{
    return reinterpret_cast<unsigned char const*>(stdArray32.data());
}

static unsigned char const* StardardArrayToArrayPtr64(const std::array<char, 64>& stdArray64)
{
    return reinterpret_cast<unsigned char const*>(stdArray64.data());
}

static std::array<char, 32> ArrayPtrToStandardArray32(const unsigned char* pArray)
{
    //TODO (bdap): Improve this conversion function
    std::array<char, 32> arr32;
    for(unsigned int i = 0; i < 32; i++) {
         arr32[i] = (char)pArray[i];
    }
    return arr32;
}

std::vector<unsigned char> GetLinkSharedPubKey(const CKeyEd25519& dhtKey, const std::vector<unsigned char>& vchOtherPubKey)
{
    std::array<char, 32> sharedSeed = GetLinkSharedPrivateKey(dhtKey, vchOtherPubKey);
    CKeyEd25519 sharedKey(sharedSeed);
    return sharedKey.GetPubKey();
}

std::array<char, 32> GetLinkSharedPrivateKey(const CKeyEd25519& dhtKey, const std::vector<unsigned char>& vchOtherPubKey)
{
    // convert private key
    unsigned char const* private_key = StardardArrayToArrayPtr64(dhtKey.privateKey);
    // convert public key
    unsigned char const* public_key;
    std::array<char, ED25519_PUBLIC_KEY_BYTE_LENGTH> arrPubKey;
    std::string strRecipientPubKey = StringFromVch(vchOtherPubKey);
    aux::from_hex(strRecipientPubKey, arrPubKey.data());
    public_key = StardardArrayToArrayPtr32(arrPubKey);
    // get shared secret key
    std::array<char, 32> secret;
    unsigned char* shared_secret = reinterpret_cast<unsigned char*>(secret.data());
    ed25519_key_exchange(shared_secret, public_key, private_key);
    std::array<char, 32> sharedSeed = ArrayPtrToStandardArray32(shared_secret);
    return sharedSeed;
}

std::vector<unsigned char> EncodedPubKeyToBytes(const std::vector<unsigned char>& vchEncodedPubKey)
{
    // TODO (bdap): Use a more efficient way to convert the hex encoded pubkey to bytes
    std::vector<unsigned char> vchPubKeyBytes;
    std::string strEncodedPubKey = StringFromVch(vchEncodedPubKey);
    std::array<char, ED25519_PUBLIC_KEY_BYTE_LENGTH> pubkey;
    aux::from_hex(strEncodedPubKey, pubkey.data());
    for(unsigned int i = 0; i < sizeof(pubkey); i++) {
        vchPubKeyBytes.push_back(pubkey[i]);
    }
    return vchPubKeyBytes;
}