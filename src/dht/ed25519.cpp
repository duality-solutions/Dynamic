// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "ed25519.h"

#include "crypto/hmac_sha512.h"
#include "hash.h"
#include "random.h"
#include "util.h"

#include <libtorrent/ed25519.hpp>
#include <libtorrent/hex.hpp>
#include <libtorrent/kademlia/ed25519.hpp>
#include <libtorrent/kademlia/types.hpp>

#include <array>
#include <assert.h>
#include <iomanip> // std::setw
#include <tuple>
#include <vector>

using namespace libtorrent;

// TODO (BDAP): Implement check Ed25519 keys

CKeyEd25519::CKeyEd25519(const std::array<char, 32>& _seed)
{
    Set(_seed);
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

void CKeyEd25519::Set(const std::array<char, 32>& _seed)
{
    seed = _seed;
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

bool CKeyEd25519::Derive(CKeyEd25519& keyChild, const unsigned int nChild, const uint256& cc) const
{
    uint256 ccChild; // ChainCode
    std::vector<unsigned char, secure_allocator<unsigned char> > vout(64);
    BIP32Hash(cc, nChild, 0, begin(), vout.data());
    std::array<char, ED25519_PRIVATE_SEED_BYTE_LENGTH> newSeed = ConvertSecureVector32ToArray(vout);
    CKeyEd25519 newKey(newSeed);
    // Use Hiffe-Heilman key exchange to derive new key from current key
    std::array<char, 32> derivedSeed = GetLinkSharedPrivateKey(newKey, GetPubKey());
    CKeyEd25519 derivedKey(derivedSeed);
    keyChild = derivedKey;
    return true;
}

bool CEd25519ExtKey::Derive(CEd25519ExtKey& out, unsigned int _nChild) const
{
    out.nDepth = nDepth + 1;
    CKeyID id = key.GetID();
    memcpy(&out.vchFingerprint[0], &id, 4);
    out.nChild = _nChild;
    return key.Derive(out.key, _nChild, out.chaincode);
}

void CEd25519ExtKey::SetMaster(const unsigned char* seed, unsigned int nSeedLen)
{
    static const unsigned char hashkey[] = {'B', 'i', 't', 'c', 'o', 'i', 'n', ' ', 's', 'e', 'e', 'd'};
    std::vector<unsigned char, secure_allocator<unsigned char> > vout(64);
    CHMAC_SHA512(hashkey, sizeof(hashkey)).Write(seed, nSeedLen).Finalize(vout.data());
    std::array<char, ED25519_PRIVATE_SEED_BYTE_LENGTH> newSeed = ConvertSecureVector32ToArray(vout);
    key.Set(newSeed);
    memcpy(chaincode.begin(), &vout[32], 32);
    nDepth = 0;
    nChild = 0;
    memset(vchFingerprint, 0, sizeof(vchFingerprint));
}
/*
CEd25519ExtPubKey CEd25519ExtKey::Neuter() const
{
    CExtPubKey ret;
    ret.nDepth = nDepth;
    memcpy(&ret.vchFingerprint[0], &vchFingerprint[0], 4);
    ret.nChild = nChild;
    ret.pubkey = key.GetPubKey();
    ret.chaincode = chaincode;
    return ret;
}
*/
void CEd25519ExtKey::Encode(unsigned char code[74]) const
{
    code[0] = nDepth;
    memcpy(code + 1, vchFingerprint, 4);
    code[5] = (nChild >> 24) & 0xFF;
    code[6] = (nChild >> 16) & 0xFF;
    code[7] = (nChild >> 8) & 0xFF;
    code[8] = (nChild >> 0) & 0xFF;
    memcpy(code + 9, chaincode.begin(), 32);
    code[41] = 0;
    memcpy(code + 42, key.begin(), 32);
}

static std::array<char, 32> Bip32ArrayPtrToStandardArray32(const unsigned char* pArray, const unsigned int nStart)
{
    //TODO (bdap): Improve this conversion function
    std::array<char, 32> arr32;
    for(unsigned int i = nStart; i < (nStart + 32); i++) {
         arr32[i] = (char)pArray[i];
    }
    return arr32;
}

void CEd25519ExtKey::Decode(const unsigned char code[74])
{
    nDepth = code[0];
    memcpy(vchFingerprint, code + 1, 4);
    nChild = (code[5] << 24) | (code[6] << 16) | (code[7] << 8) | code[8];
    memcpy(chaincode.begin(), code + 9, 32);
    std::array<char, 32> seed = Bip32ArrayPtrToStandardArray32(code, 42);
    key.Set(seed);
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

std::string CharVectorToByteArrayString(const std::vector<unsigned char>& vchData)
{
    if (vchData.size() == 0)
        return "";

    std::stringstream ss;
    for(unsigned int i = 0; i < vchData.size(); i++) {
        ss << std::hex << std::setfill('0');
        ss  << "0x" << std::setw(2)  << static_cast<unsigned>(vchData[i]) << ",";
        ss <<  (((i + 1) % 16 == 0) ? "\n" : "");
    }
    ss.seekp(-1, std::ios_base::end);
    ss << "\n";
    return ss.str();
}

std::array<char, 32> ConvertSecureVector32ToArray(const std::vector<unsigned char, secure_allocator<unsigned char> >& vIn)
{
    std::array<char, 32> arrReturn;
    for(unsigned int i = 0; i < 32; i++) {
         arrReturn[i] = (char)vIn[i];
    }
    return arrReturn;
}