// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_CURVE25519_H
#define DYNAMIC_DHT_CURVE25519_H

#define COMPILER_ASSERT(X) (void) sizeof(char[(X) ? 1 : -1])

#ifndef CRYPTO_ALIGN
# if defined(__INTEL_COMPILER) || defined(_MSC_VER)
#  define CRYPTO_ALIGN(x) __declspec(align(x))
# else
#  define CRYPTO_ALIGN(x) __attribute__ ((aligned(x)))
# endif
#endif

#define crypto_hash_sha512_BYTES 64U
#define crypto_scalarmult_curve25519_BYTES 32U

int Ed25519_To_Curve25519_PubKey(unsigned char* curve25519_pubkey, const unsigned char* ed25519_pubkey);
int Ed25519_To_Curve25519_PrivKey(unsigned char* curve25519_privkey, const unsigned char* ed25519_privkey);

#endif // DYNAMIC_DHT_CURVE25519_H