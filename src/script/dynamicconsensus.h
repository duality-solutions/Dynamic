// Copyright (c) 2009-2019 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DYNAMICCONSENSUS_H
#define DYNAMIC_DYNAMICCONSENSUS_H

#if defined(BUILD_DYNAMIC_INTERNAL) && defined(HAVE_CONFIG_H)
#include "config/dynamic-config.h"
#if defined(_WIN32)
#if defined(DLL_EXPORT)
#if defined(HAVE_FUNC_ATTRIBUTE_DLLEXPORT)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL
#endif
#endif
#elif defined(HAVE_FUNC_ATTRIBUTE_VISIBILITY)
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif
#elif defined(MSC_VER) && !defined(STATIC_LIBDYNAMICCONSENSUS)
#define EXPORT_SYMBOL __declspec(dllimport)
#endif

#ifndef EXPORT_SYMBOL
#define EXPORT_SYMBOL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DYNAMICCONSENSUS_API_VER 0

typedef enum dynamicconsensus_error_t {
    dynamicconsensus_ERR_OK = 0,
    dynamicconsensus_ERR_TX_INDEX,
    dynamicconsensus_ERR_TX_SIZE_MISMATCH,
    dynamicconsensus_ERR_TX_DESERIALIZE,
    dynamicconsensus_ERR_INVALID_FLAGS,
} dynamicconsensus_error;

/** Script verification flags */
enum {
    dynamicconsensus_SCRIPT_FLAGS_VERIFY_NONE                = 0,
    dynamicconsensus_SCRIPT_FLAGS_VERIFY_P2SH                = (1U << 0),  // evaluate P2SH (BIP16) subscripts
    dynamicconsensus_SCRIPT_FLAGS_VERIFY_DERSIG              = (1U << 2),  // enforce strict DER (BIP66) compliance
    dynamicconsensus_SCRIPT_FLAGS_VERIFY_NULLDUMMY           = (1U << 4),  // enforce NULLDUMMY (BIP147)
    dynamicconsensus_SCRIPT_FLAGS_VERIFY_CHECKLOCKTIMEVERIFY = (1U << 9),  // enable CHECKLOCKTIMEVERIFY (BIP65)
    dynamicconsensus_SCRIPT_FLAGS_VERIFY_CHECKSEQUENCEVERIFY = (1U << 10), // enable CHECKSEQUENCEVERIFY (BIP112)
    dynamicconsensus_SCRIPT_FLAGS_VERIFY_ALL = dynamicconsensus_SCRIPT_FLAGS_VERIFY_P2SH | dynamicconsensus_SCRIPT_FLAGS_VERIFY_DERSIG |
                                               dynamicconsensus_SCRIPT_FLAGS_VERIFY_NULLDUMMY | dynamicconsensus_SCRIPT_FLAGS_VERIFY_CHECKLOCKTIMEVERIFY |
                                               dynamicconsensus_SCRIPT_FLAGS_VERIFY_CHECKSEQUENCEVERIFY
};

/// Returns 1 if the input nIn of the serialized transaction pointed to by
/// txTo correctly spends the scriptPubKey pointed to by scriptPubKey under
/// the additional constraints specified by flags.
/// If not NULL, err will contain an error/success code for the operation
EXPORT_SYMBOL int dynamicconsensus_verify_script(const unsigned char* scriptPubKey, unsigned int scriptPubKeyLen, const unsigned char* txTo, unsigned int txToLen, unsigned int nIn, unsigned int flags, dynamicconsensus_error* err);

EXPORT_SYMBOL unsigned int dynamicconsensus_version();

#ifdef __cplusplus
} // extern "C"
#endif

#undef EXPORT_SYMBOL

#endif // DYNAMIC_DYNAMICCONSENSUS_H
