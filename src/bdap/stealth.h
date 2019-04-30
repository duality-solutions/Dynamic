// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2018 The Particl Core developers
// Copyright (c) 2014 The ShadowCoin developers
// Distributed under the MIT/X11 software license, see the accompanying
// file license.txt or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_KEY_STEALTH_H
#define DYNAMIC_KEY_STEALTH_H

#include <key.h>
#include <serialize.h>
#include <uint256.h>

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <inttypes.h>

class CScript;

typedef std::vector<uint8_t> ec_point;

const uint32_t MAX_STEALTH_NARRATION_SIZE = 48;
const uint32_t MIN_STEALTH_RAW_SIZE = 1 + 33 + 1 + 33 + 1 + 1; // without checksum (4bytes) or version (1byte)
const size_t EC_SECRET_SIZE = 32;
const size_t EC_COMPRESSED_SIZE = 33;
const size_t EC_UNCOMPRESSED_SIZE = 65;

struct stealth_prefix
{
    uint8_t number_bits;
    uint32_t bitfield;
};

class CStealthAddress
{
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    uint8_t options;
    uint8_t prefix_number_bits;
    uint32_t prefix_bitfield;
    int number_signatures;
    ec_point scan_pubkey;
    ec_point spend_pubkey;

    CKey scan_secret;       // Better to store the scan secret here as it's needed often
    CKeyID spend_secret_id; // store the spend secret in a keystore

    CStealthAddress()
    {
        options = 0;
        number_signatures = 0;
        prefix_number_bits = 0;
    };

    CStealthAddress(const CKey& scanKey, const CKey& spendKey);
    CStealthAddress(const CKey& scanKey, const CKey& spendKey, const uint8_t bits, const uint32_t prefix);

    bool IsNull() const { return (scan_pubkey.size() == 0 || spend_pubkey.size() == 0 || spend_secret_id.IsNull()); }

    bool SetEncoded(const std::string &encodedAddress);
    std::string Encoded() const;
    std::string ToString() const {return Encoded();};

    int FromRaw(const uint8_t *p, size_t nSize);
    int ToRaw(std::vector<uint8_t> &raw) const;

    int SetScanPubKey(const CPubKey &pk);

    CKeyID GetSpendKeyID() const;

    bool operator <(const CStealthAddress &y) const
    {
        return memcmp(&scan_pubkey[0], &y.scan_pubkey[0], EC_COMPRESSED_SIZE) < 0;
    }

    bool operator ==(const CStealthAddress &y) const
    {
        return memcmp(&scan_pubkey[0], &y.scan_pubkey[0], EC_COMPRESSED_SIZE) == 0;
    }

    inline CStealthAddress operator=(const CStealthAddress& b) {
        nVersion = b.nVersion;
        options = b.options;
        prefix_number_bits = b.prefix_number_bits;
        prefix_bitfield = b.prefix_bitfield;
        number_signatures = b.number_signatures;
        scan_pubkey = b.scan_pubkey;
        spend_pubkey = b.spend_pubkey;
        scan_secret = b.scan_secret;
        spend_secret_id = b.spend_secret_id;
        return *this;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(options);
        READWRITE(prefix_number_bits);
        READWRITE(prefix_bitfield);
        READWRITE(number_signatures);
        READWRITE(scan_pubkey);
        READWRITE(spend_pubkey);
        spend_secret_id = GetSpendKeyID();
        if (ser_action.ForRead()) {
            std::vector<unsigned char> vchScanKey;
            READWRITE(vchScanKey);
            scan_secret.Set(vchScanKey.begin(), vchScanKey.end(), true);
        } else {
            std::vector<unsigned char> vchScanKey(scan_secret.begin(), scan_secret.end());
            READWRITE(vchScanKey);
        }
    }
};

int SecretToPublicKey(const CKey &secret, ec_point &out);

int StealthShared(const CKey &secret, const ec_point &pubkey, CKey &sharedSOut);
int StealthSecret(const CKey &secret, const ec_point &pubkey, const ec_point &pkSpend, CKey &sharedSOut, ec_point &pkOut);
int StealthSecretSpend(const CKey &scanSecret, const ec_point &ephemPubkey, const CKey &spendSecret, CKey &secretOut);
int StealthSharedToSecretSpend(const CKey &sharedS, const CKey &spendSecret, CKey &secretOut);

int StealthSharedToPublicKey(const ec_point &pkSpend, const CKey &sharedS, ec_point &pkOut);

bool IsStealthAddress(const std::string &encodedAddress);

inline uint32_t SetStealthMask(uint8_t nBits)
{
    return (nBits == 32 ? 0xFFFFFFFF : ((1<<nBits)-1));
};

uint32_t FillStealthPrefix(uint8_t nBits, uint32_t nBitfield);

bool ExtractStealthPrefix(const char *pPrefix, uint32_t &nPrefix);

int MakeStealthData(const stealth_prefix& prefix, const CKey& sShared, const CPubKey& pkEphem, std::vector<uint8_t>& vData, uint32_t& nStealthPrefix, std::string& sError);

int PrepareStealthOutput(const CStealthAddress& sx, CScript& scriptPubKey, std::vector<uint8_t>& vData, std::string& sError);

void ECC_Start_Stealth();
void ECC_Stop_Stealth();


#endif  // DYNAMIC_KEY_STEALTH_H

