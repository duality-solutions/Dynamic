// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_SWAP_SS58_H
#define DYNAMIC_SWAP_SS58_H

#include "uint256.h"

#include <string>
#include <vector>

//See https://docs.substrate.io/v3/advanced/ss58

const std::vector<uint16_t> vAcceptedAddressTypes = {42, 128}; // just accept Substrate and Clover address types for now.

class CSS58
{
private:
    std::string strAddress;
    std::vector<uint8_t> vchPrefix;
    std::vector<uint8_t> vchAddress;
    std::vector<uint8_t> vchAddressType;
    std::vector<uint8_t> vchPublicKey;
    std::vector<uint8_t> vchAddressCheckSum;
    uint512 calulatedhash;
    std::vector<uint8_t> vchCalulatedCheckSum;
    uint32_t type;

public:
    static const int CURRENT_VERSION = 1;
    int nVersion;
    int nLength;
    bool fValid;
    std::string strError = "";

    inline void SetNull()
    {
        nVersion = CSS58::CURRENT_VERSION;
        nLength = -1;
        fValid = false;
        strError = "";
        strAddress = "";
        vchAddress.clear();
        vchAddressType.clear();
        vchPublicKey.clear();
        vchAddressCheckSum.clear();
        calulatedhash = uint512(0);
        vchCalulatedCheckSum.clear();
        type = 0;
    }

    CSS58(std::string address);

    uint32_t AddressType() const;
    std::string Address() const { return strAddress; }
    std::vector<uint8_t> AddressBytes() const { return vchAddress; }
    std::string AddressHex() const;
    std::string AddressTypeHex() const;
    std::string PublicKeyHex() const;
    std::string AddressChecksumHex() const;
    std::string CalulatedChecksumHex() const;

    uint512 CalculatedHash() const { return calulatedhash; }
    std::string CalculatedHashHex() const { return calulatedhash.GetHex(); }
    bool ValidChecksum() const { return (vchCalulatedCheckSum == vchAddressCheckSum); }
    bool Valid() const { return (fValid && ValidChecksum()); }

private:
    void calulatedChecksumHash();
    void setAddressType();
};

#endif // DYNAMIC_SWAP_SS58_H
