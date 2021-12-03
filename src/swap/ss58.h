// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_SWAP_SS58_H
#define DYNAMIC_SWAP_SS58_H

#include <string>
#include <vector>

//See https://docs.substrate.io/v3/advanced/ss58

const std::vector<uint16_t> vAcceptedAddressTypes = {42, 128}; // just accept Substrate and Clover address types for now.

class CSS58
{
private:
    std::vector<uint8_t> vchAddress;
    std::vector<uint8_t> vchAddressType;
    std::vector<uint8_t> vchPublicKey;
    std::vector<uint8_t> vchCheckSum;
    uint8_t lower;
    uint8_t upper;
    uint16_t type;

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
        vchAddress.clear();
        vchAddressType.clear();
        vchPublicKey.clear();
        vchCheckSum.clear();
        lower = 0;
        upper = 0;
        type = 0;
        strError = "";
    }

    CSS58(std::string strAddress);

    uint16_t AddressType() const;
    std::vector<uint8_t> Address() const { return vchAddress; }
    std::string AddressHex() const;
    std::string AddressTypeHex() const;
    std::string PublicKeyHex() const;
    std::string CheckSumHex() const;
};

#endif // DYNAMIC_SWAP_SS58_H
