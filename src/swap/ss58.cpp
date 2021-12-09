// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "swap/ss58.h"

#include "base58.h"
#include "bdap/utils.h"
#include "hash.h"
#include "util.h"

CSS58::CSS58(std::string address)
{
    SetNull();
    strAddress = address;
    vchPrefix = vchFromString("SS58PRE"); // SS58PRE
    fValid = DecodeBase58(strAddress, vchAddress);
    nLength = vchAddress.size();
    int prefixLen = 1;
    if (fValid) {
        if (nLength >= 35) {
            if (nLength >= 36)
                prefixLen = 2;
            vchAddressType = std::vector<uint8_t>(&vchAddress[0], &vchAddress[prefixLen]);
            setAddressType();
            vchPublicKey = std::vector<uint8_t>(&vchAddress[prefixLen], &vchAddress[prefixLen + 32]);
            vchAddressCheckSum = std::vector<uint8_t>(&vchAddress[prefixLen + 32], &vchAddress[nLength]);
        } else {
            strError = strprintf("BadLength: Invalid SS58 address size %d", vchAddress.size());
            fValid = false;
            return;
        }
    } else {
        strError = strprintf("BadBase58: Failed to decode base58 address");
        return;
    }
    calulatedChecksumHash();
    if (strError == "" && std::count(vAcceptedAddressTypes.begin(), vAcceptedAddressTypes.end(), type) == 0)
    {
        strError = strprintf("AddressType: Invalid SS58 address type (%d)", type);
        fValid = false;
    } else {
        // ToDo: Check if checksum prefix matches the checksum hash prefix
        fValid = true;
    }
}

std::string CSS58::AddressHex() const
{
    if (vchAddress.size() > 0)
        return "0x" + CharVectorToHexString(vchAddress);

    return "";
}

uint32_t CSS58::AddressType() const
{
    return type;
}

std::string CSS58::PublicKeyHex() const
{
    if (vchPublicKey.size() > 0)
        return "0x" + CharVectorToHexString(vchPublicKey);

    return "";
}

std::string CSS58::AddressChecksumHex() const
{
    if (vchAddressCheckSum.size() > 0)
        return "0x" + CharVectorToHexString(vchAddressCheckSum);

    return "";
}

std::string CSS58::CalulatedChecksumHex() const
{
    if (vchAddressCheckSum.size() > 0)
        return "0x" + CharVectorToHexString(vchCalulatedCheckSum);

    return "";
}

void CSS58::setAddressType()
{
    if (vchAddressType.size() > 0 && vchAddressType[0] >= 64) {
        // Full Format
        // https://github.com/paritytech/substrate/blob/master/primitives/core/src/crypto.rs
        // weird bit manipulation owing to the combination of LE encoding and missing two
        // bits from the left.
        // d[0] d[1] are: 01aaaaaa bbcccccc
        // they make the LE-encoded 16-bit value: aaaaaabb 00cccccc
        // so the lower byte is formed of aaaaaabb and the higher byte is 00cccccc
        uint8_t lower = vchAddressType[0] << 2 | vchAddressType[1] >> 6;
        uint8_t upper = vchAddressType[1] & 0b00111111; //63 0x3F 0b00111111 '?'
        type = (uint32_t)(256 * upper + lower);
    } else if (vchAddressType.size() > 0) {
        // Simple Format
        type = (uint32_t)vchAddressType[0];
    }
}

void CSS58::calulatedChecksumHash()
{
    std::vector<uint8_t> vchImageData = vchPrefix;
    vchImageData.insert(vchImageData.end(), vchAddress.begin(), vchAddress.end());
    vchImageData.resize(nLength + vchPrefix.size() - vchAddressCheckSum.size());
    calulatedhash = HashBlake2b_512(vchImageData.begin(), vchImageData.end());
    std::vector<uint8_t> vchHash = ParseHex(calulatedhash.GetHex());
    size_t hashSize = vchHash.size();
    if (hashSize >= 32 && hashSize > vchAddressCheckSum.size() + 1) {
        vchCalulatedCheckSum.clear();
        //reverse bytes by starting from the right
        for (size_t i = 1; i < vchAddressCheckSum.size() + 1; i++) {
          vchCalulatedCheckSum.push_back(vchHash[hashSize - i]);
        }
        if (vchCalulatedCheckSum != vchAddressCheckSum) {
            if (strError == "") {
                strError = "Checksum does not match.";
            } else {
                strError += ", Checksum does not match.";
            }
            
        }
    }
}
