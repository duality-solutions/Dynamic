// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "swap/ss58.h"

#include "base58.h"
#include "bdap/utils.h"

//#include "chainparams.h"

// ToDo: check address type matches Params().nSubstrateType
// ToDo: checksum or error = InvalidChecksum

CSS58::CSS58(std::string strAddress)
{
    SetNull();

    fValid = DecodeBase58(strAddress, vchAddress);
    nLength = vchAddress.size();
    int prefixLen = 1;
    if (fValid) {
        if (nLength >= 35) {
            if (nLength >= 36)
                prefixLen = 2;
            vchAddressType = std::vector<uint8_t>(&vchAddress[0], &vchAddress[prefixLen]);
            if (vchAddressType[0] >= 64) {
            	// https://github.com/paritytech/substrate/blob/master/primitives/core/src/crypto.rs
                // weird bit manipulation owing to the combination of LE encoding and missing two
                // bits from the left.
                // d[0] d[1] are: 01aaaaaa bbcccccc
                // they make the LE-encoded 16-bit value: aaaaaabb 00cccccc
                // so the lower byte is formed of aaaaaabb and the higher byte is 00cccccc
                lower = vchAddressType[0] << 2 | vchAddressType[1] >> 6;
                upper = vchAddressType[1] & 0b00111111; //63 0x3F 0b00111111 '?'
                type = (uint16_t)(256 * upper + lower);
            } else {
            	type = (uint16_t)vchAddressType[0];
            }
            vchPublicKey = std::vector<uint8_t>(&vchAddress[prefixLen], &vchAddress[prefixLen + 32]);
            vchCheckSum = std::vector<uint8_t>(&vchAddress[prefixLen + 32], &vchAddress[nLength]);
        } else {
            strError = strprintf("BadLength: Invalid SS58 address size %d", vchAddress.size());
            fValid = false;
        }
    } else {
        strError = strprintf("BadBase58: Failed to decode base58 address");
    }
}

std::string CSS58::AddressHex() const
{
    if (vchAddress.size() > 0)
        return "0x" + CharVectorToHexString(vchAddress);

    return "";
}

uint16_t CSS58::AddressType() const
{
    return type;
}

std::string CSS58::PublicKeyHex() const
{
    if (vchPublicKey.size() > 0)
        return "0x" + CharVectorToHexString(vchPublicKey);

    return "";
}

std::string CSS58::CheckSumHex() const
{
    if (vchCheckSum.size() > 0)
        return "0x" + CharVectorToHexString(vchCheckSum);

    return "";
}
