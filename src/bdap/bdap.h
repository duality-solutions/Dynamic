// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_H
#define DYNAMIC_BDAP_H

#include <vector>
#include <string>

namespace BDAP {
    enum ObjectType {
        DEFAULT_TYPE = 0,
        USER_ACCOUNT = 1,
        GROUP = 2,
        DEVICE_ACCOUNT = 3,
        DOMAIN_ACCOUNT = 4,
        ORGANIZATIONAL_UNIT = 5,
        CERTIFICATE = 6,
        AUDIT = 7,
        CHANNEL = 8,
        CHECKPOINT = 9,
        BINDING_LINK = 10,
        IDENTITY = 11,
        IDENTITY_VERIFICATION = 12,
        SMART_CONTRACT = 13
    };
}

typedef std::vector<unsigned char> CharString;
typedef std::vector<CharString> vchCharString;
typedef std::pair<uint32_t, CharString> CheckPoint;
typedef std::vector<CheckPoint> vCheckPoints; // << height, block hash >>

static constexpr unsigned int ACTIVATE_BDAP_HEIGHT        = 10; // TODO: Change for mainnet or spork activate (???)
static constexpr unsigned int MAX_OBJECT_NAME_LENGTH      = 63;
static constexpr unsigned int MAX_OBJECT_FULL_PATH_LENGTH = (MAX_OBJECT_NAME_LENGTH * 3) + 2; // domain + ou + object name + 2 dot chars
static constexpr unsigned int MAX_COMMON_NAME_LENGTH      = 95;
static constexpr unsigned int MAX_ORG_NAME_LENGTH         = 95;
static constexpr unsigned int MAX_WALLET_ADDRESS_LENGTH   = 102; // Stealth addresses are 102 chars in length.  Regular addresses are 34 chars.
static constexpr unsigned int MAX_RESOURCE_POINTER_LENGTH = 127;
static constexpr unsigned int MAX_KEY_LENGTH              = 156;
static constexpr unsigned int MAX_DESCRIPTION_LENGTH      = 256;
static constexpr unsigned int MAX_CERTIFICATE_LENGTH      = 512;
static constexpr unsigned int MAX_CERTIFICATE_NAME        = 63;
static constexpr unsigned int MAX_SIGNATURE_LENGTH        = 65; // https://bitcoin.stackexchange.com/questions/12554/why-the-signature-is-always-65-13232-bytes-long
static constexpr unsigned int MAX_PRIVATE_DATA_LENGTH     = 512; // Pay per byte for hosting on chain
static constexpr unsigned int MAX_NUMBER_CHECKPOINTS      = 25; // Pay per byte for hosting on chain
static constexpr unsigned int MAX_CHECKPOINT_HASH_LENGTH  = 64;
static constexpr unsigned int SECONDS_PER_DAY             = 86400; // Number of seconds per day.
static const std::string DEFAULT_PUBLIC_DOMAIN            = "bdap.io";
static const std::string DEFAULT_PUBLIC_OU                = "public";
static const std::string DEFAULT_PUBLIC_USER_OU           = "users";
static const std::string DEFAULT_PUBLIC_GROUP_OU          = "groups";
static const std::string DEFAULT_ADMIN_OU                 = "admin";
static const std::string DEFAULT_ORGANIZATION_NAME        = "Duality Blockchain Solutions";
static const std::string DEFAULT_OID_PREFIX               = "0.0.0"; //TODO (bdap): get real OID prefix

inline const CharString ConvertConstantToCharString (const std::string strConvert)
{
    CharString vchConvert(strConvert.begin(), strConvert.end());
    return vchConvert;
};

static const CharString vchDefaultDomainName = ConvertConstantToCharString(DEFAULT_PUBLIC_DOMAIN);
static const CharString vchDefaultPublicOU = ConvertConstantToCharString(DEFAULT_PUBLIC_OU);
static const CharString vchDefaultUserOU = ConvertConstantToCharString(DEFAULT_PUBLIC_USER_OU);
static const CharString vchDefaultGroupOU = ConvertConstantToCharString(DEFAULT_PUBLIC_GROUP_OU);
static const CharString vchDefaultAdminOU = ConvertConstantToCharString(DEFAULT_ADMIN_OU);
static const CharString vchDefaultOrganizationName = ConvertConstantToCharString(DEFAULT_ORGANIZATION_NAME);
static const CharString vchDefaultOIDPrefix = ConvertConstantToCharString(DEFAULT_OID_PREFIX);

#endif // DYNAMIC_BDAP_H