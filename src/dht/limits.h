// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_LIMITS_H
#define DYNAMIC_DHT_LIMITS_H

/**!
These limit classes define allowed salts (records) that the DHT will accept for storage.
The first version contains a list of acceptable salts but in future versions, this list will
be dynamic.  Application developers that want to store data in the DHT can purchase a certificate
that allows their custom op code.
*/

#include <string>
#include <vector>

class CAllowDataCode {
public:
    std::string strSalt;
    uint32_t nMaximumSlots;
    unsigned int nStartHeight;
    uint64_t nExpireTime;

    CAllowDataCode(const std::string& salt, const uint32_t maxslots, const unsigned int& start, const uint64_t expire) :
        strSalt(salt), nMaximumSlots(maxslots), nStartHeight(start), nExpireTime(expire) {}

};

bool CheckSalt(const std::string& strSalt, const unsigned int nHeight, std::string& strErrorMessage);
bool CheckPubKey(const std::vector<unsigned char>& vchPubKey);

#endif // DYNAMIC_DHT_LIMITS_H
