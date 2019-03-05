// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_DHT_OPERATIONS_H
#define DYNAMIC_DHT_OPERATIONS_H

#include "dht/session.h"

/** Submit a get mutable entry to the libtorrent DHT */
bool SubmitGetDHTMutableData(const std::array<char, 32>& public_key, const std::string& entrySalt);
/** Get a mutable entry in the libtorrent DHT */
bool GetDHTMutableData(const std::array<char, 32>& public_key, const std::string& entrySalt, const int64_t& timeout, 
                             std::string& entryValue, int64_t& lastSequence, bool& fWaitForAuthoritative);

/** Set a mutable entry in the libtorrent DHT */
//bool PutDHTMutableData(const std::array<char, 32>& public_key, const std::array<char, 64>& private_key, const std::string& entrySalt, const int64_t& lastSequence
//                        ,char const* dhtValue, std::string& message);

#endif // DYNAMIC_DHT_OPERATIONS_H
