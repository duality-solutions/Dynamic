// Copyright (c) 2018 Duality Blockchain Solutions Developers
// TODO: Add License

#ifndef DYNAMIC_DHT_OPERATIONS_H
#define DYNAMIC_DHT_OPERATIONS_H

#include "dht/session.h"

/** Get a mutable entry in the libtorrent DHT */
bool GetDHTMutableData(const std::array<char, 32>& public_key, const std::string& entrySalt, std::string& entryValue, int64_t& lastSequence, bool fWaitForAuthoritative);

/** Set a mutable entry in the libtorrent DHT */
bool PutDHTMutableData(const std::array<char, 32>& public_key, const std::array<char, 64>& private_key, const std::string& entrySalt, const int64_t& lastSequence
                        ,char const* dhtValue, std::string& message);

#endif // DYNAMIC_DHT_OPERATIONS_H
