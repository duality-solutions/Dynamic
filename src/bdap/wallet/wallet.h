// Copyright (c) 2016-2019 Duality Blockchain Solutions
// Copyright (c) 2009-2019 The Bitcoin Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_BDAP_WALLET_H
#define BITCOIN_BDAP_WALLET_H

#include "serialize.h"

#include <vector>

/** An Ed key pool entry */
class CEdKeyPool
{
public:
    int64_t nTime;
    std::vector<unsigned char> edPubKey;
    bool fInternal; // for change outputs

    CEdKeyPool();
    CEdKeyPool(const std::vector<unsigned char>& edPubKeyIn, bool fInternalIn);

    ADD_SERIALIZE_METHODS;

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action)
    {
        int nVersion = s.GetVersion();
        if (!(s.GetType() & SER_GETHASH))
        READWRITE(nVersion);
        READWRITE(nTime);
        READWRITE(edPubKey);
        if (ser_action.ForRead()) {
            try {
                READWRITE(fInternal);
            } catch (std::ios_base::failure&) {
                /* flag as external address if we can't read the internal boolean
                   (this will be the case for any wallet before the HD chain split version) */
                fInternal = false;
            }
        } else {
            READWRITE(fInternal);
        }
    }
};

#endif // BITCOIN_BDAP_WALLET_H