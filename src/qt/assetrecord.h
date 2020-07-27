// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2019 The Raven Core developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_ASSETRECORD_H
#define DYNAMIC_QT_ASSETRECORD_H

#include "amount.h"
#include "tinyformat.h"

#include <math.h>

/** UI model for unspent assets.
 */
class AssetRecord
{
public:

    AssetRecord():
            name(""), quantity(0), units(0), fIsAdministrator(false)
    {
    }

    AssetRecord(const std::string _name, const CAmount& _quantity, const int _units, const bool _fIsAdministrator):
            name(_name), quantity(_quantity), units(_units), fIsAdministrator(_fIsAdministrator)
    {
    }

    std::string formattedQuantity() const {
        bool sign = quantity < 0;
        int64_t n_abs = (sign ? -quantity : quantity);
        int64_t quotient = n_abs / COIN;
        int64_t remainder = n_abs % COIN;
        remainder = remainder / pow(10, 8 - units);

        if (remainder == 0) {
            return strprintf("%s%d", sign ? "-" : "", quotient);
        }
        else {
            return strprintf("%s%d.%0" + std::to_string(units) + "d", sign ? "-" : "", quotient, remainder);
        }
    }

    /** @name Immutable attributes
      @{*/
    std::string name;
    CAmount quantity;
    int units;
    bool fIsAdministrator;
    /**@}*/

};

#endif // DYNAMIC_QT_ASSETRECORD_H
