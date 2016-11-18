// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_SSNOTIFICATIONINTERFACE_H
#define DARKSILK_SSNOTIFICATIONINTERFACE_H

#include "validationinterface.h"

class CSSNotificationInterface : public CValidationInterface
{
public:
    // virtual CSSNotificationInterface();
    CSSNotificationInterface();
    virtual ~CSSNotificationInterface();

protected:
    // CValidationInterface
    void UpdatedBlockTip(const CBlockIndex *pindex);

private:
};

#endif // DARKSILK_SSNOTIFICATIONINTERFACE_H
