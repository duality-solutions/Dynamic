// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_PSNOTIFICATIONINTERFACE_H
#define DYNAMIC_PSNOTIFICATIONINTERFACE_H

#include "validationinterface.h"

class CPSNotificationInterface : public CValidationInterface
{
public:
    // virtual CPSNotificationInterface();
    CPSNotificationInterface();
    virtual ~CPSNotificationInterface();

protected:
    // CValidationInterface
    void UpdatedBlockTip(const CBlockIndex *pindex);
    void SyncTransaction(const CTransaction &tx, const CBlock *pblock);

private:
};

#endif // DYNAMIC_PSNOTIFICATIONINTERFACE_H
