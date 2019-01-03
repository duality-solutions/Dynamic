// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_ACTIVEDYNODE_H
#define DYNAMIC_ACTIVEDYNODE_H

#include "chainparams.h"
#include "key.h"
#include "net.h"
#include "primitives/transaction.h"

#ifdef ENABLE_WALLET
#include "wallet/wallet.h"
#endif //ENABLE_WALLET

class CActiveDynode;

static const int ACTIVE_DYNODE_INITIAL = 0; // initial state
static const int ACTIVE_DYNODE_SYNC_IN_PROCESS = 1;
static const int ACTIVE_DYNODE_INPUT_TOO_NEW = 2;
static const int ACTIVE_DYNODE_NOT_CAPABLE = 3;
static const int ACTIVE_DYNODE_STARTED = 4;

extern CActiveDynode activeDynode;

// Responsible for activating the Dynode and pinging the network
class CActiveDynode
{
public:
    enum dynode_type_enum_t {
        DYNODE_UNKNOWN = 0,
        DYNODE_REMOTE = 1
    };

private:
    // critical section to protect the inner data structures
    mutable CCriticalSection cs;

    dynode_type_enum_t eType;

    bool fPingerEnabled;

    /// Ping Dynode
    bool SendDynodePing(CConnman& connman);

    //  sentinel ping data
    int64_t nSentinelPingTime;
    uint32_t nSentinelVersion;

public:
    // Keys for the active Dynode
    CPubKey pubKeyDynode;
    CKey keyDynode;

    // Initialized while registering Dynode
    COutPoint outpoint;
    CService service;

    int nState; // should be one of ACTIVE_DYNODE_XXXX
    std::string strNotCapableReason;

    CActiveDynode()
        : eType(DYNODE_UNKNOWN),
          fPingerEnabled(false),
          pubKeyDynode(),
          keyDynode(),
          outpoint(),
          service(),
          nState(ACTIVE_DYNODE_INITIAL)
    {
    }

    /// Manage state of active Dynode
    void ManageState(CConnman& connman);

    std::string GetStateString() const;
    std::string GetStatus() const;
    std::string GetTypeString() const;

    bool UpdateSentinelPing(int version);

    void DoMaintenance(CConnman &connman);

private:
    void ManageStateInitial(CConnman& connman);
    void ManageStateRemote();
};

#endif // DYNAMIC_ACTIVEDYNODE_H
