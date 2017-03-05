// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash CoreDevelopers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_ACTIVEDYNODE_H
#define DYNAMIC_ACTIVEDYNODE_H

#include "key.h"
#include "net.h"
#include "wallet/wallet.h"

class CActiveDynode;

static const int ACTIVE_DYNODE_INITIAL          = 0; // initial state
static const int ACTIVE_DYNODE_SYNC_IN_PROCESS  = 1;
static const int ACTIVE_DYNODE_INPUT_TOO_NEW    = 2;
static const int ACTIVE_DYNODE_NOT_CAPABLE      = 3;
static const int ACTIVE_DYNODE_STARTED          = 4;

extern CActiveDynode activeDynode;

// Responsible for activating the Dynode and pinging the network
class CActiveDynode
{
public:
    enum dynode_type_enum_t {
        DYNODE_UNKNOWN = 0,
        DYNODE_REMOTE  = 1,
        DYNODE_LOCAL   = 2
    };

private:
    // critical section to protect the inner data structures
    mutable CCriticalSection cs;

    dynode_type_enum_t eType;

    bool fPingerEnabled;

    /// Ping Dynode
    bool SendDynodePing();

public:
    // Keys for the active Dynode
    CPubKey pubKeyDynode;
    CKey keyDynode;

    // Initialized while registering Dynode
    CTxIn vin;
    CService service;

    int nState; // should be one of ACTIVE_DYNODE_XXXX
    std::string strNotCapableReason;

    CActiveDynode()
        : eType(DYNODE_UNKNOWN),
          fPingerEnabled(false),
          pubKeyDynode(),
          keyDynode(),
          vin(),
          service(),
          nState(ACTIVE_DYNODE_INITIAL)
    {}

    /// Manage state of active Dynode
    void ManageState();

    std::string GetStateString() const;
    std::string GetStatus() const;
    std::string GetTypeString() const;

private:
    void ManageStateInitial();
    void ManageStateRemote();
    void ManageStateLocal();
};

#endif // DYNAMIC_ACTIVEDYNODE_H
