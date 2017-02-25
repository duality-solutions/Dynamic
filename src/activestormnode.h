// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash CoreDevelopers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_ACTIVESTORMNODE_H
#define DARKSILK_ACTIVESTORMNODE_H

#include "key.h"
#include "net.h"
#include "wallet/wallet.h"

class CActiveStormnode;

static const int ACTIVE_STORMNODE_INITIAL          = 0; // initial state
static const int ACTIVE_STORMNODE_SYNC_IN_PROCESS  = 1;
static const int ACTIVE_STORMNODE_INPUT_TOO_NEW    = 2;
static const int ACTIVE_STORMNODE_NOT_CAPABLE      = 3;
static const int ACTIVE_STORMNODE_STARTED          = 4;

extern CActiveStormnode activeStormnode;

// Responsible for activating the Stormnode and pinging the network
class CActiveStormnode
{
public:
    enum stormnode_type_enum_t {
        STORMNODE_UNKNOWN = 0,
        STORMNODE_REMOTE  = 1,
        STORMNODE_LOCAL   = 2
    };

private:
    // critical section to protect the inner data structures
    mutable CCriticalSection cs;

    stormnode_type_enum_t eType;

    bool fPingerEnabled;

    /// Ping Stormnode
    bool SendStormnodePing();

public:
    // Keys for the active Stormnode
    CPubKey pubKeyStormnode;
    CKey keyStormnode;

    // Initialized while registering Stormnode
    CTxIn vin;
    CService service;

    int nState; // should be one of ACTIVE_STORMNODE_XXXX
    std::string strNotCapableReason;

    CActiveStormnode()
        : eType(STORMNODE_UNKNOWN),
          fPingerEnabled(false),
          pubKeyStormnode(),
          keyStormnode(),
          vin(),
          service(),
          nState(ACTIVE_STORMNODE_INITIAL)
    {}

    /// Manage state of active Stormnode
    void ManageState();

    std::string GetStateString() const;
    std::string GetStatus() const;
    std::string GetTypeString() const;

private:
    void ManageStateInitial();
    void ManageStateRemote();
    void ManageStateLocal();
};

#endif // DARKSILK_ACTIVESTORMNODE_H
