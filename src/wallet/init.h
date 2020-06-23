// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Core developers
// Copyright (c) 2017-2019 The Raven Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_WALLET_INIT_H
#define DYNAMIC_WALLET_INIT_H

#include <string>

class CRPCTable;
class CScheduler;

//! Return the wallets help message.
std::string GetWalletHelpString(bool showDebug);

//! Wallets parameter interaction
bool WalletParameterInteraction();

//! Register wallet RPCs.
void RegisterWalletRPC(CRPCTable &tableRPC);

#endif // DYNAMIC_WALLET_INIT_H
