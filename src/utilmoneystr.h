// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2009-2017 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/**
 * Money parsing/formatting utilities.
 */
#ifndef DYNAMIC_UTILMONEYSTR_H
#define DYNAMIC_UTILMONEYSTR_H

#include "amount.h"

#include <stdint.h>
#include <string>

std::string FormatMoney(const CAmount& n);
bool ParseMoney(const std::string& str, CAmount& nRet);
bool ParseMoney(const char* pszIn, CAmount& nRet);

#endif // DYNAMIC_UTILMONEYSTR_H
