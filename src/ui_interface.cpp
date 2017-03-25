// Copyright (c) 2010-2016 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "ui_interface.h"
#include "util.h"

CClientUIInterface uiInterface;

std::string AmountHighWarn(const std::string& optname)
{
    return strprintf(_("%s is set very high!"), optname);
}
