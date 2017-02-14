// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2013-2017 Emercoin Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_HOOKS_H
#define DARKSILK_HOOKS_H

class CScript;
class CTransaction;
class CBlockIndex;
class CTxOut;
struct nameTempProxy;

#include <map>
#include <vector>
#include <string>
using namespace std;

class CHooks
{
public:
    virtual bool IsNameFeeEnough(const CTransaction& tx, const CAmount& txFee) = 0;
    virtual bool CheckInputs(const CTransaction& tx, const CBlockIndex* pindexBlock, std::vector<nameTempProxy> &vName, const CDiskTxPos& pos, const CAmount& txFee) = 0;
    virtual bool DisconnectInputs(const CTransaction& tx) = 0;
    virtual bool ConnectBlock(CBlockIndex* pindex, const std::vector<nameTempProxy> &vName) = 0;
    virtual bool ExtractAddress(const CScript& script, std::string& address) = 0;
    virtual void AddToPendingNames(const CTransaction& tx) = 0;
    virtual bool IsNameScript(CScript scr) = 0;
    virtual bool getNameValue(const string& sName, string& sValue) = 0;
    virtual bool DumpToTextFile() = 0;
};

extern CHooks* InitHook();
extern std::string GetDefaultDataDirSuffix();
extern CHooks* hooks;

#endif //DARKSILK_HOOKS_H
