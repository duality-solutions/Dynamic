// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "swap/swapdata.h"

#include "base58.h"
#include "bdap/utils.h"
#include "chainparams.h"
#include "coins.h"
#include "core_io.h"
#include "policy/policy.h"
#include "serialize.h"
#include "streams.h"
#include "uint256.h"
#include "utilmoneystr.h"
#include "validation.h"

#include <univalue.h>

std::string CSwapData::Address() const
{
    return EncodeBase58(vSwapData);
}

void CSwapData::Serialize(std::vector<unsigned char>& vchData) 
{
    CDataStream dsSwapData(SER_NETWORK, PROTOCOL_VERSION);
    dsSwapData << *this;
    vchData = std::vector<unsigned char>(dsSwapData.begin(), dsSwapData.end());
}

bool CSwapData::UnserializeFromData(const std::vector<unsigned char>& vchData, const std::vector<unsigned char>& vchHash) 
{
    try {
        CDataStream dsSwapData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsSwapData >> *this;

        std::vector<unsigned char> vchSwapData;
        Serialize(vchSwapData);
        const uint256& calculatedHash = Hash(vchSwapData.begin(), vchSwapData.end());
        const std::vector<unsigned char>& vchRandSwapData = vchFromValue(calculatedHash.GetHex());
        if(vchRandSwapData != vchHash) {
            SetNull();
            return false;
        }
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

bool CSwapData::UnserializeFromData(const std::vector<unsigned char>& vchData) 
{
    try {
        CDataStream dsSwapData(vchData, SER_NETWORK, PROTOCOL_VERSION);
        dsSwapData >> *this;

        std::vector<unsigned char> vchSwapData;
        Serialize(vchSwapData);
    } catch (std::exception& e) {
        SetNull();
        return false;
    }
    return true;
}

CSwapData::CSwapData(const CTransactionRef& tx, const int& height)
{
    SetNull();

    if (tx->nVersion == SWAP_TX_VERSION)
    {
        CAmount nValueIn = 0;
        CCoinsViewCache view(pcoinsTip);
        for (const CTxIn& txin : tx->vin) {
            const Coin& coin = view.AccessCoin(txin.prevout);
            nValueIn += coin.out.nValue;
        }
        // This assumes there is only one swap output per transaction
        int nCurrentIndex = 0;
        CAmount nValueOut = 0;
        for (const auto& txout : tx->vout) {
            if (txout.IsData()) {
                std::vector<unsigned char> vchData;
                if (txout.GetData(vchData)) {
                    if (vchData.size() >= 32) {
                        vSwapData = vchData;
                        Amount = txout.nValue;
                        TxId = tx->GetHash();
                        nOut = nCurrentIndex;
                        nHeight = height;
                    }
                }
            }
            nValueOut += txout.nValue;
            nCurrentIndex++;
        }
        if (nValueIn > nValueOut) {
            Fee = (nValueIn - nValueOut);
        } else {
            Fee = 0;
            LogPrintf("%s Error getting Fee %s, nValueIn %s, nValueOut %s\n", __func__, FormatMoney((nValueIn - nValueOut)), FormatMoney(nValueIn), FormatMoney(nValueOut));
        }
    }
}

CAmount CSwapData::GetFee() const
{
    if (Fee <= 0) {
        CTransactionRef tx;
        uint256 hashBlock;
        if (!GetTransaction(TxId, tx, Params().GetConsensus(), hashBlock, true)) {
            LogPrintf("%s Unable to get %s transaction \n", __func__, TxId.GetHex());
            return 0;
        }
        CAmount nValueIn = 0;
        CCoinsViewCache view(pcoinsTip);
        size_t index = 0;
        for (const CTxIn& txin : tx->vin) {
            const Coin& coin = view.AccessCoin(txin.prevout);
            if (coin.out.nValue <= 0) {
                LogPrintf("%s Unable to get %s transaction value\n", __func__, TxId.GetHex(), ScriptToAsmStr(coin.out.scriptPubKey));
            }
            nValueIn += coin.out.nValue;
            index ++;
        }
        CAmount nValueOut = 0;
        for (const auto& txout : tx->vout) {
            nValueOut += txout.nValue;
        }

        if (nValueIn > nValueOut) {
            return (nValueIn - nValueOut);
        } else {
            LogPrintf("%s - Txid %s nValueIn (%s) is less than nValueOut (%s)\n", __func__, TxId.GetHex(), FormatMoney(nValueIn), FormatMoney(nValueOut));
            return 0;
        }
    } else {
        return Fee;
    }
}

std::string CSwapData::ToString() const
{
    return strprintf(
            "CSwapData(\n"
            "    nVersion                   = %d\n"
            "    vSwapData                  = %s\n"
            "    Amount                     = %s\n"
            "    Fee                        = %s\n"
            "    TxId                       = %s\n"
            "    nOut                       = %d\n"
            "    nHeight                    = %d\n"
            ")\n",
            nVersion,
            Address(),
            FormatMoney(Amount),
            FormatMoney(GetFee()),
            TxId.ToString(),
            nOut,
            nHeight
        );
}

std::vector<unsigned char> CSwapData::vchTxId() const
{
    return vchFromString(TxId.ToString());
}