// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "swap/swapdata.h"

#include "base58.h"
#include "bdap/utils.h"
#include "policy/policy.h"
#include "serialize.h"
#include "streams.h"
#include "uint256.h"
#include "utilmoneystr.h"

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
        // This assumes there is only one swap output per transaction
        int nCurrentIndex = 0;
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
                        return;
                    }
                }
            }
            nCurrentIndex++;
        }
    }
}

std::string CSwapData::ToString() const
{
    return strprintf(
            "CSwapData(\n"
            "    nVersion                   = %d\n"
            "    vSwapData                  = %s\n"
            "    Amount                     = %s\n"
            "    TxId                       = %s\n"
            "    nOut                       = %d\n"
            "    nHeight                    = %d\n"
            ")\n",
            nVersion,
            Address(),
            FormatMoney(Amount),
            TxId.ToString(),
            nOut,
            nHeight
        );
}

std::vector<unsigned char> CSwapData::vchTxId() const
{
    return vchFromString(TxId.ToString());
}