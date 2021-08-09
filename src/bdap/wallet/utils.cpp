// Copyright (c) 2019-2021 Duality Blockchain Solutions
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/utils.h"

#include "base58.h"
#include "chainparams.h"
#include "coins.h"
#include "core_io.h"
#include "policy/policy.h"
#include "serialize.h"
#include "uint256.h"
#include "validation.h"
#include "wallet/wallet.h"
#include "utils.h"
#include "utiltime.h"
#include "script/script.h"

#include <univalue.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

#include <ctime>

void CreateRecipient(const CScript& scriptPubKey, CRecipient& recipient)
{
    CRecipient recp = {scriptPubKey, recipient.nAmount, false};
    recipient = recp;
    CTxOut txout(recipient.nAmount, scriptPubKey);
    size_t nSize = GetSerializeSize(txout, SER_DISK, 0) + 148u;
    recipient.nAmount = 3 * minRelayTxFee.GetFee(nSize);
}

void CreateFeeRecipient(CScript& scriptPubKey, const std::vector<unsigned char>& data, CRecipient& recipient)
{
    // add hash to data output (must match hash in inputs check with the tx scriptpubkey hash)
    uint256 hash = Hash(data.begin(), data.end());
    std::vector<unsigned char> vchHashRand = vchFromValue(hash.GetHex());
    scriptPubKey << vchHashRand;
    CRecipient recp = {scriptPubKey, 0, false};
    recipient = recp;
}

bool GetScriptOpTypeValue(const std::vector<CRecipient>& vecSend, CScript& bdapOpScript, std::string& strOpType, std::vector<unsigned char>& vchValue)
{
    LogPrint("bdap", "%s -- vecSend size = %u \n", __func__, vecSend.size());
    for (const CRecipient& rec : vecSend) {
        CScript script = rec.scriptPubKey;
        if (!script.IsUnspendable()) {
            if (ExtractOpTypeValue(script, strOpType, vchValue)) {
                bdapOpScript = script;
                break;
            }
        }
    }
    if (strOpType.size() > 0) {
        return true;
    }
    return false;
}
