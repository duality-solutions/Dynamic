// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "transactionrecord.h"

#include "base58.h"
#include "bdap/domainentry.h"
#include "bdap/utils.h"
#include "consensus/consensus.h"
#include "fluid/fluid.h"
#include "instantsend.h"
#include "policy/policy.h"
#include "privatesend.h"
#include "timedata.h"
#include "validation.h"
#include "wallet/wallet.h"

#include <stdint.h>

/* Return positive answer if transaction should be shown in list.
 */
bool TransactionRecord::showTransaction(const CWalletTx& wtx)
{
    if (wtx.IsCoinBase()) {
        // Ensures we show generated coins / mined transactions at depth 1
        if (!wtx.IsInMainChain()) {
            return false;
        }
    }
    return true;
}

/*
 * Decompose CWallet transaction to model transaction records.
 */
QList<TransactionRecord> TransactionRecord::decomposeTransaction(const CWallet* wallet, const CWalletTx& wtx)
{
    QList<TransactionRecord> parts;
    int64_t nTime = wtx.GetTxTime();
    CAmount nCredit = wtx.GetCredit(ISMINE_ALL);
    CAmount nDebit = wtx.GetDebit(ISMINE_ALL);
    CAmount nNet = nCredit - nDebit;
    uint256 hash = wtx.GetHash();
    std::map<std::string, std::string> mapValue = wtx.mapValue;

    if (nNet > 0 || wtx.IsCoinBase()) {
        //
        // Credit
        //
        for (unsigned int i = 0; i < wtx.tx->vout.size(); i++) {
            const CTxOut& txout = wtx.tx->vout[i];
            isminetype mine = wallet->IsMine(txout);
            if (mine) {
                TransactionRecord sub(hash, nTime);
                CTxDestination address;
                sub.idx = i; // vout index
                sub.credit = txout.nValue;
                sub.involvesWatchAddress = mine & ISMINE_WATCH_ONLY;
                if (ExtractDestination(txout.scriptPubKey, address) && IsMine(*wallet, address)) {
                    // Received by Dynamic Address
                    sub.type = TransactionRecord::RecvWithAddress;
                    sub.address = CDynamicAddress(address).ToString();
                } else {
                    // Received by IP connection (deprecated features), or a multisignature or other non-simple transaction
                    sub.type = TransactionRecord::RecvFromOther;
                    sub.address = mapValue["from"];
                }
                if (IsTransactionFluid(txout.scriptPubKey)) {
                    // Fluid type
                    sub.type = TransactionRecord::Fluid;
                } else if (wtx.IsCoinBase()) {
                    // Generated
                    sub.type = TransactionRecord::Generated;
                }

                parts.append(sub);
            }
        }
    } else {
        bool fAllFromMeDenom = true;
        int nFromMe = 0;
        bool involvesWatchAddress = false;
        isminetype fAllFromMe = ISMINE_SPENDABLE;
        for (const CTxIn& txin : wtx.tx->vin) {
            if (wallet->IsMine(txin)) {
                fAllFromMeDenom = fAllFromMeDenom && wallet->IsDenominated(txin.prevout);
                nFromMe++;
            }
            isminetype mine = wallet->IsMine(txin);
            if (mine & ISMINE_WATCH_ONLY)
                involvesWatchAddress = true;
            if (fAllFromMe > mine)
                fAllFromMe = mine;
        }

        isminetype fAllToMe = ISMINE_SPENDABLE;
        bool fAllToMeDenom = true;
        int nToMe = 0;
        for (const CTxOut& txout : wtx.tx->vout) {
            if (wallet->IsMine(txout)) {
                fAllToMeDenom = fAllToMeDenom && CPrivateSend::IsDenominatedAmount(txout.nValue);
                nToMe++;
            }
            isminetype mine = wallet->IsMine(txout);
            if (mine & ISMINE_WATCH_ONLY)
                involvesWatchAddress = true;
            if (fAllToMe > mine)
                fAllToMe = mine;
        }

        if (fAllFromMeDenom && fAllToMeDenom && nFromMe * nToMe) {
            parts.append(TransactionRecord(hash, nTime, TransactionRecord::PrivateSendDenominate, "", -nDebit, nCredit));
            parts.last().involvesWatchAddress = false; // maybe pass to TransactionRecord as constructor argument
        } else if (fAllFromMe && fAllToMe) {
            // Payment to self
            // TODO: this section still not accurate but covers most cases,
            // might need some additional work however

            TransactionRecord sub(hash, nTime);
            // Payment to self by default
            sub.type = TransactionRecord::SendToSelf;
            sub.address = "";

            if (mapValue["PS"] == "1") {
                sub.type = TransactionRecord::PrivateSend;
                CTxDestination address;
                if (ExtractDestination(wtx.tx->vout[0].scriptPubKey, address)) {
                    // Sent to Dynamic Address
                    sub.address = CDynamicAddress(address).ToString();
                } else {
                    // Sent to IP, or other non-address transaction like OP_EVAL
                    sub.address = mapValue["to"];
                }
            } else {
                sub.idx = parts.size();
                if (wtx.tx->vin.size() == 1 && wtx.tx->vout.size() == 1 && CPrivateSend::IsCollateralAmount(nDebit) && CPrivateSend::IsCollateralAmount(nCredit) && CPrivateSend::IsCollateralAmount(-nNet)) {
                    sub.type = TransactionRecord::PrivateSendCollateralPayment;
                } else {
                    for (const auto& txout : wtx.tx->vout) {
                        if (txout.nValue == CPrivateSend::GetMaxCollateralAmount()) {
                            sub.type = TransactionRecord::PrivateSendMakeCollaterals;
                            break;
                        }
                        if (CPrivateSend::IsDenominatedAmount(txout.nValue)) {
                            sub.type = TransactionRecord::PrivateSendCreateDenominations;
                            break;
                        }
                        if (IsTransactionFluid(txout.scriptPubKey)) {
                            sub.type = TransactionRecord::Fluid;
                            break;
                        }
                    }
                }
            }
            CAmount nChange = wtx.GetChange();

            sub.debit = -(nDebit - nChange);
            sub.credit = nCredit - nChange;
            parts.append(sub);
            parts.last().involvesWatchAddress = involvesWatchAddress; // maybe pass to TransactionRecord as constructor argument
        } else if (fAllFromMe) {
            //
            // Debit
            //
            CAmount nTxFee = nDebit - wtx.tx->GetValueOut();
            bool fDone = false;
            if (wtx.tx->vin.size() == 1 && wtx.tx->vout.size() == 1 && CPrivateSend::IsCollateralAmount(nDebit) && nCredit == 0 // OP_RETURN
                && CPrivateSend::IsCollateralAmount(-nNet)) {
                TransactionRecord sub(hash, nTime);
                sub.idx = 0;
                sub.type = TransactionRecord::PrivateSendCollateralPayment;
                sub.debit = -nDebit;
                parts.append(sub);
                fDone = true;
            }
            for (unsigned int nOut = 0; nOut < wtx.tx->vout.size() && !fDone; nOut++) {
                const CTxOut& txout = wtx.tx->vout[nOut];
                TransactionRecord sub(hash, nTime);
                sub.idx = parts.size();
                sub.involvesWatchAddress = involvesWatchAddress;

                if (wtx.tx->nVersion == BDAP_TX_VERSION && GetBDAPOpType(txout) > 0) {
                    continue;
                }
                if (wallet->IsMine(txout)) {
                    // Ignore parts sent to self, as this is usually the change
                    // from a transaction sent back to our own address.
                    continue;
                }

                // Do not display stealth OP_RETURN outputs.
                if (txout.IsData() && txout.nValue == 0)
                    continue;

                CTxDestination address;
                if (ExtractDestination(txout.scriptPubKey, address)) {
                    // Sent to Dynamic Address
                    sub.type = TransactionRecord::SendToAddress;
                    sub.address = CDynamicAddress(address).ToString();
                } else {
                    // Sent to IP, or other non-address transaction like OP_EVAL
                    sub.type = TransactionRecord::SendToOther;
                    sub.address = mapValue["to"];
                }
                if (wtx.tx->nVersion == BDAP_TX_VERSION && txout.scriptPubKey.IsUnspendable() && IsBDAPDataOutput(txout)) {
                    int op1, op2;
                    std::vector<std::vector<unsigned char> > vvchBDAPArgs;
                    CScript scriptOp;
                    if (GetBDAPOpScript(wtx.tx, scriptOp, vvchBDAPArgs, op1, op2)) {
                        std::string errorMessage;
                        std::string strOpType = GetBDAPOpTypeString(op1, op2);
                        if (strOpType == "bdap_new_account" || strOpType == "bdap_update_account" || strOpType == "bdap_delete_account") {
                            std::vector<unsigned char> vchData;
                            std::vector<unsigned char> vchHash;
                            CDomainEntry entry;
                            GetBDAPData(txout, vchData, vchHash);
                            entry.UnserializeFromData(vchData, vchHash);
                            if (strOpType == "bdap_new_account" && entry.ObjectTypeString() == "User Entry") {
                                sub.type = TransactionRecord::NewDomainUser;
                            } else if (strOpType == "bdap_update_account" && entry.ObjectTypeString() == "User Entry") {
                                sub.type = TransactionRecord::UpdateDomainUser;
                            } else if (strOpType == "bdap_delete_account" && entry.ObjectTypeString() == "User Entry") {
                                sub.type = TransactionRecord::DeleteDomainUser;
                            } else if (strOpType == "bdap_revoke_account" && entry.ObjectTypeString() == "User Entry") {
                                sub.type = TransactionRecord::RevokeDomainUser;
                            } else if (strOpType == "bdap_new_account" && entry.ObjectTypeString() == "Group Entry") {
                                sub.type = TransactionRecord::NewDomainGroup;
                            } else if (strOpType == "bdap_update_account" && entry.ObjectTypeString() == "Group Entry") {
                                sub.type = TransactionRecord::UpdateDomainGroup;
                            } else if (strOpType == "bdap_delete_account" && entry.ObjectTypeString() == "Group Entry") {
                                sub.type = TransactionRecord::DeleteDomainGroup;
                            } else if (strOpType == "bdap_revoke_account" && entry.ObjectTypeString() == "Group Entry") {
                                sub.type = TransactionRecord::RevokeDomainGroup;
                            }
                        }
                        else if (strOpType == "bdap_new_link_request" || strOpType == "bdap_update_link_request" || strOpType == "bdap_delete_link_request") {
                            sub.type = TransactionRecord::LinkRequest;
                        }
                        else if (strOpType == "bdap_new_link_accept" || strOpType == "bdap_update_link_accept" || strOpType == "bdap_delete_link_accept") {
                            sub.type = TransactionRecord::LinkAccept;
                        }
                    }
                }
                if (IsTransactionFluid(txout.scriptPubKey)) {
                    sub.type = TransactionRecord::Fluid;
                } else if (mapValue["PS"] == "1") {
                    sub.type = TransactionRecord::PrivateSend;
                }

                CAmount nValue = txout.nValue;
                /* Add fee to first output */
                if (nTxFee > 0) {
                    nValue += nTxFee;
                    nTxFee = 0;
                }
                sub.debit = -nValue;

                parts.append(sub);
            }
        } else {
            //
            // Mixed debit transaction, can't break down payees
            //
            parts.append(TransactionRecord(hash, nTime, TransactionRecord::Other, "", nNet, 0));
            parts.last().involvesWatchAddress = involvesWatchAddress;
        }
    }

    return parts;
}

void TransactionRecord::updateStatus(const CWalletTx& wtx)
{
    AssertLockHeld(cs_main);
    // Determine transaction status

    // Find the block the tx is in
    CBlockIndex* pindex = NULL;
    BlockMap::iterator mi = mapBlockIndex.find(wtx.hashBlock);
    if (mi != mapBlockIndex.end())
        pindex = (*mi).second;

    // Sort order, unrecorded transactions sort to the top
    status.sortKey = strprintf("%010d-%01d-%010u-%03d",
        (pindex ? pindex->nHeight : std::numeric_limits<int>::max()),
        (wtx.IsCoinBase() ? 1 : 0),
        wtx.nTimeReceived,
        idx);
    status.countsForBalance = wtx.IsTrusted() && !(wtx.GetBlocksToMaturity() > 0);
    status.depth = wtx.GetDepthInMainChain();
    status.cur_num_blocks = chainActive.Height();
    status.cur_num_is_locks = nCompleteTXLocks;

    if (!CheckFinalTx(wtx)) {
        if (wtx.tx->nLockTime < LOCKTIME_THRESHOLD) {
            status.status = TransactionStatus::OpenUntilBlock;
            status.open_for = wtx.tx->nLockTime - chainActive.Height();
        } else {
            status.status = TransactionStatus::OpenUntilDate;
            status.open_for = wtx.tx->nLockTime;
        }
    }
    // For generated transactions, determine maturity
    else if (type == TransactionRecord::Generated) {
        if (wtx.GetBlocksToMaturity() > 0) {
            status.status = TransactionStatus::Immature;

            if (wtx.IsInMainChain()) {
                status.matures_in = wtx.GetBlocksToMaturity();

                // Check if the block was requested by anyone
                if (GetAdjustedTime() - wtx.nTimeReceived > 2 * 60 && wtx.GetRequestCount() == 0)
                    status.status = TransactionStatus::MaturesWarning;
            } else {
                status.status = TransactionStatus::NotAccepted;
            }
        } else {
            status.status = TransactionStatus::Confirmed;
        }
    } else {
        status.lockedByInstantSend = wtx.IsLockedByInstantSend();
        if (status.depth < 0) {
            status.status = TransactionStatus::Conflicted;
        } else if (GetAdjustedTime() - wtx.nTimeReceived > 2 * 60 && wtx.GetRequestCount() == 0) {
            status.status = TransactionStatus::Offline;
        } else if (status.depth == 0) {
            status.status = TransactionStatus::Unconfirmed;
            if (wtx.isAbandoned())
                status.status = TransactionStatus::Abandoned;
        } else if (status.depth < RecommendedNumConfirmations) {
            status.status = TransactionStatus::Confirming;
        } else {
            status.status = TransactionStatus::Confirmed;
        }
    }
}

bool TransactionRecord::statusUpdateNeeded()
{
    AssertLockHeld(cs_main);
    return status.cur_num_blocks != chainActive.Height() || status.cur_num_is_locks != nCompleteTXLocks;
}

QString TransactionRecord::getTxID() const
{
    return QString::fromStdString(hash.ToString());
}

int TransactionRecord::getOutputIndex() const
{
    return idx;
}