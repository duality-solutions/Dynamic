// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Core developers
// Copyright (c) 2017-2020 The Raven Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "wallet/init.h"

#include "keepass.h"
#include "net.h"
#include "rpc/wallet.h"
#include "util.h"
#include "utilmoneystr.h"
#include "validation.h"
#include "wallet/wallet.h"

std::string GetWalletHelpString(bool showDebug)
{
    std::string strUsage = HelpMessageGroup(_("Wallet options:"));
    strUsage += HelpMessageOpt("-disablewallet", _("Do not load the wallet and disable wallet RPC calls"));
    strUsage += HelpMessageOpt("-keypool=<n>", strprintf(_("Set key pool size to <n> (default: %u)"), DEFAULT_KEYPOOL_SIZE));
    strUsage += HelpMessageOpt("-fallbackfee=<amt>", strprintf(_("A fee rate (in %s/kB) that will be used when fee estimation has insufficient data (default: %s)"),
                                                         CURRENCY_UNIT, FormatMoney(DEFAULT_FALLBACK_FEE)));
    strUsage += HelpMessageOpt("-mintxfee=<amt>", strprintf(_("Fees (in %s/kB) smaller than this are considered zero fee for transaction creation (default: %s)"),
                                                      CURRENCY_UNIT, FormatMoney(DEFAULT_TRANSACTION_MINFEE)));
    strUsage += HelpMessageOpt("-paytxfee=<amt>", strprintf(_("Fee (in %s/kB) to add to transactions you send (default: %s)"),
                                                      CURRENCY_UNIT, FormatMoney(payTxFee.GetFeePerK())));
    strUsage += HelpMessageOpt("-rescan", _("Rescan the block chain for missing wallet transactions on startup"));
    strUsage += HelpMessageOpt("-salvagewallet", _("Attempt to recover private keys from a corrupt wallet on startup"));
    if (showDebug)
        strUsage += HelpMessageOpt("-sendfreetransactions", strprintf(_("Send transactions as zero-fee transactions if possible (default: %u)"), DEFAULT_SEND_FREE_TRANSACTIONS));
    strUsage += HelpMessageOpt("-spendzeroconfchange", strprintf(_("Spend unconfirmed change when sending transactions (default: %u)"), DEFAULT_SPEND_ZEROCONF_CHANGE));
    strUsage += HelpMessageOpt("-txconfirmtarget=<n>", strprintf(_("If paytxfee is not set, include enough fee so transactions begin confirmation on average within n blocks (default: %u)"), DEFAULT_TX_CONFIRM_TARGET));
    strUsage += HelpMessageOpt("-usehd", _("Use hierarchical deterministic key generation (HD) after BIP39/BIP44. Only has effect during wallet creation/first start") + " " + strprintf(_("(default: %u)"), DEFAULT_USE_HD_WALLET));
    strUsage += HelpMessageOpt("-mnemonic", _("User defined mnemonic for HD wallet (bip39). Only has effect during wallet creation/first start (default: randomly generated)"));
    strUsage += HelpMessageOpt("-mnemonicpassphrase", _("User defined mnemonic passphrase for HD wallet (BIP39). Only has effect during wallet creation/first start (default: empty string)"));
    strUsage += HelpMessageOpt("-hdseed", _("User defined seed for HD wallet (should be in hex). Only has effect during wallet creation/first start (default: randomly generated)"));
    strUsage += HelpMessageOpt("-upgradewallet", _("Upgrade wallet to latest format on startup"));
    strUsage += HelpMessageOpt("-wallet=<file>", _("Specify wallet file (within data directory)") + " " + strprintf(_("(default: %s)"), DEFAULT_WALLET_DAT));
    strUsage += HelpMessageOpt("-walletbroadcast", _("Make the wallet broadcast transactions") + " " + strprintf(_("(default: %u)"), DEFAULT_WALLETBROADCAST));
    strUsage += HelpMessageOpt("-walletnotify=<cmd>", _("Execute command when a wallet transaction changes (%s in cmd is replaced by TxID)"));
    strUsage += HelpMessageOpt("-zapwallettxes=<mode>", _("Delete all wallet transactions and only recover those parts of the blockchain through -rescan on startup") +
                                                            " " + _("(1 = keep tx meta data e.g. account owner and payment request information, 2 = drop tx meta data)"));
    strUsage += HelpMessageOpt("-createwalletbackups=<n>", strprintf(_("Number of automatic wallet backups (default: %u)"), nWalletBackups));
    strUsage += HelpMessageOpt("-walletbackupsdir=<dir>", _("Specify full path to directory for automatic wallet backups (must exist)"));
    strUsage += HelpMessageOpt("-keepass", strprintf(_("Use KeePass 2 integration using KeePassHttp plugin (default: %u)"), 0));
    strUsage += HelpMessageOpt("-keepassport=<port>", strprintf(_("Connect to KeePassHttp on port <port> (default: %u)"), DEFAULT_KEEPASS_HTTP_PORT));
    strUsage += HelpMessageOpt("-keepasskey=<key>", _("KeePassHttp key for AES encrypted communication with KeePass"));
    strUsage += HelpMessageOpt("-keepassid=<name>", _("KeePassHttp id for the established association"));
    strUsage += HelpMessageOpt("-keepassname=<name>", _("Name to construct url for KeePass entry that stores the wallet passphrase"));

    if (showDebug) {
        strUsage += HelpMessageGroup(_("Wallet debugging/testing options:"));

        strUsage += HelpMessageOpt("-dblogsize=<n>", strprintf("Flush wallet database activity from memory to disk log every <n> megabytes (default: %u)", DEFAULT_WALLET_DBLOGSIZE));
        strUsage += HelpMessageOpt("-flushwallet", strprintf("Run a thread to flush wallet periodically (default: %u)", DEFAULT_FLUSHWALLET));
        strUsage += HelpMessageOpt("-privdb", strprintf("Sets the DB_PRIVATE flag in the wallet db environment (default: %u)", DEFAULT_WALLET_PRIVDB));
        strUsage += HelpMessageOpt("-walletrejectlongchains", strprintf(_("Wallet will not create transactions that violate mempool chain limits (default: %u)"), DEFAULT_WALLET_REJECT_LONG_CHAINS));
    }

    return strUsage;
}

bool WalletParameterInteraction()
{
    if (gArgs.GetBoolArg("-disablewallet", DEFAULT_DISABLE_WALLET))
        return true;

    if (gArgs.GetBoolArg("-blocksonly", DEFAULT_BLOCKSONLY) && gArgs.SoftSetBoolArg("-walletbroadcast", false)) {
        LogPrintf("%s: parameter interaction: -blocksonly=1 -> setting -walletbroadcast=0\n", __func__);
    }

    if (gArgs.GetBoolArg("-salvagewallet", false) && gArgs.SoftSetBoolArg("-rescan", true)) {
        // Rewrite just private keys: rescan to find transactions
        LogPrintf("%s: parameter interaction: -salvagewallet=1 -> setting -rescan=1\n", __func__);
    }

    // -zapwallettx implies a rescan
    if (gArgs.GetBoolArg("-zapwallettxes", false) && gArgs.SoftSetBoolArg("-rescan", true)) {
        LogPrintf("%s: parameter interaction: -zapwallettxes=<mode> -> setting -rescan=1\n", __func__);
    }

    if (gArgs.GetBoolArg("-sysperms", false))
        return InitError("-sysperms is not allowed in combination with enabled wallet functionality");
    if (gArgs.GetArg("-prune", 0) && gArgs.GetBoolArg("-rescan", false))
        return InitError(_("Rescans are not possible in pruned mode. You will need to use -reindex which will download the whole blockchain again."));

    if (::minRelayTxFee.GetFeePerK() > HIGH_TX_FEE_PER_KB)
        InitWarning(AmountHighWarn("-minrelaytxfee") + " " +
                    _("The wallet will avoid paying less than the minimum relay fee."));

    if (gArgs.IsArgSet("-mintxfee")) {
        CAmount n = 0;
        if (!ParseMoney(gArgs.GetArg("-mintxfee", ""), n) || 0 == n)
            return InitError(AmountErrMsg("mintxfee", gArgs.GetArg("-mintxfee", "")));
        if (n > HIGH_TX_FEE_PER_KB)
            InitWarning(AmountHighWarn("-mintxfee") + " " +
                        _("This is the minimum transaction fee you pay on every transaction."));
        CWallet::minTxFee = CFeeRate(n);
    }
    if (gArgs.IsArgSet("-fallbackfee")) {
        CAmount nFeePerK = 0;
        if (!ParseMoney(gArgs.GetArg("-fallbackfee", ""), nFeePerK))
            return InitError(strprintf(_("Invalid amount for -fallbackfee=<amount>: '%s'"), gArgs.GetArg("-fallbackfee", "")));
        if (nFeePerK > HIGH_TX_FEE_PER_KB)
            InitWarning(AmountHighWarn("-fallbackfee is set very high! This is the transaction fee you may pay when fee estimates are not available."));
        CWallet::fallbackFee = CFeeRate(nFeePerK);
    }
    if (gArgs.IsArgSet("-paytxfee")) {
        CAmount nFeePerK = 0;
        if (!ParseMoney(gArgs.GetArg("-paytxfee", ""), nFeePerK))
            return InitError(AmountErrMsg("paytxfee", gArgs.GetArg("-paytxfee", "")));
        if (nFeePerK > HIGH_TX_FEE_PER_KB)
            InitWarning(AmountHighWarn("-paytxfee is set very high! This is the transaction fee you will pay if you send a transaction."));
        payTxFee = CFeeRate(nFeePerK, 1000);
        if (payTxFee < ::minRelayTxFee) {
            return InitError(strprintf(_("Invalid amount for -paytxfee=<amount>: '%s' (must be at least %s)"),
                gArgs.GetArg("-paytxfee", ""), ::minRelayTxFee.ToString()));
        }
    }
    if (gArgs.IsArgSet("-maxtxfee")) {
        CAmount nMaxFee = 0;
        if (!ParseMoney(gArgs.GetArg("-maxtxfee", ""), nMaxFee))
            return InitError(AmountErrMsg("maxtxfee", gArgs.GetArg("-maxtxfee", "")));
        if (nMaxFee > HIGH_MAX_TX_FEE)
            InitWarning(_("-maxtxfee is set very high! Fees this large could be paid on a single transaction."));
        maxTxFee = nMaxFee;
        if (CFeeRate(maxTxFee, 1000) < ::minRelayTxFee) {
            return InitError(strprintf(_("Invalid amount for -maxtxfee=<amount>: '%s' (must be at least the minrelay fee of %s to prevent stuck transactions)"),
                gArgs.GetArg("-maxtxfee", ""), ::minRelayTxFee.ToString()));
        }
    }
    nTxConfirmTarget = gArgs.GetArg("-txconfirmtarget", DEFAULT_TX_CONFIRM_TARGET);
    bSpendZeroConfChange = gArgs.GetBoolArg("-spendzeroconfchange", DEFAULT_SPEND_ZEROCONF_CHANGE);
    fSendFreeTransactions = gArgs.GetBoolArg("-sendfreetransactions", DEFAULT_SEND_FREE_TRANSACTIONS);

    if (fSendFreeTransactions && gArgs.GetArg("-limitfreerelay", DEFAULT_LIMITFREERELAY) <= 0)
        return InitError("Creation of free transactions with their relay disabled is not supported.");

    if (gArgs.IsArgSet("-walletbackupsdir")) {
        if (!boost::filesystem::is_directory(gArgs.GetArg("-walletbackupsdir", ""))) {
            LogPrintf("%s: Warning: incorrect parameter -walletbackupsdir, path must exist! Using default path.\n", __func__);
            InitWarning("Warning: incorrect parameter -walletbackupsdir, path must exist! Using default path.\n");

            gArgs.ForceRemoveArg("-walletbackupsdir");
        }
    }

    return true;
}

void RegisterWalletRPC(CRPCTable &t)
{
    if (gArgs.GetBoolArg("-disablewallet", false)) return;

    RegisterWalletRPCCommands(t);
}