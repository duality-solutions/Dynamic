// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_QT_SANDSTORMCONFIG_H
#define DARKSILK_QT_SANDSTORMCONFIG_H

#include <QDialog>

namespace Ui {
    class SandstormConfig;
}
class WalletModel;

/** Multifunctional dialog to ask for passphrases. Used for encryption, unlocking, and changing the passphrase.
 */
class SandstormConfig : public QDialog
{
    Q_OBJECT

public:

    SandstormConfig(QWidget *parent = 0);
    ~SandstormConfig();

    void setModel(WalletModel *model);


private:
    Ui::SandstormConfig *ui;
    WalletModel *model;
    void configure(bool enabled, int coins, int rounds);

private Q_SLOTS:

    void clickBasic();
    void clickHigh();
    void clickMax();
};

#endif // DARKSILK_QT_SANDSTORMCONFIG_H
