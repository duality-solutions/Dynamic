// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_SWAPDIALOG_H
#define DYNAMIC_QT_SWAPDIALOG_H

#include "amount.h"

#include <QDialog>
#include <QAbstractButton>

namespace Ui
{
class SwapDialog;
}

class WalletModel;

class SwapDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SwapDialog(QWidget* parent);
    ~SwapDialog();

    void setWalletModel(WalletModel* _walletModel);

protected Q_SLOTS:

private Q_SLOTS:
    void buttonBoxClicked(QAbstractButton*);

private:
    Ui::SwapDialog* ui;
    WalletModel* walletModel;
    CAmount totalBalance;
    CAmount immatureBalance;
    CAmount lockedBalance;
    CAmount swapBalance;
    bool walletLocked;

    QString getSwapAddress();
    bool swapDynamic();
};

#endif // DYNAMIC_QT_SWAPDIALOG_H
