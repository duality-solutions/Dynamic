// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_TRANSACTIONDESCDIALOG_H
#define DYNAMIC_QT_TRANSACTIONDESCDIALOG_H

#include <QDialog>

namespace Ui
{
class TransactionDescDialog;
}

QT_BEGIN_NAMESPACE
class QModelIndex;
QT_END_NAMESPACE

/** Dialog showing transaction details. */
class TransactionDescDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TransactionDescDialog(const QModelIndex& idx, QWidget* parent = 0);
    ~TransactionDescDialog();

private:
    Ui::TransactionDescDialog* ui;
};

#endif // DYNAMIC_QT_TRANSACTIONDESCDIALOG_H
