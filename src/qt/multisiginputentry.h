// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef MULTISIGINPUTENTRY_H
#define MULTISIGINPUTENTRY_H

#include <QFrame>

#include "uint256.h"
#include "amount.h"

class CTxIn;
class WalletModel;
class PlatformStyle;

namespace Ui
{
    class MultisigInputEntry;
}

class MultisigInputEntry : public QFrame
{
    Q_OBJECT;

  public:
    explicit MultisigInputEntry(const PlatformStyle *platformStyle, QWidget *parent = 0);
    ~MultisigInputEntry();
    void setModel(WalletModel *model);
    bool validate();
    CTxIn getInput();
    CAmount getAmount();
    QString getRedeemScript();
    void setTransactionId(QString transactionId);
    void setTransactionOutputIndex(int index);

  public Q_SLOTS:
    void setRemoveEnabled(bool enabled);
    void clear();

  Q_SIGNALS:
    void removeEntry(MultisigInputEntry *entry);
    void updateAmount();

  private:
    Ui::MultisigInputEntry *ui;
    WalletModel *model;
    uint256 txHash;
    const PlatformStyle *platformStyle;

  private Q_SLOTS:
    void on_transactionId_textChanged(const QString &transactionId);
    void on_pasteTransactionIdButton_clicked();
    void on_deleteButton_clicked();
    void on_transactionOutput_currentIndexChanged(int index);
    void on_pasteRedeemScriptButton_clicked();
};

#endif // MULTISIGINPUTENTRY_H 
