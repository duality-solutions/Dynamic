// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPPAGE_H
#define BDAPPAGE_H

#include "platformstyle.h"

#include "walletmodel.h"

#include <QPushButton>
#include <QWidget>

#include <memory>

namespace Ui
{
class BdapPage;
}

class BdapPage : public QWidget
{
    Q_OBJECT

public:
    explicit BdapPage(const PlatformStyle* platformStyle, QWidget* parent = 0);
    ~BdapPage();

    void setModel(WalletModel* model);

private:
    Ui::BdapPage* ui;
    WalletModel* model;
    std::unique_ptr<WalletModel::UnlockContext> unlockContext;



private Q_SLOTS:


};

#endif // BDAPPAGE_H
