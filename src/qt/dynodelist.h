// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_DYNODELIST_H
#define DYNAMIC_QT_DYNODELIST_H

#include "platformstyle.h"

#include "primitives/transaction.h"
#include "sync.h"
#include "util.h"

#include <QMenu>
#include <QTimer>
#include <QWidget>

#define MY_DYNODELIST_UPDATE_SECONDS 60
#define DYNODELIST_UPDATE_SECONDS 15
#define DYNODELIST_FILTER_COOLDOWN_SECONDS 3

namespace Ui
{
class DynodeList;
}

class ClientModel;
class WalletModel;

QT_BEGIN_NAMESPACE
class QModelIndex;
QT_END_NAMESPACE

/** Dynode Manager page widget */
class DynodeList : public QWidget
{
    Q_OBJECT

public:
    explicit DynodeList(const PlatformStyle* platformStyle, QWidget* parent = 0);
    ~DynodeList();

    void setClientModel(ClientModel* clientModel);
    void setWalletModel(WalletModel* walletModel);
    void StartAlias(std::string strAlias);
    void StartAll(std::string strCommand = "start-all");

private:
    QMenu* contextMenu;
    int64_t nTimeFilterUpdated;
    bool fFilterUpdated;

public Q_SLOTS:
    void updateMyDynodeInfo(QString strAlias, QString strAddr, const COutPoint& outpoint);
    void updateMyNodeList(bool fForce = false);
    void updateNodeList();

Q_SIGNALS:

private:
    QTimer* timer;
    Ui::DynodeList* ui;
    ClientModel* clientModel;
    WalletModel* walletModel;
    // Protects tableWidgetDynodes
    CCriticalSection cs_dnlist;

    // Protects tableWidgetMyDynodes
    CCriticalSection cs_mydnlist;
    QString strCurrentFilter;

private Q_SLOTS:
    void showContextMenu(const QPoint&);
    void on_filterLineEdit_textChanged(const QString& strFilterIn);
    void on_startButton_clicked();
    void on_startAllButton_clicked();
    void on_startMissingButton_clicked();
    void on_tableWidgetMyDynodes_itemSelectionChanged();
    void on_UpdateButton_clicked();
};
#endif // DYNAMIC_QT_DYNODELIST_H
