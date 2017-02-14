// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_QT_STORMNODELIST_H
#define DARKSILK_QT_STORMNODELIST_H

#include "stormnode.h"
#include "platformstyle.h"
#include "sync.h"
#include "util.h"

#include <QMenu>
#include <QTimer>
#include <QWidget>

#define MY_STORMNODELIST_UPDATE_SECONDS                 60
#define STORMNODELIST_UPDATE_SECONDS                    15
#define STORMNODELIST_FILTER_COOLDOWN_SECONDS            3

namespace Ui {
    class StormnodeList;
}

class ClientModel;
class WalletModel;

QT_BEGIN_NAMESPACE
class QModelIndex;
QT_END_NAMESPACE

/** Stormnode Manager page widget */
class StormnodeList : public QWidget
{
    Q_OBJECT

public:
    explicit StormnodeList(const PlatformStyle *platformStyle, QWidget *parent = 0);
    ~StormnodeList();

    void setClientModel(ClientModel *clientModel);
    void setWalletModel(WalletModel *walletModel);
    void StartAlias(std::string strAlias);
    void StartAll(std::string strCommand = "start-all");

private:
    QMenu *contextMenu;
    int64_t nTimeFilterUpdated;
    bool fFilterUpdated;

public Q_SLOTS:
    void updateMyStormnodeInfo(QString strAlias, QString strAddr, stormnode_info_t& infoSn);
    void updateMyNodeList(bool fForce = false);
    void updateNodeList();

Q_SIGNALS:

private:
    QTimer *timer;
    Ui::StormnodeList *ui;
    ClientModel *clientModel;
    WalletModel *walletModel;
    // Protects tableWidgetStormnodes
    CCriticalSection cs_snlist;

    // Protects tableWidgetMyStormnodes
    CCriticalSection cs_mysnlist;
    QString strCurrentFilter;

private Q_SLOTS:
    void showContextMenu(const QPoint &);
    void on_filterLineEdit_textChanged(const QString &strFilterIn);
    void on_startButton_clicked();
    void on_startAllButton_clicked();
    void on_startMissingButton_clicked();
    void on_tableWidgetMyStormnodes_itemSelectionChanged();
    void on_UpdateButton_clicked();
};
#endif // DARKSILK_QT_STORMNODELIST_H
