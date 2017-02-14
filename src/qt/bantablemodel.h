// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_QT_BANTABLEMODEL_H
#define DARKSILK_QT_BANTABLEMODEL_H

#include "net.h"

#include <QAbstractTableModel>
#include <QStringList>

class ClientModel;
class BanTablePriv;

struct CCombinedBan {
    CSubNet subnet;
    CBanEntry banEntry;
};

class BannedNodeLessThan
{
public:
    BannedNodeLessThan(int nColumn, Qt::SortOrder fOrder) :
        column(nColumn), order(fOrder) {}
    bool operator()(const CCombinedBan& left, const CCombinedBan& right) const;

private:
    int column;
    Qt::SortOrder order;
};

/**
   Qt model providing information about connected peers, similar to the
   "getpeerinfo" RPC call. Used by the rpc console UI.
 */
class BanTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit BanTableModel(ClientModel *parent = 0);
    void startAutoRefresh();
    void stopAutoRefresh();

    enum ColumnIndex {
        Address = 0,
        Bantime = 1
    };

    /** @name Methods overridden from QAbstractTableModel
        @{*/
    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const;
    QVariant data(const QModelIndex &index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    QModelIndex index(int row, int column, const QModelIndex &parent) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;
    void sort(int column, Qt::SortOrder order);
    bool shouldShow();
    /*@}*/

public Q_SLOTS:
    void refresh();

private:
    ClientModel *clientModel;
    QStringList columns;
    BanTablePriv *priv;
};

#endif // DARKSILK_QT_BANTABLEMODEL_H
