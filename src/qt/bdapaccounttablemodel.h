// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_BDAPACCOUNTTABLEMODEL_H
#define DYNAMIC_QT_BDAPACCOUNTTABLEMODEL_H

#include "net.h"
#include "net_processing.h" // For CNodeStateStats

#include <QAbstractTableModel>
#include <QStringList>
#include <QTableWidget>
#include <QLabel>

class BdapPage;
class BdapAccountTablePriv;

QT_BEGIN_NAMESPACE
class QTimer;
QT_END_NAMESPACE

struct CNodeCombinedStats {
    CNodeStats nodeStats;
    CNodeStateStats nodeStateStats;
    bool fNodeStateStatsAvailable;
};


/**
   Qt model providing information about connected peers, similar to the
   "getpeerinfo" RPC call. Used by the rpc console UI.
 */
class BdapAccountTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit BdapAccountTableModel(BdapPage* parent = 0);
    ~BdapAccountTableModel();
    const CNodeCombinedStats* getNodeStats(int idx);
    int getRowByNodeId(NodeId nodeid);
    void startAutoRefresh();
    void stopAutoRefresh();
    void refreshUsers();
    void refreshGroups();

    enum ColumnIndex {
        CommonName = 0,
        ObjectFullPath = 1,
        ExpirationDate = 2
    };

    /** @name Methods overridden from QAbstractTableModel
        @{*/
    int rowCount(const QModelIndex& parent) const;
    int columnCount(const QModelIndex& parent) const;
    QVariant data(const QModelIndex& index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    QModelIndex index(int row, int column, const QModelIndex& parent) const;
    Qt::ItemFlags flags(const QModelIndex& index) const;
    void sort(int column, Qt::SortOrder order);
    /*@}*/

public Q_SLOTS:
    void refresh();
    void getDetails(int row, int column);

private:
    BdapPage* bdapPage;
    QStringList columns;
    std::unique_ptr<BdapAccountTablePriv> priv;
    QTimer* timer;
    int currentIndex;
    QTableWidget* userTable;
    QTableWidget* groupTable;
    QLabel* userStatus;
    QLabel* groupStatus;
    bool myUsersChecked;
    bool myGroupsChecked;
    std::string searchUserCommon;
    std::string searchUserPath;
    std::string searchGroupCommon;
    std::string searchGroupPath;
    
};

#endif // DYNAMIC_QT_BDAPACCOUNTTABLEMODEL_H