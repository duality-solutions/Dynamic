// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_BDAPACCOUNTTABLEMODEL_H
#define DYNAMIC_QT_BDAPACCOUNTTABLEMODEL_H

#include <QAbstractTableModel>
#include <QStringList>
#include <QTableWidget>
#include <QLabel>

#include <memory>

class BdapPage;
class BdapAccountTablePriv;

QT_BEGIN_NAMESPACE
class QTimer;
QT_END_NAMESPACE

struct CAccountStats {
    unsigned int count;
};

/**
   Qt model providing information about BDAP users and groups.
 */
class BdapAccountTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit BdapAccountTableModel(BdapPage* parent = 0);
    ~BdapAccountTableModel();

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