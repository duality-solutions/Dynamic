// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdapaccounttablemodel.h"
#include "bdappage.h"
#include "guiconstants.h"
#include "guiutil.h"
#include "sync.h"
#include "validation.h" // for cs_main
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"

#include <QDebug>
#include <QList>
#include <QTimer>
#include <QTableWidget>

// private implementation
class BdapAccountTablePriv
{
public:
    /** Local cache of peer information */
    QList<CNodeCombinedStats> cachedNodeStats;
    /** Column to sort nodes by */
    int sortColumn;
    /** Order (ascending or descending) to sort nodes by */
    Qt::SortOrder sortOrder;
    /** Index of rows by node ID */
    std::map<NodeId, int> mapNodeRows;

    /** Populate tableWidget_Users via RPC call */
    void refreshAccounts(QTableWidget* inputtable, bool filterOn = false)
    {

        JSONRPCRequest jreq;
        std::vector<std::string> params;
        int nNewRow = 0;
    
        std::string keyName {""};
        std::string getName {""};
        std::string getPath {""};
        std::string getExpirationDate {""};
        std::string tableWidgetName {""};

        LogPrintf("DEBUGGER CHECKBOX --%s %s-- \n", __func__, filterOn);
    
        if (!inputtable->objectName().isEmpty()) tableWidgetName = inputtable->objectName().toStdString();

        //Execute RPC call (List All Users)
        if (tableWidgetName == "tableWidget_Groups") {
            jreq.params = RPCConvertValues("getgroups", params);
            jreq.strMethod = "getgroups";
        } else {
            if (filterOn) {
                jreq.params = RPCConvertValues("mybdapaccounts", params);
                jreq.strMethod = "mybdapaccounts";
            } else {
                jreq.params = RPCConvertValues("getusers", params);
                jreq.strMethod = "getusers";
            } //(filterOn)
        }
        UniValue result = tableRPC.execute(jreq);

        inputtable->clearContents();
        inputtable->setRowCount(0);
        inputtable->setColumnCount(0);
        inputtable->setSortingEnabled(true);
        inputtable->setColumnCount(3);
        inputtable->setColumnWidth(0, 275); //Common Name (fixed)
        inputtable->setColumnWidth(1, 200); //Object Full Path (fixed)
        //inputtable->setColumnWidth(2, 175);

        inputtable->setHorizontalHeaderItem(0, new QTableWidgetItem(QString::fromStdString("Common Name")));
        inputtable->setHorizontalHeaderItem(1, new QTableWidgetItem(QString::fromStdString("Object Full Path")));
        inputtable->setHorizontalHeaderItem(2, new QTableWidgetItem(QString::fromStdString("Expiration Date")));

        //Parse results and populate QWidgetTable
        for (size_t i {0} ; i < result.size() ; ++i) {
            getName = "";
            getPath = "";
            getExpirationDate = "";

            nNewRow = inputtable->rowCount();
            inputtable->insertRow(nNewRow);

            for (size_t j {0} ; j < result[i].size() ; ++j) {
                keyName = "";
                keyName = result[i].getKeys()[j];

                // "common_name", "object_full_path"
                if (keyName == "common_name") getName = result[i].getValues()[j].get_str();
                if (keyName == "object_full_path") getPath = result[i].getValues()[j].get_str();
                //if (keyName == "expires_on") getExpirationDate = std::to_string(result[i].getValues()[j].get_int64());
                if (keyName == "expires_on") getExpirationDate = DateTimeStrFormat("%Y-%m-%d", result[i].getValues()[j].get_int64());

                //LogPrintf("DEBUGGER LOOP --%s %s: %s-- \n", __func__, result[i].getKeys()[j], result[i].getValues()[j].get_str());
            }
            QTableWidgetItem* commonNameItem = new QTableWidgetItem(QString::fromStdString(getName));
            QTableWidgetItem* fullPathItem = new QTableWidgetItem(QString::fromStdString(getPath));
            QTableWidgetItem* expirationDateItem = new QTableWidgetItem(QString::fromStdString(getExpirationDate));
            inputtable->setItem(nNewRow, 0, commonNameItem);
            inputtable->setItem(nNewRow, 1, fullPathItem);
            inputtable->setItem(nNewRow, 2, expirationDateItem);

        };

    }

    int size() const
    {
        return cachedNodeStats.size();
    }

    CNodeCombinedStats* index(int idx)
    {
        if (idx >= 0 && idx < cachedNodeStats.size())
            return &cachedNodeStats[idx];

        return 0;
    }
};

BdapAccountTableModel::BdapAccountTableModel(BdapPage* parent) : QAbstractTableModel(parent),
                                                      bdapPage(parent),
                                                      timer(0)
{
    
    currentIndex = bdapPage->getCurrentIndex();
    userTable = bdapPage->getUserTable();
    groupTable = bdapPage->getGroupTable();

        
    columns << tr("Common Name") << tr("Object Full Path") << tr("Expiration Date");
    priv.reset(new BdapAccountTablePriv());
    // default to unsorted
    priv->sortColumn = -1;

    //refresh();
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), SLOT(refresh()));
    timer->setInterval(25000); //MODEL_UPDATE_DELAY originally
    startAutoRefresh();


/*
    // set up timer for auto refresh
    timer = new QTimer(this);
    if (currentIndex == 1) { //Groups Table
        connect(timer, SIGNAL(timeout()), SLOT(refresh(bdapPage->getGroupTable())));
    } else { //Users Table
        connect(bdapPage->getUserTable(), SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getDetails(int,int)));
        connect(timer, SIGNAL(timeout()), SLOT(refresh(bdapPage->getUserTable())));
    };
    timer->setInterval(MODEL_UPDATE_DELAY);

    // load initial data
    if (currentIndex == 1) {
        refresh(bdapPage->getGroupTable());
    } else {
        refresh(bdapPage->getUserTable());
    };
*/

    //refresh(bdapPage->getUserTable());

    //LogPrintf("DEBUGGER TABLE --%s %s-- \n", __func__, bdapPage->getUserTable()->rowCount());
        //LogPrintf("DEBUGGER TABLE OUT --%s %s-- \n", __func__, bdapPage->getUserTable()->rowCount());
        //LogPrintf("DEBUGGER TABLE OUT --%s %s-- \n", __func__, bdapPage->getUserTable()->columnCount());

}

BdapAccountTableModel::~BdapAccountTableModel()
{
    // Intentionally left empty
}

void BdapAccountTableModel::startAutoRefresh()
{
    timer->start();
}

void BdapAccountTableModel::stopAutoRefresh()
{
    timer->stop();
}

int BdapAccountTableModel::rowCount(const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    return priv->size();
}

int BdapAccountTableModel::columnCount(const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    return columns.length();
}

QVariant BdapAccountTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    return QVariant();
}

QVariant BdapAccountTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal) {
        if (role == Qt::DisplayRole && section < columns.size()) {
            return columns[section];
        }
    }
    return QVariant();
}

Qt::ItemFlags BdapAccountTableModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    Qt::ItemFlags retval = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    return retval;
}

QModelIndex BdapAccountTableModel::index(int row, int column, const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    CNodeCombinedStats* data = priv->index(row);

    if (data)
        return createIndex(row, column, data);
    return QModelIndex();
}

const CNodeCombinedStats* BdapAccountTableModel::getNodeStats(int idx)
{
    return priv->index(idx);
}

void BdapAccountTableModel::refresh()
{
    myUsersChecked = bdapPage->getMyUserCheckBoxChecked();

    Q_EMIT layoutAboutToBeChanged();
    priv->refreshAccounts(userTable,myUsersChecked);
    priv->refreshAccounts(groupTable);
    Q_EMIT layoutChanged();
}

void BdapAccountTableModel::refreshUsers()
{
    myUsersChecked = bdapPage->getMyUserCheckBoxChecked();

    Q_EMIT layoutAboutToBeChanged();
    priv->refreshAccounts(userTable,myUsersChecked);
    Q_EMIT layoutChanged();
}

void BdapAccountTableModel::refreshGroups()
{
    Q_EMIT layoutAboutToBeChanged();
    priv->refreshAccounts(groupTable);
    Q_EMIT layoutChanged();
}


int BdapAccountTableModel::getRowByNodeId(NodeId nodeid)
{
    std::map<NodeId, int>::iterator it = priv->mapNodeRows.find(nodeid);
    if (it == priv->mapNodeRows.end())
        return -1;

    return it->second;
}

void BdapAccountTableModel::sort(int column, Qt::SortOrder order)
{
    priv->sortColumn = column;
    priv->sortOrder = order;
    refresh();
}


void BdapAccountTableModel::getDetails(int row, int column)
{
    //QObject* obj = sender();
    QTableWidget* inputtable = qobject_cast<QTableWidget*>(sender());


    LogPrintf("DEBUGGER TABLE SELECT --%s %s %s %s-- \n", __func__, row, column, inputtable->item(row,column)->text().toStdString());

} //getDetails


