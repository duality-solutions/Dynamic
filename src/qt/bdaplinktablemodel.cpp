// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdaplinktablemodel.h"
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
#include <QHeaderView>

#include <boost/algorithm/string.hpp>

// private implementation
class BdapLinkTablePriv
{
public:
    /** Local cache of peer information */
    QList<CAccountStats2> cachedAccountStats;
    /** Column to sort nodes by */
    int sortColumn;
    /** Order (ascending or descending) to sort nodes by */
    Qt::SortOrder sortOrder;

    /** Populate tableWidget_Users via RPC call */
    void refreshLinks(QTableWidget* inputtable, QLabel* statusDisplay)
    {

        JSONRPCRequest jreq;
        std::vector<std::string> params;
        int nNewRow = 0;
        int recordsFound = 0;
    
        std::string keyName {""};
        std::string getRequestor {""};
        std::string getRecipient {""};
        std::string getDate {""};
        std::string tableWidgetName {""};
        std::string outputmessage = "";

        bool hasValues = false;

        if (!inputtable->objectName().isEmpty()) tableWidgetName = inputtable->objectName().toStdString();

        //check if table has been previously sorted
        if (!inputtable->objectName().isEmpty()){
            if (inputtable->rowCount() > 0) {
                hasValues = true;
                sortColumn = inputtable->horizontalHeader()->sortIndicatorSection();
                sortOrder = inputtable->horizontalHeader()->sortIndicatorOrder();
            } //if rowcount
        } //if not isempty


        //Execute proper RPC call 
        if (tableWidgetName == "tableWidgetComplete") {
                params.push_back("complete");
                jreq.params = RPCConvertValues("link", params);
                jreq.strMethod = "link";
        } else if (tableWidgetName == "tableWidgetPendingAccept") {
                params.push_back("pending");
                params.push_back("accept");
                jreq.params = RPCConvertValues("link", params);
                jreq.strMethod = "link";
        } else if (tableWidgetName == "tableWidgetPendingRequest") {
                params.push_back("pending");
                params.push_back("request");
                jreq.params = RPCConvertValues("link", params);
                jreq.strMethod = "link";
        } else {
            return;
        }
        
        UniValue result = UniValue(UniValue::VOBJ);

        //Handle RPC errors
        try {
            result = tableRPC.execute(jreq);
        } catch (const UniValue& objError) {
            std::string message = find_value(objError, "message").get_str();
            outputmessage = message;
            QMessageBox::critical(0, "BDAP Error", QString::fromStdString(outputmessage));
            return;
        } catch (const std::exception& e) {
            outputmessage = e.what();
            QMessageBox::critical(0, "BDAP Error", QString::fromStdString(outputmessage));
            return;
        }

        inputtable->clearContents();
        inputtable->setRowCount(0);
        inputtable->setColumnCount(0);
        inputtable->setSortingEnabled(true);
        inputtable->setColumnCount(3);
        //inputtable->setColumnWidth(0, COMMONNAME_COLWIDTH); //Common Name (fixed)
        //inputtable->setColumnWidth(1, FULLPATH_COLWIDTH); //Object Full Path (fixed)
        //inputtable->setColumnWidth(2, 100);

        inputtable->setHorizontalHeaderItem(0, new QTableWidgetItem(QString::fromStdString("Requestor")));
        inputtable->setHorizontalHeaderItem(1, new QTableWidgetItem(QString::fromStdString("Recipient")));
        inputtable->setHorizontalHeaderItem(2, new QTableWidgetItem(QString::fromStdString("Date")));

        //make columns resize dynamically
        QHeaderView* header = inputtable->horizontalHeader();
        header->setSectionResizeMode(QHeaderView::Stretch);

        //Parse results and populate QWidgetTable
        for (size_t i {0} ; i < result.size() ; ++i) {
            getRequestor = "";
            getRecipient = "";
            getDate = "";


            for (size_t j {0} ; j < result[i].size() ; ++j) {
                keyName = "";
                keyName = result[i].getKeys()[j];

                // "requestor_fqdn", "recipient_fqdn", "time"
                if (keyName == "requestor_fqdn") getRequestor = result[i].getValues()[j].get_str();
                if (keyName == "recipient_fqdn") getRecipient = result[i].getValues()[j].get_str();
                if (keyName == "time") getDate = DateTimeStrFormat("%Y-%m-%d", result[i].getValues()[j].get_int64());

            }

            //add row if all criteria have been met
            //if ( ((searchCommon == "") && (searchPath == "")) || (((boost::algorithm::to_lower_copy(getName)).find(boost::algorithm::to_lower_copy(searchCommon)) != std::string::npos) && ((boost::algorithm::to_lower_copy(getPath)).find(boost::algorithm::to_lower_copy(searchPath)) != std::string::npos)) ) {
                nNewRow = inputtable->rowCount();
                inputtable->insertRow(nNewRow);
                QTableWidgetItem* requestorItem = new QTableWidgetItem(QString::fromStdString(getIdFromPath(getRequestor)));
                QTableWidgetItem* recipientItem = new QTableWidgetItem(QString::fromStdString(getIdFromPath(getRecipient)));
                QTableWidgetItem* dateItem = new QTableWidgetItem(QString::fromStdString(getDate));
                inputtable->setItem(nNewRow, 0, requestorItem);
                inputtable->setItem(nNewRow, 1, recipientItem);
                inputtable->setItem(nNewRow, 2, dateItem);
                recordsFound++;

            //} //if searchcommon


        }; //for loop

        //if we saved the previous state, apply to current results
        if (hasValues) {
            inputtable->horizontalHeader()->setSortIndicator(sortColumn, sortOrder);
        }

        statusDisplay->setText(QString::fromStdString(("(" + std::to_string(recordsFound) +")")));

    } //refreshLinks

    int size() const
    {
        return cachedAccountStats.size();
    }

    CAccountStats2* index(int idx)
    {
        if (idx >= 0 && idx < cachedAccountStats.size())
            return &cachedAccountStats[idx];

        return 0;
    }


    std::string getIdFromPath(std::string inputstring) {
        std::string returnvalue = inputstring;
        std::vector<std::string> results;

        boost::split(results, inputstring, [](char c){return c == '@';});

        if (results.size() > 0) {
            returnvalue = results[0];
        }

        return returnvalue;

    } //getIdFromPath

}; //BdapLinkTablePriv



BdapLinkTableModel::BdapLinkTableModel(BdapPage* parent) : QAbstractTableModel(parent),
                                                      bdapPage(parent),
                                                      timer(0)
{
    completeTable = bdapPage->getCompleteTable();
    completeStatus = bdapPage->getLinkCompleteRecords();

    pendingAcceptTable = bdapPage->getPendingAcceptTable();;
    pendingAcceptStatus = bdapPage->getPendingAcceptRecords();

    pendingRequestTable = bdapPage->getPendingRequestTable();;
    pendingRequestStatus = bdapPage->getPendingRequestRecords();

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), SLOT(refresh()));
    timer->setInterval(60000); //MODEL_UPDATE_DELAY originally    

    priv.reset(new BdapLinkTablePriv());
    // default to unsorted
    priv->sortColumn = -1;

}

BdapLinkTableModel::~BdapLinkTableModel()
{
    // Intentionally left empty
}

int BdapLinkTableModel::rowCount(const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    return priv->size();
}

int BdapLinkTableModel::columnCount(const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    return columns.length();
}

QVariant BdapLinkTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    return QVariant();
}

QVariant BdapLinkTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal) {
        if (role == Qt::DisplayRole && section < columns.size()) {
            return columns[section];
        }
    }
    return QVariant();
}

Qt::ItemFlags BdapLinkTableModel::flags(const QModelIndex& index) const
{
    if (!index.isValid())
        return 0;

    Qt::ItemFlags retval = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    return retval;
}

QModelIndex BdapLinkTableModel::index(int row, int column, const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    CAccountStats2* data = priv->index(row);

    if (data)
        return createIndex(row, column, data);
    return QModelIndex();
}


void BdapLinkTableModel::sort(int column, Qt::SortOrder order)
{
    priv->sortColumn = column;
    priv->sortOrder = order;
    refresh();
}

void BdapLinkTableModel::refresh()
{


}


void BdapLinkTableModel::refreshComplete()
{
    Q_EMIT layoutAboutToBeChanged();
    priv->refreshLinks(completeTable,completeStatus);
    Q_EMIT layoutChanged();
}


void BdapLinkTableModel::refreshPendingAccept()
{
    Q_EMIT layoutAboutToBeChanged();
    priv->refreshLinks(pendingAcceptTable,pendingAcceptStatus);
    Q_EMIT layoutChanged();
}


void BdapLinkTableModel::refreshPendingRequest()
{
    Q_EMIT layoutAboutToBeChanged();
    priv->refreshLinks(pendingRequestTable,pendingRequestStatus);
    Q_EMIT layoutChanged();
}
