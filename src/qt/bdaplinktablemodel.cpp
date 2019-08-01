// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdaplinktablemodel.h"
#include "bdappage.h"
#include "guiconstants.h"
#include "guiutil.h"
#include "spork.h"
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
    void refreshLinks(QTableWidget* inputtable, QLabel* statusDisplay, std::string searchRequestor = "", std::string searchRecipient = "")
    {
        if (!sporkManager.IsSporkActive(SPORK_30_ACTIVATE_BDAP))
            return;

        JSONRPCRequest jreq;
        std::vector<std::string> params;
        int nNewRow = 0;
        int recordsFound = 0;
    
        std::string keyName {""};
        std::string getRequestor {""};
        std::string getRecipient {""};
        std::string getDate {""};
        std::string getMessage {""};
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
            QMessageBox::critical(0, "BDAP Error", QObject::tr(outputmessage.c_str()));
            return;
        } catch (const std::exception& e) {
            outputmessage = e.what();
            QMessageBox::critical(0, "BDAP Error", QObject::tr(outputmessage.c_str()));
            return;
        }

        inputtable->clearContents();
        inputtable->setRowCount(0);
        inputtable->setColumnCount(0);
        inputtable->setSortingEnabled(true);
        inputtable->setColumnCount(3);

        inputtable->setHorizontalHeaderItem(0, new QTableWidgetItem(QObject::tr("Requestor")));
        inputtable->setHorizontalHeaderItem(1, new QTableWidgetItem(QObject::tr("Recipient")));
        inputtable->setHorizontalHeaderItem(2, new QTableWidgetItem(QObject::tr("Date")));

        //make columns resize dynamically
        QHeaderView* header = inputtable->horizontalHeader();
        header->setSectionResizeMode(QHeaderView::Stretch);

        //Parse results and populate QWidgetTable
        for (size_t i {0} ; i < result.size() - 1 ; ++i) {
            getRequestor = "";
            getRecipient = "";
            getDate = "";


            for (size_t j {0} ; j < result[i].size() ; ++j) {
                keyName = "";
                keyName = result[i].getKeys()[j];

                // "requestor_fqdn", "recipient_fqdn", "time", "link_message"
                if (keyName == "requestor_fqdn") getRequestor = getIdFromPath(result[i].getValues()[j].get_str());
                if (keyName == "recipient_fqdn") getRecipient = getIdFromPath(result[i].getValues()[j].get_str());
                if (keyName == "time") getDate = DateTimeStrFormat("%Y-%m-%d", result[i].getValues()[j].get_int64());
                if (keyName == "link_message") getMessage = result[i].getValues()[j].get_str();

            }

            //add row if all criteria have been met
            if ( ((searchRequestor == "") && (searchRecipient == "")) || (((boost::algorithm::to_lower_copy(getRequestor)).find(boost::algorithm::to_lower_copy(searchRequestor)) != std::string::npos) && ((boost::algorithm::to_lower_copy(getRecipient)).find(boost::algorithm::to_lower_copy(searchRecipient)) != std::string::npos)) ) {
                nNewRow = inputtable->rowCount();
                inputtable->insertRow(nNewRow);
                QTableWidgetItem* requestorItem = new QTableWidgetItem(QString::fromStdString(getRequestor));
                QTableWidgetItem* recipientItem = new QTableWidgetItem(QString::fromStdString(getRecipient));
                QTableWidgetItem* dateItem = new QTableWidgetItem(QString::fromStdString(getDate));
                requestorItem->setToolTip(QString::fromStdString(getMessage));
                recipientItem->setToolTip(QString::fromStdString(getMessage));
                dateItem->setToolTip(QString::fromStdString(getMessage));
                inputtable->setItem(nNewRow, 0, requestorItem);
                inputtable->setItem(nNewRow, 1, recipientItem);
                inputtable->setItem(nNewRow, 2, dateItem);
                recordsFound++;
            } //if searchRequestor...


        } //for loop

        //if we saved the previous state, apply to current results
        if (hasValues) {
            inputtable->horizontalHeader()->setSortIndicator(sortColumn, sortOrder);
        }

        std::string statusString = "(" + std::to_string(recordsFound) + ")";

        statusDisplay->setText(QObject::tr(statusString.c_str()));

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

    priv.reset(new BdapLinkTablePriv());
    // default to unsorted
    priv->sortColumn = -1;

    refreshAll();

    //comment out timer, but keep for possible future use
    /*
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), SLOT(refresh()));
    timer->setInterval(60000); //MODEL_UPDATE_DELAY originally 
    startAutoRefresh();  
    */

}

BdapLinkTableModel::~BdapLinkTableModel()
{
    // Intentionally left empty
}

void BdapLinkTableModel::startAutoRefresh()
{
    timer->start();
}

void BdapLinkTableModel::stopAutoRefresh()
{
    timer->stop();
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
    refreshAll();

}

void BdapLinkTableModel::refreshAll()
{
    refreshComplete();
    refreshPendingAccept();
    refreshPendingRequest();

} //refreshAll

void BdapLinkTableModel::refreshComplete()
{
    searchCompleteRequestor = bdapPage->getCompleteRequestorSearch();
    searchCompleteRecipient = bdapPage->getCompleteRecipientSearch();

    Q_EMIT layoutAboutToBeChanged();
    priv->refreshLinks(completeTable,completeStatus,searchCompleteRequestor,searchCompleteRecipient);
    Q_EMIT layoutChanged();
}

void BdapLinkTableModel::refreshPendingAccept()
{
    searchPARequestor = bdapPage->getPARequestorSearch();
    searchPARecipient = bdapPage->getPARecipientSearch();
    
    Q_EMIT layoutAboutToBeChanged();
    priv->refreshLinks(pendingAcceptTable,pendingAcceptStatus,searchPARequestor,searchPARecipient);
    Q_EMIT layoutChanged();
}

void BdapLinkTableModel::refreshPendingRequest()
{
    searchPRRequestor = bdapPage->getPRRequestorSearch();
    searchPRRecipient = bdapPage->getPRRecipientSearch();

    Q_EMIT layoutAboutToBeChanged();
    priv->refreshLinks(pendingRequestTable,pendingRequestStatus,searchPRRequestor,searchPRRecipient);
    Q_EMIT layoutChanged();
}