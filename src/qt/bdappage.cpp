// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/fees.h"
#include "bdapaccounttablemodel.h"
#include "bdapaddlinkdialog.h"
#include "bdapadduserdialog.h"
#include "bdapfeespopup.h"
#include "bdaplinkdetaildialog.h"
#include "bdaplinktablemodel.h"
#include "bdappage.h"
#include "bdapupdateaccountdialog.h"
#include "bdapuserdetaildialog.h"
#include "clientmodel.h"
#include "dynode-sync.h"
#include "guiutil.h"
#include "optionsmodel.h"
#include "rpcclient.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "ui_bdappage.h"
#include "walletmodel.h"

#include <stdio.h>

#include <boost/algorithm/string.hpp>

#include <QTableWidget>

BdapPage::BdapPage(const PlatformStyle* platformStyle, QWidget* parent) : QWidget(parent),
                                                                            ui(new Ui::BdapPage),
                                                                            clientModel(0),
                                                                            model(0),
                                                                            bdapAccountTableModel(0)
{
    ui->setupUi(this);
    
    evaluateTransactionButtons();

    bdapAccountTableModel = new BdapAccountTableModel(this);
    bdapLinkTableModel = new BdapLinkTableModel(this);

    ui->lineEditUserCommonNameSearch->setFixedWidth(COMMONNAME_COLWIDTH);
    ui->lineEditUserFullPathSearch->setFixedWidth(FULLPATH_COLWIDTH);

    ui->lineEditGroupCommonNameSearch->setFixedWidth(COMMONNAME_COLWIDTH);
    ui->lineEditGroupFullPathSearch->setFixedWidth(FULLPATH_COLWIDTH);

    //Users tab
    connect(ui->pushButton_All, SIGNAL(clicked()), this, SLOT(listAllUsers()));
    connect(ui->addUser, SIGNAL(clicked()), this, SLOT(addUser()));
    connect(ui->pushButtonUpdateUser, SIGNAL(clicked()), this, SLOT(updateUser()));
    connect(ui->deleteUser, SIGNAL(clicked()), this, SLOT(deleteUser()));

    connect(ui->checkBoxMyUsers, SIGNAL(clicked()), this, SLOT(listAllUsers()));
    connect(ui->lineEditUserCommonNameSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllUsers()));
    connect(ui->lineEditUserFullPathSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllUsers()));

    connect(ui->tableWidget_Users, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getUserDetails(int,int)));

    //Groups tab
    connect(ui->pushButton_AllGroups, SIGNAL(clicked()), this, SLOT(listAllGroups()));
    connect(ui->addGroup, SIGNAL(clicked()), this, SLOT(addGroup()));
    connect(ui->pushButtonUpdateGroup, SIGNAL(clicked()), this, SLOT(updateGroup()));
    connect(ui->deleteGroup, SIGNAL(clicked()), this, SLOT(deleteGroup()));

    connect(ui->checkBoxMyGroups, SIGNAL(clicked()), this, SLOT(listAllGroups()));
    connect(ui->lineEditGroupCommonNameSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllGroups()));
    connect(ui->lineEditGroupFullPathSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllGroups()));

    connect(ui->tableWidget_Groups, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getGroupDetails(int,int)));

    //Links tab
    connect(ui->pushButtonRefreshAllLinks, SIGNAL(clicked()), this, SLOT(listLinksAll()));

    connect(ui->lineEditCompleteRequestorSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listLinksComplete()));
    connect(ui->lineEditCompleteRecipientSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listLinksComplete()));
    
    connect(ui->lineEditPARequestorSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listPendingAccept()));
    connect(ui->lineEditPARecipientSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listPendingAccept()));

    connect(ui->lineEditPRRequestorSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listPendingRequest()));
    connect(ui->lineEditPRRecipientSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listPendingRequest()));

    connect(ui->pushButtonAccept, SIGNAL(clicked()), this, SLOT(acceptLink()));

    connect(ui->tableWidgetPendingAccept, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getLinkDetails(int,int)));
    connect(ui->tableWidgetPendingRequest, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getLinkDetails(int,int)));
    connect(ui->tableWidgetComplete, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getLinkDetails(int,int)));

    connect(ui->pushButtonAddLink, SIGNAL(clicked()), this, SLOT(addLink()));
}

BdapPage::~BdapPage()
{
    delete ui;
}

void BdapPage::setModel(WalletModel* model)
{
    this->model = model;

}

void BdapPage::setClientModel(ClientModel* _clientModel)
{
    this->clientModel = _clientModel;

    if (_clientModel) {
        connect(_clientModel, SIGNAL(numBlocksChanged(int, QDateTime, double, bool)), this, SLOT(updateBDAPLists()));
    }
}

void BdapPage::evaluateTransactionButtons()
{
    int currentIndex = ui->tabWidget->currentIndex();
    bool myUsersChecked = ui->checkBoxMyUsers->isChecked(); 
    bool myGroupsChecked = ui->checkBoxMyGroups->isChecked();

    switch (currentIndex) {
        case 0: //Users
            ui->pushButtonUpdateUser->setVisible(myUsersChecked);
            ui->deleteUser->setVisible(myUsersChecked);   
            break;
        case 1: //Groups
            ui->pushButtonUpdateGroup->setVisible(myGroupsChecked);
            ui->deleteGroup->setVisible(myGroupsChecked);
            break;

    }; //end switch
} //evaluateTransactionButtons

//Links tab =========================================================================
void BdapPage::listLinksAll()
{
    bdapLinkTableModel->refreshAll();
} //listLinksAll

void BdapPage::listLinksComplete()
{
    bdapLinkTableModel->refreshComplete();
} //listLinksComplete

void BdapPage::listPendingAccept()
{
    bdapLinkTableModel->refreshPendingAccept();
} //listPendingAccept

void BdapPage::listPendingRequest()
{
    bdapLinkTableModel->refreshPendingRequest();
} //listPendingRequest

//Groups tab ========================================================================
void BdapPage::listAllGroups()
{
    evaluateTransactionButtons();

    bdapAccountTableModel->refreshGroups();
} //listAllGroups

void BdapPage::addGroup()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create BDAP objects while wallet is not synced."));
        return;
    }

    BdapAddUserDialog dlg(this,model->getOptionsModel()->getDisplayUnit(),BDAP::ObjectType::BDAP_GROUP);
    dlg.setWindowTitle(QObject::tr("Add BDAP Group"));
    dlg.exec();
    if (dlg.result() == 1) {
        bdapAccountTableModel->refreshGroups();
    }
} //addGroup

void BdapPage::deleteGroup()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create/modify BDAP objects while wallet is not synced."));
        return;
    }

    QMessageBox::StandardButton reply;
    std::string account = "";
    std::string displayedMessage = "";

    QItemSelectionModel* selectionModel = ui->tableWidget_Groups->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : -1;

    if (nSelectedRow == -1) return; //do nothing if no rows are selected

    account = ui->tableWidget_Groups->item(nSelectedRow,1)->text().toStdString();
    displayedMessage = "Are you sure you want to delete \"" + account + "\""; //std::to_string(nSelectedRow);

    reply = QMessageBox::question(this, QObject::tr("Confirm Delete Account"), QObject::tr(displayedMessage.c_str()), QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        executeDeleteAccount(account, BDAP::ObjectType::BDAP_GROUP);

    };
} //deleteGroup

void BdapPage::updateGroup()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create/modify BDAP objects while wallet is not synced."));
        return;
    }

    std::string account = "";
    std::string commonName = "";
    std::string expirationDate = "";
    
    QItemSelectionModel* selectionModel = ui->tableWidget_Groups->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : -1;

    if (nSelectedRow == -1) return; //do nothing if no rows are selected

    account = ui->tableWidget_Groups->item(nSelectedRow,1)->text().toStdString();
    commonName = ui->tableWidget_Groups->item(nSelectedRow,0)->text().toStdString();
    expirationDate = ui->tableWidget_Groups->item(nSelectedRow,2)->text().toStdString();

    BdapUpdateAccountDialog dlg(this,BDAP::ObjectType::BDAP_GROUP,account,commonName,expirationDate,model->getOptionsModel()->getDisplayUnit());
    dlg.setWindowTitle(QObject::tr("Update BDAP Group"));
    
    dlg.exec();
    if (dlg.result() == 1) {
        bdapAccountTableModel->refreshGroups();
    }

} //updateGroup

void BdapPage::getGroupDetails(int row, int column)
{
    BdapUserDetailDialog dlg(this,BDAP::ObjectType::BDAP_GROUP,ui->tableWidget_Groups->item(row,1)->text().toStdString());
    dlg.setWindowTitle(QObject::tr("BDAP Group Detail"));
    dlg.exec();
} //getGroupDetails

void BdapPage::updateBDAPLists()
{
    if (dynodeSync.IsBlockchainSynced())  {
        evaluateTransactionButtons();

        bdapAccountTableModel->refreshUsers();
        bdapAccountTableModel->refreshGroups();
        bdapLinkTableModel->refreshComplete();
        bdapLinkTableModel->refreshPendingAccept();
        bdapLinkTableModel->refreshPendingRequest();
    }
} //updateBDAPLists

//Users tab =========================================================================
void BdapPage::listAllUsers()
{
    evaluateTransactionButtons();

    bdapAccountTableModel->refreshUsers();

} //listAllUsers

void BdapPage::addUser()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create BDAP objects while wallet is not synced."));
        return;
    }

    BdapAddUserDialog dlg(this,model->getOptionsModel()->getDisplayUnit());
    dlg.exec();
    if (dlg.result() == 1) {
        bdapAccountTableModel->refreshUsers();
    }
} //addUser

void BdapPage::getUserDetails(int row, int column)
{
    BdapUserDetailDialog dlg(this,BDAP::ObjectType::BDAP_USER,ui->tableWidget_Users->item(row,1)->text().toStdString());
    dlg.setWindowTitle(QObject::tr("BDAP User Detail"));
    dlg.exec();
} //getUserDetails

void BdapPage::addLink()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create BDAP objects while wallet is not synced."));
        return;
    }

    BdapAddLinkDialog dlg(this, model->getOptionsModel()->getDisplayUnit());
    dlg.exec();

    if (dlg.result() == 1) {
        bdapLinkTableModel->refreshAll();
    }
    
} //addLink

void BdapPage::acceptLink()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create BDAP objects while wallet is not synced."));
        return;
    }

    QMessageBox::StandardButton reply;
    std::string requestor = "";
    std::string recipient = "";
    std::string displayedMessage = "";

    QItemSelectionModel* selectionModel = ui->tableWidgetPendingAccept->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : -1;

    if (nSelectedRow == -1) return; //do nothing if no rows are selected

    requestor = ui->tableWidgetPendingAccept->item(nSelectedRow,0)->text().toStdString();
    recipient = ui->tableWidgetPendingAccept->item(nSelectedRow,1)->text().toStdString();
    displayedMessage = "Are you sure you want to confirm link from \"" + requestor + "\" to \"" + recipient + "\"?"; //std::to_string(nSelectedRow);

    reply = QMessageBox::question(this, QObject::tr("Confirm Accept Link"), QObject::tr(displayedMessage.c_str()), QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        if (!bdapFeesPopup(this,OP_BDAP_NEW,OP_BDAP_LINK_ACCEPT,BDAP::ObjectType::BDAP_LINK_ACCEPT,model->getOptionsModel()->getDisplayUnit())) {
            return;
        }

        executeLinkTransaction(LinkActions::LINK_ACCEPT, requestor, recipient);
    }

} //acceptLink

void BdapPage::getLinkDetails(int row, int column)
{
    std::string requestor = "";
    std::string recipient = "";
    std::string displayedMessage = "";
    LinkActions actionType = LinkActions::LINK_DEFAULT;

    //figure out which table called us
    QTableWidget* tableSource = qobject_cast<QTableWidget*>(sender());

    requestor = tableSource->item(row,0)->text().toStdString();
    recipient = tableSource->item(row,1)->text().toStdString();

    if (tableSource == ui->tableWidgetPendingAccept) actionType = LinkActions::LINK_PENDING_ACCEPT_DETAIL;
    else if (tableSource == ui->tableWidgetPendingRequest) actionType = LinkActions::LINK_PENDING_REQUEST_DETAIL;
    else if (tableSource == ui->tableWidgetComplete) actionType = LinkActions::LINK_COMPLETE_DETAIL;

    executeLinkTransaction(actionType, requestor, recipient);

} //linkDetails

void BdapPage::deleteUser()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create/modify BDAP objects while wallet is not synced."));
        return;
    }

    QMessageBox::StandardButton reply;
    std::string account = "";
    std::string displayedMessage = "";

    QItemSelectionModel* selectionModel = ui->tableWidget_Users->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : -1;

    if (nSelectedRow == -1) return; //do nothing if no rows are selected

    account = ui->tableWidget_Users->item(nSelectedRow,1)->text().toStdString();
    displayedMessage = "Are you sure you want to delete \"" + account + "\""; //std::to_string(nSelectedRow);

    reply = QMessageBox::question(this, QObject::tr("Confirm Delete Account"), QObject::tr(displayedMessage.c_str()), QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        executeDeleteAccount(account, BDAP::ObjectType::BDAP_USER);
    }

} //deleteUser

void BdapPage::updateUser()
{
    if (!dynodeSync.IsBlockchainSynced())  {
        QMessageBox::information(this, QObject::tr("Wallet not synced"), QObject::tr("Cannot create/modify BDAP objects while wallet is not synced."));
        return;
    }

    std::string account = "";
    std::string commonName = "";
    std::string expirationDate = "";
    
    QItemSelectionModel* selectionModel = ui->tableWidget_Users->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : -1;

    if (nSelectedRow == -1) return; //do nothing if no rows are selected

    account = ui->tableWidget_Users->item(nSelectedRow,1)->text().toStdString();
    commonName = ui->tableWidget_Users->item(nSelectedRow,0)->text().toStdString();
    expirationDate = ui->tableWidget_Users->item(nSelectedRow,2)->text().toStdString();

    BdapUpdateAccountDialog dlg(this,BDAP::ObjectType::BDAP_USER,account,commonName,expirationDate,model->getOptionsModel()->getDisplayUnit());
    dlg.setWindowTitle(QObject::tr("Update BDAP User"));
    
    dlg.exec();
    if (dlg.result() == 1) {
        bdapAccountTableModel->refreshUsers();
    }
} //updateUser

void BdapPage::executeDeleteAccount(std::string account, BDAP::ObjectType accountType) {

    std::string objectID = "";
    std::vector<std::string> results;
    std::string outputmessage = "";

    boost::split(results, account, [](char c){return c == '@';});

    if (results.size() > 0) {
        objectID = results[0];
    }

    JSONRPCRequest jreq;
    std::vector<std::string> params;

    params.push_back(objectID);

    switch (accountType) {
        case (BDAP::ObjectType::BDAP_USER):
            jreq.params = RPCConvertValues("deleteuser", params);
            jreq.strMethod = "deleteuser";
            break;
        case (BDAP::ObjectType::BDAP_GROUP):
            jreq.params = RPCConvertValues("deletegroup", params);
            jreq.strMethod = "deletegroup";
            break;
        default:
            jreq.params = RPCConvertValues("deleteuser", params);
            jreq.strMethod = "deleteuser";
            break;
    } //end switch

    try {
        UniValue result = tableRPC.execute(jreq);

        outputmessage = result.getValues()[0].get_str();
        BdapUserDetailDialog dlg(this,accountType,"",result,true);

        if (accountType == BDAP::ObjectType::BDAP_USER) {
            dlg.setWindowTitle(QObject::tr("Successfully deleted user"));
        } else  { //only other option for now is group
            dlg.setWindowTitle(QObject::tr("Successfully deleted group"));
        }; //end accountType if

        dlg.exec();
        if (accountType == BDAP::ObjectType::BDAP_USER) bdapAccountTableModel->refreshUsers();
        else if (accountType == BDAP::ObjectType::BDAP_GROUP) bdapAccountTableModel->refreshGroups();

        return;
    } catch (const UniValue& objError) {
        std::string message = find_value(objError, "message").get_str();
        outputmessage = message;
    } catch (const std::exception& e) {
        outputmessage = e.what();
    }

    QMessageBox::critical(this, "BDAP Error", QObject::tr(outputmessage.c_str()));

} //executeDeleteAccount

void BdapPage::executeLinkTransaction(LinkActions actionType, std::string requestor, std::string recipient) {

    std::string outputmessage = "";
    bool displayMessage = false;
    JSONRPCRequest jreq;
    std::vector<std::string> params;

    switch (actionType) {
        case (LinkActions::LINK_ACCEPT):
            params.push_back("accept");
            params.push_back(recipient);            
            params.push_back(requestor);            
            jreq.params = RPCConvertValues("link", params);
            jreq.strMethod = "link";
            displayMessage = true;
            break;
        case (LinkActions::LINK_PENDING_ACCEPT_DETAIL):
            params.push_back("pending");
            params.push_back("accept");
            params.push_back(requestor);            
            params.push_back(recipient);            
            jreq.params = RPCConvertValues("link", params);
            jreq.strMethod = "link";
            break;
        case (LinkActions::LINK_PENDING_REQUEST_DETAIL):
            params.push_back("pending");
            params.push_back("request");
            params.push_back(requestor);            
            params.push_back(recipient);            
            jreq.params = RPCConvertValues("link", params);
            jreq.strMethod = "link";
            break;
        case (LinkActions::LINK_COMPLETE_DETAIL):
            params.push_back("complete");
            params.push_back(requestor);            
            params.push_back(recipient);            
            jreq.params = RPCConvertValues("link", params);
            jreq.strMethod = "link";
            break;
        default:
            params.push_back("complete");
            jreq.params = RPCConvertValues("link", params);
            jreq.strMethod = "link";
            break;
    } //end switch

    try {
        UniValue resultToPass = UniValue(UniValue::VOBJ);
        
        UniValue result = tableRPC.execute(jreq);

        //NOTE: this payload is a list of details that contains one item for everything (so far) EXCEPT for LINK_ACCEPT
        if (actionType != LinkActions::LINK_ACCEPT) {
            resultToPass = result[0];
        } else {
            resultToPass = result;
        }

        BdapLinkDetailDialog dlg(this,actionType,"","",resultToPass,displayMessage);

        if (actionType == LinkActions::LINK_ACCEPT) {
            dlg.setWindowTitle(QObject::tr("Successfully accepted link"));
        } else if (actionType == LinkActions::LINK_PENDING_ACCEPT_DETAIL) { 
            dlg.setWindowTitle(QObject::tr("BDAP Pending Accept Link Detail"));
        } else if (actionType == LinkActions::LINK_PENDING_REQUEST_DETAIL) {
            dlg.setWindowTitle(QObject::tr("BDAP Pending Request Link Detail"));
        } else if (actionType == LinkActions::LINK_COMPLETE_DETAIL) {
            dlg.setWindowTitle(QObject::tr("BDAP Complete Link Detail"));
        } //end actionType if

        dlg.exec();

        if (actionType == LinkActions::LINK_ACCEPT) bdapLinkTableModel->refreshAll();

        return;
    } catch (const UniValue& objError) {
        std::string message = find_value(objError, "message").get_str();
        outputmessage = message;
        QMessageBox::critical(this, "BDAP Error", QObject::tr(outputmessage.c_str()));
    } catch (const std::exception& e) {
        outputmessage = e.what();
        QMessageBox::critical(this, "BDAP Error", QObject::tr(outputmessage.c_str()));
    }

} //executeLinkTransaction

BdapAccountTableModel* BdapPage::getBdapAccountTableModel()
{
    return bdapAccountTableModel;
}

BdapLinkTableModel* BdapPage::getBdapLinkTableModel()
{
    return bdapLinkTableModel;
}

QTableWidget* BdapPage::getCompleteTable() 
{ 
    return ui->tableWidgetComplete; 
}

QTableWidget* BdapPage::getPendingAcceptTable() 
{ 
    return ui->tableWidgetPendingAccept; 
    
}

QTableWidget* BdapPage::getPendingRequestTable() 
{ 
    return ui->tableWidgetPendingRequest; 
    
}

QTableWidget* BdapPage::getUserTable() 
{ 
    return ui->tableWidget_Users; 
}

QTableWidget* BdapPage::getGroupTable() 
{ 
    return ui->tableWidget_Groups; 
}

QLabel* BdapPage::getUserStatus()
{
    return ui->labelUserStatus;
}

QLabel* BdapPage::getLinkCompleteRecords()
{
    return ui->labelCompleteRecords;
}

QLabel* BdapPage::getPendingAcceptRecords()
{
    return ui->labelPARecords;
}

QLabel* BdapPage::getPendingRequestRecords()
{
    return ui->labelPRRecords;
}

QLabel* BdapPage::getGroupStatus()
{
    return ui->labelGroupStatus;
}

int BdapPage::getCurrentIndex() 
{ 
    return ui->tabWidget->currentIndex(); 
}

bool BdapPage::getMyUserCheckBoxChecked() 
{ 
    return ui->checkBoxMyUsers->isChecked(); 
}

bool BdapPage::getMyGroupCheckBoxChecked() 
{ 
    return ui->checkBoxMyGroups->isChecked(); 
}

std::string BdapPage::getCommonUserSearch()
{
    return ui->lineEditUserCommonNameSearch->text().toStdString();
}

std::string BdapPage::getPathUserSearch()
{
    return ui->lineEditUserFullPathSearch->text().toStdString();
}

std::string BdapPage::getCommonGroupSearch()
{
    return ui->lineEditGroupCommonNameSearch->text().toStdString();
}

std::string BdapPage::getPathGroupSearch()
{
    return ui->lineEditGroupFullPathSearch->text().toStdString();
}

std::string BdapPage::getCompleteRequestorSearch()
{
    return ui->lineEditCompleteRequestorSearch->text().toStdString();
}

std::string BdapPage::getCompleteRecipientSearch()
{
    return ui->lineEditCompleteRecipientSearch->text().toStdString();
}

std::string BdapPage::getPARequestorSearch()
{
    return ui->lineEditPARequestorSearch->text().toStdString();
}

std::string BdapPage::getPARecipientSearch()
{
    return ui->lineEditPARecipientSearch->text().toStdString();
}

std::string BdapPage::getPRRequestorSearch()
{
    return ui->lineEditPRRequestorSearch->text().toStdString();
}

std::string BdapPage::getPRRecipientSearch()
{
    return ui->lineEditPRRecipientSearch->text().toStdString();
}