// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdappage.h"
#include "ui_bdappage.h"
#include "bdapadduserdialog.h"
#include "bdapupdateaccountdialog.h"
#include "bdapuserdetaildialog.h"
#include "guiutil.h"
#include "walletmodel.h"
#include "bdapaccounttablemodel.h"
#include "bdaplinktablemodel.h"

#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"

#include <stdio.h>

#include <boost/algorithm/string.hpp>

#include <QTableWidget>

BdapPage::BdapPage(const PlatformStyle* platformStyle, QWidget* parent) : QWidget(parent),
                                                                            ui(new Ui::BdapPage),
                                                                            bdapAccountTableModel(0)
{
    ui->setupUi(this);
    
    evaluateTransactionButtons();

    bdapAccountTableModel = new BdapAccountTableModel(this);
    bdapLinkTableModel = new BdapLinkTableModel(this);

    ui->lineEditUserCommonNameSearch->setFixedWidth(COMMONNAME_COLWIDTH);
    ui->lineEditUserFullPathSearch->setFixedWidth(FULLPATH_COLWIDTH);
    ui->lineEditUserCommonNameSearch->setPlaceholderText("Enter common name to search");
    ui->lineEditUserFullPathSearch->setPlaceholderText("Enter object full path to search");

    ui->lineEditGroupCommonNameSearch->setFixedWidth(COMMONNAME_COLWIDTH);
    ui->lineEditGroupFullPathSearch->setFixedWidth(FULLPATH_COLWIDTH);
    ui->lineEditGroupCommonNameSearch->setPlaceholderText("Enter common name to search");
    ui->lineEditGroupFullPathSearch->setPlaceholderText("Enter object full path to search");

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
    connect(ui->pushButtonRefreshComplete, SIGNAL(clicked()), this, SLOT(listLinksComplete()));
    connect(ui->pushButtonRefreshPendingAccept, SIGNAL(clicked()), this, SLOT(listPendingAccept()));
    connect(ui->pushButtonRefreshPendingRequest, SIGNAL(clicked()), this, SLOT(listPendingRequest()));

    

}

BdapPage::~BdapPage()
{
    delete ui;
}

void BdapPage::setModel(WalletModel* model)
{
    this->model = model;
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
    BdapAddUserDialog dlg(this,BDAP::ObjectType::BDAP_GROUP);
    dlg.setWindowTitle(QString::fromStdString("Add BDAP Group"));
    dlg.exec();
} //addGroup


void BdapPage::deleteGroup()
{
    QMessageBox::StandardButton reply;
    std::string account = "";
    std::string displayedMessage = "";

    QItemSelectionModel* selectionModel = ui->tableWidget_Groups->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : -1;

    if (nSelectedRow == -1) return; //do nothing if no rows are selected

    account = ui->tableWidget_Groups->item(nSelectedRow,1)->text().toStdString();
    displayedMessage = "Are you sure you want to delete \"" + account + "\""; //std::to_string(nSelectedRow);

    reply = QMessageBox::question(this, "Confirm Delete Account", QString::fromStdString(displayedMessage), QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        executeDeleteAccount(account, BDAP::ObjectType::BDAP_GROUP);

    };
} //deleteGroup

void BdapPage::updateGroup()
{
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

    BdapUpdateAccountDialog dlg(this,BDAP::ObjectType::BDAP_GROUP,account,commonName,expirationDate);
    dlg.setWindowTitle(QString::fromStdString("Update BDAP Group"));
    
    dlg.exec();

} //updateGroup


void BdapPage::getGroupDetails(int row, int column)
{
    BdapUserDetailDialog dlg(this,BDAP::ObjectType::BDAP_GROUP,ui->tableWidget_Groups->item(row,1)->text().toStdString());
    dlg.setWindowTitle(QString::fromStdString("BDAP Group Detail"));
    dlg.exec();
} //getGroupDetails



//Users tab =========================================================================
void BdapPage::listAllUsers()
{
    evaluateTransactionButtons();

    bdapAccountTableModel->refreshUsers();

} //listAllUsers


void BdapPage::addUser()
{
    BdapAddUserDialog dlg(this);
    //connect(&dlg, SIGNAL(cmdToConsole(QString)),rpcConsole, SIGNAL(cmdRequest(QString)));
    dlg.exec();
} //addUser



void BdapPage::getUserDetails(int row, int column)
{
    BdapUserDetailDialog dlg(this,BDAP::ObjectType::BDAP_USER,ui->tableWidget_Users->item(row,1)->text().toStdString());
    dlg.setWindowTitle(QString::fromStdString("BDAP User Detail"));
    dlg.exec();
} //getUserDetails



void BdapPage::deleteUser()
{
    QMessageBox::StandardButton reply;
    std::string account = "";
    std::string displayedMessage = "";

    QItemSelectionModel* selectionModel = ui->tableWidget_Users->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : -1;

    if (nSelectedRow == -1) return; //do nothing if no rows are selected

    account = ui->tableWidget_Users->item(nSelectedRow,1)->text().toStdString();
    displayedMessage = "Are you sure you want to delete \"" + account + "\""; //std::to_string(nSelectedRow);

    reply = QMessageBox::question(this, "Confirm Delete Account", QString::fromStdString(displayedMessage), QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        executeDeleteAccount(account, BDAP::ObjectType::BDAP_USER);

    };

} //deleteUser

void BdapPage::updateUser()
{
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

    BdapUpdateAccountDialog dlg(this,BDAP::ObjectType::BDAP_USER,account,commonName,expirationDate);
    dlg.setWindowTitle(QString::fromStdString("Update BDAP User"));
    
    dlg.exec();

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
                dlg.setWindowTitle(QString::fromStdString("Successfully deleted user"));
            } else  { //only other option for now is group
                dlg.setWindowTitle(QString::fromStdString("Successfully deleted group"));
            }; //end accountType if

            dlg.exec();
            return;
        } catch (const UniValue& objError) {
            std::string message = find_value(objError, "message").get_str();
            outputmessage = message;
        } catch (const std::exception& e) {
            outputmessage = e.what();
        }

        QMessageBox::critical(this, "BDAP Error", QString::fromStdString(outputmessage));

} //executeDeleteAccount


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







