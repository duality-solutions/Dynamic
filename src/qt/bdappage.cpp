#include "bdappage.h"
#include "ui_bdappage.h"
#include "bdapadduserdialog.h"
#include "bdapuserdetaildialog.h"
#include "guiutil.h"
#include "walletmodel.h"
#include "bdapaccounttablemodel.h"

#include "rpcregister.h" //NEED TO MOVE
#include "rpcserver.h" //NEED TO MOVE
#include "rpcclient.h" //NEED TO MOVE

#include <stdio.h>
#include <boost/algorithm/string.hpp>

#include <QTableWidget>

BdapPage::BdapPage(const PlatformStyle* platformStyle, QWidget* parent) : QWidget(parent),
                                                                            ui(new Ui::BdapPage),
                                                                            bdapAccountTableModel(0)
{
    ui->setupUi(this);
    
    //Initialize QWidgetTable names
    // if (ui->tableWidget_Users->objectName().isEmpty())
    //     ui->tableWidget_Users->setObjectName(QStringLiteral("BDAPUsersTable"));

    // if (ui->tableWidget_Groups->objectName().isEmpty())
    //     ui->tableWidget_Groups->setObjectName(QStringLiteral("BDAPGroupsTable"));


    bdapAccountTableModel = new BdapAccountTableModel(this);

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
    connect(ui->deleteUser, SIGNAL(clicked()), this, SLOT(deleteUser()));

    connect(ui->checkBoxMyUsers, SIGNAL(clicked()), this, SLOT(listAllUsers()));
    connect(ui->lineEditUserCommonNameSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllUsers()));
    connect(ui->lineEditUserFullPathSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllUsers()));

    connect(ui->tableWidget_Users, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getUserDetails(int,int)));


    //Groups tab
    connect(ui->pushButton_AllGroups, SIGNAL(clicked()), this, SLOT(listAllGroups()));
    connect(ui->addGroup, SIGNAL(clicked()), this, SLOT(addGroup()));
    connect(ui->deleteGroup, SIGNAL(clicked()), this, SLOT(deleteGroup()));

    connect(ui->checkBoxMyGroups, SIGNAL(clicked()), this, SLOT(listAllGroups()));
    connect(ui->lineEditGroupCommonNameSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllGroups()));
    connect(ui->lineEditGroupFullPathSearch, SIGNAL(textChanged(const QString &)), this, SLOT(listAllGroups()));


    connect(ui->tableWidget_Groups, SIGNAL(cellDoubleClicked(int,int)), this, SLOT(getGroupDetails(int,int)));

   LogPrintf("DEBUGGER TABLE 1--%s %s-- \n", __func__, ui->tableWidget_Users->rowCount());
   LogPrintf("DEBUGGER TABLENAME 1--%s %s-- \n", __func__, ui->tableWidget_Users->objectName().toStdString());


}

BdapPage::~BdapPage()
{
    delete ui;
}

void BdapPage::setModel(WalletModel* model)
{
    this->model = model;
}

//Groups tab ========================================================================
void BdapPage::listAllGroups()
{
    //ui->lineEdit_GroupSearch->setPlaceholderText("All Groups");

    //bdapAccountTableModel = new BdapAccountTableModel(this);

    bdapAccountTableModel->refreshGroups();

    LogPrintf("DEBUGGER TAB --%s %s-- \n", __func__, ui->tabWidget->currentIndex());

} //listAllGroups

void BdapPage::listMyGroups()
{
    //ui->lineEdit_GroupSearch->setPlaceholderText("My Groups");
} //listMyGroups


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

void BdapPage::getGroupDetails(int row, int column)
{
    //LogPrintf("DEBUGGER USERDETAIL --%s made it here-- \n", __func__);

    BdapUserDetailDialog dlg(this,BDAP::ObjectType::BDAP_GROUP,ui->tableWidget_Groups->item(row,1)->text().toStdString());
    dlg.setWindowTitle(QString::fromStdString("BDAP Group Detail"));
    dlg.exec();
} //getGroupDetails



//Users tab =========================================================================
void BdapPage::listAllUsers()
{

    bdapAccountTableModel->refreshUsers();

    LogPrintf("DEBUGGER TAB --%s %s-- \n", __func__, ui->tabWidget->currentIndex());


} //listAllUsers

void BdapPage::listMyUsers()
{

} //listMyUsers

void BdapPage::addUser()
{
    BdapAddUserDialog dlg(this);
    //connect(&dlg, SIGNAL(cmdToConsole(QString)),rpcConsole, SIGNAL(cmdRequest(QString)));
    dlg.exec();
} //addUser




void BdapPage::getUserDetails(int row, int column)
{
    LogPrintf("DEBUGGER USERDETAIL --%s made it here-- \n", __func__);

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


        UniValue rpc_result(UniValue::VOBJ);

        try {
            UniValue result = tableRPC.execute(jreq);

            outputmessage = result.getValues()[0].get_str();
            BdapUserDetailDialog dlg(this,accountType,"",result);

            if (accountType == BDAP::ObjectType::BDAP_USER) {
                dlg.setWindowTitle(QString::fromStdString("Successfully deleted user"));
            } else  { //only other option for now is group
                dlg.setWindowTitle(QString::fromStdString("Successfully deleted group"));
            }; //end accountType if

            dlg.exec();
            return;
        } catch (const UniValue& objError) {
            rpc_result = JSONRPCReplyObj(NullUniValue, objError, jreq.id);
            LogPrintf("DEBUGGER ADDUSER ERROR1--%s-- \n", __func__);
            std::string message = find_value(objError, "message").get_str();
            LogPrintf("DEBUGGER ADDUSER ERROR1--%s %s-- \n", __func__, message);
            outputmessage = message;
        } catch (const std::exception& e) {
            rpc_result = JSONRPCReplyObj(NullUniValue,
            JSONRPCError(RPC_PARSE_ERROR, e.what()), jreq.id);
            LogPrintf("DEBUGGER ADDUSER ERROR2--%s-- \n", __func__);
            outputmessage = e.what();
        }

        QMessageBox::critical(this, "BDAP Error", QString::fromStdString(outputmessage));

} //executeDeleteAccount


BdapAccountTableModel* BdapPage::getBdapAccountTableModel()
{
    return bdapAccountTableModel;
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







