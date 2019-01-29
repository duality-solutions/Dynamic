#include "bdappage.h"
#include "ui_bdappage.h"
#include "bdapadduserdialog.h"
#include "guiutil.h"
#include "walletmodel.h"
#include "bdapaccounttablemodel.h"

#include "rpcregister.h" //NEED TO MOVE
#include "rpcserver.h" //NEED TO MOVE
#include "rpcclient.h" //NEED TO MOVE

#include <stdio.h>
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



    ui->lineEditUserCommonNameSearch->setFixedWidth(275);
    ui->lineEditUserFullPathSearch->setFixedWidth(200);
    ui->lineEditUserCommonNameSearch->setPlaceholderText("Enter common name to search");
    ui->lineEditUserFullPathSearch->setPlaceholderText("Enter full object path to search");

    ui->lineEditGroupCommonNameSearch->setFixedWidth(275);
    ui->lineEditGroupFullPathSearch->setFixedWidth(200);
    ui->lineEditGroupCommonNameSearch->setPlaceholderText("Enter common name to search");
    ui->lineEditGroupFullPathSearch->setPlaceholderText("Enter full object path to search");

    //Users tab
    connect(ui->pushButton_All, SIGNAL(clicked()), this, SLOT(listAllUsers()));
    connect(ui->addUser, SIGNAL(clicked()), this, SLOT(addUser()));
    connect(ui->deleteUser, SIGNAL(clicked()), this, SLOT(deleteUser()));

    //Groups tab
    connect(ui->pushButton_AllGroups, SIGNAL(clicked()), this, SLOT(listAllGroups()));
    connect(ui->addGroup, SIGNAL(clicked()), this, SLOT(addGroup()));
    connect(ui->deleteGroup, SIGNAL(clicked()), this, SLOT(deleteGroup()));


   LogPrintf("DEBUGGER TABLE 1--%s %s-- \n", __func__, ui->tableWidget_Users->rowCount());
   LogPrintf("DEBUGGER TABLENAME 1--%s %s-- \n", __func__, ui->tableWidget_Users->objectName().toStdString());


    //bdapAccountTableModel = new BdapAccountTableModel(this);



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

    bdapAccountTableModel = new BdapAccountTableModel(this);

    LogPrintf("DEBUGGER TAB --%s %s-- \n", __func__, ui->tabWidget->currentIndex());

} //listAllGroups

void BdapPage::listMyGroups()
{
    //ui->lineEdit_GroupSearch->setPlaceholderText("My Groups");
} //listMyGroups

void BdapPage::addGroup()
{
    //ui->lineEdit_GroupSearch->setPlaceholderText("Add Group");
} //addGroup

void BdapPage::deleteGroup()
{
    //ui->lineEdit_GroupSearch->setPlaceholderText("Delete Group");
} //deleteGroup



//Users tab =========================================================================
void BdapPage::listAllUsers()
{
    //ui->lineEdit_UserSearch->setPlaceholderText("All Users");

    bdapAccountTableModel = new BdapAccountTableModel(this);

    LogPrintf("DEBUGGER TAB --%s %s-- \n", __func__, ui->tabWidget->currentIndex());


} //listAllUsers

void BdapPage::listMyUsers()
{
    //ui->lineEdit_UserSearch->setPlaceholderText("My Users");
} //listMyUsers

void BdapPage::addUser()
{
    //ui->lineEdit_UserSearch->setPlaceholderText("Add User");
    BdapAddUserDialog dlg(this);
    //connect(&dlg, SIGNAL(cmdToConsole(QString)),rpcConsole, SIGNAL(cmdRequest(QString)));
    dlg.exec();
} //addUser

void BdapPage::deleteUser()
{
    //ui->lineEdit_UserSearch->setPlaceholderText("Delete User");
} //deleteUser

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


int BdapPage::getCurrentIndex() 
{ 
    return ui->tabWidget->currentIndex(); 
}








