#include "bdapadduserdialog.h"
#include "ui_bdapadduserdialog.h"

#include "guiutil.h"

#include <stdio.h>


BdapAddUserDialog::BdapAddUserDialog(QWidget *parent) : QDialog(parent),
                                                        ui(new Ui::BdapAddUserDialog)
{
    ui->setupUi(this);

    connect(ui->addUser, SIGNAL(clicked()), this, SLOT(goAddUser()));
    connect(ui->cancel, SIGNAL(clicked()), this, SLOT(goCancel()));
 

}

BdapAddUserDialog::~BdapAddUserDialog()
{
    delete ui;
}


void BdapAddUserDialog::goAddUser()
{
    ui->lineEdit_userID->setPlaceholderText("Add User");
} //goAddUser

void BdapAddUserDialog::goCancel()
{
    ui->lineEdit_userID->setPlaceholderText("Cancel");
    QDialog::reject(); //cancelled
} //goAddUser














