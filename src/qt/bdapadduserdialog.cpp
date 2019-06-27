// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdapadduserdialog.h"
#include "bdapfeespopup.h"
#include "bdappage.h"
#include "bdapuserdetaildialog.h"
#include "ui_bdapadduserdialog.h"

#include "bdap/fees.h"
#include "guiutil.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"
#include "util.h"

#include <boost/algorithm/string.hpp>
#include <stdio.h>

BdapAddUserDialog::BdapAddUserDialog(QWidget *parent, int DynamicUnits, BDAP::ObjectType accountType) : QDialog(parent),
                                                        ui(new Ui::BdapAddUserDialog)
{
    //By default, accountType is USER. so only change stuff if different
    ui->setupUi(this);
    inputAccountType = accountType;
    nDynamicUnits = DynamicUnits;

    if (inputAccountType == BDAP::ObjectType::BDAP_GROUP) {

        ui->labelUserId->setText(QObject::tr("Group ID:"));
        ui->addUser->setText(QObject::tr("Add Group"));

    } //if inputAccountType

    connect(ui->addUser, SIGNAL(clicked()), this, SLOT(goAddUser()));
    connect(ui->cancel, SIGNAL(clicked()), this, SLOT(goCancel()));
    connect(ui->pushButtonOK, SIGNAL(clicked()), this, SLOT(goCancel()));
 
    ui->labelErrorMsg->setVisible(false);
    ui->pushButtonOK->setVisible(false);

}

BdapAddUserDialog::~BdapAddUserDialog()
{
    delete ui;
}

void BdapAddUserDialog::goAddUser()
{
    std::string accountID = "";
    std::string commonName = "";
    std::string registrationMonths = "";
    JSONRPCRequest jreq;
    std::vector<std::string> params;

    std::string outputmessage = "";

    int32_t regMonths = DEFAULT_REGISTRATION_MONTHS;

    accountID = ui->lineEdit_userID->text().toStdString();
    commonName = ui->lineEdit_commonName->text().toStdString();
    registrationMonths = ui->lineEdit_registrationMonths->text().toStdString();
    
    if (registrationMonths.length() >> 0) 
    {
        try {
            regMonths = std::stoi(registrationMonths);
        } catch (std::exception& e) {
            QMessageBox::critical(this, QObject::tr("BDAP Error"),QObject::tr("Registration months must be a number."));
            return;
        }

        CAmount tmpAmount;
        if ( (!ParseFixedPoint(registrationMonths, 0, &tmpAmount)) || (regMonths <= 0) ) {
            QMessageBox::critical(this, QObject::tr("BDAP Error"),QObject::tr("Registration months cannot be less than or equal to zero, and must be a whole number (no decimals)."));
            return;
        }
    }

    ui->lineEdit_userID->setReadOnly(true);
    ui->lineEdit_commonName->setReadOnly(true);
    ui->lineEdit_registrationMonths->setReadOnly(true);

    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Text,Qt::darkGray);
    ui->lineEdit_userID->setPalette(*palette);
    ui->lineEdit_commonName->setPalette(*palette);
    ui->lineEdit_registrationMonths->setPalette(*palette);

    ui->labelErrorMsg->setVisible(true);
    ui->pushButtonOK->setVisible(true);

    ui->addUser->setVisible(false);
    ui->cancel->setVisible(false);

    if (!bdapFeesPopup(this,OP_BDAP_NEW,OP_BDAP_ACCOUNT_ENTRY,inputAccountType,nDynamicUnits,regMonths)) {
        goClose();
        return;
    }

    params.push_back(accountID);
    params.push_back(commonName);
    if (registrationMonths.length() >> 0) params.push_back(std::to_string(regMonths));

    if (inputAccountType == BDAP::ObjectType::BDAP_USER) {
        jreq.params = RPCConvertValues("adduser", params);
        jreq.strMethod = "adduser";

    } else { //only other option for now is group
        jreq.params = RPCConvertValues("addgroup", params);
        jreq.strMethod = "addgroup";

    }; //end inputAccountType if

    try {
        UniValue result = tableRPC.execute(jreq);

        outputmessage = result.getValues()[0].get_str();
        BdapUserDetailDialog dlg(this,inputAccountType,"",result,true);

        if (inputAccountType == BDAP::ObjectType::BDAP_USER) {
            dlg.setWindowTitle(QObject::tr("Successfully added user"));
        } else  { //only other option for now is group
           dlg.setWindowTitle(QObject::tr("Successfully added group"));
        }; //end inputAccountType if

        dlg.exec();
        goClose();
    } catch (const UniValue& objError) {
        std::string message = find_value(objError, "message").get_str();
        outputmessage = ignoreErrorCode(message);
    } catch (const std::exception& e) {
        outputmessage = e.what();
    }

    ui->labelErrorMsg->setText(QObject::tr(outputmessage.c_str()));

} //goAddUser

void BdapAddUserDialog::goCancel()
{
    QDialog::reject(); //cancelled
} //goCancel

void BdapAddUserDialog::goClose()
{
    QDialog::accept(); //accepted
} //goClose

std::string BdapAddUserDialog::ignoreErrorCode(const std::string input)
{
    //assuming error code is in the following format: ERROR CODE - ERROR MESSAGE
    std::vector<std::string> results;
    std::string returnvalue = input;

    boost::split(results, input, [](char c){return c == '-';});

    if (results.size() > 1) {
        returnvalue = results[1];
    }

    return returnvalue;
} //ignoreErrorCode