// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdapfeespopup.h"
#include "bdapupdateaccountdialog.h"
#include "bdapuserdetaildialog.h"
#include "ui_bdapupdateaccountdialog.h"

#include "bdap/fees.h"
#include "guiutil.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"
#include "util.h"

#include <stdio.h>

#include <boost/algorithm/string.hpp>

BdapUpdateAccountDialog::BdapUpdateAccountDialog(QWidget *parent, BDAP::ObjectType accountType, std::string account, std::string commonName, std::string expirationDate, int DynamicUnits) : QDialog(parent),
                                                        ui(new Ui::BdapUpdateAccountDialog)
{
    //By default, accountType is USER. so only change stuff if different
    ui->setupUi(this);
    inputAccountType = accountType;
    nDynamicUnits = DynamicUnits;

    std::string objectID = "";
    std::vector<std::string> results;

    boost::split(results, account, [](char c){return c == '@';});

    if (results.size() > 0) {
        objectID = results[0];
    }

    if (inputAccountType == BDAP::ObjectType::BDAP_GROUP) {
        ui->labelID->setText(QString::fromStdString("Group ID:"));
    } //if inputAccountType

    ui->lineEditID->setText(QString::fromStdString(objectID));
    ui->lineEditCommonName->setText(QString::fromStdString(commonName));
    ui->labelExpirationDateInfo->setText(QString::fromStdString(expirationDate)); 

    connect(ui->pushButtonUpdate, SIGNAL(clicked()), this, SLOT(updateAccount()));
    connect(ui->pushButtonCancel, SIGNAL(clicked()), this, SLOT(goCancel()));
    connect(ui->pushButtonOK, SIGNAL(clicked()), this, SLOT(goCancel()));
 
    ui->labelErrorMsg->setVisible(false);
    ui->pushButtonOK->setVisible(false);
}

BdapUpdateAccountDialog::~BdapUpdateAccountDialog()
{
    delete ui;
}

void BdapUpdateAccountDialog::updateAccount()
{
    std::string accountID = "";
    std::string commonName = "";
    std::string registrationMonths = "";
    JSONRPCRequest jreq;
    std::vector<std::string> params;
    int32_t regMonths = 0; //DEFAULT_REGISTRATION_MONTHS;

    std::string outputmessage = "";

    accountID = ui->lineEditID->text().toStdString();
    commonName = ui->lineEditCommonName->text().toStdString();
    registrationMonths = ui->lineEditRegistrationMonths->text().toStdString();

    if (registrationMonths.length() >> 0) 
    {
        try {
            regMonths = std::stoi(registrationMonths);
        } catch (std::exception& e) {
            QMessageBox::critical(this, QObject::tr("BDAP Error"),QObject::tr("Registration months must be a number."));
            return;
        }
        
        CAmount tmpAmount;
        if ( (!ParseFixedPoint(registrationMonths, 0, &tmpAmount)) || (regMonths < 0) ) {
            QMessageBox::critical(this, QObject::tr("BDAP Error"),QObject::tr("Additional months cannot be less than zero, and must be a whole number (no decimals)."));
            return;
        }
    }

    ui->lineEditID->setReadOnly(true);
    ui->lineEditCommonName->setReadOnly(true);
    ui->lineEditRegistrationMonths->setReadOnly(true);

    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Text,Qt::darkGray);
    ui->lineEditID->setPalette(*palette);
    ui->lineEditCommonName->setPalette(*palette);
    ui->lineEditRegistrationMonths->setPalette(*palette);

    ui->labelErrorMsg->setVisible(true);
    ui->pushButtonOK->setVisible(true);

    ui->pushButtonUpdate->setVisible(false);
    ui->pushButtonCancel->setVisible(false);
    
    params.push_back(accountID);
    params.push_back(commonName);

    //TODO: Front end GUI changed this parameter to be ADDITIONAL DAYS from current expiration date.
    //RPC command needs to be updated to reflect this change (so no entry, means expiration date stays the same. Value would mean number of days extended)
    if (registrationMonths.length() >> 0) params.push_back(std::to_string(regMonths));

    if (inputAccountType == BDAP::ObjectType::BDAP_USER) {
        jreq.params = RPCConvertValues("updateuser", params);
        jreq.strMethod = "updateuser";

    } else { //only other option for now is group
        jreq.params = RPCConvertValues("updategroup", params);
        jreq.strMethod = "updategroup";

    } //end inputAccountType if

    if (!bdapFeesPopup(this,OP_BDAP_MODIFY,OP_BDAP_ACCOUNT_ENTRY,inputAccountType,nDynamicUnits,regMonths)) {
        goClose();
        return;
    }

    try {
        UniValue result = tableRPC.execute(jreq);

        outputmessage = result.getValues()[0].get_str();
        BdapUserDetailDialog dlg(this,inputAccountType,"",result,true);

        if (inputAccountType == BDAP::ObjectType::BDAP_USER) {
            dlg.setWindowTitle(QObject::tr("Successfully updated user"));
        } else  { //only other option for now is group
           dlg.setWindowTitle(QObject::tr("Successfully updated group"));
        }; //end inputAccountType if

        dlg.exec();
        goClose();
        return;
    } catch (const UniValue& objError) {
        std::string message = find_value(objError, "message").get_str();
        outputmessage = ignoreErrorCode(message);
    } catch (const std::exception& e) {
        outputmessage = e.what();
    }

    ui->labelErrorMsg->setText(QObject::tr(outputmessage.c_str()));
} //updateAccount

void BdapUpdateAccountDialog::goCancel()
{
    QDialog::reject(); //cancelled
} //goCancel

void BdapUpdateAccountDialog::goClose()
{
    QDialog::accept(); //accepted
} //goClose

std::string BdapUpdateAccountDialog::ignoreErrorCode(const std::string input)
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