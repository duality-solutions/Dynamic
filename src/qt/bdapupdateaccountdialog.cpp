// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdapupdateaccountdialog.h"
#include "ui_bdapupdateaccountdialog.h"
#include "bdapuserdetaildialog.h"

#include "guiutil.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"
#include "util.h"

#include <stdio.h>

#include <boost/algorithm/string.hpp>


BdapUpdateAccountDialog::BdapUpdateAccountDialog(QWidget *parent, BDAP::ObjectType accountType, std::string account, std::string commonName, std::string expirationDate) : QDialog(parent),
                                                        ui(new Ui::BdapUpdateAccountDialog)
{
    //By default, accountType is USER. so only change stuff if different
    
    ui->setupUi(this);
    inputAccountType = accountType;

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
    ui->lineEditRegistrationDays->setPlaceholderText(QString::fromStdString("Expiration date: " + expirationDate));


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
    std::string registrationDays = "";
    JSONRPCRequest jreq;
    std::vector<std::string> params;

    std::string outputmessage = "";

    accountID = ui->lineEditID->text().toStdString();
    commonName = ui->lineEditCommonName->text().toStdString();
    registrationDays = ui->lineEditRegistrationDays->text().toStdString();

    ui->lineEditID->setReadOnly(true);
    ui->lineEditCommonName->setReadOnly(true);
    ui->lineEditRegistrationDays->setReadOnly(true);

    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Text,Qt::darkGray);
    ui->lineEditID->setPalette(*palette);
    ui->lineEditCommonName->setPalette(*palette);
    ui->lineEditRegistrationDays->setPalette(*palette);
    

    ui->labelErrorMsg->setVisible(true);
    ui->pushButtonOK->setVisible(true);

    ui->pushButtonUpdate->setVisible(false);
    ui->pushButtonCancel->setVisible(false);

    
    params.push_back(accountID);
    params.push_back(commonName);
    if (registrationDays.length() >> 0) params.push_back(registrationDays);

    if (inputAccountType == BDAP::ObjectType::BDAP_USER) {
        jreq.params = RPCConvertValues("updateuser", params);
        jreq.strMethod = "updateuser";

    } else { //only other option for now is group
        jreq.params = RPCConvertValues("updategroup", params);
        jreq.strMethod = "updategroup";

    }; //end inputAccountType if


    try {
        UniValue result = tableRPC.execute(jreq);

        outputmessage = result.getValues()[0].get_str();
        BdapUserDetailDialog dlg(this,inputAccountType,"",result,true);

        if (inputAccountType == BDAP::ObjectType::BDAP_USER) {
            dlg.setWindowTitle(QString::fromStdString("Successfully updated user"));
        } else  { //only other option for now is group
           dlg.setWindowTitle(QString::fromStdString("Successfully updated group"));
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

    ui->labelErrorMsg->setText(QString::fromStdString(outputmessage));
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














