// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdapuserdetaildialog.h"
#include "ui_bdapuserdetaildialog.h"

#include "guiutil.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"
#include "util.h"

#include <stdio.h>

#include <boost/algorithm/string.hpp>


BdapUserDetailDialog::BdapUserDetailDialog(QWidget *parent, BDAP::ObjectType accountType, const std::string& accountID, const UniValue& resultinput, bool displayInfo) : QDialog(parent),
                                                        ui(new Ui::BdapUserDetailDialog)
{
    ui->setupUi(this);

    ui->labelinfoHeader->setVisible(displayInfo);
    ui->labelinfoHeader->setText(QString::fromStdString(TRANSACTION_MESSAGE));

    connect(ui->pushButtonOK, SIGNAL(clicked()), this, SLOT(goCancel()));

    populateValues(accountType,accountID,resultinput);

}

BdapUserDetailDialog::~BdapUserDetailDialog()
{
    delete ui;
}


void BdapUserDetailDialog::populateValues(BDAP::ObjectType accountType, const std::string& accountID, const UniValue& resultinput)
{
    std::vector<std::string> results;
    std::string objectID = "";
    std::string keyName = "";
    std::string commonName = "";
    std::string walletAddress = "";
    std::string publicKey = "";
    std::string linkAddress = "";
    std::string txId = "";
    std::string expirationDate = "";
    std::string expired = "";
    std::string timeValue = "";
    std::string fullPath = "";
    std::string outputmessage = "";

    UniValue result = UniValue(UniValue::VOBJ);
 
    if (resultinput.size() == 0) {
        JSONRPCRequest jreq;
        std::vector<std::string> params;

        boost::split(results, accountID, [](char c){return c == '@';});

        if (results.size() > 0) {
            objectID = results[0];
            //ui->lineEditCommonName->setText(QString::fromStdString(results[0]));
        }

        params.push_back(objectID);

        switch (accountType) {
            case (BDAP::ObjectType::BDAP_USER):
                jreq.params = RPCConvertValues("getuserinfo", params);
                jreq.strMethod = "getuserinfo";
                break;
            case (BDAP::ObjectType::BDAP_GROUP):
                jreq.params = RPCConvertValues("getgroupinfo", params);
                jreq.strMethod = "getgroupinfo";
                break;
            default:
                jreq.params = RPCConvertValues("getuserinfo", params);
                jreq.strMethod = "getuserinfo";
                break;
        } //end switch

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

    } //if resultinput.size() = 0
    else {
        result = resultinput;
    }



    for (size_t i {0} ; i < result.size() ; ++i) {
        keyName = "";
        keyName = result.getKeys()[i];
        if (keyName == "common_name") commonName = result.getValues()[i].get_str();
        if (keyName == "object_full_path") fullPath = result.getValues()[i].get_str();
        if (keyName == "wallet_address") walletAddress = result.getValues()[i].get_str();
        if (keyName == "dht_publickey") publicKey = result.getValues()[i].get_str();
        if (keyName == "link_address") linkAddress = result.getValues()[i].get_str();
        if (keyName == "txid") txId = result.getValues()[i].get_str();
        if (keyName == "expires_on") expirationDate = DateTimeStrFormat("%Y-%m-%d", result.getValues()[i].get_int64());
        if (keyName == "expired") expired = ((result.getValues()[i].get_bool())?"true":"false");
        if (keyName == "time") timeValue = DateTimeStrFormat("%Y-%m-%d %H:%M", result.getValues()[i].get_int64());
    } //for i

    ui->lineEditCommonName->setText(QString::fromStdString(commonName));
    ui->lineEditPath->setText(QString::fromStdString(fullPath));
    ui->lineEditWalletAddress->setText(QString::fromStdString(walletAddress));
    ui->lineEditPublicKey->setText(QString::fromStdString(publicKey));
    ui->lineEditLinkAddress->setText(QString::fromStdString(linkAddress));
    ui->lineEditTXID->setText(QString::fromStdString(txId));
    ui->lineEditExpirationDate->setText(QString::fromStdString(expirationDate));
    ui->lineEditExpired->setText(QString::fromStdString(expired));
    ui->lineEditTime->setText(QString::fromStdString(timeValue));

} //populateValues


void BdapUserDetailDialog::goCancel()
{
    QDialog::reject(); //cancelled
} //goCancel




















