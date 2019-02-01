#include "bdapuserdetaildialog.h"
#include "ui_bdapuserdetaildialog.h"

#include "guiutil.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"
#include "util.h"

#include <stdio.h>
#include <boost/algorithm/string.hpp>


BdapUserDetailDialog::BdapUserDetailDialog(QWidget *parent, BDAP::ObjectType accountType, const std::string& accountID) : QDialog(parent),
                                                        ui(new Ui::BdapUserDetailDialog)
{
    ui->setupUi(this);

    connect(ui->pushButtonOK, SIGNAL(clicked()), this, SLOT(goCancel()));

    populateValues(accountType,accountID);

}

BdapUserDetailDialog::~BdapUserDetailDialog()
{
    delete ui;
}


void BdapUserDetailDialog::populateValues(BDAP::ObjectType accountType, const std::string& accountID)
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

    UniValue result = tableRPC.execute(jreq);

    LogPrintf("DEBUGGER USERDETAIL --%s %s %s-- \n", __func__, objectID, result.size());

    for (size_t i {0} ; i < result.size() ; ++i) {
        keyName = "";
        keyName = result.getKeys()[i];
        if (keyName == "common_name") commonName = result.getValues()[i].get_str();
        if (keyName == "wallet_address") walletAddress = result.getValues()[i].get_str();
        if (keyName == "dht_publickey") publicKey = result.getValues()[i].get_str();
        if (keyName == "link_address") linkAddress = result.getValues()[i].get_str();
        if (keyName == "txid") txId = result.getValues()[i].get_str();
        if (keyName == "expires_on") expirationDate = DateTimeStrFormat("%Y-%m-%d", result.getValues()[i].get_int64());
        if (keyName == "expired") expired = ((result.getValues()[i].get_bool())?"true":"false");
        if (keyName == "time") timeValue = DateTimeStrFormat("%Y-%m-%d %H:%M", result.getValues()[i].get_int64());
    } //for i

    ui->lineEditCommonName->setText(QString::fromStdString(commonName));
    ui->lineEditPath->setText(QString::fromStdString(accountID));
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




















