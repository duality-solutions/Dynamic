// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdapaddlinkdialog.h"
#include "bdapfeespopup.h"
#include "bdaplinkdetaildialog.h"
#include "bdappage.h"
#include "ui_bdapaddlinkdialog.h"

#include "bdap/fees.h"
#include "guiutil.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"
#include "util.h"

#include <boost/algorithm/string.hpp>
#include <stdio.h>

BdapAddLinkDialog::BdapAddLinkDialog(QWidget *parent, int DynamicUnits) : QDialog(parent),
                                                        ui(new Ui::BdapAddLinkDialog)
{
    ui->setupUi(this);

    nDynamicUnits = DynamicUnits;

    connect(ui->pushButtonCancel, SIGNAL(clicked()), this, SLOT(goCancel()));
    connect(ui->pushButtonAddLink, SIGNAL(clicked()), this, SLOT(addLink()));

    QStringList fromList;
    QStringList toList;
    std::vector<std::string> accountListFrom; 
    std::vector<std::string> accountListTo; 

    //setup autocomplete for FROM input
    populateList(accountListFrom,LinkUserType::LINK_REQUESTOR);

    for (size_t i = 0; i < accountListFrom.size(); ++i) {
        fromList << accountListFrom[i].c_str();
    }

    autoCompleterFrom = new QCompleter(fromList, this);
    ui->lineEditRequestor->setCompleter(autoCompleterFrom);
    autoCompleterFrom->popup()->installEventFilter(this);

    //setup autocomplete for TO input
    populateList(accountListTo,LinkUserType::LINK_RECIPIENT);

    for (size_t i = 0; i < accountListTo.size(); ++i) {
        toList << accountListTo[i].c_str();
    }

    autoCompleterTo = new QCompleter(toList, this);
    ui->lineEditRecipient->setCompleter(autoCompleterTo);
    autoCompleterTo->popup()->installEventFilter(this);
}

BdapAddLinkDialog::~BdapAddLinkDialog()
{
    delete ui;
}

void BdapAddLinkDialog::populateList(std::vector<std::string> &inputList, LinkUserType userType) {

    JSONRPCRequest jreq;
    std::vector<std::string> params;
    std::string outputmessage = "";
    std::string getaccountID = "";
    std::string keyName = "";

    UniValue result = UniValue(UniValue::VOBJ);

    switch (userType) {
        case LinkUserType::LINK_REQUESTOR:
            params.push_back("users");
            jreq.params = RPCConvertValues("mybdapaccounts", params);
            jreq.strMethod = "mybdapaccounts";
            break;
        case LinkUserType::LINK_RECIPIENT:
            jreq.params = RPCConvertValues("getusers", params);
            jreq.strMethod = "getusers";
            break;
        default:
            params.push_back("users");
            jreq.params = RPCConvertValues("mybdapaccounts", params);
            jreq.strMethod = "mybdapaccounts";
            break;

    }; //switch userType

    //Handle RPC errors
    try {
        result = tableRPC.execute(jreq);
    } catch (const UniValue& objError) {
        std::string message = find_value(objError, "message").get_str();
        outputmessage = message;
        QMessageBox::critical(0, "BDAP Error", QObject::tr(outputmessage.c_str()));
        return;
    } catch (const std::exception& e) {
        outputmessage = e.what();
        QMessageBox::critical(0, "BDAP Error", QObject::tr(outputmessage.c_str()));
        return;
    }

    for (size_t i {0} ; i < result.size() ; ++i) {
        getaccountID = "";

        for (size_t j {0} ; j < result[i].size() ; ++j) {
            keyName = "";
            keyName = result[i].getKeys()[j];
            if (keyName == "object_full_path") getaccountID = getIdFromPath(result[i].getValues()[j].get_str());
        } //for loop j
            inputList.push_back(getaccountID);
    }; //for loop i

} //populateList

void BdapAddLinkDialog::addLink()
{
    std::string requestor = "";
    std::string recipient = "";
    std::string linkMessage = "";
    std::string outputmessage = "";

    JSONRPCRequest jreq;
    std::vector<std::string> params;

    requestor = ui->lineEditRequestor->text().toStdString();
    recipient = ui->lineEditRecipient->text().toStdString();
    linkMessage = ui->lineEditLinkMessage->text().toStdString();

    if ((requestor == "") || (recipient == "") || (linkMessage == "")) {
        QMessageBox::critical(this, "BDAP Add Link Error", QObject::tr("Requestor, Recipient and Link Message are required fields"));
        return;
    } //if requestor

    if (!bdapFeesPopup(this,OP_BDAP_NEW,OP_BDAP_LINK_REQUEST,BDAP::ObjectType::BDAP_LINK_REQUEST,nDynamicUnits)) {
        goClose();
        return;
    }

    params.push_back("request");
    params.push_back(requestor);
    params.push_back(recipient);
    params.push_back(linkMessage);
    jreq.params = RPCConvertValues("link", params);
    jreq.strMethod = "link";

    try {
        UniValue result = tableRPC.execute(jreq);

        outputmessage = result.getValues()[0].get_str();
        BdapLinkDetailDialog dlg(this,LinkActions::LINK_REQUEST,"","",result,true);
        dlg.setWindowTitle(QObject::tr("Successfully added link"));

        dlg.exec();
        goClose();
    } catch (const UniValue& objError) {
        std::string message = find_value(objError, "message").get_str();
        outputmessage = ignoreErrorCode(message);
        QMessageBox::critical(this, "BDAP Add Link Error", QObject::tr(outputmessage.c_str()));
        return;
    } catch (const std::exception& e) {
        outputmessage = e.what();
        QMessageBox::critical(this, "BDAP Add Link Error", QObject::tr(outputmessage.c_str()));
        return;
    }
} //addLink

void BdapAddLinkDialog::goCancel()
{
    QDialog::reject(); //cancelled
} //goCancel

void BdapAddLinkDialog::goClose()
{
    QDialog::accept(); //accepted
} //goClose

std::string BdapAddLinkDialog::ignoreErrorCode(const std::string input)
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

std::string BdapAddLinkDialog::getIdFromPath(std::string inputstring) {
    std::string returnvalue = inputstring;
    std::vector<std::string> results;

    boost::split(results, inputstring, [](char c){return c == '@';});

    if (results.size() > 0) {
        returnvalue = results[0];
    }

    return returnvalue;

} //getIdFromPath