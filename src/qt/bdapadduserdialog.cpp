#include "bdapadduserdialog.h"
#include "ui_bdapadduserdialog.h"
#include "bdapuserdetaildialog.h"

#include "guiutil.h"
#include "rpcregister.h"
#include "rpcserver.h"
#include "rpcclient.h"
#include "util.h"

#include <stdio.h>
#include <boost/algorithm/string.hpp>


BdapAddUserDialog::BdapAddUserDialog(QWidget *parent) : QDialog(parent),
                                                        ui(new Ui::BdapAddUserDialog)
{
    ui->setupUi(this);

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
    JSONRPCRequest jreq;
    std::vector<std::string> params;

    std::string outputmessage = "";

    accountID = ui->lineEdit_userID->text().toStdString();
    commonName = ui->lineEdit_commonName->text().toStdString();

    ui->lineEdit_userID->setReadOnly(true);
    ui->lineEdit_commonName->setReadOnly(true);
    ui->lineEdit_registrationDays->setReadOnly(true);

    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Background,Qt::gray);
    palette->setColor(QPalette::Text,Qt::darkGray);
    ui->lineEdit_userID->setAutoFillBackground(true);
    ui->lineEdit_commonName->setAutoFillBackground(true);
    ui->lineEdit_registrationDays->setAutoFillBackground(true);
    ui->lineEdit_userID->setPalette(*palette);
    ui->lineEdit_commonName->setPalette(*palette);
    ui->lineEdit_registrationDays->setPalette(*palette);
    



    //ui->lineEdit_registrationDays->setText(QString::fromStdString(accountID));

    ui->labelErrorMsg->setVisible(true);
    ui->pushButtonOK->setVisible(true);

    ui->addUser->setVisible(false);
    ui->cancel->setVisible(false);

    
    params.push_back(accountID);
    params.push_back(commonName);
    jreq.params = RPCConvertValues("adduser", params);
    jreq.strMethod = "adduser";


    LogPrintf("DEBUGGER ADDUSER 1--%s -- \n", __func__);


/* VERSION1
    try {
        UniValue result = tableRPC.execute(jreq);
    } catch (const std::exception& e) {
        //return error("%s: Deserialize or I/O error - %s", __func__, e.what());
        error = e.what();
        LogPrintf("DEBUGGER ADDUSER 2--%s %s-- \n", __func__, error);
        return;
    }
*/


    UniValue rpc_result(UniValue::VOBJ);

    try {
        //jreq.parse(req);

        UniValue result = tableRPC.execute(jreq);

        outputmessage = result.getValues()[0].get_str();
        //rpc_result = JSONRPCReplyObj(result, NullUniValue, jreq.id);
        //LogPrintf("DEBUGGER ADDUSER 2--%s %s-- \n", __func__, rpc_result.size());
        //outputmessage = rpc_result.getValues()[0].get_str();  //std::to_string(rpc_result.size());
        //LogPrintf("DEBUGGER ADDUSER 3--%s -- \n", __func__);
        BdapUserDetailDialog dlg(this,BDAP::ObjectType::BDAP_USER,"",result);
        dlg.setWindowTitle(QString::fromStdString("Successfully added user"));
        dlg.exec();
        goClose();
    } catch (const UniValue& objError) {
        rpc_result = JSONRPCReplyObj(NullUniValue, objError, jreq.id);
        LogPrintf("DEBUGGER ADDUSER ERROR1--%s-- \n", __func__);
        std::string message = find_value(objError, "message").get_str();
        LogPrintf("DEBUGGER ADDUSER ERROR1--%s %s-- \n", __func__, message);
        outputmessage = ignoreErrorCode(message);
    } catch (const std::exception& e) {
        rpc_result = JSONRPCReplyObj(NullUniValue,
            JSONRPCError(RPC_PARSE_ERROR, e.what()), jreq.id);
        LogPrintf("DEBUGGER ADDUSER ERROR2--%s-- \n", __func__);
        outputmessage = e.what();
    }



    //LogPrintf("DEBUGGER ADDUSER 2--%s %s-- \n", __func__, result.size());




    ui->labelErrorMsg->setText(QString::fromStdString(outputmessage));


    //ui->textEditResults->setText(QString::fromStdString(std::to_string(result.size())));

} //goAddUser

void BdapAddUserDialog::goCancel()
{
    ui->lineEdit_userID->setPlaceholderText("Cancel");
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














