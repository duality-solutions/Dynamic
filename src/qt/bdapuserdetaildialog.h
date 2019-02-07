// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPUSERDETAILDIALOG_H
#define BDAPUSERDETAILDIALOG_H

#include "platformstyle.h"
#include "bdap/bdap.h"

#include <univalue.h>
#include <QPushButton>
#include <QDialog>

#include <memory>

const std::string TRANSACTION_MESSAGE = "Please note that your transaction will not be reflected until the next block.";

namespace Ui
{
class BdapUserDetailDialog;
}

class BdapUserDetailDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BdapUserDetailDialog(QWidget *parent = 0, BDAP::ObjectType accountType = BDAP::ObjectType::BDAP_USER, const std::string& accountID = "", const UniValue& resultinput = UniValue(UniValue::VOBJ), bool displayInfo = false);
    ~BdapUserDetailDialog();



private:
    Ui::BdapUserDetailDialog* ui;

    void populateValues(BDAP::ObjectType accountType, const std::string& accountID, const UniValue& resultinput);




private Q_SLOTS:
    void goCancel();


};

#endif // BDAPUSERDETAILDIALOG_H
