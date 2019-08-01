// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPLINKDETAILDIALOG_H
#define BDAPLINKDETAILDIALOG_H

#include "bdappage.h"
#include "platformstyle.h"
#include "bdap/bdap.h"

#include <univalue.h>

#include <QPushButton>
#include <QDialog>

#include <memory>

const std::string LINK_TRANSACTION_MESSAGE = "Please note that your transaction will not be reflected until the next block.";;

namespace Ui
{
class BdapLinkDetailDialog;
}

class BdapLinkDetailDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BdapLinkDetailDialog(QWidget *parent = 0, LinkActions actionType = LinkActions::LINK_DEFAULT, const std::string& requestor = "", const std::string& recipient = "", const UniValue& resultinput = UniValue(UniValue::VOBJ), bool displayInfo = false);
    ~BdapLinkDetailDialog();

private:
    Ui::BdapLinkDetailDialog* ui;

    void populateValues(LinkActions accountType, const std::string& requestor, const std::string& recipient, const UniValue& resultinput);

private Q_SLOTS:
    void goClose();

};

#endif // BDAPLINKDETAILDIALOG_H