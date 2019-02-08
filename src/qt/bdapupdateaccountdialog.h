// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPUPDATEACCOUNTDIALOG_H
#define BDAPUPDATEACCOUNTDIALOG_H

#include "platformstyle.h"
#include "bdap/bdap.h"

#include <QPushButton>
#include <QDialog>

#include <memory>

namespace Ui
{
class BdapUpdateAccountDialog;
}

class BdapUpdateAccountDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BdapUpdateAccountDialog(QWidget *parent = 0, BDAP::ObjectType accountType = BDAP::ObjectType::BDAP_USER, std::string account = "", std::string commonName = "", std::string expirationDate = "");
    ~BdapUpdateAccountDialog();



private:
    Ui::BdapUpdateAccountDialog* ui;
    std::string ignoreErrorCode(const std::string input);
    BDAP::ObjectType inputAccountType;




private Q_SLOTS:

    void goCancel();
    void goClose();
    void updateAccount();

};

#endif // BDAPUPDATEACCOUNTDIALOG_H
