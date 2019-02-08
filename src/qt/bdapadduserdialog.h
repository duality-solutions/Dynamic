// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPADDUSERDIALOG_H
#define BDAPADDUSERDIALOG_H

#include "platformstyle.h"
#include "bdap/bdap.h"

#include <QPushButton>
#include <QDialog>

#include <memory>

namespace Ui
{
class BdapAddUserDialog;
}

class BdapAddUserDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BdapAddUserDialog(QWidget *parent = 0, BDAP::ObjectType accountType = BDAP::ObjectType::BDAP_USER);
    ~BdapAddUserDialog();



private:
    Ui::BdapAddUserDialog* ui;
    std::string ignoreErrorCode(const std::string input);
    BDAP::ObjectType inputAccountType;




private Q_SLOTS:

    void goAddUser();
    void goCancel();
    void goClose();

};

#endif // BDAPADDUSERDIALOG_H
