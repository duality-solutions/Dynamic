// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPADDLINKDIALOG_H
#define BDAPADDLINKDIALOG_H

#include "platformstyle.h"

#include <QPushButton>
#include <QDialog>

#include <memory>

namespace Ui
{
class BdapAddLinkDialog;
}

class BdapAddLinkDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BdapAddLinkDialog(QWidget *parent = 0);
    ~BdapAddLinkDialog();



private:
    Ui::BdapAddLinkDialog* ui;
    std::string ignoreErrorCode(const std::string input);




private Q_SLOTS:

    void goClose();
    void goCancel();
    void addLink();

};

#endif // BDAPADDLINKDIALOG_H
