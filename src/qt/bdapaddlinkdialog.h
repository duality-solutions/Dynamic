// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPADDLINKDIALOG_H
#define BDAPADDLINKDIALOG_H

#include "platformstyle.h"

#include <QCompleter>
#include <QPushButton>
#include <QDialog>
#include <QThread>

#include <memory>

enum LinkUserType {
    LINK_REQUESTOR = 0,
    LINK_RECIPIENT = 1
};

namespace Ui
{
class BdapAddLinkDialog;
}

class BdapAddLinkDialog : public QDialog
{
    Q_OBJECT

public:
    explicit BdapAddLinkDialog(QWidget *parent = 0, int DynamicUnits = 0);
    ~BdapAddLinkDialog();

private:
    Ui::BdapAddLinkDialog* ui;
    std::string ignoreErrorCode(const std::string input);
    int nDynamicUnits;

    QCompleter* autoCompleterFrom;
    QCompleter* autoCompleterTo;

    std::string getIdFromPath(std::string inputstring);
    void populateList(std::vector<std::string> &inputList, LinkUserType userType);

private Q_SLOTS:

    void goClose();
    void goCancel();
    void addLink();

};

#endif // BDAPADDLINKDIALOG_H