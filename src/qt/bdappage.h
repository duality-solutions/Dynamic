// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPPAGE_H
#define BDAPPAGE_H

#include "platformstyle.h"
#include "walletmodel.h"
#include "bdap/bdap.h"

#include <QPushButton>
#include <QWidget>
#include <memory>


class BdapAccountTableModel;
class QTableWidget;
class QLabel;

const int COMMONNAME_COLWIDTH = 450;
const int FULLPATH_COLWIDTH = 350;


namespace Ui
{
class BdapPage;
}

class BdapPage : public QWidget
{
    Q_OBJECT

public:
    explicit BdapPage(const PlatformStyle* platformStyle, QWidget* parent = 0);
    ~BdapPage();

    void setModel(WalletModel* model);
    BdapAccountTableModel* getBdapAccountTableModel();
    QTableWidget* getUserTable();
    QTableWidget* getGroupTable();
    QLabel* getUserStatus();
    QLabel* getGroupStatus();
    bool getMyUserCheckBoxChecked();
    bool getMyGroupCheckBoxChecked();
    int getCurrentIndex();
    std::string getCommonUserSearch();
    std::string getPathUserSearch();
    std::string getCommonGroupSearch();
    std::string getPathGroupSearch();
    void evaluateTransactionButtons();




private:
    Ui::BdapPage* ui;
    WalletModel* model;
    std::unique_ptr<WalletModel::UnlockContext> unlockContext;
    BdapAccountTableModel* bdapAccountTableModel;
    void executeDeleteAccount(std::string account, BDAP::ObjectType accountType);



private Q_SLOTS:

    void listAllUsers();
    void addUser();
    void deleteUser();
    void updateUser();
    void getUserDetails(int row, int column);

    void listAllGroups();
    void addGroup();
    void deleteGroup();
    void updateGroup();
    void getGroupDetails(int row, int column);

};

#endif // BDAPPAGE_H
