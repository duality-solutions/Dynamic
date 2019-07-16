// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BDAPPAGE_H
#define BDAPPAGE_H

#include "bdap/bdap.h"
#include "dynamicunits.h"
#include "platformstyle.h"
#include "walletmodel.h"

#include <QPushButton>
#include <QWidget>

#include <memory>

class BdapAccountTableModel;
class BdapLinkTableModel;
class ClientModel;
class QTableWidget;
class QLabel;

const int COMMONNAME_COLWIDTH = 450;
const int FULLPATH_COLWIDTH = 350;

enum LinkActions {
    LINK_DEFAULT = 0,
    LINK_ACCEPT = 1,
    LINK_REQUEST = 2,
    LINK_PENDING_ACCEPT_DETAIL = 3,
    LINK_PENDING_REQUEST_DETAIL = 4,
    LINK_COMPLETE_DETAIL = 5
};

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

    void setClientModel(ClientModel* clientModel);
    void setModel(WalletModel* model);
    BdapAccountTableModel* getBdapAccountTableModel();
    BdapLinkTableModel* getBdapLinkTableModel();
    QTableWidget* getUserTable();
    QTableWidget* getGroupTable();

    QTableWidget* getCompleteTable();
    QTableWidget* getPendingAcceptTable();
    QTableWidget* getPendingRequestTable();
    
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
    QLabel* getLinkCompleteRecords();
    QLabel* getPendingAcceptRecords();
    QLabel* getPendingRequestRecords();
    std::string getCompleteRequestorSearch();    
    std::string getCompleteRecipientSearch();    
    std::string getPARequestorSearch();    
    std::string getPARecipientSearch();    
    std::string getPRRequestorSearch();    
    std::string getPRRecipientSearch();  

private:
    Ui::BdapPage* ui;
    ClientModel* clientModel;
    WalletModel* model;
    std::unique_ptr<WalletModel::UnlockContext> unlockContext;
    BdapAccountTableModel* bdapAccountTableModel;
    BdapLinkTableModel* bdapLinkTableModel;
    void executeDeleteAccount(std::string account, BDAP::ObjectType accountType);
    void executeLinkTransaction(LinkActions actionType, std::string requestor, std::string recipient);

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

    void listLinksAll();
    void listLinksComplete();
    void listPendingAccept();
    void listPendingRequest();
    void acceptLink();
    void addLink();
    void getLinkDetails(int row, int column);

    void updateBDAPLists();

};

#endif // BDAPPAGE_H