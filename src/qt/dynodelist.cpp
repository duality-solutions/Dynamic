// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynodelist.h"
#include "ui_dynodelist.h"

#include "clientmodel.h"
#include "guiutil.h"
#include "walletmodel.h"

#include "activedynode.h"
#include "dynode-sync.h"
#include "dynodeconfig.h"
#include "dynodeman.h"
#include "init.h"
#include "sync.h"
#include "wallet/wallet.h"

#include <QMessageBox>
#include <QTimer>

int GetOffsetFromUtc()
{
#if QT_VERSION < 0x050200
    const QDateTime dateTime1 = QDateTime::currentDateTime();
    const QDateTime dateTime2 = QDateTime(dateTime1.date(), dateTime1.time(), Qt::UTC);
    return dateTime1.secsTo(dateTime2);
#else
    return QDateTime::currentDateTime().offsetFromUtc();
#endif
}

DynodeList::DynodeList(const PlatformStyle* platformStyle, QWidget* parent) : QWidget(parent),
                                                                              ui(new Ui::DynodeList),
                                                                              clientModel(0),
                                                                              walletModel(0)
{
    ui->setupUi(this);

    ui->startButton->setEnabled(true);

    int columnAliasWidth = 100;
    int columnAddressWidth = 200;
    int columnProtocolWidth = 60;
    int columnStatusWidth = 80;
    int columnActiveWidth = 130;
    int columnLastSeenWidth = 130;

    ui->tableWidgetMyDynodes->setColumnWidth(0, columnAliasWidth);
    ui->tableWidgetMyDynodes->setColumnWidth(1, columnAddressWidth);
    ui->tableWidgetMyDynodes->setColumnWidth(2, columnProtocolWidth);
    ui->tableWidgetMyDynodes->setColumnWidth(3, columnStatusWidth);
    ui->tableWidgetMyDynodes->setColumnWidth(4, columnActiveWidth);
    ui->tableWidgetMyDynodes->setColumnWidth(5, columnLastSeenWidth);

    ui->tableWidgetDynodes->setColumnWidth(0, columnAddressWidth);
    ui->tableWidgetDynodes->setColumnWidth(1, columnProtocolWidth);
    ui->tableWidgetDynodes->setColumnWidth(2, columnStatusWidth);
    ui->tableWidgetDynodes->setColumnWidth(3, columnActiveWidth);
    ui->tableWidgetDynodes->setColumnWidth(4, columnLastSeenWidth);

    ui->tableWidgetMyDynodes->setContextMenuPolicy(Qt::CustomContextMenu);

    QAction* startAliasAction = new QAction(tr("Start alias"), this);
    contextMenu = new QMenu();
    contextMenu->addAction(startAliasAction);
    connect(ui->tableWidgetMyDynodes, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
    connect(startAliasAction, SIGNAL(triggered()), this, SLOT(on_startButton_clicked()));

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateNodeList()));
    connect(timer, SIGNAL(timeout()), this, SLOT(updateMyNodeList()));
    timer->start(1000);

    fFilterUpdated = false;
    nTimeFilterUpdated = GetTime();
    updateNodeList();
}

DynodeList::~DynodeList()
{
    delete ui;
}

void DynodeList::setClientModel(ClientModel* model)
{
    this->clientModel = model;
    if (model) {
        // try to update list when Dynode count changes
        connect(clientModel, SIGNAL(strDynodesChanged(QString)), this, SLOT(updateNodeList()));
    }
}

void DynodeList::setWalletModel(WalletModel* model)
{
    this->walletModel = model;
}

void DynodeList::showContextMenu(const QPoint& point)
{
    QTableWidgetItem* item = ui->tableWidgetMyDynodes->itemAt(point);
    if (item)
        contextMenu->exec(QCursor::pos());
}

void DynodeList::StartAlias(std::string strAlias)
{
    std::string strStatusHtml;
    strStatusHtml += "<center>Alias: " + strAlias;

    for (const auto& dne : dynodeConfig.getEntries()) {
        if (dne.getAlias() == strAlias) {
            std::string strError;
            CDynodeBroadcast dnb;

            bool fSuccess = CDynodeBroadcast::Create(dne.getIp(), dne.getPrivKey(), dne.getTxHash(), dne.getOutputIndex(), strError, dnb);

            int nDoS;
            if (fSuccess && !dnodeman.CheckDnbAndUpdateDynodeList(NULL, dnb, nDoS, *g_connman)) {
                strError = "Failed to verify DNB";
                fSuccess = false;
            }

            if (fSuccess) {
                strStatusHtml += "<br>Successfully started Dynode.";
                dnodeman.NotifyDynodeUpdates(*g_connman);
            } else {
                strStatusHtml += "<br>Failed to start Dynode.<br>Error: " + strError;
            }
            break;
        }
    }
    strStatusHtml += "</center>";

    QMessageBox msg;
    msg.setText(QString::fromStdString(strStatusHtml));
    msg.exec();

    updateMyNodeList(true);
}

void DynodeList::StartAll(std::string strCommand)
{
    int nCountSuccessful = 0;
    int nCountFailed = 0;
    std::string strFailedHtml;

    for (const auto& dne : dynodeConfig.getEntries()) {
        std::string strError;
        CDynodeBroadcast dnb;

        int32_t nOutputIndex = 0;
        if (!ParseInt32(dne.getOutputIndex(), &nOutputIndex)) {
            continue;
        }

        COutPoint outpoint = COutPoint(uint256S(dne.getTxHash()), nOutputIndex);

        if (strCommand == "start-missing" && dnodeman.Has(outpoint))
            continue;

        bool fSuccess = CDynodeBroadcast::Create(dne.getIp(), dne.getPrivKey(), dne.getTxHash(), dne.getOutputIndex(), strError, dnb);

        int nDoS;
        if (fSuccess && !dnodeman.CheckDnbAndUpdateDynodeList(NULL, dnb, nDoS, *g_connman)) {
            strError = "Failed to verify DNB";
            fSuccess = false;
        }

        if (fSuccess) {
            nCountSuccessful++;
            dnodeman.NotifyDynodeUpdates(*g_connman);
        } else {
            nCountFailed++;
            strFailedHtml += "\nFailed to start " + dne.getAlias() + ". Error: " + strError;
        }
    }

    std::string returnObj;
    returnObj = strprintf("Successfully started %d Dynodes, failed to start %d, total %d", nCountSuccessful, nCountFailed, nCountFailed + nCountSuccessful);
    if (nCountFailed > 0) {
        returnObj += strFailedHtml;
    }

    QMessageBox msg;
    msg.setText(QString::fromStdString(returnObj));
    msg.exec();

    updateMyNodeList(true);
}

void DynodeList::updateMyDynodeInfo(QString strAlias, QString strAddr, const COutPoint& outpoint)
{
    bool fOldRowFound = false;
    int nNewRow = 0;

    for (int i = 0; i < ui->tableWidgetMyDynodes->rowCount(); i++) {
        if (ui->tableWidgetMyDynodes->item(i, 0)->text() == strAlias) {
            fOldRowFound = true;
            nNewRow = i;
            break;
        }
    }

    if (nNewRow == 0 && !fOldRowFound) {
        nNewRow = ui->tableWidgetMyDynodes->rowCount();
        ui->tableWidgetMyDynodes->insertRow(nNewRow);
    }

    dynode_info_t infoDn;
    bool fFound = dnodeman.GetDynodeInfo(outpoint, infoDn);

    QTableWidgetItem* aliasItem = new QTableWidgetItem(strAlias);
    QTableWidgetItem* addrItem = new QTableWidgetItem(fFound ? QString::fromStdString(infoDn.addr.ToString()) : strAddr);
    QTableWidgetItem* protocolItem = new QTableWidgetItem(QString::number(fFound ? infoDn.nProtocolVersion : -1));
    QTableWidgetItem* statusItem = new QTableWidgetItem(QString::fromStdString(fFound ? CDynode::StateToString(infoDn.nActiveState) : "MISSING"));
    QTableWidgetItem* activeSecondsItem = new QTableWidgetItem(QString::fromStdString(DurationToDHMS(fFound ? (infoDn.nTimeLastPing - infoDn.sigTime) : 0)));
    QTableWidgetItem* lastSeenItem = new QTableWidgetItem(QString::fromStdString(DateTimeStrFormat("%Y-%m-%d %H:%M",
        fFound ? infoDn.nTimeLastPing + GetOffsetFromUtc() : 0)));
    QTableWidgetItem* pubkeyItem = new QTableWidgetItem(QString::fromStdString(fFound ? CDynamicAddress(infoDn.pubKeyCollateralAddress.GetID()).ToString() : ""));

    ui->tableWidgetMyDynodes->setItem(nNewRow, 0, aliasItem);
    ui->tableWidgetMyDynodes->setItem(nNewRow, 1, addrItem);
    ui->tableWidgetMyDynodes->setItem(nNewRow, 2, protocolItem);
    ui->tableWidgetMyDynodes->setItem(nNewRow, 3, statusItem);
    ui->tableWidgetMyDynodes->setItem(nNewRow, 4, activeSecondsItem);
    ui->tableWidgetMyDynodes->setItem(nNewRow, 5, lastSeenItem);
    ui->tableWidgetMyDynodes->setItem(nNewRow, 6, pubkeyItem);
}

void DynodeList::updateMyNodeList(bool fForce)
{
    TRY_LOCK(cs_mydnlist, fLockAcquired);
    if (!fLockAcquired) {
        return;
    }
    static int64_t nTimeMyListUpdated = 0;

    // automatically update my Dynode list only once in MY_DYNODELIST_UPDATE_SECONDS seconds,
    // this update still can be triggered manually at any time via button click
    int64_t nSecondsTillUpdate = nTimeMyListUpdated + MY_DYNODELIST_UPDATE_SECONDS - GetTime();
    ui->secondsLabel->setText(QString::number(nSecondsTillUpdate));

    if (nSecondsTillUpdate > 0 && !fForce)
        return;
    nTimeMyListUpdated = GetTime();

    // Find selected row
    QItemSelectionModel* selectionModel = ui->tableWidgetMyDynodes->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();
    int nSelectedRow = selected.count() ? selected.at(0).row() : 0;

    ui->tableWidgetDynodes->setSortingEnabled(false);
    for (const auto& dne : dynodeConfig.getEntries()) {
        int32_t nOutputIndex = 0;
        if (!ParseInt32(dne.getOutputIndex(), &nOutputIndex)) {
            continue;
        }

        updateMyDynodeInfo(QString::fromStdString(dne.getAlias()), QString::fromStdString(dne.getIp()), COutPoint(uint256S(dne.getTxHash()), nOutputIndex));
    }
    ui->tableWidgetMyDynodes->selectRow(nSelectedRow);
    ui->tableWidgetDynodes->setSortingEnabled(true);

    // reset "timer"
    ui->secondsLabel->setText("0");
}

void DynodeList::updateNodeList()
{
    TRY_LOCK(cs_dnlist, fLockAcquired);
    if (!fLockAcquired) {
        return;
    }

    static int64_t nTimeListUpdated = GetTime();

    // to prevent high cpu usage update only once in DYNODELIST_UPDATE_SECONDS seconds
    // or DYNODELIST_FILTER_COOLDOWN_SECONDS seconds after filter was last changed
    int64_t nSecondsToWait = fFilterUpdated ? nTimeFilterUpdated - GetTime() + DYNODELIST_FILTER_COOLDOWN_SECONDS : nTimeListUpdated - GetTime() + DYNODELIST_UPDATE_SECONDS;

    if (fFilterUpdated)
        ui->countLabel->setText(QString::fromStdString(strprintf("Please wait... %d", nSecondsToWait)));
    if (nSecondsToWait > 0)
        return;

    nTimeListUpdated = GetTime();
    fFilterUpdated = false;

    QString strToFilter;
    ui->countLabel->setText("Updating...");
    ui->tableWidgetDynodes->setSortingEnabled(false);
    ui->tableWidgetDynodes->clearContents();
    ui->tableWidgetDynodes->setRowCount(0);
    std::map<COutPoint, CDynode> mapDynodes = dnodeman.GetFullDynodeMap();
    int offsetFromUtc = GetOffsetFromUtc();

    for (const auto& dnpair : mapDynodes) {
        CDynode dn = dnpair.second;
        // populate list
        // Address, Protocol, Status, Active Seconds, Last Seen, Pub Key
        QTableWidgetItem* addressItem = new QTableWidgetItem(QString::fromStdString(dn.addr.ToString()));
        QTableWidgetItem* protocolItem = new QTableWidgetItem(QString::number(dn.nProtocolVersion));
        QTableWidgetItem* statusItem = new QTableWidgetItem(QString::fromStdString(dn.GetStatus()));
        QTableWidgetItem* activeSecondsItem = new QTableWidgetItem(QString::fromStdString(DurationToDHMS(dn.lastPing.sigTime - dn.sigTime)));
        QTableWidgetItem* lastSeenItem = new QTableWidgetItem(QString::fromStdString(DateTimeStrFormat("%Y-%m-%d %H:%M", dn.lastPing.sigTime + offsetFromUtc)));
        QTableWidgetItem* pubkeyItem = new QTableWidgetItem(QString::fromStdString(CDynamicAddress(dn.pubKeyCollateralAddress.GetID()).ToString()));

        if (strCurrentFilter != "") {
            strToFilter = addressItem->text() + " " +
                          protocolItem->text() + " " +
                          statusItem->text() + " " +
                          activeSecondsItem->text() + " " +
                          lastSeenItem->text() + " " +
                          pubkeyItem->text();
            if (!strToFilter.contains(strCurrentFilter))
                continue;
        }

        ui->tableWidgetDynodes->insertRow(0);
        ui->tableWidgetDynodes->setItem(0, 0, addressItem);
        ui->tableWidgetDynodes->setItem(0, 1, protocolItem);
        ui->tableWidgetDynodes->setItem(0, 2, statusItem);
        ui->tableWidgetDynodes->setItem(0, 3, activeSecondsItem);
        ui->tableWidgetDynodes->setItem(0, 4, lastSeenItem);
        ui->tableWidgetDynodes->setItem(0, 5, pubkeyItem);
    }

    ui->countLabel->setText(QString::number(ui->tableWidgetDynodes->rowCount()));
    ui->tableWidgetDynodes->setSortingEnabled(true);
}

void DynodeList::on_filterLineEdit_textChanged(const QString& strFilterIn)
{
    strCurrentFilter = strFilterIn;
    nTimeFilterUpdated = GetTime();
    fFilterUpdated = true;
    ui->countLabel->setText(QString::fromStdString(strprintf("Please wait... %d", DYNODELIST_FILTER_COOLDOWN_SECONDS)));
}

void DynodeList::on_startButton_clicked()
{
    std::string strAlias;
    {
        LOCK(cs_mydnlist);
        // Find selected node alias
        QItemSelectionModel* selectionModel = ui->tableWidgetMyDynodes->selectionModel();
        QModelIndexList selected = selectionModel->selectedRows();

        if (selected.count() == 0)
            return;

        QModelIndex index = selected.at(0);
        int nSelectedRow = index.row();
        strAlias = ui->tableWidgetMyDynodes->item(nSelectedRow, 0)->text().toStdString();
    }

    // Display message box
    QMessageBox::StandardButton retval = QMessageBox::question(this, tr("Confirm Dynode start"),
        tr("Are you sure you want to start Dynode %1?").arg(QString::fromStdString(strAlias)),
        QMessageBox::Yes | QMessageBox::Cancel,
        QMessageBox::Cancel);

    if (retval != QMessageBox::Yes)
        return;

    WalletModel::EncryptionStatus encStatus = walletModel->getEncryptionStatus();

    if (encStatus == walletModel->Locked || encStatus == walletModel->UnlockedForMixingOnly) {
        WalletModel::UnlockContext ctx(walletModel->requestUnlock());

        if (!ctx.isValid())
            return; // Unlock wallet was cancelled

        StartAlias(strAlias);
        return;
    }

    StartAlias(strAlias);
}

void DynodeList::on_startAllButton_clicked()
{
    // Display message box
    QMessageBox::StandardButton retval = QMessageBox::question(this, tr("Confirm all Dynodes start"),
        tr("Are you sure you want to start ALL Dynodes?"),
        QMessageBox::Yes | QMessageBox::Cancel,
        QMessageBox::Cancel);

    if (retval != QMessageBox::Yes)
        return;

    WalletModel::EncryptionStatus encStatus = walletModel->getEncryptionStatus();

    if (encStatus == walletModel->Locked || encStatus == walletModel->UnlockedForMixingOnly) {
        WalletModel::UnlockContext ctx(walletModel->requestUnlock());

        if (!ctx.isValid())
            return; // Unlock wallet was cancelled

        StartAll();
        return;
    }

    StartAll();
}

void DynodeList::on_startMissingButton_clicked()
{
    if (!dynodeSync.IsDynodeListSynced()) {
        QMessageBox::critical(this, tr("Command is not available right now"),
            tr("You can't use this command until Dynode list is synced"));
        return;
    }

    // Display message box
    QMessageBox::StandardButton retval = QMessageBox::question(this,
        tr("Confirm missing Dynodes start"),
        tr("Are you sure you want to start MISSING Dynodes?"),
        QMessageBox::Yes | QMessageBox::Cancel,
        QMessageBox::Cancel);

    if (retval != QMessageBox::Yes)
        return;

    WalletModel::EncryptionStatus encStatus = walletModel->getEncryptionStatus();

    if (encStatus == walletModel->Locked || encStatus == walletModel->UnlockedForMixingOnly) {
        WalletModel::UnlockContext ctx(walletModel->requestUnlock());

        if (!ctx.isValid())
            return; // Unlock wallet was cancelled

        StartAll("start-missing");
        return;
    }

    StartAll("start-missing");
}

void DynodeList::on_tableWidgetMyDynodes_itemSelectionChanged()
{
    if (ui->tableWidgetMyDynodes->selectedItems().count() > 0) {
        ui->startButton->setEnabled(true);
    }
}

void DynodeList::on_UpdateButton_clicked()
{
    updateMyNodeList(true);
}
