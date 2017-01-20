// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "stormnodelist.h"
#include "ui_stormnodelist.h"

#include "activestormnode.h"
#include "clientmodel.h"
#include "init.h"
#include "guiutil.h"
#include "stormnode-sync.h"
#include "stormnodeconfig.h"
#include "stormnodeman.h"
#include "sync.h"
#include "wallet/wallet.h"
#include "walletmodel.h"

#include <QTimer>
#include <QMessageBox>

CCriticalSection cs_stormnodes;

StormnodeList::StormnodeList(const PlatformStyle *platformStyle, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StormnodeList),
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

    ui->tableWidgetMyStormnodes->setColumnWidth(0, columnAliasWidth);
    ui->tableWidgetMyStormnodes->setColumnWidth(1, columnAddressWidth);
    ui->tableWidgetMyStormnodes->setColumnWidth(2, columnProtocolWidth);
    ui->tableWidgetMyStormnodes->setColumnWidth(3, columnStatusWidth);
    ui->tableWidgetMyStormnodes->setColumnWidth(4, columnActiveWidth);
    ui->tableWidgetMyStormnodes->setColumnWidth(5, columnLastSeenWidth);

    ui->tableWidgetStormnodes->setColumnWidth(0, columnAddressWidth);
    ui->tableWidgetStormnodes->setColumnWidth(1, columnProtocolWidth);
    ui->tableWidgetStormnodes->setColumnWidth(2, columnStatusWidth);
    ui->tableWidgetStormnodes->setColumnWidth(3, columnActiveWidth);
    ui->tableWidgetStormnodes->setColumnWidth(4, columnLastSeenWidth);

    ui->tableWidgetMyStormnodes->setContextMenuPolicy(Qt::CustomContextMenu);

    QAction *startAliasAction = new QAction(tr("Start alias"), this);
    contextMenu = new QMenu();
    contextMenu->addAction(startAliasAction);
    connect(ui->tableWidgetMyStormnodes, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
    connect(startAliasAction, SIGNAL(triggered()), this, SLOT(on_startButton_clicked()));

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateNodeList()));
    connect(timer, SIGNAL(timeout()), this, SLOT(updateMyNodeList()));
    timer->start(1000);

    fFilterUpdated = false;
    nTimeFilterUpdated = GetTime();
    updateNodeList();
}

StormnodeList::~StormnodeList()
{
    delete ui;
}

void StormnodeList::setClientModel(ClientModel *model)
{
    this->clientModel = model;
    if(model) {
        // try to update list when Stormnode count changes
        connect(clientModel, SIGNAL(strStormnodesChanged(QString)), this, SLOT(updateNodeList()));
    }
}

void StormnodeList::setWalletModel(WalletModel *model)
{
    this->walletModel = model;
}

void StormnodeList::showContextMenu(const QPoint &point)
{
    QTableWidgetItem *item = ui->tableWidgetMyStormnodes->itemAt(point);
    if(item) contextMenu->exec(QCursor::pos());
}

void StormnodeList::StartAlias(std::string strAlias)
{
    std::string strStatusHtml;
    strStatusHtml += "<center>Alias: " + strAlias;

    BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
        if(sne.getAlias() == strAlias) {
            std::string strError;
            CStormnodeBroadcast snb;

            bool fSuccess = CStormnodeBroadcast::Create(sne.getIp(), sne.getPrivKey(), sne.getTxHash(), sne.getOutputIndex(), strError, snb);

            if(fSuccess) {
                strStatusHtml += "<br>Successfully started Stormnode.";
                snodeman.UpdateStormnodeList(snb);
                snb.Relay();
                snodeman.NotifyStormnodeUpdates();
            } else {
                strStatusHtml += "<br>Failed to start Stormnode.<br>Error: " + strError;
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

void StormnodeList::StartAll(std::string strCommand)
{
    int nCountSuccessful = 0;
    int nCountFailed = 0;
    std::string strFailedHtml;

    BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
        std::string strError;
        CStormnodeBroadcast snb;

        CTxIn txin = CTxIn(uint256S(sne.getTxHash()), uint32_t(atoi(sne.getOutputIndex().c_str())));
        CStormnode *psn = snodeman.Find(txin);

        if(strCommand == "start-missing" && psn) continue;

        bool fSuccess = CStormnodeBroadcast::Create(sne.getIp(), sne.getPrivKey(), sne.getTxHash(), sne.getOutputIndex(), strError, snb);

        if(fSuccess) {
            nCountSuccessful++;
            snodeman.UpdateStormnodeList(snb);
            snb.Relay();
            snodeman.NotifyStormnodeUpdates();
        } else {
            nCountFailed++;
            strFailedHtml += "\nFailed to start " + sne.getAlias() + ". Error: " + strError;
        }
    }
    pwalletMain->Lock();

    std::string returnObj;
    returnObj = strprintf("Successfully started %d Stormnodes, failed to start %d, total %d", nCountSuccessful, nCountFailed, nCountFailed + nCountSuccessful);
    if (nCountFailed > 0) {
        returnObj += strFailedHtml;
    }

    QMessageBox msg;
    msg.setText(QString::fromStdString(returnObj));
    msg.exec();

    updateMyNodeList(true);
}

void StormnodeList::updateMyStormnodeInfo(QString strAlias, QString strAddr, CStormnode *psn)
{
    LOCK(cs_snlistupdate);
    bool fOldRowFound = false;
    int nNewRow = 0;

    for(int i = 0; i < ui->tableWidgetMyStormnodes->rowCount(); i++) {
        if(ui->tableWidgetMyStormnodes->item(i, 0)->text() == strAlias) {
            fOldRowFound = true;
            nNewRow = i;
            break;
        }
    }

    if(nNewRow == 0 && !fOldRowFound) {
        nNewRow = ui->tableWidgetMyStormnodes->rowCount();
        ui->tableWidgetMyStormnodes->insertRow(nNewRow);
    }

    QTableWidgetItem *aliasItem = new QTableWidgetItem(strAlias);
    QTableWidgetItem *addrItem = new QTableWidgetItem(psn ? QString::fromStdString(psn->addr.ToString()) : strAddr);
    QTableWidgetItem *protocolItem = new QTableWidgetItem(QString::number(psn ? psn->nProtocolVersion : -1));
    QTableWidgetItem *statusItem = new QTableWidgetItem(QString::fromStdString(psn ? psn->GetStatus() : "MISSING"));
    QTableWidgetItem *activeSecondsItem = new QTableWidgetItem(QString::fromStdString(DurationToDHMS(psn ? (psn->lastPing.sigTime - psn->sigTime) : 0)));
    QTableWidgetItem *lastSeenItem = new QTableWidgetItem(QString::fromStdString(DateTimeStrFormat("%Y-%m-%d %H:%M", psn ? psn->lastPing.sigTime + QDateTime::currentDateTime().offsetFromUtc() : 0)));
    QTableWidgetItem *pubkeyItem = new QTableWidgetItem(QString::fromStdString(psn ? CDarkSilkAddress(psn->pubKeyCollateralAddress.GetID()).ToString() : ""));

    ui->tableWidgetMyStormnodes->setItem(nNewRow, 0, aliasItem);
    ui->tableWidgetMyStormnodes->setItem(nNewRow, 1, addrItem);
    ui->tableWidgetMyStormnodes->setItem(nNewRow, 2, protocolItem);
    ui->tableWidgetMyStormnodes->setItem(nNewRow, 3, statusItem);
    ui->tableWidgetMyStormnodes->setItem(nNewRow, 4, activeSecondsItem);
    ui->tableWidgetMyStormnodes->setItem(nNewRow, 5, lastSeenItem);
    ui->tableWidgetMyStormnodes->setItem(nNewRow, 6, pubkeyItem);
}

void StormnodeList::updateMyNodeList(bool fForce)
{
    static int64_t nTimeMyListUpdated = 0;

    // automatically update my Stormnode list only once in MY_STORMNODELIST_UPDATE_SECONDS seconds,
    // this update still can be triggered manually at any time via button click
    int64_t nSecondsTillUpdate = nTimeMyListUpdated + MY_STORMNODELIST_UPDATE_SECONDS - GetTime();
    ui->secondsLabel->setText(QString::number(nSecondsTillUpdate));

    if(nSecondsTillUpdate > 0 && !fForce) return;
    nTimeMyListUpdated = GetTime();

    ui->tableWidgetStormnodes->setSortingEnabled(false);
    BOOST_FOREACH(CStormnodeConfig::CStormnodeEntry sne, stormnodeConfig.getEntries()) {
        CTxIn txin = CTxIn(uint256S(sne.getTxHash()), uint32_t(atoi(sne.getOutputIndex().c_str())));
        CStormnode *psn = snodeman.Find(txin);

        updateMyStormnodeInfo(QString::fromStdString(sne.getAlias()), QString::fromStdString(sne.getIp()), psn);
    }
    ui->tableWidgetStormnodes->setSortingEnabled(true);

    // reset "timer"
    ui->secondsLabel->setText("0");
}

void StormnodeList::updateNodeList()
{
    static int64_t nTimeListUpdated = GetTime();

    // to prevent high cpu usage update only once in STORMNODELIST_UPDATE_SECONDS seconds
    // or STORMNODELIST_FILTER_COOLDOWN_SECONDS seconds after filter was last changed
    int64_t nSecondsToWait = fFilterUpdated
                            ? nTimeFilterUpdated - GetTime() + STORMNODELIST_FILTER_COOLDOWN_SECONDS
                            : nTimeListUpdated - GetTime() + STORMNODELIST_UPDATE_SECONDS;

    if(fFilterUpdated) ui->countLabel->setText(QString::fromStdString(strprintf("Please wait... %d", nSecondsToWait)));
    if(nSecondsToWait > 0) return;

    nTimeListUpdated = GetTime();
    fFilterUpdated = false;

    TRY_LOCK(cs_stormnodes, lockStormnodes);
    if(!lockStormnodes) return;

    QString strToFilter;
    ui->countLabel->setText("Updating...");
    ui->tableWidgetStormnodes->setSortingEnabled(false);
    ui->tableWidgetStormnodes->clearContents();
    ui->tableWidgetStormnodes->setRowCount(0);
    std::vector<CStormnode> vStormnodes = snodeman.GetFullStormnodeVector();

    BOOST_FOREACH(CStormnode& sn, vStormnodes)
    {
        // populate list
        // Address, Protocol, Status, Active Seconds, Last Seen, Pub Key
        QTableWidgetItem *addressItem = new QTableWidgetItem(QString::fromStdString(sn.addr.ToString()));
        QTableWidgetItem *protocolItem = new QTableWidgetItem(QString::number(sn.nProtocolVersion));
        QTableWidgetItem *statusItem = new QTableWidgetItem(QString::fromStdString(sn.GetStatus()));
        QTableWidgetItem *activeSecondsItem = new QTableWidgetItem(QString::fromStdString(DurationToDHMS(sn.lastPing.sigTime - sn.sigTime)));
        QTableWidgetItem *lastSeenItem = new QTableWidgetItem(QString::fromStdString(DateTimeStrFormat("%Y-%m-%d %H:%M", sn.lastPing.sigTime + QDateTime::currentDateTime().offsetFromUtc())));
        QTableWidgetItem *pubkeyItem = new QTableWidgetItem(QString::fromStdString(CDarkSilkAddress(sn.pubKeyCollateralAddress.GetID()).ToString()));

        if (strCurrentFilter != "")
        {
            strToFilter =   addressItem->text() + " " +
                            protocolItem->text() + " " +
                            statusItem->text() + " " +
                            activeSecondsItem->text() + " " +
                            lastSeenItem->text() + " " +
                            pubkeyItem->text();
            if (!strToFilter.contains(strCurrentFilter)) continue;
        }

        ui->tableWidgetStormnodes->insertRow(0);
        ui->tableWidgetStormnodes->setItem(0, 0, addressItem);
        ui->tableWidgetStormnodes->setItem(0, 1, protocolItem);
        ui->tableWidgetStormnodes->setItem(0, 2, statusItem);
        ui->tableWidgetStormnodes->setItem(0, 3, activeSecondsItem);
        ui->tableWidgetStormnodes->setItem(0, 4, lastSeenItem);
        ui->tableWidgetStormnodes->setItem(0, 5, pubkeyItem);
    }

    ui->countLabel->setText(QString::number(ui->tableWidgetStormnodes->rowCount()));
    ui->tableWidgetStormnodes->setSortingEnabled(true);
}

void StormnodeList::on_filterLineEdit_textChanged(const QString &strFilterIn)
{
    strCurrentFilter = strFilterIn;
    nTimeFilterUpdated = GetTime();
    fFilterUpdated = true;
    ui->countLabel->setText(QString::fromStdString(strprintf("Please wait... %d", STORMNODELIST_FILTER_COOLDOWN_SECONDS)));
}

void StormnodeList::on_startButton_clicked()
{
    // Find selected node alias
    QItemSelectionModel* selectionModel = ui->tableWidgetMyStormnodes->selectionModel();
    QModelIndexList selected = selectionModel->selectedRows();

    if(selected.count() == 0) return;

    QModelIndex index = selected.at(0);
    int nSelectedRow = index.row();
    std::string strAlias = ui->tableWidgetMyStormnodes->item(nSelectedRow, 0)->text().toStdString();

    // Display message box
    QMessageBox::StandardButton retval = QMessageBox::question(this, tr("Confirm Stormnode start"),
        tr("Are you sure you want to start Stormnode %1?").arg(QString::fromStdString(strAlias)),
        QMessageBox::Yes | QMessageBox::Cancel,
        QMessageBox::Cancel);

    if(retval != QMessageBox::Yes) return;

    WalletModel::EncryptionStatus encStatus = walletModel->getEncryptionStatus();

    if(encStatus == walletModel->Locked || encStatus == walletModel->UnlockedForMixingOnly) {
        WalletModel::UnlockContext ctx(walletModel->requestUnlock());

        if(!ctx.isValid()) return; // Unlock wallet was cancelled

        StartAlias(strAlias);
        return;
    }

    StartAlias(strAlias);
}

void StormnodeList::on_startAllButton_clicked()
{
    // Display message box
    QMessageBox::StandardButton retval = QMessageBox::question(this, tr("Confirm all Stormnodes start"),
        tr("Are you sure you want to start ALL Stormnodes?"),
        QMessageBox::Yes | QMessageBox::Cancel,
        QMessageBox::Cancel);

    if(retval != QMessageBox::Yes) return;

    WalletModel::EncryptionStatus encStatus = walletModel->getEncryptionStatus();

    if(encStatus == walletModel->Locked || encStatus == walletModel->UnlockedForMixingOnly) {
        WalletModel::UnlockContext ctx(walletModel->requestUnlock());

        if(!ctx.isValid()) return; // Unlock wallet was cancelled

        StartAll();
        return;
    }

    StartAll();
}

void StormnodeList::on_startMissingButton_clicked()
{

    if(!stormnodeSync.IsStormnodeListSynced()) {
        QMessageBox::critical(this, tr("Command is not available right now"),
            tr("You can't use this command until Stormnode list is synced"));
        return;
    }

    // Display message box
    QMessageBox::StandardButton retval = QMessageBox::question(this,
        tr("Confirm missing Stormnodes start"),
        tr("Are you sure you want to start MISSING Stormnodes?"),
        QMessageBox::Yes | QMessageBox::Cancel,
        QMessageBox::Cancel);

    if(retval != QMessageBox::Yes) return;

    WalletModel::EncryptionStatus encStatus = walletModel->getEncryptionStatus();

    if(encStatus == walletModel->Locked || encStatus == walletModel->UnlockedForMixingOnly) {
        WalletModel::UnlockContext ctx(walletModel->requestUnlock());

        if(!ctx.isValid()) return; // Unlock wallet was cancelled

        StartAll("start-missing");
        return;
    }

    StartAll("start-missing");
}

void StormnodeList::on_tableWidgetMyStormnodes_itemSelectionChanged()
{
    if(ui->tableWidgetMyStormnodes->selectedItems().count() > 0) {
        ui->startButton->setEnabled(true);
    }
}

void StormnodeList::on_UpdateButton_clicked()
{
    updateMyNodeList(true);
}
