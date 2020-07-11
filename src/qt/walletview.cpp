// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "walletview.h"

#include "addressbookpage.h"
#include "askpassphrasedialog.h"
#include "bdappage.h"
#include "clientmodel.h"
#include "dynamicgui.h"
#include "guiutil.h"
#include "miningpage.h"
#include "multisenddialog.h"
#include "optionsmodel.h"
#include "overviewpage.h"
#include "platformstyle.h"
#include "receivecoinsdialog.h"
#include "sendcoinsdialog.h"
#include "signverifymessagedialog.h"
#include "transactionrecord.h"
#include "transactiontablemodel.h"
#include "transactionview.h"
#include "validation.h"
#include "walletmodel.h"

#include "dynodeconfig.h"
#include "ui_interface.h"

#include "assettablemodel.h"
#include "assetsdialog.h"
#include "createassetdialog.h"
#include "reissueassetdialog.h"
#include "restrictedassetsdialog.h"

#include <QAction>
#include <QActionGroup>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressDialog>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>

WalletView::WalletView(const PlatformStyle* _platformStyle, QWidget* parent) : QStackedWidget(parent),
                                                                               clientModel(0),
                                                                               walletModel(0),
                                                                               platformStyle(_platformStyle)
{
    // Create tabs
    overviewPage = new OverviewPage(platformStyle);


    sendCoinsPage = new SendCoinsDialog(platformStyle);

    receiveCoinsPage = new ReceiveCoinsDialog(platformStyle);

    miningPage = new MiningPage(platformStyle);
    bdapPage = new BdapPage(platformStyle);

    transactionsPage = new QWidget(this);
    QVBoxLayout* vbox = new QVBoxLayout();
    QHBoxLayout* hbox_buttons = new QHBoxLayout();
    transactionView = new TransactionView(platformStyle, this);
    vbox->addWidget(transactionView);
    QPushButton* exportButton = new QPushButton(tr("&Export"), this);
    exportButton->setToolTip(tr("Export the data in the current tab to a file"));
    if (platformStyle->getImagesOnButtons()) {
        QString theme = GUIUtil::getThemeName();
        exportButton->setIcon(QIcon(":/icons/" + theme + "/export"));
    }
    hbox_buttons->addStretch();

    assetsPage = new AssetsDialog(platformStyle);
    createAssetsPage = new CreateAssetDialog(platformStyle);
    manageAssetsPage = new ReissueAssetDialog(platformStyle);
    restrictedAssetsPage = new RestrictedAssetsDialog(platformStyle);

    usedSendingAddressesPage = new AddressBookPage(platformStyle, AddressBookPage::ForEditing, AddressBookPage::SendingTab, this);
    usedReceivingAddressesPage = new AddressBookPage(platformStyle, AddressBookPage::ForEditing, AddressBookPage::ReceivingTab, this);

    // Sum of selected transactions
    QLabel* transactionSumLabel = new QLabel();                // Label
    transactionSumLabel->setObjectName("transactionSumLabel"); // Label ID as CSS-reference
    transactionSumLabel->setText(tr("Selected amount:"));
    hbox_buttons->addWidget(transactionSumLabel);

    transactionSum = new QLabel();                   // Amount
    transactionSum->setObjectName("transactionSum"); // Label ID as CSS-reference
    transactionSum->setMinimumSize(200, 8);
    transactionSum->setTextInteractionFlags(Qt::TextSelectableByMouse);
    hbox_buttons->addWidget(transactionSum);

    hbox_buttons->addWidget(exportButton);
    vbox->addLayout(hbox_buttons);
    transactionsPage->setLayout(vbox);

    QSettings settings;
    if (settings.value("fShowDynodesTab").toBool()) {
        dynodeListPage = new DynodeList(platformStyle);
    }

    addWidget(overviewPage);
    addWidget(sendCoinsPage);
    addWidget(receiveCoinsPage);
    addWidget(transactionsPage);
    if (settings.value("fShowDynodesTab").toBool()) {
        addWidget(dynodeListPage);
    }
    addWidget(miningPage);
    addWidget(bdapPage);

    /** ASSET START */
    addWidget(assetsPage);
    addWidget(createAssetsPage);
    addWidget(manageAssetsPage);
    addWidget(restrictedAssetsPage);
    /** ASSET END */

    // Clicking on a transaction on the overview pre-selects the transaction on the transaction history page
    connect(overviewPage, SIGNAL(transactionClicked(QModelIndex)), transactionView, SLOT(focusTransaction(QModelIndex)));
    connect(overviewPage, SIGNAL(outOfSyncWarningClicked()), this, SLOT(requestedSyncWarningInfo()));

    // Double-clicking on a transaction on the transaction history page shows details
    connect(transactionView, SIGNAL(doubleClicked(QModelIndex)), transactionView, SLOT(showDetails()));

    // Update wallet with sum of selected transactions
    connect(transactionView, SIGNAL(trxAmount(QString)), this, SLOT(trxAmount(QString)));

    // Clicking on "Export" allows to export the transaction list
    connect(exportButton, SIGNAL(clicked()), transactionView, SLOT(exportClicked()));

    // Pass through messages from sendCoinsPage
    connect(sendCoinsPage, SIGNAL(message(QString, QString, unsigned int)), this, SIGNAL(message(QString, QString, unsigned int)));

    // Pass through messages from transactionView
    connect(transactionView, SIGNAL(message(QString, QString, unsigned int)), this, SIGNAL(message(QString, QString, unsigned int)));

    /** ASSET START */
    connect(assetsPage, SIGNAL(message(QString,QString,unsigned int)), this, SIGNAL(message(QString,QString,unsigned int)));
    connect(createAssetsPage, SIGNAL(message(QString,QString,unsigned int)), this, SIGNAL(message(QString,QString,unsigned int)));
    connect(manageAssetsPage, SIGNAL(message(QString,QString,unsigned int)), this, SIGNAL(message(QString,QString,unsigned int)));
    connect(restrictedAssetsPage, SIGNAL(message(QString,QString,unsigned int)), this, SIGNAL(message(QString,QString,unsigned int)));
    connect(overviewPage, SIGNAL(assetSendClicked(QModelIndex)), assetsPage, SLOT(focusAsset(QModelIndex)));
    connect(overviewPage, SIGNAL(assetIssueSubClicked(QModelIndex)), createAssetsPage, SLOT(focusSubAsset(QModelIndex)));
    connect(overviewPage, SIGNAL(assetIssueUniqueClicked(QModelIndex)), createAssetsPage, SLOT(focusUniqueAsset(QModelIndex)));
    connect(overviewPage, SIGNAL(assetReissueClicked(QModelIndex)), manageAssetsPage, SLOT(focusReissueAsset(QModelIndex)));
    /** ASSET END */
}

WalletView::~WalletView()
{
}

void WalletView::setDynamicGUI(DynamicGUI* gui)
{
    if (gui) {
        // Clicking on a transaction on the overview page simply sends you to transaction history page
        connect(overviewPage, SIGNAL(transactionClicked(QModelIndex)), gui, SLOT(gotoHistoryPage()));

        // Clicking on a asset menu item Send
        connect(overviewPage, SIGNAL(assetSendClicked(QModelIndex)), gui, SLOT(gotoAssetsPage()));

        // Clicking on a asset menu item Issue Sub
        connect(overviewPage, SIGNAL(assetIssueSubClicked(QModelIndex)), gui, SLOT(gotoCreateAssetsPage()));

        // Clicking on a asset menu item Issue Unique
        connect(overviewPage, SIGNAL(assetIssueUniqueClicked(QModelIndex)), gui, SLOT(gotoCreateAssetsPage()));

        // Clicking on a asset menu item Reissue
        connect(overviewPage, SIGNAL(assetReissueClicked(QModelIndex)), gui, SLOT(gotoManageAssetsPage()));

        // Receive and report messages
        connect(this, SIGNAL(message(QString, QString, unsigned int)), gui, SLOT(message(QString, QString, unsigned int)));

        // Pass through encryption status changed signals
        connect(this, SIGNAL(encryptionStatusChanged(int)), gui, SLOT(setEncryptionStatus(int)));

        // Pass through transaction notifications
        connect(this, SIGNAL(incomingTransaction(QString, int, CAmount, QString, QString, QString, QString)), gui, SLOT(incomingTransaction(QString, int, CAmount, QString, QString, QString, QString)));

        // Clicking on the lock icon will open the passphrase dialog
        connect(gui->labelWalletEncryptionIcon, SIGNAL(clicked()), this, SLOT(on_labelWalletEncryptionIcon_clicked()));

        // Connect HD enabled state signal
        connect(this, SIGNAL(hdEnabledStatusChanged(int)), gui, SLOT(setHDStatus(int)));

        // Pass through checkAssets calls to the GUI
        connect(this, SIGNAL(checkAssets()), gui, SLOT(checkAssets()));
    }
}

void WalletView::setClientModel(ClientModel* _clientModel)
{
    this->clientModel = _clientModel;

    overviewPage->setClientModel(_clientModel);
    sendCoinsPage->setClientModel(_clientModel);
    bdapPage->setClientModel(_clientModel);
    QSettings settings;
    if (settings.value("fShowDynodesTab").toBool()) {
        dynodeListPage->setClientModel(_clientModel);
    }
}

void WalletView::setWalletModel(WalletModel* _walletModel)
{
    this->walletModel = _walletModel;
    // Put transaction list in tabs
    overviewPage->setWalletModel(_walletModel);
    receiveCoinsPage->setModel(_walletModel);
    sendCoinsPage->setModel(_walletModel);
    usedReceivingAddressesPage->setModel(_walletModel->getAddressTableModel());
    usedSendingAddressesPage->setModel(_walletModel->getAddressTableModel());
    transactionView->setModel(_walletModel);
    QSettings settings;
    if (settings.value("fShowDynodesTab").toBool()) {
        dynodeListPage->setWalletModel(_walletModel);
    }
    miningPage->setModel(_walletModel);
    bdapPage->setModel(_walletModel);

    /** ASSET START */
    assetsPage->setModel(_walletModel);
    createAssetsPage->setModel(_walletModel);
    manageAssetsPage->setModel(_walletModel);
    restrictedAssetsPage->setModel(_walletModel);

    if (_walletModel) {
        // Receive and pass through messages from wallet model
        connect(_walletModel, SIGNAL(message(QString, QString, unsigned int)), this, SIGNAL(message(QString, QString, unsigned int)));

        // Handle changes in encryption status
        connect(_walletModel, SIGNAL(encryptionStatusChanged(int)), this, SIGNAL(encryptionStatusChanged(int)));
        updateEncryptionStatus();

        // update HD status
        Q_EMIT hdEnabledStatusChanged(walletModel->hdEnabled());

        // Balloon pop-up for new transaction
        connect(_walletModel->getTransactionTableModel(), SIGNAL(rowsInserted(QModelIndex, int, int)),
            this, SLOT(processNewTransaction(QModelIndex, int, int)));

        // Ask for passphrase if needed
        connect(_walletModel, SIGNAL(requireUnlock(bool)), this, SLOT(unlockWallet(bool)));

        // Show progress dialog
        connect(_walletModel, SIGNAL(showProgress(QString, int)), this, SLOT(showProgress(QString, int)));
    }
}

void WalletView::processNewTransaction(const QModelIndex& parent, int start, int end)
{
    // Prevent balloon-spam when initial block download is in progress
    if (!walletModel || !clientModel || clientModel->inInitialBlockDownload())
        return;

    TransactionTableModel* ttm = walletModel->getTransactionTableModel();
    if (!ttm || ttm->processingQueuedTransactions())
        return;

    QModelIndex index = ttm->index(start, 0, parent);
    QSettings settings;
    if (!settings.value("fShowPrivateSendPopups").toBool()) {
        QVariant nType = ttm->data(index, TransactionTableModel::TypeRole);
        if (nType == TransactionRecord::PrivateSendDenominate ||
            nType == TransactionRecord::PrivateSendCollateralPayment ||
            nType == TransactionRecord::PrivateSendMakeCollaterals ||
            nType == TransactionRecord::PrivateSendCreateDenominations)
            return;
    }

    /** ASSET START */
    // With the addition of asset transactions, there can be multiple transaction that need notifications
    // so we need to loop through all new transaction that were added to the transaction table and display
    // notifications for each individual transaction
    QString assetName = "";
    for (int i = start; i <= end; i++) {
    QString date = ttm->index(start, TransactionTableModel::Date, parent).data().toString();
    qint64 amount = ttm->index(start, TransactionTableModel::Amount, parent).data(Qt::EditRole).toULongLong();
    QString type = ttm->index(start, TransactionTableModel::Type, parent).data().toString();
    QString address = ttm->data(index, TransactionTableModel::AddressRole).toString();
    QString label = ttm->data(index, TransactionTableModel::LabelRole).toString();
    assetName = ttm->data(index, TransactionTableModel::AssetNameRole).toString();

    Q_EMIT incomingTransaction(date, walletModel->getOptionsModel()->getDisplayUnit(), amount, type, address, label, assetName);
    }

    /** Everytime we get an new transaction. We should check to see if assets are enabled or not */
    overviewPage->showAssets();
    transactionView->showAssets();
    Q_EMIT checkAssets();

    assetsPage->processNewTransaction();
    createAssetsPage->updateAssetList();
    manageAssetsPage->updateAssetsList();
    /** ASSET END */

}

void WalletView::gotoOverviewPage()
{
    setCurrentWidget(overviewPage);
    Q_EMIT checkAssets();
}

void WalletView::gotoSendCoinsPage(QString addr)
{
    setCurrentWidget(sendCoinsPage);

    if (!addr.isEmpty())
        sendCoinsPage->setAddress(addr);
}

void WalletView::gotoReceiveCoinsPage()
{
    setCurrentWidget(receiveCoinsPage);
}

void WalletView::gotoHistoryPage()
{
    setCurrentWidget(transactionsPage);
}

void WalletView::gotoDynodePage()
{
    QSettings settings;
    if (settings.value("fShowDynodesTab").toBool()) {
        setCurrentWidget(dynodeListPage);
    }
}

void WalletView::gotoMiningPage()
{
    setCurrentWidget(miningPage);
}

void WalletView::gotoBdapPage()
{
    setCurrentWidget(bdapPage);
}

void WalletView::gotoSignMessageTab(QString addr)
{
    // calls show() in showTab_SM()
    SignVerifyMessageDialog* signVerifyMessageDialog = new SignVerifyMessageDialog(platformStyle, this);
    signVerifyMessageDialog->setAttribute(Qt::WA_DeleteOnClose);
    signVerifyMessageDialog->setModel(walletModel);
    signVerifyMessageDialog->showTab_SM(true);

    if (!addr.isEmpty())
        signVerifyMessageDialog->setAddress_SM(addr);
}

void WalletView::gotoVerifyMessageTab(QString addr)
{
    // calls show() in showTab_VM()
    SignVerifyMessageDialog* signVerifyMessageDialog = new SignVerifyMessageDialog(platformStyle, this);
    signVerifyMessageDialog->setAttribute(Qt::WA_DeleteOnClose);
    signVerifyMessageDialog->setModel(walletModel);
    signVerifyMessageDialog->showTab_VM(true);

    if (!addr.isEmpty())
        signVerifyMessageDialog->setAddress_VM(addr);
}

void WalletView::gotoMultiSendDialog()
{
    MultiSendDialog* multiSendDialog = new MultiSendDialog(platformStyle, this);
    multiSendDialog->setModel(walletModel);
    multiSendDialog->show();
}

bool WalletView::handlePaymentRequest(const SendCoinsRecipient& recipient)
{
    return sendCoinsPage->handlePaymentRequest(recipient);
}

void WalletView::showOutOfSyncWarning(bool fShow)
{
    overviewPage->showOutOfSyncWarning(fShow);
}

void WalletView::updateEncryptionStatus()
{
    Q_EMIT encryptionStatusChanged(walletModel->getEncryptionStatus());
}

void WalletView::encryptWallet(bool status)
{
    if (!walletModel)
        return;
    AskPassphraseDialog dlg(status ? AskPassphraseDialog::Encrypt : AskPassphraseDialog::Decrypt, this);
    dlg.setModel(walletModel);
    dlg.exec();

    updateEncryptionStatus();
}

void WalletView::backupWallet()
{
    QString filename = GUIUtil::getSaveFileName(this,
        tr("Backup Wallet"), QString(),
        tr("Wallet Data (*.dat)"), nullptr);

    if (filename.isEmpty())
        return;

    if (!walletModel->backupWallet(filename)) {
        Q_EMIT message(tr("Backup Failed"), tr("There was an error trying to save the wallet data to %1.").arg(filename),
            CClientUIInterface::MSG_ERROR);
    } else {
        Q_EMIT message(tr("Backup Successful"), tr("The wallet data was successfully saved to %1.").arg(filename),
            CClientUIInterface::MSG_INFORMATION);
    }
}

void WalletView::changePassphrase()
{
    AskPassphraseDialog dlg(AskPassphraseDialog::ChangePass, this);
    dlg.setModel(walletModel);
    dlg.exec();
}

void WalletView::unlockWallet(bool fForMixingOnly)
{
    if (!walletModel)
        return;
    // Unlock wallet when requested by wallet model

    if (walletModel->getEncryptionStatus() == WalletModel::Locked || walletModel->getEncryptionStatus() == WalletModel::UnlockedForMixingOnly) {
        AskPassphraseDialog dlg(fForMixingOnly ? AskPassphraseDialog::UnlockMixing : AskPassphraseDialog::Unlock, this);
        dlg.setModel(walletModel);
        dlg.exec();
    }
}

void WalletView::lockWallet()
{
    if (!walletModel)
        return;

    walletModel->setWalletLocked(true);
}

void WalletView::usedSendingAddresses()
{
    if (!walletModel)
        return;

    usedSendingAddressesPage->show();
    usedSendingAddressesPage->raise();
    usedSendingAddressesPage->activateWindow();
}

void WalletView::usedReceivingAddresses()
{
    if (!walletModel)
        return;

    usedReceivingAddressesPage->show();
    usedReceivingAddressesPage->raise();
    usedReceivingAddressesPage->activateWindow();
}

void WalletView::showProgress(const QString& title, int nProgress)
{
    if (nProgress == 0) {
        progressDialog = new QProgressDialog(title, "", 0, 100);
        progressDialog->setWindowTitle(tr(PACKAGE_NAME));
        progressDialog->setWindowModality(Qt::ApplicationModal);
        progressDialog->setMinimumDuration(0);
        progressDialog->setCancelButton(0);
        progressDialog->setAutoClose(false);
        progressDialog->setValue(0);
    } else if (nProgress == 100) {
        if (progressDialog) {
            progressDialog->close();
            progressDialog->deleteLater();
        }
    } else if (progressDialog)
        progressDialog->setValue(nProgress);
}

void WalletView::on_labelWalletEncryptionIcon_clicked(bool fForMixingOnly)
{
    if (!walletModel)
        return;

    if (walletModel->getEncryptionStatus() == WalletModel::Unlocked) {
        walletModel->setWalletLocked(true);
    } else {
        AskPassphraseDialog dlg(fForMixingOnly ? AskPassphraseDialog::UnlockMixing : AskPassphraseDialog::Unlock, this);
        dlg.setModel(walletModel);
        dlg.exec();
    }
}

void WalletView::requestedSyncWarningInfo()
{
    Q_EMIT outOfSyncWarningClicked();
}

/** Update wallet with the sum of the selected transactions */
void WalletView::trxAmount(QString amount)
{
    transactionSum->setText(amount);
}

bool fFirstVisit = true;
/** ASSET START */
void WalletView::gotoAssetsPage()
{
    if (fFirstVisit){
        fFirstVisit = false;
        assetsPage->handleFirstSelection();
    }
    setCurrentWidget(assetsPage);
    assetsPage->focusAssetListBox();
}

void WalletView::gotoCreateAssetsPage()
{
    setCurrentWidget(createAssetsPage);
}

void WalletView::gotoManageAssetsPage()
{
    setCurrentWidget(manageAssetsPage);
}

void WalletView::gotoRestrictedAssetsPage()
{
    setCurrentWidget(restrictedAssetsPage);
}
/** ASSET END */
