// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2019 The Raven Core developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_CREATEASSETDIALOG_H
#define DYNAMIC_QT_CREATEASSETDIALOG_H

#include "walletmodel.h"

#include <QDialog>

class PlatformStyle;
class WalletModel;
class ClientModel;

namespace Ui {
    class CreateAssetDialog;
}

QT_BEGIN_NAMESPACE
class QModelIndex;
class QStringListModel;
class QSortFilterProxyModel;
class QCompleter;
QT_END_NAMESPACE

/** Dialog showing transaction details. */
class CreateAssetDialog : public QDialog
{
Q_OBJECT

public:
    explicit CreateAssetDialog(const PlatformStyle *platformStyle, QWidget *parent = 0);
    ~CreateAssetDialog();

    void setClientModel(ClientModel *clientModel);
    void setModel(WalletModel *model);

    int type;
    QString format;


    void setupCoinControlFrame(const PlatformStyle *platformStyle);
    void setupAssetDataView(const PlatformStyle *platformStyle);
    void setupFeeControl(const PlatformStyle *platformStyle);

    void updateAssetList();
    void updateAssetListForRestrictedIssuance();
    void updateAssetListForSubQualifierIssuance();
    void restrictedAssetSelected();
    void restrictedAssetNotSelected();

    void clear();
    void selectTypeName(int type, QString name);

    QStringListModel* stringModel;
    QSortFilterProxyModel* proxy;
    QCompleter* completer;

private:
    Ui::CreateAssetDialog *ui;
    ClientModel *clientModel;
    WalletModel *model;
    bool fFeeMinimized;
    const PlatformStyle *platformStyle;

    bool checkedAvailablity = false;

    void toggleIPFSText();
    void setUpValues();
    void showMessage(QString string);
    void showValidMessage(QString string);
    void showInvalidVerifierStringMessage(QString string);
    void hideInvalidVerifierStringMessage();
    void hideMessage();
    void disableCreateButton();
    void enableCreateButton();
    void CheckFormState();
    void updatePresentedAssetName(QString name);
    QString GetSpecialCharacter();
    QString GetAssetName();
    void UpdateAssetNameMaxSize();
    void UpdateAssetNameToUpper();
    void setUniqueSelected();
    void setQualifierSelected();
    void clearSelected();

    //CoinControl
    // Update the passed in CCoinControl with state from the GUI
    void updateCoinControlState(CCoinControl& ctrl);

    //Fee
    void updateFeeMinimizedLabel();
    void minimizeFeeSection(bool fMinimize);

    //Validation
    bool checkIPFSHash(QString hash);

private Q_SLOTS:
    void ipfsStateChanged();
    void checkAvailabilityClicked();
    void onNameChanged(QString name);
    void onAddressNameChanged(QString address);
    void onIPFSHashChanged(QString hash);
    void onCreateAssetClicked();
    void onUnitChanged(int value);
    void onChangeAddressChanged(QString changeAddress);
    void onAssetTypeActivated(int index);
    void onAssetListActivated(int index);
    void onClearButtonClicked();
    void onVerifierStringChanged(QString verifier);

    //CoinControl
    void coinControlFeatureChanged(bool);
    void coinControlButtonClicked();
    void coinControlChangeChecked(int);
    void coinControlChangeEdited(const QString &);
    void coinControlClipboardQuantity();
    void coinControlClipboardAmount();
    void coinControlClipboardFee();
    void coinControlClipboardAfterFee();
    void coinControlClipboardBytes();
    void coinControlClipboardLowOutput();
    void coinControlClipboardChange();
    void coinControlUpdateLabels();

    //Fee
    void on_buttonChooseFee_clicked();
    void on_buttonMinimizeFee_clicked();
    void setMinimumFee();
    void updateFeeSectionControls();
    void updateMinFeeLabel();
    void updateSmartFeeLabel();
    void feeControlFeatureChanged(bool);

    void setBalance(const CAmount& balance, const CAmount& total, const CAmount &stake, const CAmount& unconfirmedBalance, const CAmount& immatureBalance, const CAmount& anonymizedBalance, const CAmount& watchBalance, const CAmount& watchStake, const CAmount& watchUnconfirmedBalance, const CAmount& watchImmatureBalance);
    void updateDisplayUnit();


    void focusSubAsset(const QModelIndex& index);
    void focusUniqueAsset(const QModelIndex& index);

protected:
    bool eventFilter( QObject* sender, QEvent* event);


Q_SIGNALS:
    // Fired when a message should be reported to the user
    void message(const QString &title, const QString &message, unsigned int style);
};

#endif // DYNAMIC_QT_CREATEASSETDIALOG_H
