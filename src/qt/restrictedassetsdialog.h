﻿// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2019 The Raven Core developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_RESTRICTEDASSETSSDIALOG_H
#define DYNAMIC_QT_RESTRICTEDASSETSSDIALOG_H

#include "walletmodel.h"

#include <QDialog>
#include <QMessageBox>
#include <QString>

class AssetFilterProxy;
class AssignQualifier;
class ClientModel;
class MyRestrictedAssetsFilterProxy;
class MyRestrictedAssetsTableModel;
class PlatformStyle;
class SendAssetsEntry;
class SendCoinsRecipient;
class QSortFilterProxyModel;

namespace Ui {
    class RestrictedAssetsDialog;
}

QT_BEGIN_NAMESPACE
class QUrl;
QT_END_NAMESPACE

/** Dialog for sending Dynamic */
class RestrictedAssetsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RestrictedAssetsDialog(const PlatformStyle *platformStyle, QWidget *parent = 0);
    ~RestrictedAssetsDialog();

    void setClientModel(ClientModel *clientModel);
    void setModel(WalletModel *model);
    void setupStyling(const PlatformStyle *platformStyle);

    /** Set up the tab chain manually, as Qt messes up the tab chain by default in some cases (issue https://bugreports.qt-project.org/browse/QTBUG-10907).
     */
    QWidget *setupTabChain(QWidget *prev);
public Q_SLOTS:
    void setBalance(const CAmount& balance, const CAmount& total, const CAmount &stake, const CAmount& unconfirmedBalance, const CAmount& immatureBalance, const CAmount& anonymizedBalance, const CAmount& watchBalance, const CAmount& watchStake, const CAmount& watchUnconfirmedBalance, const CAmount& watchImmatureBalance);


private:
    Ui::RestrictedAssetsDialog *ui;
    ClientModel *clientModel;
    WalletModel *model;
    const PlatformStyle *platformStyle;
    AssetFilterProxy *assetFilterProxy;
    QSortFilterProxyModel *myRestrictedAssetsFilterProxy;

    MyRestrictedAssetsTableModel *myRestrictedAssetsModel;

private Q_SLOTS:
    void updateDisplayUnit();
    void assignQualifierClicked();
    void freezeAddressClicked();


    Q_SIGNALS:
            // Fired when a message should be reported to the user
            void message(const QString &title, const QString &message, unsigned int style);
};

#endif // DYNAMIC_QT_RESTRICTEDASSETSSDIALOG_H
