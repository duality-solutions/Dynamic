// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2019 The Raven Core developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_FREEZEADDRESS_H
#define DYNAMIC_QT_FREEZEADDRESS_H

#include "amount.h"

#include <memory>

#include <QMenu>
#include <QWidget>

class ClientModel;
class PlatformStyle;
class WalletModel;
class QStringListModel;
class QSortFilterProxyModel;
class QCompleter;
class AssetFilterProxy;


namespace Ui {
    class FreezeAddress;
}

QT_BEGIN_NAMESPACE
class QModelIndex;
QT_END_NAMESPACE

/** Overview ("home") page widget */
class FreezeAddress : public QWidget
{
    Q_OBJECT

public:
    explicit FreezeAddress(const PlatformStyle *_platformStyle, QWidget *parent = 0);
    ~FreezeAddress();

    void setClientModel(ClientModel *clientModel);
    void setWalletModel(WalletModel *walletModel);
    void showOutOfSyncWarning(bool fShow);
    Ui::FreezeAddress* getUI();
    bool eventFilter(QObject* object, QEvent* event);

    void enableSubmitButton();
    void showWarning(QString string, bool failure = true);
    void hideWarning();

    AssetFilterProxy *assetFilterProxy;
    QCompleter* completer;

    void clear();

private:
    Ui::FreezeAddress *ui;
    ClientModel *clientModel;
    WalletModel *walletModel;
    const PlatformStyle *platformStyle;

private Q_SLOTS:
    void check();
    void dataChanged();
    void globalOptionSelected();
    void changeAddressChanged(int);
};

#endif // DYNAMIC_QT_FREEZEADDRESS_H
