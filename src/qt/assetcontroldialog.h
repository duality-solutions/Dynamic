// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2019 The Raven Core developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_ASSETCONTROLDIALOG_H
#define DYNAMIC_QT_ASSETCONTROLDIALOG_H

#include "amount.h"

#include <QAbstractButton>
#include <QAction>
#include <QDialog>
#include <QList>
#include <QMenu>
#include <QPoint>
#include <QString>
#include <QTreeWidgetItem>


class PlatformStyle;
class WalletModel;

class CCoinControl;

class QStringListModel;
class QSortFilterProxyModel;
class QCompleter;

namespace Ui {
    class AssetControlDialog;
}

#define ASYMP_UTF8 "\xE2\x89\x88"

class CAssetControlWidgetItem : public QTreeWidgetItem
{
public:
    explicit CAssetControlWidgetItem(QTreeWidget *parent, int type = Type) : QTreeWidgetItem(parent, type) {}
    explicit CAssetControlWidgetItem(int type = Type) : QTreeWidgetItem(type) {}
    explicit CAssetControlWidgetItem(QTreeWidgetItem *parent, int type = Type) : QTreeWidgetItem(parent, type) {}

    bool operator<(const QTreeWidgetItem &other) const;
};

class AssetControlDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AssetControlDialog(const PlatformStyle *platformStyle, QWidget *parent = 0);
    ~AssetControlDialog();

    void setModel(WalletModel *model);

    // static because also called from sendcoinsdialog
    static void updateLabels(WalletModel*, QDialog*);

    //update the list of assets
    void updateAssetList(bool fSetOnStart = false);

    static QList<CAmount> payAmounts;
    static CCoinControl* assetControl;
    static bool fSubtractFeeFromAmount;
    bool fOnStartUp;

    QStringListModel* stringModel;
    QSortFilterProxyModel* proxy;
    QCompleter* completer;

private:
    Ui::AssetControlDialog *ui;
    WalletModel *model;
    int sortColumn;
    Qt::SortOrder sortOrder;

    QMenu *contextMenu;
    QTreeWidgetItem *contextMenuItem;
    QAction *copyTransactionHashAction;
    QAction *lockAction;
    QAction *unlockAction;

    const PlatformStyle *platformStyle;

    void sortView(int, Qt::SortOrder);
    void updateView();

    enum
    {
        COLUMN_CHECKBOX = 0,
        COLUMN_ASSET_NAME,
        COLUMN_AMOUNT,
        COLUMN_LABEL,
        COLUMN_ADDRESS,
        COLUMN_DATE,
        COLUMN_CONFIRMATIONS,
        COLUMN_TXHASH,
        COLUMN_VOUT_INDEX,
    };
    friend class CAssetControlWidgetItem;

private Q_SLOTS:
    void showMenu(const QPoint &);
    void copyAmount();
    void copyLabel();
    void copyAddress();
    void copyTransactionHash();
    void lockCoin();
    void unlockCoin();
    void clipboardQuantity();
    void clipboardAmount();
    void clipboardFee();
    void clipboardAfterFee();
    void clipboardBytes();
    void clipboardLowOutput();
    void clipboardChange();
    void radioTreeMode(bool);
    void radioListMode(bool);
    void viewItemChanged(QTreeWidgetItem*, int);
    void headerSectionClicked(int);
    void buttonBoxClicked(QAbstractButton*);
    void buttonSelectAllClicked();
    void updateLabelLocked();
    void viewAdministratorClicked();
    void onAssetSelected(QString name);
};

#endif // DYNAMIC_QT_ASSETCONTROLDIALOG_H
