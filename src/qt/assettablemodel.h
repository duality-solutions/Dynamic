// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2017-2019 The Raven Core developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_ASSETTABLEMODEL_H
#define DYNAMIC_QT_ASSETTABLEMODEL_H

#include "amount.h"

#include <QAbstractTableModel>
#include <QStringList>

class AssetRecord;
class CAssets;
class AssetTablePriv;
class WalletModel;


/** Models assets portion of wallet as table of owned assets.
 */
class AssetTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit AssetTableModel(WalletModel *parent = 0);
    ~AssetTableModel();

    enum ColumnIndex {
        Name = 0,
        Quantity = 1
    };

    /** Roles to get specific information from a transaction row.
        These are independent of column.
    */
    enum RoleIndex {
        /** Net amount of transaction */
            AmountRole = 100,
        /** RVN or name of an asset */
            AssetNameRole = 101,
        /** Formatted amount, without brackets when unconfirmed */
            FormattedAmountRole = 102,
        /** AdministratorRole */
            AdministratorRole = 103
    };

    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const;
    QVariant data(const QModelIndex &index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    QModelIndex index(int row, int column, const QModelIndex & parent = QModelIndex()) const;
    QString formatTooltip(const AssetRecord *rec) const;
    QString formatAssetName(const AssetRecord *wtx) const;
    QString formatAssetQuantity(const AssetRecord *wtx) const;

    void checkBalanceChanged();

private:
    WalletModel *walletModel;
    QStringList columns;
    AssetTablePriv *priv;

    friend class AssetTablePriv;
};

#endif // DYNAMIC_QT_ASSETTABLEMODEL_H
