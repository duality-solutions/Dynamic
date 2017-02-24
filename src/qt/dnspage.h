// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2013-2017 Emercoin Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DNSPAGE_H
#define DNSPAGE_H

#include <QDialog>
#include <QSortFilterProxyModel>

namespace Ui {
    class DNSPage;
}
class WalletModel;
class NameTableModel;

QT_BEGIN_NAMESPACE
class QTableView;
class QItemSelection;
class QMenu;
class QModelIndex;
QT_END_NAMESPACE

class NameFilterProxyModel : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    explicit NameFilterProxyModel(QObject *parent = 0);

    void setNameSearch(const QString &search);
    void setValueSearch(const QString &search);
    void setAddressSearch(const QString &search);

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const;
    bool lessThan(const QModelIndex &left, const QModelIndex &right) const;

private:
    QString nameSearch, valueSearch, addressSearch;
};

/** Page for managing names */
class DNSPage : public QDialog
{
    Q_OBJECT

public:
    explicit DNSPage(QWidget *parent = 0);
    ~DNSPage();

    void setModel(WalletModel *walletModel);
    std::vector<unsigned char> importedAsBinaryFile;
    QString               importedAsTextFile;

private:
    Ui::DNSPage *ui;
    NameTableModel *model;
    WalletModel *walletModel;
    NameFilterProxyModel *proxyModel;
    QMenu *contextMenu;

public Q_SLOTS:
    void exportClicked();

    void changedNameFilter(const QString &filter);
    void changedValueFilter(const QString &filter);
    void changedAddressFilter(const QString &filter);

private Q_SLOTS:
    void on_submitNameButton_clicked();

    bool eventFilter(QObject *object, QEvent *event);
    void selectionChanged();

    /** Spawn contextual menu (right mouse menu) for name table entry */
    void contextualMenu(const QPoint &point);

    void onCopyNameAction();
    void onCopyValueAction();
    void onCopyAddressAction();
    void onCopyAllAction();
    void onSaveValueAsBinaryAction();

    void on_txTypeSelector_currentIndexChanged(const QString &txType);
    void on_cbMyNames_stateChanged(int arg1);
    void on_cbOtherNames_stateChanged(int arg1);
    void on_cbExpired_stateChanged(int arg1);
    void on_importValueButton_clicked();
    void on_registerValue_textChanged();
    void on_tableView_doubleClicked(const QModelIndex& index);

Q_SIGNALS:
    void doubleClicked(const QModelIndex&);
};

#endif // DNSPAGE_H
