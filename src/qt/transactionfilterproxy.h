// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_TRANSACTIONFILTERPROXY_H
#define DYNAMIC_QT_TRANSACTIONFILTERPROXY_H

#include "amount.h"

#include <QDateTime>
#include <QSortFilterProxyModel>

/** Filter the transaction list according to pre-specified rules. */
class TransactionFilterProxy : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    explicit TransactionFilterProxy(QObject* parent = 0);

    /** Earliest date that can be represented (far in the past) */
    static const QDateTime MIN_DATE;
    /** Last date that can be represented (far in the future) */
    static const QDateTime MAX_DATE;
    /** Type filter bit field (all types) */
    static const quint32 ALL_TYPES = 0xFFFFFFFF;
    /** Type filter bit field (all types but PrivateSend-SPAM) */
    static const quint32 COMMON_TYPES = 4223; //TODO Change this bit filter to include BDAP as a common type

    static quint32 TYPE(int type) { return 1 << type; }

    enum WatchOnlyFilter {
        WatchOnlyFilter_All,
        WatchOnlyFilter_Yes,
        WatchOnlyFilter_No
    };

    enum InstantSendFilter
    {
        InstantSendFilter_All,
        InstantSendFilter_Yes,
        InstantSendFilter_No
    };

    void setDateRange(const QDateTime& from, const QDateTime& to);
    void setAddressPrefix(const QString& addrPrefix);
    /**
      @note Type filter takes a bit field created with TYPE() or ALL_TYPES
     */
    void setTypeFilter(quint32 modes);
    void setMinAmount(const CAmount& minimum);
    void setWatchOnlyFilter(WatchOnlyFilter filter);
    void setInstantSendFilter(InstantSendFilter filter);

    /** Set maximum number of rows returned, -1 if unlimited. */
    void setLimit(int limit);

    /** Set whether to show conflicted transactions. */
    void setShowInactive(bool showInactive);

    /** Set whether to Hide orphans. */
    void setHideOrphans(bool fHide);

    int rowCount(const QModelIndex& parent = QModelIndex()) const;
    static bool isOrphan(const int status, const int type);

protected:
    bool filterAcceptsRow(int source_row, const QModelIndex& source_parent) const;

private:
    QDateTime dateFrom;
    QDateTime dateTo;
    QString addrPrefix;
    quint32 typeFilter;
    WatchOnlyFilter watchOnlyFilter;
    InstantSendFilter instantsendFilter;
    CAmount minAmount;
    int limitRows;
    bool showInactive;
    bool fHideOrphans;
};

#endif // DYNAMIC_QT_TRANSACTIONFILTERPROXY_H
