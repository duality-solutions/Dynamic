// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "transactiontablemodel.h"

#include "addresstablemodel.h"
#include "guiconstants.h"
#include "guiutil.h"
#include "optionsmodel.h"
#include "platformstyle.h"
#include "transactiondesc.h"
#include "transactionrecord.h"
#include "walletmodel.h"

#include "core_io.h"
#include "sync.h"
#include "uint256.h"
#include "util.h"
#include "validation.h"
#include "wallet/wallet.h"

#include <QColor>
#include <QDateTime>
#include <QDebug>
#include <QIcon>
#include <QList>

static int column_alignments[] = {
    Qt::AlignLeft | Qt::AlignVCenter, /* status */
    Qt::AlignLeft | Qt::AlignVCenter, /* watchonly */
    Qt::AlignLeft|Qt::AlignVCenter, /* instantsend */
    Qt::AlignLeft | Qt::AlignVCenter, /* date */
    Qt::AlignLeft | Qt::AlignVCenter, /* type */
    Qt::AlignLeft | Qt::AlignVCenter, /* address */
    Qt::AlignLeft | Qt::AlignVCenter,  /* amount */
    Qt::AlignLeft|Qt::AlignVCenter /* assetName */
};

// Comparison operator for sort/binary search of model tx list
struct TxLessThan {
    bool operator()(const TransactionRecord& a, const TransactionRecord& b) const
    {
        return a.hash < b.hash;
    }
    bool operator()(const TransactionRecord& a, const uint256& b) const
    {
        return a.hash < b;
    }
    bool operator()(const uint256& a, const TransactionRecord& b) const
    {
        return a < b.hash;
    }
};

// Private implementation
class TransactionTablePriv
{
public:
    TransactionTablePriv(CWallet* _wallet, TransactionTableModel* _parent) : wallet(_wallet),
                                                                             parent(_parent)
    {
    }

    CWallet* wallet;
    TransactionTableModel* parent;

    /* Local cache of wallet.
     * As it is in the same order as the CWallet, by definition
     * this is sorted by sha256.
     */
    QList<TransactionRecord> cachedWallet;

    /* Query entire wallet anew from core.
     */
    void refreshWallet()
    {
        qDebug() << "TransactionTablePriv::refreshWallet";
        cachedWallet.clear();
        {
            LOCK2(cs_main, wallet->cs_wallet);
            for (std::map<uint256, CWalletTx>::iterator it = wallet->mapWallet.begin(); it != wallet->mapWallet.end(); ++it) {
                if (TransactionRecord::showTransaction(it->second))
                    cachedWallet.append(TransactionRecord::decomposeTransaction(wallet, it->second));
            }
        }
    }

    /* Update our model of the wallet incrementally, to synchronize our model of the wallet
       with that of the core.

       Call with transaction that was added, removed or changed.
     */
    void updateWallet(const uint256& hash, int status, bool showTransaction)
    {
        qDebug() << "TransactionTablePriv::updateWallet: " + QString::fromStdString(hash.ToString()) + " " + QString::number(status);

        // Find bounds of this transaction in model
        QList<TransactionRecord>::iterator lower = qLowerBound(
            cachedWallet.begin(), cachedWallet.end(), hash, TxLessThan());
        QList<TransactionRecord>::iterator upper = qUpperBound(
            cachedWallet.begin(), cachedWallet.end(), hash, TxLessThan());
        int lowerIndex = (lower - cachedWallet.begin());
        int upperIndex = (upper - cachedWallet.begin());
        bool inModel = (lower != upper);

        if (status == CT_UPDATED) {
            if (showTransaction && !inModel)
                status = CT_NEW; /* Not in model, but want to show, treat as new */
            if (!showTransaction && inModel)
                status = CT_DELETED; /* In model, but want to hide, treat as deleted */
        }

        qDebug() << "    inModel=" + QString::number(inModel) +
                        " Index=" + QString::number(lowerIndex) + "-" + QString::number(upperIndex) +
                        " showTransaction=" + QString::number(showTransaction) + " derivedStatus=" + QString::number(status);

        switch (status) {
        case CT_NEW:
            if (inModel) {
                qWarning() << "TransactionTablePriv::updateWallet: Warning: Got CT_NEW, but transaction is already in model";
                break;
            }
            if (showTransaction) {
                LOCK2(cs_main, wallet->cs_wallet);
                // Find transaction in wallet
                std::map<uint256, CWalletTx>::iterator mi = wallet->mapWallet.find(hash);
                if (mi == wallet->mapWallet.end()) {
                    qWarning() << "TransactionTablePriv::updateWallet: Warning: Got CT_NEW, but transaction is not in wallet";
                    break;
                }
                // Added -- insert at the right position
                QList<TransactionRecord> toInsert =
                    TransactionRecord::decomposeTransaction(wallet, mi->second);
                if (!toInsert.isEmpty()) /* only if something to insert */
                {
                    parent->beginInsertRows(QModelIndex(), lowerIndex, lowerIndex + toInsert.size() - 1);
                    int insert_idx = lowerIndex;
                    Q_FOREACH (const TransactionRecord& rec, toInsert) {
                        cachedWallet.insert(insert_idx, rec);
                        insert_idx += 1;
                    }
                    parent->endInsertRows();
                }
            }
            break;
        case CT_DELETED:
            if (!inModel) {
                qWarning() << "TransactionTablePriv::updateWallet: Warning: Got CT_DELETED, but transaction is not in model";
                break;
            }
            // Removed -- remove entire transaction from table
            parent->beginRemoveRows(QModelIndex(), lowerIndex, upperIndex - 1);
            cachedWallet.erase(lower, upper);
            parent->endRemoveRows();
            break;
        case CT_UPDATED:
            // Miscellaneous updates -- nothing to do, status update will take care of this, and is only computed for
            // visible transactions.
            break;
        }
    }

    int size()
    {
        return cachedWallet.size();
    }

    TransactionRecord* index(int idx)
    {
        if (idx >= 0 && idx < cachedWallet.size()) {
            TransactionRecord* rec = &cachedWallet[idx];

            // Get required locks upfront. This avoids the GUI from getting
            // stuck if the core is holding the locks for a longer time - for
            // example, during a wallet rescan.
            //
            // If a status update is needed (blocks came in since last check),
            //  update the status of this transaction from the wallet. Otherwise,
            // simply re-use the cached status.
            TRY_LOCK(cs_main, lockMain);
            if (lockMain) {
                TRY_LOCK(wallet->cs_wallet, lockWallet);
                if (lockWallet && rec->statusUpdateNeeded()) {
                    std::map<uint256, CWalletTx>::iterator mi = wallet->mapWallet.find(rec->hash);

                    if (mi != wallet->mapWallet.end()) {
                        rec->updateStatus(mi->second);
                    }
                }
            }
            return rec;
        }
        return 0;
    }

    QString describe(TransactionRecord* rec, int unit)
    {
        {
            LOCK2(cs_main, wallet->cs_wallet);
            std::map<uint256, CWalletTx>::iterator mi = wallet->mapWallet.find(rec->hash);
            if (mi != wallet->mapWallet.end()) {
                return TransactionDesc::toHTML(wallet, mi->second, rec, unit);
            }
        }
        return QString();
    }

    QString getTxHex(TransactionRecord* rec)
    {
        LOCK2(cs_main, wallet->cs_wallet);
        std::map<uint256, CWalletTx>::iterator mi = wallet->mapWallet.find(rec->hash);
        if (mi != wallet->mapWallet.end()) {
            std::string strHex = EncodeHexTx(static_cast<CTransaction>(mi->second));
            return QString::fromStdString(strHex);
        }
        return QString();
    }
};

TransactionTableModel::TransactionTableModel(const PlatformStyle* _platformStyle, CWallet* _wallet, WalletModel* parent) : QAbstractTableModel(parent),
                                                                                                                           wallet(_wallet),
                                                                                                                           walletModel(parent),
                                                                                                                           priv(new TransactionTablePriv(_wallet, this)),
                                                                                                                           fProcessingQueuedTransactions(false),
                                                                                                                           platformStyle(_platformStyle)
{
    columns << QString() << QString() << QString() << tr("Date") << tr("Type") << tr("Address / Label") << DynamicUnits::getAmountColumnTitle(walletModel->getOptionsModel()->getDisplayUnit()) << tr("Asset");
    priv->refreshWallet();

    connect(walletModel->getOptionsModel(), SIGNAL(displayUnitChanged(int)), this, SLOT(updateDisplayUnit()));

    subscribeToCoreSignals();
}

TransactionTableModel::~TransactionTableModel()
{
    unsubscribeFromCoreSignals();
    delete priv;
}

/** Updates the column title to "Amount (DisplayUnit)" and emits headerDataChanged() signal for table headers to react. */
//void TransactionTableModel::updateAmountColumnTitle()
//{
//    columns[Amount] = DynamicUnits::getAmountColumnTitle(walletModel->getOptionsModel()->getDisplayUnit());
//    Q_EMIT headerDataChanged(Qt::Horizontal, Amount, Amount);
//}

void TransactionTableModel::updateTransaction(const QString& hash, int status, bool showTransaction)
{
    uint256 updated;
    updated.SetHex(hash.toStdString());

    priv->updateWallet(updated, status, showTransaction);
}

void TransactionTableModel::updateConfirmations()
{
    // Blocks came in since last poll.
    // Invalidate status (number of confirmations) and (possibly) description
    //  for all rows. Qt is smart enough to only actually request the data for the
    //  visible rows.
    Q_EMIT dataChanged(index(0, Status), index(priv->size() - 1, Status));
    Q_EMIT dataChanged(index(0, ToAddress), index(priv->size() - 1, ToAddress));
}

int TransactionTableModel::rowCount(const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    return priv->size();
}

int TransactionTableModel::columnCount(const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    return columns.length();
}

QString TransactionTableModel::formatTxStatus(const TransactionRecord* wtx) const
{
    QString status;

    switch (wtx->status.status) {
    case TransactionStatus::OpenUntilBlock:
        status = tr("Open for %n more block(s)", "", wtx->status.open_for);
        break;
    case TransactionStatus::OpenUntilDate:
        status = tr("Open until %1").arg(GUIUtil::dateTimeStr(wtx->status.open_for));
        break;
    case TransactionStatus::Offline:
        status = tr("Offline");
        break;
    case TransactionStatus::Unconfirmed:
        status = tr("Unconfirmed");
        break;
    case TransactionStatus::Abandoned:
        status = tr("Abandoned");
        break;
    case TransactionStatus::Confirming:
        status = tr("Confirming (%1 of %2 recommended confirmations)").arg(wtx->status.depth).arg(TransactionRecord::RecommendedNumConfirmations);
        break;
    case TransactionStatus::Confirmed:
        status = tr("Confirmed (%1 confirmations)").arg(wtx->status.depth);
        break;
    case TransactionStatus::Conflicted:
        status = tr("Conflicted");
        break;
    case TransactionStatus::Immature:
        status = tr("Immature (%1 confirmations, will be available after %2)").arg(wtx->status.depth).arg(wtx->status.depth + wtx->status.matures_in);
        break;
    case TransactionStatus::MaturesWarning:
        status = tr("This block was not received by any other nodes and will probably not be accepted!");
        break;
    case TransactionStatus::NotAccepted:
        status = tr("Orphan Block - Generated but not accepted. This does not impact your holdings.");
        break;
    }

    return status;
}

QString TransactionTableModel::formatTxDate(const TransactionRecord* wtx) const
{
    if (wtx->time) {
        return GUIUtil::dateTimeStr(wtx->time);
    }
    return QString();
}

/* Look up address in address book, if found return label (address)
   otherwise just return (address)
 */
QString TransactionTableModel::lookupAddress(const std::string& address, bool tooltip) const
{
    QString label = walletModel->getAddressTableModel()->labelForAddress(QString::fromStdString(address));
    QString description;
    if (!label.isEmpty()) {
        description += label;
    }
    if (label.isEmpty() || tooltip) {
        description += QString(" (") + QString::fromStdString(address) + QString(")");
    }
    return description;
}

QString TransactionTableModel::formatTxType(const TransactionRecord* wtx) const
{
    switch (wtx->type) {
    case TransactionRecord::Fluid:
        return tr("Fluid");
    case TransactionRecord::RecvWithAddress:
        return tr("Received with");
    case TransactionRecord::RecvFromOther:
        return tr("Received from");
    case TransactionRecord::RecvWithPrivateSend:
        return tr("Received via PrivateSend");
    case TransactionRecord::SendToAddress:
    case TransactionRecord::SendToOther:
        return tr("Sent to");
    case TransactionRecord::SendToSelf:
        return tr("Payment to yourself");
    case TransactionRecord::DNReward:
        return tr("Dynode Reward");
    case TransactionRecord::Generated:
        return tr("Mined");
    case TransactionRecord::Stake:
        return tr("Stake");
    case TransactionRecord::NewDomainUser:
    case TransactionRecord::UpdateDomainUser:
    case TransactionRecord::DeleteDomainUser:
    case TransactionRecord::RevokeDomainUser:
        return tr("BDAP User Entry");
    case TransactionRecord::NewDomainGroup:
    case TransactionRecord::UpdateDomainGroup:
    case TransactionRecord::DeleteDomainGroup:
    case TransactionRecord::RevokeDomainGroup:
        return tr("BDAP Group Entry");
    case TransactionRecord::LinkRequest:
        return tr("BDAP Link Request");
    case TransactionRecord::LinkAccept:
        return tr("BDAP Link Accepted");
    case TransactionRecord::NewAudit:
        return tr("BDAP Audit");
    case TransactionRecord::PrivateSendDenominate:
        return tr("PrivateSend Denominate");
    case TransactionRecord::PrivateSendCollateralPayment:
        return tr("PrivateSend Collateral Payment");
    case TransactionRecord::PrivateSendMakeCollaterals:
        return tr("PrivateSend Make Collateral Inputs");
    case TransactionRecord::PrivateSendCreateDenominations:
        return tr("PrivateSend Create Denominations");
    case TransactionRecord::PrivateSend:
        return tr("PrivateSend");
    case TransactionRecord::Issue:
        return tr("Asset Issued");
    case TransactionRecord::Reissue:
        return tr("Asset Reissued");
    case TransactionRecord::TransferFrom:
        return tr("Assets Received");
    case TransactionRecord::TransferTo:
        return tr("Assets Sent");
    default:
        return QString();
    }
}

QVariant TransactionTableModel::txAddressDecoration(const TransactionRecord* wtx) const
{
    QString theme = GUIUtil::getThemeName();
    switch (wtx->type) {
    case TransactionRecord::Fluid:
        return QIcon(":/icons/" + theme + "/fluid");
    case TransactionRecord::DNReward:
        return QIcon(":/icons/" + theme + "/dynode_network");
    case TransactionRecord::Generated:
        return QIcon(":/icons/" + theme + "/tx_mined");
    case TransactionRecord::Stake:
        return QIcon(":/icons/" + theme + "/pos");
    case TransactionRecord::RecvWithPrivateSend:
    case TransactionRecord::RecvWithAddress:
    case TransactionRecord::RecvFromOther:
        return QIcon(":/icons/" + theme + "/tx_input");
    case TransactionRecord::PrivateSend:
    case TransactionRecord::SendToAddress:
    case TransactionRecord::SendToOther:
        return QIcon(":/icons/" + theme + "/tx_output");
    case TransactionRecord::NewDomainUser:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::UpdateDomainUser:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::DeleteDomainUser:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::RevokeDomainUser:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::NewDomainGroup:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::UpdateDomainGroup:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::DeleteDomainGroup:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::RevokeDomainGroup:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::NewAudit:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::LinkRequest:
    case TransactionRecord::LinkAccept:
        return QIcon(":/icons/" + theme + "/bdap");
    case TransactionRecord::Issue:
    case TransactionRecord::Reissue:
    case TransactionRecord::TransferFrom:
        return QIcon(":/icons/tx_asset_input");
    case TransactionRecord::TransferTo:
        return QIcon(":/icons/tx_asset_output");
    default:
        return QIcon(":/icons/" + theme + "/tx_inout");
    }
}

QString TransactionTableModel::formatTxToAddress(const TransactionRecord* wtx, bool tooltip) const
{
    QString watchAddress;
    if (tooltip) {
        // Mark transactions involving watch-only addresses by adding " (watch-only)"
        watchAddress = wtx->involvesWatchAddress ? QString(" (") + tr("watch-only") + QString(")") : "";
    }

    switch (wtx->type) {
    case TransactionRecord::Fluid:
        return tr("Fluid"); // //TODO: Add Fluid Operation Type here
    case TransactionRecord::RecvFromOther:
        return QString::fromStdString(wtx->address) + watchAddress;
    case TransactionRecord::RecvWithAddress:
    case TransactionRecord::RecvWithPrivateSend:
    case TransactionRecord::SendToAddress:
    case TransactionRecord::DNReward:
    case TransactionRecord::Generated:
    case TransactionRecord::Stake:
    case TransactionRecord::PrivateSend:
        return lookupAddress(wtx->address, tooltip) + watchAddress;
    case TransactionRecord::SendToOther:
        return QString::fromStdString(wtx->address) + watchAddress;
    case TransactionRecord::NewDomainUser:
        return tr("New Directory User");
    case TransactionRecord::UpdateDomainUser:
        return tr("Update Directory User");
    case TransactionRecord::DeleteDomainUser:
        return tr("Delete Directory User");
    case TransactionRecord::RevokeDomainUser:
        return tr("Revoke Directory User");
    case TransactionRecord::NewDomainGroup:
        return tr("New Directory Group");
    case TransactionRecord::UpdateDomainGroup:
        return tr("Update Directory Group");
    case TransactionRecord::DeleteDomainGroup:
        return tr("Delete Directory Group");
    case TransactionRecord::RevokeDomainGroup:
        return tr("Revoke Directory Group");
    case TransactionRecord::LinkRequest:
        return tr("Link Request");
    case TransactionRecord::LinkAccept:
        return tr("Link Accepted");
    case TransactionRecord::NewAudit:
        return tr("New Audit Entry");
    case TransactionRecord::SendToSelf:
    default:
        return tr("(n/a)") + watchAddress;
    }
}

QVariant TransactionTableModel::addressColor(const TransactionRecord* wtx) const
{
    // Show addresses without label in a less visible color
    switch (wtx->type) {
    case TransactionRecord::Fluid:
    case TransactionRecord::RecvWithAddress:
    case TransactionRecord::SendToAddress:
    case TransactionRecord::DNReward: {
        QString label = walletModel->getAddressTableModel()->labelForAddress(QString::fromStdString(wtx->address));
        if (label.isEmpty())
            return COLOR_BAREADDRESS;
    }
    case TransactionRecord::Generated:
    case TransactionRecord::Stake:
    case TransactionRecord::PrivateSend:
    case TransactionRecord::RecvWithPrivateSend:
    case TransactionRecord::NewDomainUser:
    case TransactionRecord::NewAudit:
    case TransactionRecord::UpdateDomainUser:
    case TransactionRecord::DeleteDomainUser:
    case TransactionRecord::RevokeDomainUser:
    case TransactionRecord::NewDomainGroup:
    case TransactionRecord::UpdateDomainGroup:
    case TransactionRecord::DeleteDomainGroup:
    case TransactionRecord::RevokeDomainGroup:
    case TransactionRecord::LinkRequest:
    case TransactionRecord::LinkAccept: {
        QString label = walletModel->getAddressTableModel()->labelForAddress(QString::fromStdString(wtx->address));
        if (label.isEmpty())
            return COLOR_BAREADDRESS;
    } break;
    case TransactionRecord::SendToSelf:
    case TransactionRecord::PrivateSendCreateDenominations:
    case TransactionRecord::PrivateSendDenominate:
    case TransactionRecord::PrivateSendMakeCollaterals:
    case TransactionRecord::PrivateSendCollateralPayment:
        return COLOR_BAREADDRESS;
    default:
        break;
    }
    return QVariant();
}

QString TransactionTableModel::formatTxAmount(const TransactionRecord* wtx, bool showUnconfirmed, DynamicUnits::SeparatorStyle separators) const
{
    QString str = DynamicUnits::format(walletModel->getOptionsModel()->getDisplayUnit(), wtx->credit + wtx->debit, false, separators);
    if (showUnconfirmed) {
        if (!wtx->status.countsForBalance) {
            str = QString("[") + str + QString("]");
        }
    }
    return QString(str);
}

QVariant TransactionTableModel::txStatusDecoration(const TransactionRecord* wtx) const
{
    QString theme = GUIUtil::getThemeName();
    switch (wtx->status.status) {
    case TransactionStatus::OpenUntilBlock:
    case TransactionStatus::OpenUntilDate:
        return COLOR_TX_STATUS_OPENUNTILDATE;
    case TransactionStatus::Offline:
        return COLOR_TX_STATUS_OFFLINE;
    case TransactionStatus::Unconfirmed:
        return QIcon(":/icons/" + theme + "/transaction_0");
    case TransactionStatus::Abandoned:
        return QIcon(":/icons/" + theme + "/transaction_abandoned");
    case TransactionStatus::Confirming:
        switch (wtx->status.depth) {
        case 1: 
            return QIcon(":/icons/" + theme + "/transaction_1");
        case 2: 
            return QIcon(":/icons/" + theme + "/transaction_2");
        case 3: 
            return QIcon(":/icons/" + theme + "/transaction_3");
        case 4: 
            return QIcon(":/icons/" + theme + "/transaction_4");
        default: 
            return QIcon(":/icons/" + theme + "/transaction_5");
        };
    case TransactionStatus::Confirmed:
        return QIcon(":/icons/" + theme + "/transaction_confirmed");
    case TransactionStatus::Conflicted:
        return QIcon(":/icons/" + theme + "/transaction_conflicted");
    case TransactionStatus::Immature: {
        int total = wtx->status.depth + wtx->status.matures_in;
        int part = (wtx->status.depth * 5 / total) + 1;
        return QIcon(QString(":/icons/" + theme + "/transaction_%1").arg(part));
    }
    case TransactionStatus::MaturesWarning:
    case TransactionStatus::NotAccepted:
        return QIcon(":/icons/" + theme + "/transaction_0");
    default:
        return COLOR_BLACK;
    }
}

QVariant TransactionTableModel::txWatchonlyDecoration(const TransactionRecord* wtx) const
{
    QString theme = GUIUtil::getThemeName();
    if (wtx->involvesWatchAddress)
        return QIcon(":/icons/" + theme + "/eye");
    else
        return QVariant();
}

QVariant TransactionTableModel::txInstantSendDecoration(const TransactionRecord *wtx) const
{
    if (wtx->status.lockedByInstantSend) {
        QString theme = GUIUtil::getThemeName();
        return QIcon(":/icons/" + theme + "/verify");
    }
    return QVariant();
}

QString TransactionTableModel::formatTooltip(const TransactionRecord* rec) const
{
    QString tooltip = formatTxStatus(rec) + QString("\n") + formatTxType(rec);
    if (rec->type == TransactionRecord::RecvFromOther || rec->type == TransactionRecord::SendToOther ||
        rec->type == TransactionRecord::SendToAddress || rec->type == TransactionRecord::RecvWithAddress || rec->type == TransactionRecord::DNReward) {
        tooltip += QString(" ") + formatTxToAddress(rec, true);
    }
    return tooltip;
}

QVariant TransactionTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();
    TransactionRecord* rec = static_cast<TransactionRecord*>(index.internalPointer());

    switch (role) {
    case RawDecorationRole:
        switch (index.column()) {
        case Status:
            return txStatusDecoration(rec);
        case Watchonly:
            return txWatchonlyDecoration(rec);
        case InstantSend:
            return txInstantSendDecoration(rec);
        case ToAddress:
            return txAddressDecoration(rec);
        case AssetName:
            return QString::fromStdString(rec->assetName);
        }
        break;
    case Qt::DecorationRole: {
        return qvariant_cast<QIcon>(index.data(RawDecorationRole));
    }
    case Qt::DisplayRole:
        switch (index.column()) {
        case Date:
            return formatTxDate(rec);
        case Type:
            return formatTxType(rec);
        case ToAddress:
            return formatTxToAddress(rec, false);
        case Amount:
            return formatTxAmount(rec, true, DynamicUnits::separatorAlways);
        case AssetName:
            if (rec->assetName != "DYN")
               return QString::fromStdString(rec->assetName);
            else
               return QString(DynamicUnits::name(walletModel->getOptionsModel()->getDisplayUnit()));
        }
        break;
    case Qt::EditRole:
        // Edit role is used for sorting, so return the unformatted values
        switch (index.column()) {
        case Status:
            return QString::fromStdString(rec->status.sortKey);
        case Date:
            return rec->time;
        case Type:
            return formatTxType(rec);
        case Watchonly:
            return (rec->involvesWatchAddress ? 1 : 0);
        case InstantSend:
            return (rec->status.lockedByInstantSend ? 1 : 0);
        case ToAddress:
            return formatTxToAddress(rec, true);
        case Amount:
            return qint64(rec->credit + rec->debit);
        case AssetName:
            return QString::fromStdString(rec->assetName);
        }
        break;
    case Qt::ToolTipRole:
        return formatTooltip(rec);
    case Qt::TextAlignmentRole:
        return column_alignments[index.column()];
    case Qt::ForegroundRole:
        // Use the "danger" color for abandoned transactions
        if (rec->status.status == TransactionStatus::Abandoned) {
            return COLOR_TX_STATUS_DANGER;
        }
        if(rec->status.lockedByInstantSend) {
            return COLOR_TX_STATUS_LOCKED;
        }
        // Non-confirmed (but not immature) as transactions are grey
        if (!rec->status.countsForBalance && rec->status.status != TransactionStatus::Immature) {
            return COLOR_UNCONFIRMED;
        }
        // Fluid Transactions
        if (rec->type == TransactionRecord::Fluid) {
            return COLOR_FLUID_TX;
        }
        // Dynode Rewards
        if (rec->type == TransactionRecord::DNReward) {
                return COLOR_DYNODE_REWARD;
        }
        // Generated Rewards
        if (rec->type == TransactionRecord::Generated) {
                return COLOR_GENERATED;
        }
        // Stake Rewards
        if (rec->type == TransactionRecord::Stake) {
            if (rec->status.status == TransactionStatus::Conflicted || rec->status.status == TransactionStatus::NotAccepted)
                return COLOR_ORPHAN;
            else
                return COLOR_STAKE;
        }
        if (index.column() == Amount && (rec->credit + rec->debit) < 0) {
            return COLOR_NEGATIVE;
        }
        if (index.column() == ToAddress) {
            return addressColor(rec);
        }
        if(index.column() == AssetName)
        {
            if (rec->assetName != "DYN")
               return platformStyle->AssetTxColor();
        }
        // To avoid overriding above conditional formats a default text color for this QTableView is not defined in stylesheet,
        // so we must always return a color here
        return COLOR_BLACK;
    case TypeRole:
        return rec->type;
    case DateRole:
        return QDateTime::fromTime_t(static_cast<uint>(rec->time));
    case WatchonlyRole:
        return rec->involvesWatchAddress;
    case WatchonlyDecorationRole:
        return txWatchonlyDecoration(rec);
    case InstantSendRole:
        return rec->status.lockedByInstantSend;
    case InstantSendDecorationRole:
        return txInstantSendDecoration(rec);
    case LongDescriptionRole:
        return priv->describe(rec, walletModel->getOptionsModel()->getDisplayUnit());
    case AddressRole:
        return QString::fromStdString(rec->address);
    case LabelRole:
        return walletModel->getAddressTableModel()->labelForAddress(QString::fromStdString(rec->address));
    case AmountRole:
        return qint64(rec->credit + rec->debit);
    case TxIDRole:
        return rec->getTxID();
    case TxHashRole:
        return QString::fromStdString(rec->hash.ToString());
    case TxHexRole:
        return priv->getTxHex(rec);
    case TxPlainTextRole: {
        QString details;
        QString txLabel = walletModel->getAddressTableModel()->labelForAddress(QString::fromStdString(rec->address));

        details.append(formatTxDate(rec));
        details.append(" ");
        details.append(formatTxStatus(rec));
        details.append(". ");
        if (!formatTxType(rec).isEmpty()) {
            details.append(formatTxType(rec));
            details.append(" ");
        }
        if (!rec->address.empty()) {
            if (txLabel.isEmpty())
                details.append(tr("(no label)") + " ");
            else {
                details.append("(");
                details.append(txLabel);
                details.append(") ");
            }
            details.append(QString::fromStdString(rec->address));
            details.append(" ");
        }
        details.append(formatTxAmount(rec, false, DynamicUnits::separatorNever));
        return details;
    }
    case ConfirmedRole:
        return rec->status.countsForBalance;
    case FormattedAmountRole:
        // Used for copy/export, so don't include separators
        return formatTxAmount(rec, false, DynamicUnits::separatorNever);
    case AssetNameRole:
        {
            QString assetName;
            if (rec->assetName != "DYN")
               assetName.append(QString::fromStdString(rec->assetName));
            else
               assetName.append(QString(DynamicUnits::name(walletModel->getOptionsModel()->getDisplayUnit())));
            return assetName;
        }
    case StatusRole:
        return rec->status.status;
    }
    return QVariant();
}

QVariant TransactionTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal) {
        if (role == Qt::DisplayRole) {
            return columns[section];
        } else if (role == Qt::TextAlignmentRole) {
            return column_alignments[section];
        } else if (role == Qt::ToolTipRole) {
            switch (section) {
            case Status:
                return tr("Transaction status. Hover over this field to show number of confirmations.");
            case Date:
                return tr("Date and time that the transaction was received.");
            case Type:
                return tr("Type of transaction.");
            case Watchonly:
                return tr("Whether or not a watch-only address is involved in this transaction.");
            case InstantSend:
                return tr("Whether or not this transaction was locked by InstantSend.");
            case ToAddress:
                return tr("User-defined intent/purpose of the transaction.");
            case Amount:
                return tr("Amount removed from or added to balance.");
            case AssetName:
                return tr("The asset (or DYN) removed or added to balance.");
            }
        }
    }
    return QVariant();
}

QModelIndex TransactionTableModel::index(int row, int column, const QModelIndex& parent) const
{
    Q_UNUSED(parent);
    TransactionRecord* data = priv->index(row);
    if (data) {
        return createIndex(row, column, priv->index(row));
    }
    return QModelIndex();
}

void TransactionTableModel::updateDisplayUnit()
{
    // emit dataChanged to update Amount column with the current unit
    //updateAmountColumnTitle();
    Q_EMIT dataChanged(index(0, Amount), index(priv->size() - 1, Amount));
}

// queue notifications to show a non freezing progress dialog e.g. for rescan
struct TransactionNotification {
public:
    TransactionNotification() {}
    TransactionNotification(uint256 _hash, ChangeType _status, bool _showTransaction) : hash(_hash), status(_status), showTransaction(_showTransaction) {}

    void invoke(QObject* ttm)
    {
        QString strHash = QString::fromStdString(hash.GetHex());
        qDebug() << "NotifyTransactionChanged: " + strHash + " status= " + QString::number(status);
        QMetaObject::invokeMethod(ttm, "updateTransaction", Qt::QueuedConnection,
            Q_ARG(QString, strHash),
            Q_ARG(int, status),
            Q_ARG(bool, showTransaction));
    }

private:
    uint256 hash;
    ChangeType status;
    bool showTransaction;
};

static bool fQueueNotifications = false;
static std::vector<TransactionNotification> vQueueNotifications;

static void NotifyTransactionChanged(TransactionTableModel* ttm, CWallet* wallet, const uint256& hash, ChangeType status)
{
    // Find transaction in wallet
    std::map<uint256, CWalletTx>::iterator mi = wallet->mapWallet.find(hash);
    // Determine whether to show transaction or not (determine this here so that no relocking is needed in GUI thread)
    bool inWallet = mi != wallet->mapWallet.end();
    bool showTransaction = (inWallet && TransactionRecord::showTransaction(mi->second));

    TransactionNotification notification(hash, status, showTransaction);

    if (fQueueNotifications) {
        vQueueNotifications.push_back(notification);
        return;
    }
    notification.invoke(ttm);
}

static void ShowProgress(TransactionTableModel* ttm, const std::string& title, int nProgress)
{
    if (nProgress == 0)
        fQueueNotifications = true;

    if (nProgress == 100) {
        fQueueNotifications = false;
        if (vQueueNotifications.size() > 10) // prevent balloon spam, show maximum 10 balloons
            QMetaObject::invokeMethod(ttm, "setProcessingQueuedTransactions", Qt::QueuedConnection, Q_ARG(bool, true));
        for (unsigned int i = 0; i < vQueueNotifications.size(); ++i) {
            if (vQueueNotifications.size() - i <= 10)
                QMetaObject::invokeMethod(ttm, "setProcessingQueuedTransactions", Qt::QueuedConnection, Q_ARG(bool, false));

            vQueueNotifications[i].invoke(ttm);
        }
        std::vector<TransactionNotification>().swap(vQueueNotifications); // clear
    }
}

void TransactionTableModel::subscribeToCoreSignals()
{
    // Connect signals to wallet
    wallet->NotifyTransactionChanged.connect(boost::bind(NotifyTransactionChanged, this, _1, _2, _3));
    wallet->ShowProgress.connect(boost::bind(ShowProgress, this, _1, _2));
}

void TransactionTableModel::unsubscribeFromCoreSignals()
{
    // Disconnect signals from wallet
    wallet->NotifyTransactionChanged.disconnect(boost::bind(NotifyTransactionChanged, this, _1, _2, _3));
    wallet->ShowProgress.disconnect(boost::bind(ShowProgress, this, _1, _2));
}
