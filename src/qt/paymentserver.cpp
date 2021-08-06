// Copyright (c) 2016-2021 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2021 The Dash Core Developers
// Copyright (c) 2009-2021 The Bitcoin Developers
// Copyright (c) 2009-2021 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "paymentserver.h"

#include "dynamicunits.h"
#include "guiutil.h"
#include "optionsmodel.h"

#include "base58.h"
#include "chainparams.h"
#include "policy/policy.h"
#include "ui_interface.h"
#include "util.h"
#include "validation.h" // For minRelayTxFee
#include "wallet/wallet.h"

#include <cstdlib>

#include <QApplication>
#include <QByteArray>
#include <QDataStream>
#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QFileOpenEvent>
#include <QHash>
#include <QList>
#include <QLocalServer>
#include <QLocalSocket>
#include <QStringList>
#include <QUrlQuery>
#endif

const int DYNAMIC_IPC_CONNECT_TIMEOUT = 1000; // milliseconds
const QString DYNAMIC_IPC_PREFIX("dynamic:");

//
// Create a name that is unique for:
//  testnet / non-testnet
//  data directory
//
static QString ipcServerName()
{
    QString name("DynamicQt");

    // Append a simple hash of the datadir
    // Note that GetDataDir(true) returns a different path
    // for -testnet versus main net
    QString ddir(GUIUtil::boostPathToQString(GetDataDir(true)));
    name.append(QString::number(qHash(ddir)));

    return name;
}

//
// We store payment URIs and requests received before
// the main GUI window is up and ready to ask the user
// to send payment.

static QList<QString> savedPaymentRequests;

//
// Sending to the server is done synchronously, at startup.
// If the server isn't already running, startup continues,
// and the items in savedPaymentRequest will be handled
// when uiReady() is called.
//
// Warning: ipcSendCommandLine() is called early in init,
// so don't use "Q_EMIT message()", but "QMessageBox::"!
//
void PaymentServer::ipcParseCommandLine(int argc, char* argv[])
{
    for (int i = 1; i < argc; i++) {
        QString arg(argv[i]);
        if (arg.startsWith("-"))
            continue;

        // If the dynamic: URI contains a payment request, we are not able to detect the
        // network as that would require fetching and parsing the payment request.
        // That means clicking such an URI which contains a testnet payment request
        // will start a mainnet instance and throw a "wrong network" error.
        if (arg.startsWith(DYNAMIC_IPC_PREFIX, Qt::CaseInsensitive)) // dynamic: URI
        {
            savedPaymentRequests.append(arg);

            SendCoinsRecipient r;
            if (GUIUtil::parseDynamicURI(arg, &r) && !r.address.isEmpty()) {
                CDynamicAddress address(r.address.toStdString());

                if (address.IsValid(Params(CBaseChainParams::MAIN))) {
                    SelectParams(CBaseChainParams::MAIN);
                } else if (address.IsValid(Params(CBaseChainParams::TESTNET))) {
                    SelectParams(CBaseChainParams::TESTNET);
                }
            }
        }
    }
}

//
// Sending to the server is done synchronously, at startup.
// If the server isn't already running, startup continues,
// and the items in savedPaymentRequest will be handled
// when uiReady() is called.
//
bool PaymentServer::ipcSendCommandLine()
{
    bool fResult = false;
    Q_FOREACH (const QString& r, savedPaymentRequests) {
        QLocalSocket* socket = new QLocalSocket();
        socket->connectToServer(ipcServerName(), QIODevice::WriteOnly);
        if (!socket->waitForConnected(DYNAMIC_IPC_CONNECT_TIMEOUT)) {
            delete socket;
            socket = NULL;
            return false;
        }

        QByteArray block;
        QDataStream out(&block, QIODevice::WriteOnly);
        out.setVersion(QDataStream::Qt_4_0);
        out << r;
        out.device()->seek(0);

        socket->write(block);
        socket->flush();
        socket->waitForBytesWritten(DYNAMIC_IPC_CONNECT_TIMEOUT);
        socket->disconnectFromServer();

        delete socket;
        socket = NULL;
        fResult = true;
    }

    return fResult;
}

PaymentServer::PaymentServer(QObject* parent, bool startLocalServer) :
    QObject(parent),
    saveURIs(true),
    uriServer(0),
    optionsModel(0)
{
    // Install global event filter to catch QFileOpenEvents
    // on Mac: sent when you click dynamic: links
    // other OSes: helpful when dealing with payment request files
    if (parent)
        parent->installEventFilter(this);

    QString name = ipcServerName();

    // Clean up old socket leftover from a crash:
    QLocalServer::removeServer(name);

    if (startLocalServer) {
        uriServer = new QLocalServer(this);

        if (!uriServer->listen(name)) {
            // constructor is called early in init, so don't use "Q_EMIT message()" here
            QMessageBox::critical(0, tr("Payment request error"),
                tr("Cannot start dynamic: click-to-pay handler"));
        } else {
            connect(uriServer, SIGNAL(newConnection()), this, SLOT(handleURIConnection()));
        }
    }
}

PaymentServer::~PaymentServer()
{
}
//
// OSX-specific way of handling dynamic: URIs
//
bool PaymentServer::eventFilter(QObject* object, QEvent* event)
{
    if (event->type() == QEvent::FileOpen) {
        QFileOpenEvent* fileEvent = static_cast<QFileOpenEvent*>(event);
        if (!fileEvent->file().isEmpty())
            handleURIOrFile(fileEvent->file());
        else if (!fileEvent->url().isEmpty())
            handleURIOrFile(fileEvent->url().toString());

        return true;
    }

    return QObject::eventFilter(object, event);
}

void PaymentServer::uiReady()
{
    saveURIs = false;
    Q_FOREACH (const QString& s, savedPaymentRequests) {
        handleURIOrFile(s);
    }
    savedPaymentRequests.clear();
}

void PaymentServer::handleURIOrFile(const QString& s)
{
    if (saveURIs) {
        savedPaymentRequests.append(s);
        return;
    }

    if (s.startsWith(DYNAMIC_IPC_PREFIX, Qt::CaseInsensitive)) // dynamic: URI
    {
#if QT_VERSION < 0x050000
        QUrl uri(s);
#else
        QUrlQuery uri((QUrl(s)));
        // normal URI
        {
            SendCoinsRecipient recipient;
            if (GUIUtil::parseDynamicURI(s, &recipient))
            {
                if (!IsValidDestinationString(recipient.address.toStdString())) {
                    if (uri.hasQueryItem("r")) {  // payment request
                        Q_EMIT message(tr("URI handling"),
                            tr("Cannot process payment request because BIP70 is not supported.")+
                            tr("Due to widespread security flaws in BIP70 it's strongly recommended that any merchant instructions to switch wallets be ignored.")+
                            tr("If you are receiving this error you should request the merchant provide a BIP21 compatible URI."),
                            CClientUIInterface::ICON_WARNING);
                    } else {
                        Q_EMIT message(tr("URI handling"), tr("Invalid payment address %1").arg(recipient.address),
                            CClientUIInterface::MSG_ERROR);
                    }
                }
                else
                    Q_EMIT receivedPaymentRequest(recipient);
            } else
                Q_EMIT message(tr("URI handling"),
                    tr("URI cannot be parsed! This can be caused by an invalid Dynamic address or malformed URI parameters."),
                    CClientUIInterface::ICON_WARNING);

            return;
        }
    }

    if (QFile::exists(s)) // payment request file
    {
      Q_EMIT message(tr("Payment request file handling"),
          tr("Cannot process payment request because BIP70 is not supported.")+
          tr("Due to widespread security flaws in BIP70 it's strongly recommended that any merchant instructions to switch wallets be ignored.")+
          tr("If you are receiving this error you should request the merchant provide a BIP21 compatible URI."),
          CClientUIInterface::ICON_WARNING);
    }
}

void PaymentServer::handleURIConnection()
{
    QLocalSocket* clientConnection = uriServer->nextPendingConnection();

    while (clientConnection->bytesAvailable() < (int)sizeof(quint32))
        clientConnection->waitForReadyRead();

    connect(clientConnection, SIGNAL(disconnected()),
        clientConnection, SLOT(deleteLater()));

    QDataStream in(clientConnection);
    in.setVersion(QDataStream::Qt_4_0);
    if (clientConnection->bytesAvailable() < (int)sizeof(quint16)) {
        return;
    }
    QString msg;
    in >> msg;

    handleURIOrFile(msg);
}

void PaymentServer::setOptionsModel(OptionsModel *_optionsModel)
{
    this->optionsModel = _optionsModel;
}
