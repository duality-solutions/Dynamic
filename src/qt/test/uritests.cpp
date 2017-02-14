// Copyright (c) 2009-2014 The DarkSilk Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "uritests.h"

#include "guiutil.h"
#include "walletmodel.h"

#include <QUrl>

void URITests::uriTests()
{
    SendCoinsRecipient rv;
    QUrl uri;
    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?req-dontexist="));
    QVERIFY(!GUIUtil::parseDarkSilkURI(uri, &rv));

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?dontexist="));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.label == QString());
    QVERIFY(rv.amount == 0);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?label=Some Example Address"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.label == QString("Some Example Address"));
    QVERIFY(rv.amount == 0);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?amount=0.001"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.label == QString());
    QVERIFY(rv.amount == 100000);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?amount=1.001"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.label == QString());
    QVERIFY(rv.amount == 100100000);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?amount=100&label=Some Example"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.amount == 10000000000LL);
    QVERIFY(rv.label == QString("Some Example"));

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?message=Some Example Address"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.label == QString());

    QVERIFY(GUIUtil::parseDarkSilkURI("darksilk://D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?message=Some Example Address", &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.label == QString());

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?req-message=Some Example Address"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?amount=1,000&label=Some Example"));
    QVERIFY(!GUIUtil::parseDarkSilkURI(uri, &rv));

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?amount=1,000.0&label=Some Example"));
    QVERIFY(!GUIUtil::parseDarkSilkURI(uri, &rv));

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?amount=100&label=Some Example&message=Some Example Message&IS=1"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.amount == 10000000000LL);
    QVERIFY(rv.label == QString("Some Example"));
    QVERIFY(rv.message == QString("Some Example Message"));
    QVERIFY(rv.fUseInstantSend == 1);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?amount=100&label=Some Example&message=Some Example Message&IS=Something Invalid"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.address == QString("D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(rv.amount == 10000000000LL);
    QVERIFY(rv.label == QString("Some Example"));
    QVERIFY(rv.message == QString("Some Example Message"));
    QVERIFY(rv.fUseInstantSend != 1);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?IS=1"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.fUseInstantSend == 1);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf?IS=0"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.fUseInstantSend != 1);

    uri.setUrl(QString("darksilk:D8RHNF9Tf7Zsef8gMGL2fhWA9ZslrP4K5tf"));
    QVERIFY(GUIUtil::parseDarkSilkURI(uri, &rv));
    QVERIFY(rv.fUseInstantSend != 1);
}
