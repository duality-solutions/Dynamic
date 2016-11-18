// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DARKSILK_QT_DARKSILKADDRESSVALIDATOR_H
#define DARKSILK_QT_DARKSILKADDRESSVALIDATOR_H

#include <QValidator>

/** Base58 entry widget validator, checks for valid characters and
 * removes some whitespace.
 */
class DarkSilkAddressEntryValidator : public QValidator
{
    Q_OBJECT

public:
    explicit DarkSilkAddressEntryValidator(QObject *parent);

    State validate(QString &input, int &pos) const;
};

/** DarkSilk address widget validator, checks for a valid DarkSilk address.
 */
class DarkSilkAddressCheckValidator : public QValidator
{
    Q_OBJECT

public:
    explicit DarkSilkAddressCheckValidator(QObject *parent);

    State validate(QString &input, int &pos) const;
};

#endif // DARKSILK_QT_DARKSILKADDRESSVALIDATOR_H
