// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_DYNAMICADDRESSVALIDATOR_H
#define DYNAMIC_QT_DYNAMICADDRESSVALIDATOR_H

#include <QValidator>

/** Base58 entry widget validator, checks for valid characters and
 * removes some whitespace.
 */
class DynamicAddressEntryValidator : public QValidator
{
    Q_OBJECT

public:
    explicit DynamicAddressEntryValidator(QObject* parent);

    State validate(QString& input, int& pos) const;
};

/** Dynamic address widget validator, checks for a valid Dynamic address.
 */
class DynamicAddressCheckValidator : public QValidator
{
    Q_OBJECT

public:
    explicit DynamicAddressCheckValidator(QObject* parent);

    State validate(QString& input, int& pos) const;
};

#endif // DYNAMIC_QT_DYNAMICADDRESSVALIDATOR_H
