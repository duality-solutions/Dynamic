// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2009-2017 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynamicaddressvalidator.h"

#include "base58.h"

/* Base58 characters are:
     "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

  This is:
  - All numbers except for '0'
  - All upper-case letters except for 'I' and 'O'
  - All lower-case letters except for 'l'
*/

DynamicAddressEntryValidator::DynamicAddressEntryValidator(QObject *parent) :
    QValidator(parent)
{
}

QValidator::State DynamicAddressEntryValidator::validate(QString &input, int &pos) const
{
    Q_UNUSED(pos);

    // Empty address is "intermediate" input
    if (input.isEmpty())
        return QValidator::Intermediate;

    // Correction
    for (int idx = 0; idx < input.size();)
    {
        bool removeChar = false;
        QChar ch = input.at(idx);
        // Corrections made are very conservative on purpose, to avoid
        // users unexpectedly getting away with typos that would normally
        // be detected, and thus sending to the wrong address.
        switch(ch.unicode())
        {
        // Qt categorizes these as "Other_Format" not "Separator_Space"
        case 0x200B: // ZERO WIDTH SPACE
        case 0xFEFF: // ZERO WIDTH NO-BREAK SPACE
            removeChar = true;
            break;
        default:
            break;
        }

        // Remove whitespace
        if (ch.isSpace())
            removeChar = true;

        // To next character
        if (removeChar)
            input.remove(idx, 1);
        else
            ++idx;
    }

    // Validation
    QValidator::State state = QValidator::Acceptable;
    for (int idx = 0; idx < input.size(); ++idx)
    {
        int ch = input.at(idx).unicode();

        if (((ch >= '0' && ch<='9') ||
            (ch >= 'a' && ch<='z') ||
            (ch >= 'A' && ch<='Z')) &&
            ch != 'l' && ch != 'I' && ch != '0' && ch != 'O')
        {
            // Alphanumeric and not a 'forbidden' character
        }
        else
        {
            state = QValidator::Invalid;
        }
    }

    return state;
}

DynamicAddressCheckValidator::DynamicAddressCheckValidator(QObject *parent) :
    QValidator(parent)
{
}

QValidator::State DynamicAddressCheckValidator::validate(QString &input, int &pos) const
{
    Q_UNUSED(pos);
    // Validate the passed Dynamic address
    CDynamicAddress addr(input.toStdString());
    if (addr.IsValid())
        return QValidator::Acceptable;

    return QValidator::Invalid;
}
