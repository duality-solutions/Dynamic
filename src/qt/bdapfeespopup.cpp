// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdapfeespopup.h"
#include "wallet/wallet.h"

bool bdapFeesPopup(QWidget *parentDialog, const opcodetype& opCodeAction, const opcodetype& opCodeObject, BDAP::ObjectType inputAccountType, int unit, int32_t regMonths)
{
    QMessageBox::StandardButton reply;
    QString questionString = QObject::tr("Are you sure you want to add/modify BDAP object?<br />");

    questionString.append(QObject::tr("<br /><br />"));

    CAmount monthlyFee; 
    CAmount oneTimeFee;
    CAmount depositFee;
    CAmount totalAmount;
    //DynamicUnits::Unit u;
    bool displayMonths = false;
    CAmount currBalance = pwalletMain->GetBalance();

    //only display months for BDAP objects/transactions that include it. may need to expand in the future
    if ( (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_ACCOUNT_ENTRY) && ( inputAccountType == BDAP::ObjectType::BDAP_USER || inputAccountType == BDAP::ObjectType::BDAP_GROUP ) )
        displayMonths = true;

    if (!GetBDAPFees(opCodeAction,opCodeObject,inputAccountType,regMonths,monthlyFee,oneTimeFee,depositFee))
    {
        QMessageBox::critical(0, QObject::tr("Error calculating fees"),QObject::tr("Cannot calculate fees at this time."));
        return false;
    }

    totalAmount = monthlyFee + oneTimeFee + depositFee;

    if (totalAmount > currBalance)
    {
        QMessageBox::critical(parentDialog, QObject::tr("BDAP Transaction"), QObject::tr("The amount exceeds your balance."));
        return false;
    }

    questionString.append(QObject::tr("<b>%1</b> will be withdrawn from any available funds (not anonymous).<br />") .arg(DynamicUnits::formatHtmlWithUnit(unit, totalAmount)));
    questionString.append(QObject::tr("<hr>"));
    //questionString.append(QObject::tr("Current balance = <b>%1</b><br />") .arg(DynamicUnits::formatHtmlWithUnit(u, currBalance)));
    questionString.append(QObject::tr("Total amount = <b>%1</b><br />") .arg(DynamicUnits::formatHtmlWithUnit(unit, totalAmount)));
    questionString.append(QObject::tr("Monthly fee = %1" ) .arg(DynamicUnits::formatHtmlWithUnit(unit, monthlyFee)));
    if (displayMonths)
        questionString.append(QObject::tr("&nbsp;(for %1 months)") .arg(regMonths));
    questionString.append(QObject::tr("<br />"));
    questionString.append(QObject::tr("One Time Fee = %1<br />") .arg(DynamicUnits::formatHtmlWithUnit(unit, oneTimeFee)));
    questionString.append(QObject::tr("Deposit Fee = %1") .arg(DynamicUnits::formatHtmlWithUnit(unit, depositFee)));
    questionString.append(QObject::tr("<hr>"));

    reply = QMessageBox::question(parentDialog, QObject::tr("Confirm BDAP Transaction Amount"), questionString, QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        return true;
    }
    else {
        return false;
    }

} //bdapFeesPopup

