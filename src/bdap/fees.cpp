// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/fees.h"

#include "bdap/utils.h" // for GetObjectTypeString
#include "util.h" // for LogPrintf

// Default BDAP Monthly Fees
std::map<int32_t, CAmount> mapDefaultMonthlyFees = {
    {BDAP_MONTHY_USER_FEE, 50 * BDAP_CREDIT},
    {BDAP_MONTHY_GROUP_FEE, 200 * BDAP_CREDIT},
    {BDAP_MONTHY_CERTIFICATE_FEE, 100 * BDAP_CREDIT},
    {BDAP_MONTHY_SIDECHAIN_FEE, 1000 * BDAP_CREDIT},
};

// Default BDAP One Time Fees
std::map<int32_t, CAmount> mapOneTimeFees = {
    {BDAP_ONE_TIME_REQUEST_LINK_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_ACCEPT_LINK_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_AUDIT_RECORD_FEE, 100 * BDAP_CREDIT},
    {BDAP_ONE_TIME_UPDATE_USER_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_DELETE_USER_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_UPDATE_GROUP_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_DELETE_GROUP_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_UPDATE_LINK_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_DELETE_LINK_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_UPDATE_CERTIFICATE_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_DELETE_CERTIFICATE_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_UPDATE_SIDECHAIN_FEE, 1 * BDAP_CREDIT},
    {BDAP_ONE_TIME_DELETE_SIDECHAIN_FEE, 1 * BDAP_CREDIT},
};

// Default BDAP Non-Refundable Security Deposit Fees
std::map<int32_t, CAmount> mapNoRefundDeposits = {
    {BDAP_NON_REFUNDABLE_USER_DEPOSIT, 1000 * BDAP_CREDIT},
    {BDAP_NON_REFUNDABLE_GROUP_DEPOSIT, 10000 * BDAP_CREDIT},
    {BDAP_NON_REFUNDABLE_CERTIFICATE_DEPOSIT, 5000 * BDAP_CREDIT},
    {BDAP_NON_REFUNDABLE_SIDECHAIN_DEPOSIT, 25000 * BDAP_CREDIT},
};

bool GetBDAPFees(const opcodetype& opCodeAction, const opcodetype& opCodeObject, const BDAP::ObjectType objType, const uint16_t nMonths, CAmount& monthlyFee, CAmount& oneTimeFee, CAmount& depositFee)
{
    std::string strObjectType = BDAP::GetObjectTypeString((unsigned int)objType);
    LogPrintf("%s -- strObjectType = %s, OpAction %d, OpObject %d\n", __func__, strObjectType, opCodeAction, opCodeObject);
    if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_USER) {
        // new BDAP user account
        oneTimeFee = 0;
        monthlyFee = mapDefaultMonthlyFees[BDAP_MONTHY_USER_FEE] * nMonths;
        depositFee = mapNoRefundDeposits[BDAP_NON_REFUNDABLE_USER_DEPOSIT];

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_GROUP) {
        // new BDAP group account
        oneTimeFee = 0;
        monthlyFee = mapDefaultMonthlyFees[BDAP_MONTHY_GROUP_FEE] * nMonths;
        depositFee = mapNoRefundDeposits[BDAP_NON_REFUNDABLE_GROUP_DEPOSIT];

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_CERTIFICATE && objType == BDAP::ObjectType::BDAP_CERTIFICATE) {
        // new BDAP certificate
        oneTimeFee = 0;
        monthlyFee = mapDefaultMonthlyFees[BDAP_MONTHY_CERTIFICATE_FEE] * nMonths;
        depositFee = mapNoRefundDeposits[BDAP_NON_REFUNDABLE_CERTIFICATE_DEPOSIT];

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_SIDECHAIN && objType == BDAP::ObjectType::BDAP_SIDECHAIN) {
        // new BDAP sidechain entry
        oneTimeFee = 0;
        monthlyFee = mapDefaultMonthlyFees[BDAP_MONTHY_SIDECHAIN_FEE] * nMonths;
        depositFee = mapNoRefundDeposits[BDAP_NON_REFUNDABLE_SIDECHAIN_DEPOSIT];

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_LINK_REQUEST && objType == BDAP::ObjectType::BDAP_LINK_REQUEST) {
        // new BDAP sidechain entry
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_REQUEST_LINK_FEE];
        monthlyFee = 0;
        depositFee = 0;

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_LINK_ACCEPT && objType == BDAP::ObjectType::BDAP_LINK_ACCEPT) {
        // new BDAP sidechain entry
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_ACCEPT_LINK_FEE];
        monthlyFee = 0;
        depositFee = 0;

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_AUDIT && objType == BDAP::ObjectType::BDAP_AUDIT) {
        // new BDAP sidechain entry
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_AUDIT_RECORD_FEE];
        monthlyFee = 0;
        depositFee = 0;

    } else if (opCodeAction == OP_BDAP_MODIFY && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_USER) {
        // new BDAP sidechain entry
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_UPDATE_USER_FEE];
        monthlyFee = 0;
        depositFee = 0;

    } else if (opCodeAction == OP_BDAP_MODIFY && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_GROUP) {
        // new BDAP sidechain entry
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_UPDATE_GROUP_FEE];
        monthlyFee = 0;
        depositFee = 0;

    }
    else {
        LogPrintf("%s -- BDAP operation code pair (%d and %d) for %s not found or unsupported.\n", __func__, opCodeAction, opCodeObject, strObjectType);
        return false;
    }

    return true;
}