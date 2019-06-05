// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/fees.h"

#include "bdap/bdap.h" // for SECONDS_PER_DAY
#include "bdap/utils.h" // for GetObjectTypeString
#include "util.h" // for LogPrintf

#include <boost/date_time/posix_time/posix_time.hpp>

// Default BDAP Monthly Fees
std::map<int32_t, CAmount> mapDefaultMonthlyFees = {
    {BDAP_MONTHY_USER_FEE, 50 * BDAP_CREDIT},
    {BDAP_MONTHY_GROUP_FEE, 200 * BDAP_CREDIT},
    {BDAP_MONTHY_CERTIFICATE_FEE, 100 * BDAP_CREDIT},
    {BDAP_MONTHY_SIDECHAIN_FEE, 1000 * BDAP_CREDIT},
};

// Default BDAP One Time Fees
std::map<int32_t, CAmount> mapOneTimeFees = {
    {BDAP_ONE_TIME_REQUEST_LINK_FEE, 99 * BDAP_CREDIT},
    {BDAP_ONE_TIME_ACCEPT_LINK_FEE, 99 * BDAP_CREDIT},
    {BDAP_ONE_TIME_AUDIT_RECORD_FEE, 99 * BDAP_CREDIT},
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
        // new BDAP link request
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_REQUEST_LINK_FEE];
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_LINK_ACCEPT && objType == BDAP::ObjectType::BDAP_LINK_ACCEPT) {
        // new BDAP link accept
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_ACCEPT_LINK_FEE];
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_AUDIT && objType == BDAP::ObjectType::BDAP_AUDIT) {
        // new BDAP audit record
        oneTimeFee = mapOneTimeFees[BDAP_ONE_TIME_AUDIT_RECORD_FEE];
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;

    } else if (opCodeAction == OP_BDAP_MODIFY && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_USER) {
        // update BDAP user account entry
        oneTimeFee = 0;
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;

    } else if (opCodeAction == OP_BDAP_MODIFY && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_GROUP) {
        // update BDAP group account entry
        oneTimeFee = 0;
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;
    } else if (objType == BDAP::ObjectType::BDAP_DEFAULT_TYPE) {
        // ********** TODO (BDAP): Remove this
        oneTimeFee = 0;
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;
    }
    else {
        LogPrintf("%s -- BDAP operation code pair (%d and %d) for %s not found or unsupported.\n", __func__, opCodeAction, opCodeObject, strObjectType);
        return false;
    }

    return true;
}

int64_t AddMonthsToCurrentEpoch(const short nMonths)
{
    boost::gregorian::date dt = boost::gregorian::day_clock::universal_day();
    short nYear = dt.year() + ((dt.month() + nMonths)/12);
    short nMonth = (dt.month() + nMonths) % 12;
    short nDay = dt.day();
    boost::posix_time::time_duration dur = boost::posix_time::ptime(boost::gregorian::date(nYear, nMonth, nDay)) - boost::posix_time::ptime(boost::gregorian::date(1970, 1, 1));
    //LogPrintf("%s -- nYear %d, nMonth %d, nDay %d\n", __func__, nYear, nMonth, nDay);
    return dur.total_seconds() + SECONDS_PER_DAY;
}

int64_t AddMonthsToBlockTime(const uint32_t& nBlockTime, const short nMonths)
{
    boost::gregorian::date dt = boost::posix_time::from_time_t(nBlockTime).date();
    short nYear = dt.year() + ((dt.month() + nMonths)/12);
    short nMonth = (dt.month() + nMonths) % 12;
    short nDay = dt.day();
    boost::posix_time::time_duration dur = boost::posix_time::ptime(boost::gregorian::date(nYear, nMonth, nDay)) - boost::posix_time::ptime(boost::gregorian::date(1970, 1, 1));
    //LogPrintf("%s -- nYear %d, nMonth %d, nDay %d\n", __func__, nYear, nMonth, nDay);
    return dur.total_seconds() + SECONDS_PER_DAY;
}

uint16_t MonthsFromBlockToExpire(const uint32_t& nBlockTime, const uint64_t& nExpireTime)
{
    boost::gregorian::date dtBlock = boost::posix_time::from_time_t(nBlockTime).date();
    boost::gregorian::date dtExpire = boost::posix_time::from_time_t(nExpireTime).date();
    return (uint16_t)((dtExpire.year() - dtBlock.year())*12 + dtExpire.month() - dtBlock.month());
}

bool ExtractAmountsFromTx(const CTransactionRef& ptx, CAmount& dataAmount, CAmount& opAmount)
{
    bool fDataFound = false, fOpFound = false;
    for (const CTxOut& out : ptx->vout) {
        if (out.scriptPubKey.IsUnspendable() && out.scriptPubKey.size() > 40) 
        {
            dataAmount = out.nValue;
            fDataFound = true;
        }
        int op1, op2;
        std::vector<std::vector<unsigned char>> vOpArgs;
        if (DecodeBDAPScript(out.scriptPubKey, op1, op2, vOpArgs)) {
            opAmount = out.nValue;
            fOpFound = true;
        }
    }
    return (fDataFound && fOpFound);
}