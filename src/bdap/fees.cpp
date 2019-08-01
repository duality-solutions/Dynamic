// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/fees.h"

#include "bdap/bdap.h" // for SECONDS_PER_DAY
#include "bdap/utils.h" // for GetObjectTypeString
#include "util.h" // for LogPrintf

#include <boost/date_time/posix_time/posix_time.hpp>
#include <limits>
#include <map>

// Default BDAP Monthly Fees
std::map<int32_t, CFeeItem> mapDefaultMonthlyFees = {
    {BDAP_MONTHY_USER_FEE, CFeeItem(BDAP_MONTHY_USER_FEE, 15 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_MONTHY_GROUP_FEE, CFeeItem(BDAP_MONTHY_GROUP_FEE, 60 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_MONTHY_CERTIFICATE_FEE, CFeeItem(BDAP_MONTHY_CERTIFICATE_FEE, 30 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_MONTHY_SIDECHAIN_FEE, CFeeItem(BDAP_MONTHY_SIDECHAIN_FEE, 300 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
};

// Default BDAP One Time Fees
std::multimap<int32_t, CFeeItem> mapOneTimeFees = {
    {BDAP_ONE_TIME_REQUEST_LINK_FEE, CFeeItem(BDAP_ONE_TIME_REQUEST_LINK_FEE, 30 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_ONE_TIME_ACCEPT_LINK_FEE, CFeeItem(BDAP_ONE_TIME_ACCEPT_LINK_FEE, 30 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_ONE_TIME_AUDIT_RECORD_FEE, CFeeItem(BDAP_ONE_TIME_AUDIT_RECORD_FEE, 30 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
};

// Default BDAP Non-Refundable Security Deposit Fees
std::map<int32_t, CFeeItem> mapNoRefundDeposits = {
    {BDAP_NON_REFUNDABLE_USER_DEPOSIT, CFeeItem(BDAP_NON_REFUNDABLE_USER_DEPOSIT, 300 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_NON_REFUNDABLE_GROUP_DEPOSIT, CFeeItem(BDAP_NON_REFUNDABLE_GROUP_DEPOSIT, 3000 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_NON_REFUNDABLE_CERTIFICATE_DEPOSIT, CFeeItem(BDAP_NON_REFUNDABLE_CERTIFICATE_DEPOSIT, 1500 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
    {BDAP_NON_REFUNDABLE_SIDECHAIN_DEPOSIT, CFeeItem(BDAP_NON_REFUNDABLE_SIDECHAIN_DEPOSIT, 7500 * BDAP_CREDIT, 0, std::numeric_limits<unsigned int>::max())},
};

bool GetBDAPFees(const opcodetype& opCodeAction, const opcodetype& opCodeObject, const BDAP::ObjectType objType, const uint16_t nMonths, CAmount& monthlyFee, CAmount& oneTimeFee, CAmount& depositFee)
{
    std::string strObjectType = BDAP::GetObjectTypeString((unsigned int)objType);
    LogPrint("bdap", "%s -- strObjectType = %s, OpAction %d, OpObject %d\n", __func__, strObjectType, opCodeAction, opCodeObject);
    if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_USER) {
        // Fees for a new BDAP user account
        oneTimeFee = 0;
        CFeeItem feeMonthly;
        std::multimap<int32_t, CFeeItem>::iterator iMonthly = mapDefaultMonthlyFees.find(BDAP_MONTHY_USER_FEE);
        if (iMonthly != mapDefaultMonthlyFees.end()) {
            feeMonthly = iMonthly->second;
            monthlyFee = (nMonths * feeMonthly.Fee);
        }
        CFeeItem feeDeposit;
        std::multimap<int32_t, CFeeItem>::iterator iDeposit = mapNoRefundDeposits.find(BDAP_NON_REFUNDABLE_USER_DEPOSIT);
        if (iDeposit != mapNoRefundDeposits.end()) {
            feeDeposit = iDeposit->second;
            depositFee = feeDeposit.Fee;
        }

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_GROUP) {
        // Fees for a new BDAP group account
        oneTimeFee = 0;
        CFeeItem feeMonthly;
        std::multimap<int32_t, CFeeItem>::iterator iMonthly = mapDefaultMonthlyFees.find(BDAP_MONTHY_GROUP_FEE);
        if (iMonthly != mapDefaultMonthlyFees.end()) {
            feeMonthly = iMonthly->second;
            monthlyFee = (nMonths * feeMonthly.Fee);
        }
        CFeeItem feeDeposit;
        std::multimap<int32_t, CFeeItem>::iterator iDeposit = mapNoRefundDeposits.find(BDAP_NON_REFUNDABLE_GROUP_DEPOSIT);
        if (iDeposit != mapNoRefundDeposits.end()) {
            feeDeposit = iDeposit->second;
            depositFee = feeDeposit.Fee;
        }

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_CERTIFICATE && objType == BDAP::ObjectType::BDAP_CERTIFICATE) {
        // Fees for a new BDAP certificate
        oneTimeFee = 0;
        CFeeItem feeMonthly;
        std::multimap<int32_t, CFeeItem>::iterator iMonthly = mapDefaultMonthlyFees.find(BDAP_MONTHY_CERTIFICATE_FEE);
        if (iMonthly != mapDefaultMonthlyFees.end()) {
            feeMonthly = iMonthly->second;
            monthlyFee = (nMonths * feeMonthly.Fee);
        }
        CFeeItem feeDeposit;
        std::multimap<int32_t, CFeeItem>::iterator iDeposit = mapNoRefundDeposits.find(BDAP_NON_REFUNDABLE_CERTIFICATE_DEPOSIT);
        if (iDeposit != mapNoRefundDeposits.end()) {
            feeDeposit = iDeposit->second;
            depositFee = feeDeposit.Fee;
        }

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_SIDECHAIN && objType == BDAP::ObjectType::BDAP_SIDECHAIN) {
        // Fees for a new BDAP sidechain entry
        oneTimeFee = 0;
        CFeeItem feeMonthly;
        std::multimap<int32_t, CFeeItem>::iterator iMonthly = mapDefaultMonthlyFees.find(BDAP_MONTHY_SIDECHAIN_FEE);
        if (iMonthly != mapDefaultMonthlyFees.end()) {
            feeMonthly = iMonthly->second;
            monthlyFee = (nMonths * feeMonthly.Fee);
        }
        CFeeItem feeDeposit;
        std::multimap<int32_t, CFeeItem>::iterator iDeposit = mapNoRefundDeposits.find(BDAP_NON_REFUNDABLE_SIDECHAIN_DEPOSIT);
        if (iDeposit != mapNoRefundDeposits.end()) {
            feeDeposit = iDeposit->second;
            depositFee = feeDeposit.Fee;
        }

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_LINK_REQUEST && objType == BDAP::ObjectType::BDAP_LINK_REQUEST) {
        // Fees for a new BDAP link request
        CFeeItem feeOneTime;
        std::multimap<int32_t, CFeeItem>::iterator iOneTime = mapOneTimeFees.find(BDAP_ONE_TIME_REQUEST_LINK_FEE);
        if (iOneTime != mapOneTimeFees.end()) {
            feeOneTime = iOneTime->second;
            oneTimeFee = feeOneTime.Fee;
        }
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_LINK_ACCEPT && objType == BDAP::ObjectType::BDAP_LINK_ACCEPT) {
        // Fees for a new BDAP link accept
        CFeeItem feeOneTime;
        std::multimap<int32_t, CFeeItem>::iterator iOneTime = mapOneTimeFees.find(BDAP_ONE_TIME_ACCEPT_LINK_FEE);
        if (iOneTime != mapOneTimeFees.end()) {
            feeOneTime = iOneTime->second;
            oneTimeFee = feeOneTime.Fee;
        }
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;

    } else if (opCodeAction == OP_BDAP_NEW && opCodeObject == OP_BDAP_AUDIT && objType == BDAP::ObjectType::BDAP_AUDIT) {
        // Fees for a new BDAP audit record
        CFeeItem feeOneTime;
        std::multimap<int32_t, CFeeItem>::iterator iOneTime = mapOneTimeFees.find(BDAP_ONE_TIME_AUDIT_RECORD_FEE);
        if (iOneTime != mapOneTimeFees.end()) {
            feeOneTime = iOneTime->second;
            oneTimeFee = feeOneTime.Fee;
        }
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;

    } else if (opCodeAction == OP_BDAP_MODIFY && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_USER) {
        // Fees for an update BDAP user account entry
        oneTimeFee = BDAP_CREDIT;
        CFeeItem feeMonthly;
        std::multimap<int32_t, CFeeItem>::iterator iMonthly = mapDefaultMonthlyFees.find(BDAP_MONTHY_USER_FEE);
        if (iMonthly != mapDefaultMonthlyFees.end()) {
            feeMonthly = iMonthly->second;
            monthlyFee = (nMonths * feeMonthly.Fee);
        }
        if (monthlyFee == 0)
            monthlyFee = BDAP_CREDIT;
        depositFee = 0;

    } else if (opCodeAction == OP_BDAP_MODIFY && opCodeObject == OP_BDAP_ACCOUNT_ENTRY && objType == BDAP::ObjectType::BDAP_GROUP) {
        // Fees for an update BDAP group account entry
        oneTimeFee = BDAP_CREDIT;
        CFeeItem feeMonthly;
        std::multimap<int32_t, CFeeItem>::iterator iMonthly = mapDefaultMonthlyFees.find(BDAP_MONTHY_GROUP_FEE);
        if (iMonthly != mapDefaultMonthlyFees.end()) {
            feeMonthly = iMonthly->second;
            monthlyFee = (nMonths * feeMonthly.Fee);
        }
        if (monthlyFee == 0)
            monthlyFee = BDAP_CREDIT;
        depositFee = 0;

    } else {
        oneTimeFee = BDAP_CREDIT;
        monthlyFee = 0;
        depositFee = BDAP_CREDIT;
        LogPrintf("%s -- BDAP operation code pair (%d and %d) for %s not found or unsupported.\n", __func__, opCodeAction, opCodeObject, strObjectType);
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