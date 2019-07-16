// Copyright (c) 2019 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_FEES_H
#define DYNAMIC_BDAP_FEES_H

#include "amount.h"
#include "bdap/bdap.h"
#include "primitives/transaction.h"
#include "script/script.h"

#include <map>

static const int32_t BDAP_MONTHY_USER_FEE                       = 5001;
static const int32_t BDAP_MONTHY_GROUP_FEE                      = 5002;
static const int32_t BDAP_MONTHY_CERTIFICATE_FEE                = 5003;
static const int32_t BDAP_MONTHY_SIDECHAIN_FEE                  = 5004;

static const int32_t BDAP_ONE_TIME_REQUEST_LINK_FEE             = 6001;
static const int32_t BDAP_ONE_TIME_ACCEPT_LINK_FEE              = 6002;
static const int32_t BDAP_ONE_TIME_AUDIT_RECORD_FEE             = 6003;
static const int32_t BDAP_ONE_TIME_UPDATE_USER_FEE              = 6004;
static const int32_t BDAP_ONE_TIME_DELETE_USER_FEE              = 6005;
static const int32_t BDAP_ONE_TIME_UPDATE_GROUP_FEE             = 6006;
static const int32_t BDAP_ONE_TIME_DELETE_GROUP_FEE             = 6007;
static const int32_t BDAP_ONE_TIME_UPDATE_LINK_FEE              = 6008;
static const int32_t BDAP_ONE_TIME_DELETE_LINK_FEE              = 6009;
static const int32_t BDAP_ONE_TIME_UPDATE_CERTIFICATE_FEE       = 6010;
static const int32_t BDAP_ONE_TIME_DELETE_CERTIFICATE_FEE       = 6011;
static const int32_t BDAP_ONE_TIME_UPDATE_SIDECHAIN_FEE         = 6012;
static const int32_t BDAP_ONE_TIME_DELETE_SIDECHAIN_FEE         = 6013;

static const int32_t BDAP_NON_REFUNDABLE_USER_DEPOSIT           = 7001;
static const int32_t BDAP_NON_REFUNDABLE_GROUP_DEPOSIT          = 7002;
static const int32_t BDAP_NON_REFUNDABLE_CERTIFICATE_DEPOSIT    = 7003;
static const int32_t BDAP_NON_REFUNDABLE_SIDECHAIN_DEPOSIT      = 7004;

class CFeeItem {
public:
    static const int CURRENT_VERSION=1;
    int nVersion;
    int32_t nType;
    CAmount Fee;
    unsigned int nStartHeight;
    unsigned int nEndHeight;
    CFeeItem() : nVersion(CURRENT_VERSION), nType(0), Fee(0), nStartHeight(0), nEndHeight(0) {}
    CFeeItem(const int32_t& type, const CAmount& fee, const unsigned int& start, const unsigned int& end) : nVersion(CURRENT_VERSION), nType(type), Fee(fee), nStartHeight(start), nEndHeight(end) {}
};

bool GetBDAPFees(const opcodetype& opCodeAction, const opcodetype& opCodeObject, const BDAP::ObjectType objType, const uint16_t nMonths, CAmount& monthlyFee, CAmount& oneTimeFee, CAmount& depositFee);
int64_t AddMonthsToCurrentEpoch(const short nMonths);
int64_t AddMonthsToBlockTime(const uint32_t& nBlockTime, const short nMonths);
uint16_t MonthsFromBlockToExpire(const uint32_t& nBlockTime, const uint64_t& nExpireTime);
bool ExtractAmountsFromTx(const CTransactionRef& ptx, CAmount& dataAmount, CAmount& opAmount);

#endif // DYNAMIC_BDAP_FEES_H