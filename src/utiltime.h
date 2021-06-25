// Copyright (c) 2016-2021 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2021 The Dash Core Developers
// Copyright (c) 2009-2021 The Bitcoin Developers
// Copyright (c) 2009-2021 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_UTILTIME_H
#define DYNAMIC_UTILTIME_H

#include <stdint.h>
#include <string>

static constexpr unsigned int SECONDS_PER_DAY             = 86400; // Number of seconds per day.

/**
 * GetTimeMicros() and GetTimeMillis() both return the system time, but in
 * different units. GetTime() returns the sytem time in seconds, but also
 * supports mocktime, where the time can be specified by the user, eg for
 * testing (eg with the setmocktime rpc, or -mocktime argument).
 *
 * TODO: Rework these functions to be type-safe (so that we don't inadvertently
 * compare numbers with different units, or compare a mocktime to system time).
 */

int64_t GetTime();
int64_t GetTimeMillis();
int64_t GetTimeMicros();
int64_t GetSystemTimeInSeconds(); // Like GetTime(), but not mockable
int64_t GetLogTimeMicros();
void SetMockTime(int64_t nMockTimeIn);
void MilliSleep(int64_t n);

std::string DateTimeStrFormat(const char* pszFormat, int64_t nTime);
std::string DurationToDHMS(int64_t nDurationTime);
/**
 * ISO 8601 formatting is preferred. Use the FormatISO8601{DateTime,Date}
 * helper functions if possible.
 */
std::string FormatISO8601DateTime(int64_t nTime);
std::string FormatISO8601Date(int64_t nTime);
/**
 * Used for BDAP registration conversion between months and epoch
 */
int64_t AddMonthsToCurrentEpoch(const short nMonths);
int64_t AddMonthsToBlockTime(const uint32_t& nBlockTime, const short nMonths);
uint16_t MonthsFromBlockToExpire(const uint32_t& nBlockTime, const uint64_t& nExpireTime);
#endif // DYNAMIC_UTILTIME_H
