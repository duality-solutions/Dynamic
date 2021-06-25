// Copyright (c) 2016-2021 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2021 The Dash Core Developers
// Copyright (c) 2009-2021 The Bitcoin Developers
// Copyright (c) 2009-2021 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#if defined(HAVE_CONFIG_H)
#include "config/dynamic-config.h"
#endif

#include "utiltime.h"

#include "tinyformat.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

#include <ctime>

static int64_t nMockTime = 0; //! For unit testing

int64_t GetTime()
{
    if (nMockTime)
        return nMockTime;

    time_t now = time(NULL);
    assert(now > 0);
    return now;
}

void SetMockTime(int64_t nMockTimeIn)
{
    nMockTime = nMockTimeIn;
}

int64_t GetTimeMillis()
{
    int64_t now = (boost::posix_time::microsec_clock::universal_time() -
                   boost::posix_time::ptime(boost::gregorian::date(1970, 1, 1)))
                      .total_milliseconds();
    assert(now > 0);
    return now;
}

int64_t GetTimeMicros()
{
    int64_t now = (boost::posix_time::microsec_clock::universal_time() -
                   boost::posix_time::ptime(boost::gregorian::date(1970, 1, 1)))
                      .total_microseconds();
    assert(now > 0);
    return now;
}

int64_t GetSystemTimeInSeconds()
{
    return GetTimeMicros() / 1000000;
}

/** Return a time useful for the debug log */
int64_t GetLogTimeMicros()
{
    if (nMockTime)
        return nMockTime * 1000000;

    return GetTimeMicros();
}

void MilliSleep(int64_t n)
{
/**
 * Boost's sleep_for was uninterruptable when backed by nanosleep from 1.50
 * until fixed in 1.52. Use the deprecated sleep method for the broken case.
 * See: https://svn.boost.org/trac/boost/ticket/7238
 */
#if defined(HAVE_WORKING_BOOST_SLEEP_FOR)
    boost::this_thread::sleep_for(boost::chrono::milliseconds(n));
#elif defined(HAVE_WORKING_BOOST_SLEEP)
    boost::this_thread::sleep(boost::posix_time::milliseconds(n));
#else
//should never get here
#error missing boost sleep implementation
#endif
}

std::string DateTimeStrFormat(const char* pszFormat, int64_t nTime)
{
    static std::locale classic(std::locale::classic());
    // std::locale takes ownership of the pointer
    std::locale loc(classic, new boost::posix_time::time_facet(pszFormat));
    std::stringstream ss;
    ss.imbue(loc);
    ss << boost::posix_time::from_time_t(nTime);
    return ss.str();
}

std::string DurationToDHMS(int64_t nDurationTime)
{
    int seconds = nDurationTime % 60;
    nDurationTime /= 60;
    int minutes = nDurationTime % 60;
    nDurationTime /= 60;
    int hours = nDurationTime % 24;
    int days = nDurationTime / 24;
    if (days)
        return strprintf("%dd %02dh:%02dm:%02ds", days, hours, minutes, seconds);
    if (hours)
        return strprintf("%02dh:%02dm:%02ds", hours, minutes, seconds);
    return strprintf("%02dm:%02ds", minutes, seconds);
}

std::string FormatISO8601DateTime(int64_t nTime) {
    struct tm ts;
    time_t time_val = nTime;
#ifdef HAVE_GMTIME_R
    gmtime_r(&time_val, &ts);
#else
    gmtime_s(&ts, &time_val);
#endif
    return strprintf("%04i-%02i-%02iT%02i:%02i:%02iZ", ts.tm_year + 1900, ts.tm_mon + 1, ts.tm_mday, ts.tm_hour, ts.tm_min, ts.tm_sec);
}

std::string FormatISO8601Date(int64_t nTime) {
    struct tm ts;
    time_t time_val = nTime;
#ifdef HAVE_GMTIME_R
    gmtime_r(&time_val, &ts);
#else
    gmtime_s(&ts, &time_val);
#endif
    return strprintf("%04i-%02i-%02i", ts.tm_year + 1900, ts.tm_mon + 1, ts.tm_mday);
}

int64_t AddMonthsToCurrentEpoch(const short nMonths)
{
    struct std::tm epoch_date;
    epoch_date.tm_hour = 0;   epoch_date.tm_min = 0; epoch_date.tm_sec = 0;
    epoch_date.tm_year = 70; epoch_date.tm_mon = 0; epoch_date.tm_mday = 1;

    boost::gregorian::date dt = boost::gregorian::day_clock::universal_day();
    short nYear = dt.year() + ((dt.month() + nMonths)/12);
    short nMonth = (dt.month() + nMonths) % 12;
    short nDay = dt.day();
    //LogPrintf("%s -- nYear %d, nMonth %d, nDay %d\n", __func__, nYear, nMonth, nDay);
    struct std::tm month_date;
    month_date.tm_hour = 0;   month_date.tm_min = 0; month_date.tm_sec = 0;
    month_date.tm_year = nYear - 1900; month_date.tm_mon = nMonth -1; month_date.tm_mday = nDay;

    int64_t seconds = (int64_t)std::difftime(std::mktime(&month_date), std::mktime(&epoch_date));

    return seconds + SECONDS_PER_DAY;
}

int64_t AddMonthsToBlockTime(const uint32_t& nBlockTime, const short nMonths)
{
    boost::gregorian::date dt = boost::posix_time::from_time_t(nBlockTime).date();
    short nYear = dt.year() + ((dt.month() + nMonths)/12);
    short nMonth = (dt.month() + nMonths) % 12;
    short nDay = dt.day();
    //LogPrintf("%s -- nYear %d, nMonth %d, nDay %d\n", __func__, nYear, nMonth, nDay);
    struct std::tm month_date;
    month_date.tm_hour = 0;   month_date.tm_min = 0; month_date.tm_sec = 0;
    month_date.tm_year = nYear - 1900; month_date.tm_mon = nMonth -1; month_date.tm_mday = nDay;
    time_t mkTimeEnd = std::mktime(&month_date);
    time_t mkTimeStart = 0;
#if defined(WIN32) || defined(_WIN32)
    std::tm timeInfo = {};
#ifdef HAVE_GMTIME_R
    gmtime_r(&mkTimeStart, &timeInfo);
#else
    gmtime_s(&timeInfo, &mkTimeStart);
#endif
    mkTimeStart = std::mktime(&timeInfo);
#else
    struct std::tm epoch_date;
    epoch_date.tm_hour = 0;   epoch_date.tm_min = 0; epoch_date.tm_sec = 0;
    epoch_date.tm_year = 70; epoch_date.tm_mon = 0; epoch_date.tm_mday = 1;
    mkTimeStart = std::mktime(&epoch_date);
#endif
    if (mkTimeStart < 0)
        mkTimeStart == 0;

    int64_t seconds = (int64_t)std::difftime(mkTimeEnd, mkTimeStart);
    //LogPrintf("%s -- seconds %d\n", __func__, seconds);
    return seconds + SECONDS_PER_DAY;
}

uint16_t MonthsFromBlockToExpire(const uint32_t& nBlockTime, const uint64_t& nExpireTime)
{
    boost::gregorian::date dtBlock = boost::posix_time::from_time_t(nBlockTime).date();
    boost::gregorian::date dtExpire = boost::posix_time::from_time_t(nExpireTime).date();
    return (uint16_t)((dtExpire.year() - dtBlock.year())*12 + dtExpire.month() - dtBlock.month());
}
