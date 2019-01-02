// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2013-2016 The NovaCoin developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_NTP_H
#define DYNAMIC_NTP_H

// Get time from random server and return server address.
int64_t NtpGetTime(CNetAddr& ip);

// Get time from provided server.
int64_t NtpGetTime(const std::string& strHostName);

// Get time from random server.
int64_t NtpGetTime();

extern std::string strTrustedUpstream;

// NTP time samples thread.
void ThreadNtpSamples();

// NTP offset
int64_t GetNtpOffset();

#endif // DYNAMIC_NTP_H
