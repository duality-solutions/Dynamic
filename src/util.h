// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2018 The Dash Core Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

/**
 * Server/client environment: argument handling, config file parsing,
 * logging, thread wrappers
 */
#ifndef DYNAMIC_UTIL_H
#define DYNAMIC_UTIL_H

#if defined(HAVE_CONFIG_H)
#include "config/dynamic-config.h"
#endif

#include "amount.h"
#include "compat.h"
#include "tinyformat.h"
#include "utiltime.h"

#include <atomic>
#include <exception>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

#include <boost/thread/exceptions.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/signals2/signal.hpp>

#ifndef WIN32
#include <signal.h>
#endif

// Debugging macros

// Uncomment the following line to enable debugging messages
// or enable on a per file basis prior to inclusion of util.h
//#define ENABLE_DYNAMIC_DEBUG
#ifdef ENABLE_DYNAMIC_DEBUG
#define DBG( x ) x
#else
#define DBG( x ) 
#endif

//Dynamic only features

extern bool fDynodeMode;
extern bool fLiteMode;
extern int nWalletBackups;

static const bool DEFAULT_LOGTIMEMICROS  = false;
static const bool DEFAULT_LOGIPS         = false;
static const bool DEFAULT_LOGTIMESTAMPS  = true;
static const bool DEFAULT_LOGTHREADNAMES = false;

/**
 * Default average PoW block span time.
 */
static const int64_t DEFAULT_AVERAGE_POW_BLOCK_TIME = 2 * 64; // Dynamic average block span time is set to 128 seconds
/**
 * Maximum amount of time that a block timestamp is allowed to exceed the
 * current network-adjusted time before the block will be accepted.
 */
static const int64_t MAX_FUTURE_BLOCK_TIME = 12 * DEFAULT_AVERAGE_POW_BLOCK_TIME; // ~26 minutes for Dynamic or 12 blocks

/** Signals for translation. */
class CTranslationInterface
{
public:
    /** Translate a message to the native language of the user. */
    boost::signals2::signal<std::string (const char* psz)> Translate;
};

extern const std::map<std::string, std::vector<std::string> >& mapMultiArgs;
extern bool fDebug;
extern bool fPrintToConsole;
extern bool fPrintToDebugLog;
extern bool fLogTimestamps;
extern bool fLogTimeMicros;
extern bool fLogThreadNames;
extern bool fLogIPs;
extern std::atomic<bool> fReopenDebugLog;
extern CTranslationInterface translationInterface;

extern const char * const DYNAMIC_CONF_FILENAME;
extern const char * const DYNAMIC_PID_FILENAME;

/**
 * Translation function: Call Translate signal on UI interface, which returns a boost::optional result.
 * If no translation slot is registered, nothing is returned, and simply return the input.
 */
inline std::string _(const char* psz)
{
    boost::optional<std::string> rv = translationInterface.Translate(psz);
    return rv ? (*rv) : psz;
}

void SetupEnvironment();
bool SetupNetworking();

/** Return true if log accepts specified category */
bool LogAcceptCategory(const char* category);
/** Send a string to the log output */
int LogPrintStr(const std::string &str);

#define LogPrintf(...) LogPrint(NULL, __VA_ARGS__)

template<typename... Args>
static inline int LogPrint(const char* category, const char* fmt, const Args&... args)
{
    if(!LogAcceptCategory(category)) return 0;                            \
    return LogPrintStr(tfm::format(fmt, args...));
}

template<typename... Args>
bool error(const char* fmt, const Args&... args)
{
    LogPrintStr("ERROR: " + tfm::format(fmt, args...) + "\n");
    return false;
}

void PrintExceptionContinue(const std::exception *pex, const char* pszThread);
void ParseParameters(int argc, const char*const argv[]);
void FileCommit(FILE *file);
bool TruncateFile(FILE *file, unsigned int length);
int RaiseFileDescriptorLimit(int nMinFD);
void AllocateFileRange(FILE *file, unsigned int offset, unsigned int length);
bool RenameOver(boost::filesystem::path src, boost::filesystem::path dest);
bool TryCreateDirectory(const boost::filesystem::path& p);
boost::filesystem::path GetDefaultDataDir();
const boost::filesystem::path &GetDataDir(bool fNetSpecific = true);
boost::filesystem::path GetBackupsDir();
std::string GenerateRandomString(unsigned int len);
void ClearDatadirCache();
boost::filesystem::path GetConfigFile(const std::string& confPath);
boost::filesystem::path GetDynodeConfigFile();
#ifndef WIN32
boost::filesystem::path GetPidFile();
void CreatePidFile(const boost::filesystem::path &path, pid_t pid);
#endif
void ReadConfigFile(const std::string& confPath);
#ifdef WIN32
boost::filesystem::path GetSpecialFolderPath(int nFolder, bool fCreate = true);
#endif
void OpenDebugLog();
void ShrinkDebugFile();
void runCommand(const std::string& strCommand);

inline bool IsSwitchChar(char c)
{
#ifdef WIN32
    return c == '-' || c == '/';
#else
    return c == '-';
#endif
}

/**
 * Return true if the given argument has been manually set
 *
 * @param strArg Argument to get (e.g. "-foo")
 * @return true if the argument has been set
 */
bool IsArgSet(const std::string& strArg);

/**
 * Return string argument or default value
 *
 * @param strArg Argument to get (e.g. "-foo")
 * @param default (e.g. "1")
 * @return command-line argument or default value
 */
std::string GetArg(const std::string& strArg, const std::string& strDefault);

/**
 * Return integer argument or default value
 *
 * @param strArg Argument to get (e.g. "-foo")
 * @param default (e.g. 1)
 * @return command-line argument (0 if invalid number) or default value
 */
int64_t GetArg(const std::string& strArg, int64_t nDefault);

/**
 * Return boolean argument or default value
 *
 * @param strArg Argument to get (e.g. "-foo")
 * @param default (true or false)
 * @return command-line argument or default value
 */
bool GetBoolArg(const std::string& strArg, bool fDefault);

/**
 * Set an argument if it doesn't already have a value
 *
 * @param strArg Argument to set (e.g. "-foo")
 * @param strValue Value (e.g. "1")
 * @return true if argument gets set, false if it already had a value
 */
bool SoftSetArg(const std::string& strArg, const std::string& strValue);

/**
 * Set a boolean argument if it doesn't already have a value
 *
 * @param strArg Argument to set (e.g. "-foo")
 * @param fValue Value (e.g. false)
 * @return true if argument gets set, false if it already had a value
 */
bool SoftSetBoolArg(const std::string& strArg, bool fValue);

// Forces a arg setting
void ForceSetArg(const std::string& strArg, const std::string& strValue);
void ForceSetMultiArgs(const std::string& strArg, const std::vector<std::string>& values);
void ForceRemoveArg(const std::string& strArg);

/**
 * Format a string to be used as group of options in help messages
 *
 * @param message Group name (e.g. "RPC server options:")
 * @return the formatted string
 */
std::string HelpMessageGroup(const std::string& message);

/**
 * Format a string to be used as option description in help messages
 *
 * @param option Option message (e.g. "-rpcuser=<user>")
 * @param message Option description (e.g. "Username for JSON-RPC connections")
 * @return the formatted string
 */
std::string HelpMessageOpt(const std::string& option, const std::string& message);

/**
 * Return the number of cores available on the current system.
 * @note This does count virtual cores, such as those provided by HyperThreading.
 */
int GetNumCores();

void SetThreadPriority(int nPriority);
void RenameThread(const char* name);
std::string GetThreadName();

/**
 * .. and a wrapper that just calls func once
 */
template <typename Callable> void TraceThread(const char* name,  Callable func)
{
    std::string s = strprintf("dynamic-%s", name);
    RenameThread(s.c_str());
    try
    {
        LogPrintf("%s thread start\n", name);
        func();
        LogPrintf("%s thread exit\n", name);
    }
    catch (const boost::thread_interrupted&)
    {
        LogPrintf("%s thread interrupt\n", name);
        throw;
    }
    catch (const std::exception& e) {
        PrintExceptionContinue(&e, name);
        throw;
    }
    catch (...) {
        PrintExceptionContinue(NULL, name);
        throw;
    }
}

/**
 * @brief Converts version strings to 4-byte unsigned integer
 * @param strVersion version in "x.x.x" format (decimal digits only)
 * @return 4-byte unsigned integer, most significant byte is always 0
 * Throws std::bad_cast if format doesn\t match.
 */
uint32_t StringVersionToInt(const std::string& strVersion);

/**
 * @brief Converts version as 4-byte unsigned integer to string
 * @param nVersion 4-byte unsigned integer, most significant byte is always 0
 * @return version string in "x.x.x" format (last 3 bytes as version parts)
 * Throws std::bad_cast if format doesn\t match.
 */
std::string IntVersionToString(uint32_t nVersion);

/**
 * @brief Copy of the IntVersionToString, that returns "Invalid version" string
 * instead of throwing std::bad_cast
 * @param nVersion 4-byte unsigned integer, most significant byte is always 0
 * @return version string in "x.x.x" format (last 3 bytes as version parts)
 * or "Invalid version" if can't cast the given value
 */
std::string SafeIntVersionToString(uint32_t nVersion);

#endif // DYNAMIC_UTIL_H
