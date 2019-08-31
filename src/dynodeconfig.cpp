// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "dynodeconfig.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

CDynodeConfig dynodeConfig;

void CDynodeConfig::add(const std::string& alias, const std::string& ip, const std::string& privKey, const std::string& txHash, const std::string& outputIndex)
{
    CDynodeEntry cme(alias, ip, privKey, txHash, outputIndex);
    entries.push_back(cme);
}

bool CDynodeConfig::read(std::string& strErr)
{
    int linenumber = 1;
    boost::filesystem::path pathDynodeConfigFile = GetDynodeConfigFile();
    boost::filesystem::ifstream streamConfig(pathDynodeConfigFile);

    if (!streamConfig.good()) {
        FILE* configFile = fopen(pathDynodeConfigFile.string().c_str(), "a");
        if (configFile != NULL) {
            std::string strHeader = "# Dynode config file\n"
                                    "# Format: alias IP:port dynodepairingkey collateral_output_txid collateral_output_index\n"
                                    "# Example: dn1 123.123.123.123:33300 93HaYBVUCYjEMeeH1Y4sBGLALQZE1Yc1K64xiqgX37tGBDQL8Xg 2bcd3c84c84f87eaa86e4e56834c92927a07f9e18718810b92e0d0324456a67c 0\n";
            fwrite(strHeader.c_str(), std::strlen(strHeader.c_str()), 1, configFile);
            fclose(configFile);
        }
        return true; // Nothing to read, so just return
    }

    for (std::string line; std::getline(streamConfig, line); linenumber++) {
        if (line.empty())
            continue;

        std::istringstream iss(line);
        std::string comment, alias, ip, privKey, txHash, outputIndex;

        if (iss >> comment) {
            if (comment.at(0) == '#')
                continue;
            iss.str(line);
            iss.clear();
        }

        if (!(iss >> alias >> ip >> privKey >> txHash >> outputIndex)) {
            iss.str(line);
            iss.clear();
            if (!(iss >> alias >> ip >> privKey >> txHash >> outputIndex)) {
                strErr = _("Could not parse dynode.conf") + "\n" +
                         strprintf(_("Line: %d"), linenumber) + "\n\"" + line + "\"";
                streamConfig.close();
                return false;
            }
        }

        int port = 0;
        std::string hostname = "";
        SplitHostPort(ip, port, hostname);
        if (port == 0 || hostname == "") {
            strErr = _("Failed to parse host:port string") + "\n" +
                     strprintf(_("Line: %d"), linenumber) + "\n\"" + line + "\"";
            streamConfig.close();
            return false;
        }
        int mainnetDefaultPort = DEFAULT_P2P_PORT;
        if (Params().NetworkIDString() == CBaseChainParams::MAIN) {
            if (port != mainnetDefaultPort) {
                strErr = _("Invalid port detected in dynode.conf") + "\n" +
                         strprintf(_("Port: %d"), port) + "\n" +
                         strprintf(_("Line: %d"), linenumber) + "\n\"" + line + "\"" + "\n" +
                         strprintf(_("(must be %d for mainnet)"), mainnetDefaultPort);
                streamConfig.close();
                return false;
            }
        } else if (port == mainnetDefaultPort) {
            strErr = _("Invalid port detected in dynode.conf") + "\n" +
                     strprintf(_("Line: %d"), linenumber) + "\n\"" + line + "\"" + "\n" +
                     strprintf(_("(%d could be used only on mainnet)"), mainnetDefaultPort);
            streamConfig.close();
            return false;
        }


        add(alias, ip, privKey, txHash, outputIndex);
    }

    streamConfig.close();
    return true;
}
