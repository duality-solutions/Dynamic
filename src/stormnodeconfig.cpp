// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2015-2017 Silk Network Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "net.h"
#include "stormnodeconfig.h"
#include "util.h"
#include "ui_interface.h"
#include "chainparams.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

CStormnodeConfig stormnodeConfig;

void CStormnodeConfig::add(std::string alias, std::string ip, std::string privKey, std::string txHash, std::string outputIndex) {
    CStormnodeEntry cme(alias, ip, privKey, txHash, outputIndex);
    entries.push_back(cme);
}

bool CStormnodeConfig::read(std::string& strErr) {
    int linenumber = 1;
    boost::filesystem::path pathStormnodeConfigFile = GetStormnodeConfigFile();
    boost::filesystem::ifstream streamConfig(pathStormnodeConfigFile);

    if (!streamConfig.good()) {
        FILE* configFile = fopen(pathStormnodeConfigFile.string().c_str(), "a");
        if (configFile != NULL) {
            std::string strHeader = "# Stormnode config file\n"
                          "# Format: alias IP:port stormnodeprivkey collateral_output_txid collateral_output_index\n"
                          "# Example: sn1 127.0.0.2:31000 93HaYBVUCYjEMeeH1Y4sBGLALQZE1Yc1K64xiqgX37tGBDQL8Xg 2bcd3c84c84f87eaa86e4e56834c92927a07f9e18718810b92e0d0324456a67c 0\n";
            fwrite(strHeader.c_str(), std::strlen(strHeader.c_str()), 1, configFile);
            fclose(configFile);
        }
        return true; // Nothing to read, so just return
    }

    for(std::string line; std::getline(streamConfig, line); linenumber++)
    {
        if(line.empty()) continue;

        std::istringstream iss(line);
        std::string comment, alias, ip, privKey, txHash, outputIndex;

        if (iss >> comment) {
            if(comment.at(0) == '#') continue;
            iss.str(line);
            iss.clear();
        }

        if (!(iss >> alias >> ip >> privKey >> txHash >> outputIndex)) {
            iss.str(line);
            iss.clear();
            if (!(iss >> alias >> ip >> privKey >> txHash >> outputIndex)) {
                strErr = _("Could not parse stormnode.conf") + "\n" +
                        strprintf(_("Line: %d"), linenumber) + "\n\"" + line + "\"";
                streamConfig.close();
                return false;
            }
        }

        //This check has been temporarily removed as it seems to prevent Windows Stormnodes from starting
        /*
        int mainnetDefaultPort = Params(CBaseChainParams::MAIN).GetDefaultPort();
        if(Params().NetworkIDString() == CBaseChainParams::MAIN) {
            if(CService(ip).GetPort() != mainnetDefaultPort) {
                strErr = _("Invalid port detected in stormnode.conf") + "\n" +
                        strprintf(_("Line: %d"), linenumber) + "\n\"" + line + "\"" + "\n" +
                        strprintf(_("Port Used: %d"), CService(ip).GetPort()) + "\n" +
                        strprintf(_("IP Address Used: %s"), ip) + "\n" +
                        strprintf(_("Private Key Used: %s"), privKey) + "\n" +
                        strprintf(_("Tx Hash Used: %s"), txHash) + "\n" +
                        strprintf(_("Tx Output Index Used: %s"), outputIndex) + "\n" +
                        strprintf(_("(must be %d for mainnet)"), mainnetDefaultPort);
                streamConfig.close();
                return false;
            }
        } else if(CService(ip).GetPort() == mainnetDefaultPort) {
            strErr = _("Invalid port detected in stormnode.conf") + "\n" +
                    strprintf(_("Line: %d"), linenumber) + "\n\"" + line + "\"" + "\n" +
                    strprintf(_("Port Used: %d"), CService(ip).GetPort()) + "\n" +
                    strprintf(_("IP Address Used: %s"), ip) + "\n" +
                    strprintf(_("Private Key Used: %s"), privKey) + "\n" +
                    strprintf(_("Tx Hash Used: %s"), txHash) + "\n" +
                    strprintf(_("Tx Output Index Used: %s"), outputIndex) + "\n" +
                    strprintf(_("(%d could be used only on mainnet)"), mainnetDefaultPort);
            streamConfig.close();
            return false;
        }

        add(alias, ip, privKey, txHash, outputIndex);
    }*/

    streamConfig.close();
    return true;
}
