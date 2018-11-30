// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "bdap/domainentry.h"
#include "bdap/domainentrydb.h"
#include "bdap/utils.h"
#include "chain.h"
#include "core_io.h"
#include "init.h"
#include "key_io.h"
#include "merkleblock.h"
#include "rpcserver.h"
#include "script/script.h"
#include "script/standard.h"
#include "sync.h"
#include "util.h"
#include "utiltime.h"
#include "validation.h"
#include "wallet.h"

#include <univalue.h>
#include <libtorrent/hex.hpp> // for to_hex and from_hex

#include <fstream>
#include <stdint.h>

#include <boost/algorithm/string.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/foreach.hpp>

void EnsureWalletIsUnlocked();
bool EnsureWalletIsAvailable(bool avoidException);

std::string static EncodeDumpTime(int64_t nTime)
{
    return DateTimeStrFormat("%Y-%m-%dT%H:%M:%SZ", nTime);
}

int64_t static DecodeDumpTime(const std::string& str)
{
    static const boost::posix_time::ptime epoch = boost::posix_time::from_time_t(0);
    static const std::locale loc(std::locale::classic(),
        new boost::posix_time::time_input_facet("%Y-%m-%dT%H:%M:%SZ"));
    std::istringstream iss(str);
    iss.imbue(loc);
    boost::posix_time::ptime ptime(boost::date_time::not_a_date_time);
    iss >> ptime;
    if (ptime.is_not_a_date_time())
        return 0;
    return (ptime - epoch).total_seconds();
}

std::string static EncodeDumpString(const std::string& str)
{
    std::stringstream ret;
    BOOST_FOREACH (unsigned char c, str) {
        if (c <= 32 || c >= 128 || c == '%') {
            ret << '%' << HexStr(&c, &c + 1);
        } else {
            ret << c;
        }
    }
    return ret.str();
}

std::string DecodeDumpString(const std::string& str)
{
    std::stringstream ret;
    for (unsigned int pos = 0; pos < str.length(); pos++) {
        unsigned char c = str[pos];
        if (c == '%' && pos + 2 < str.length()) {
            c = (((str[pos + 1] >> 6) * 9 + ((str[pos + 1] - '0') & 15)) << 4) |
                ((str[pos + 2] >> 6) * 9 + ((str[pos + 2] - '0') & 15));
            pos += 2;
        }
        ret << c;
    }
    return ret.str();
}

UniValue importprivkey(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 3)
        throw std::runtime_error(
            "importprivkey \"dynamicprivkey\" ( \"label\" ) ( rescan )\n"
            "\nAdds a private key (as returned by dumpprivkey) to your wallet.\n"
            "\nArguments:\n"
            "1. \"dynamicprivkey\"   (string, required) The private key (see dumpprivkey)\n"
            "2. \"label\"            (string, optional, default=\"\") An optional label\n"
            "3. rescan               (boolean, optional, default=true) Rescan the wallet for transactions\n"
            "\nNote: This call can take minutes to complete if rescan is true.\n"
            "\nExamples:\n"
            "\nDump a private key\n" +
            HelpExampleCli("dumpprivkey", "\"myaddress\"") +
            "\nImport the private key with rescan\n" + HelpExampleCli("importprivkey", "\"mykey\"") +
            "\nImport using a label and without rescan\n" + HelpExampleCli("importprivkey", "\"mykey\" \"testing\" false") +
            "\nImport using default blank label and without rescan\n" + HelpExampleCli("importprivkey", "\"mykey\" \"\" false") +
            "\nAs a JSON-RPC call\n" + HelpExampleRpc("importprivkey", "\"mykey\", \"testing\", false"));


    LOCK2(cs_main, pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();

    std::string strSecret = request.params[0].get_str();
    std::string strLabel = "";
    if (request.params.size() > 1)
        strLabel = request.params[1].get_str();

    bool isskip = false;
    // Whether to perform rescan after import
    bool fRescan = false;
    if (request.params.size() > 2)
        fRescan = request.params[2].get_bool();

    if (fRescan && fPruneMode)
        throw JSONRPCError(RPC_WALLET_ERROR, "Rescan is disabled in pruned mode");

    CDynamicSecret vchSecret;
    bool fGood = vchSecret.SetString(strSecret);

    if (!fGood)
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid private key encoding");

    CKey key = vchSecret.GetKey();
    if (!key.IsValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Private key outside allowed range");

    CPubKey pubkey = key.GetPubKey();
    assert(key.VerifyPubKey(pubkey));
    CKeyID vchAddress = pubkey.GetID();
    {
        pwalletMain->MarkDirty();
        pwalletMain->SetAddressBook(vchAddress, strLabel, "receive");

        // Don't throw error in case a key is already there
        if (pwalletMain->HaveKey(vchAddress)) {
            isskip = true;
        } else {
            
            pwalletMain->mapKeyMetadata[vchAddress].nCreateTime = 1;

            if (!pwalletMain->AddKeyPubKey(key, pubkey))
                throw JSONRPCError(RPC_WALLET_ERROR, "Error adding key to wallet");

            // whenever a key is imported, we need to scan the whole chain
            pwalletMain->UpdateTimeFirstKey(1);
        }

        if (fRescan || !isskip) {
            pwalletMain->ScanForWalletTransactions(chainActive.Genesis(), true);
        }
    }

    return NullUniValue;
}

UniValue importbdapkeys(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;
    
    if (request.fHelp || request.params.size() < 4 || request.params.size() > 5)
        throw std::runtime_error(
            "importbdapkeys \"account id\" \"wallet privkey\" \"link privkey\" \"DHT privkey\" \"rescan\"\n"
            "\nAdds a private keys (as returned by dumpbdapkeys) to your wallet.\n"
            "\nArguments:\n"
            "1. \"account id\"       (string, required) The BDAP account id (see dumpbdapkeys)\n"
            "2. \"wallet privkey\"   (string, required) The BDAP account wallet private key (see dumpbdapkeys)\n"
            "3. \"link privkey\"     (string, required) The BDAP account link address private key (see dumpbdapkeys)\n"
            "4. \"DHT privkey\"      (string, required) The BDAP account DHT private key (see dumpbdapkeys)\n"
            "5. \"rescan\"           (boolean, optional, default=true) Rescan the wallet for transactions\n"
            "\nNote: This call can take minutes to complete if rescan is true.\n"
            "\nExamples:\n"
            "\nDump a private key\n"
            + HelpExampleCli("dumpbdapkeys", "\"account id\" \"wallet_key\" \"link_key\" \"dht_key\" false") +
            "\nImport the private key with rescan\n"
            + HelpExampleCli("importbdapkeys", "\"account id\" \"wallet_key\" \"link_key\" \"dht_key\"") +
            "\nImport using a label and without rescan\n"
            + HelpExampleCli("importbdapkeys", "\"account id\" \"wallet_key\" \"link_key\" \"dht_key\" false") +
            "\nAs a JSON-RPC call\n"
            + HelpExampleRpc("importbdapkeys", "\"account id\" \"wallet_key\" \"link_key\" \"dht_key\" false")
        );

    LOCK2(cs_main, pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();
 
    std::string vchObjectID = request.params[0].get_str();
    std::string strWalletPrivKey = request.params[1].get_str();
    std::string strLinkPrivKey = request.params[2].get_str();
    std::string strDHTPrivKey = request.params[3].get_str();
    // Whether to perform rescan after import
    bool fRescan = true;
    if (request.params.size() > 4)
        fRescan = request.params[2].get_bool();

    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchFromString(vchObjectID);
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry)) {
        throw JSONRPCError(RPC_TYPE_ERROR, "Can not find BDAP entry " + entry.GetFullObjectPath());
    }

    if (fRescan && fPruneMode)
        throw JSONRPCError(RPC_WALLET_ERROR, "Rescan is disabled in pruned mode");

    // Wallet address
    CDynamicSecret vchWalletSecret;
    bool fGood = vchWalletSecret.SetString(strWalletPrivKey);
    if (!fGood) throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid wallet private key encoding");
    CKey keyWallet = vchWalletSecret.GetKey();
    if (!keyWallet.IsValid()) throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Wallet private key outside allowed range");
    CPubKey pubkeyWallet = keyWallet.GetPubKey();
    assert(keyWallet.VerifyPubKey(pubkeyWallet));
    CKeyID vchWalletAddress = pubkeyWallet.GetID();
    // TODO (BDAP): Check if wallet address matches BDAP entry

    // Link address
    CDynamicSecret vchLinkSecret;
    fGood = vchLinkSecret.SetString(strLinkPrivKey);
    if (!fGood) throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid link private key encoding");
    CKey keyLink = vchLinkSecret.GetKey();
    if (!keyLink.IsValid()) throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Link private key outside allowed range");
    CPubKey pubkeyLink = keyLink.GetPubKey();
    assert(keyLink.VerifyPubKey(pubkeyLink));
    CKeyID vchLinkAddress = pubkeyLink.GetID();
    // TODO (BDAP): Check if link address matches BDAP entry
    
    // DHT key
    std::array<char, ED25519_PRIVATE_SEED_BYTE_LENGTH> arrDHTPrivSeedKey;
    libtorrent::aux::from_hex(strDHTPrivKey, arrDHTPrivSeedKey.data());
    CKeyEd25519 privDHTKey(arrDHTPrivSeedKey);
    std::vector<unsigned char> vchDHTPubKey = privDHTKey.GetPubKey();
    CKeyID dhtKeyID(Hash160(vchDHTPubKey.begin(), vchDHTPubKey.end()));

    // Add keys to local wallet database
    {
        pwalletMain->MarkDirty();
        
        if (!pwalletMain->HaveDHTKey(dhtKeyID)) {
            pwalletMain->SetAddressBook(privDHTKey.GetID(), vchObjectID, "bdap-dht-key");
            if (!pwalletMain->AddDHTKey(privDHTKey, vchDHTPubKey)) {
                throw JSONRPCError(RPC_WALLET_ERROR, "Error adding BDAP DHT key to wallet database");
            }
            pwalletMain->mapKeyMetadata[dhtKeyID].nCreateTime = 1;
        }

        // Don't throw error in case all keys already already there
        if (pwalletMain->HaveKey(vchWalletAddress) && pwalletMain->HaveKey(vchLinkAddress))
            return NullUniValue;

        // TODO (BDAP): What is nCreateTime used for?
        pwalletMain->mapKeyMetadata[vchWalletAddress].nCreateTime = 1;
        pwalletMain->mapKeyMetadata[vchLinkAddress].nCreateTime = 1;
        
        if (!pwalletMain->HaveKey(vchWalletAddress)) {
            pwalletMain->SetAddressBook(vchWalletAddress, vchObjectID, "bdap-wallet");
            if (!pwalletMain->AddKeyPubKey(keyWallet, pubkeyWallet))
                throw JSONRPCError(RPC_WALLET_ERROR, "Error adding BDAP wallet key to wallet database");
        }

        if (!pwalletMain->HaveKey(vchLinkAddress)) {
            pwalletMain->SetAddressBook(vchLinkAddress, vchObjectID, "bdap-link");
            if (!pwalletMain->AddKeyPubKey(keyLink, pubkeyLink))
                throw JSONRPCError(RPC_WALLET_ERROR, "Error adding BDAP link key to wallet database");
        }

        // whenever a key is imported, we need to scan the whole chain
        pwalletMain->UpdateTimeFirstKey(1);

        if (fRescan) {
            pwalletMain->ScanForWalletTransactions(chainActive.Genesis(), true);
        }
    }

    return NullUniValue;
}

void ImportAddress(const CDynamicAddress& address, const std::string& strLabel);
void ImportScript(const CScript& script, const std::string& strLabel, bool isRedeemScript)
{
    if (!isRedeemScript && ::IsMine(*pwalletMain, script) == ISMINE_SPENDABLE)
        throw JSONRPCError(RPC_WALLET_ERROR, "The wallet already contains the private key for this address or script");

    pwalletMain->MarkDirty();

    if (!pwalletMain->HaveWatchOnly(script) && !pwalletMain->AddWatchOnly(script, 0 /* nCreateTime */))
        throw JSONRPCError(RPC_WALLET_ERROR, "Error adding address to wallet");

    if (isRedeemScript) {
        if (!pwalletMain->HaveCScript(script) && !pwalletMain->AddCScript(script))
            throw JSONRPCError(RPC_WALLET_ERROR, "Error adding p2sh redeemScript to wallet");
        ImportAddress(CDynamicAddress(CScriptID(script)), strLabel);
    } else {
        CTxDestination destination;
        if (ExtractDestination(script, destination)) {
            pwalletMain->SetAddressBook(destination, strLabel, "receive");
        }
    }
}

void ImportAddress(const CDynamicAddress& address, const std::string& strLabel)
{
    CScript script = GetScriptForDestination(address.Get());
    ImportScript(script, strLabel, false);
    // add to address book or update label
    if (address.IsValid())
        pwalletMain->SetAddressBook(address.Get(), strLabel, "receive");
}

UniValue importaddress(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 4)
        throw std::runtime_error(
            "importaddress \"address\" ( \"label\" rescan p2sh )\n"
            "\nAdds a script (in hex) or address that can be watched as if it were in your wallet but cannot be used to spend.\n"
            "\nArguments:\n"
            "1. \"script\"           (string, required) The hex-encoded script (or address)\n"
            "2. \"label\"            (string, optional, default=\"\") An optional label\n"
            "3. rescan               (boolean, optional, default=true) Rescan the wallet for transactions\n"
            "4. p2sh                 (boolean, optional, default=false) Add the P2SH version of the script as well\n"
            "\nNote: This call can take minutes to complete if rescan is true.\n"
            "If you have the full public key, you should call importpubkey instead of this.\n"
            "\nNote: If you import a non-standard raw script in hex form, outputs sending to it will be treated\n"
            "as change, and not show up in many RPCs.\n"
            "\nExamples:\n"
            "\nImport a script with rescan\n" +
            HelpExampleCli("importaddress", "\"myscript\"") +
            "\nImport using a label without rescan\n" + HelpExampleCli("importaddress", "\"myscript\" \"testing\" false") +
            "\nAs a JSON-RPC call\n" + HelpExampleRpc("importaddress", "\"myscript\", \"testing\", false"));


    std::string strLabel = "";
    if (request.params.size() > 1)
        strLabel = request.params[1].get_str();

    // Whether to perform rescan after import
    bool fRescan = true;
    if (request.params.size() > 2)
        fRescan = request.params[2].get_bool();

    if (fRescan && fPruneMode)
        throw JSONRPCError(RPC_WALLET_ERROR, "Rescan is disabled in pruned mode");

    // Whether to import a p2sh version, too
    bool fP2SH = false;
    if (request.params.size() > 3)
        fP2SH = request.params[3].get_bool();

    LOCK2(cs_main, pwalletMain->cs_wallet);

    CDynamicAddress address(request.params[0].get_str());
    if (address.IsValid()) {
        if (fP2SH)
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Cannot use the p2sh flag with an address - use a script instead");
        ImportAddress(address, strLabel);
    } else if (IsHex(request.params[0].get_str())) {
        std::vector<unsigned char> data(ParseHex(request.params[0].get_str()));
        ImportScript(CScript(data.begin(), data.end()), strLabel, fP2SH);
    } else {
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address or script");
    }

    if (fRescan) {
        pwalletMain->ScanForWalletTransactions(chainActive.Genesis(), true);
        pwalletMain->ReacceptWalletTransactions();
    }

    return NullUniValue;
}

UniValue importprunedfunds(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 2)
        throw std::runtime_error(
            "importprunedfunds\n"
            "\nImports funds without rescan. Corresponding address or script must previously be included in wallet. Aimed towards pruned wallets. The end-user is responsible to import additional transactions that subsequently spend the imported outputs or rescan after the point in the blockchain the transaction is included.\n"
            "\nArguments:\n"
            "1. \"rawtransaction\" (string, required) A raw transaction in hex funding an already-existing address in wallet\n"
            "2. \"txoutproof\"     (string, required) The hex output from gettxoutproof that contains the transaction\n");

    CMutableTransaction tx;
    if (!DecodeHexTx(tx, request.params[0].get_str()))
        throw JSONRPCError(RPC_DESERIALIZATION_ERROR, "TX decode failed");
    uint256 hashTx = tx.GetHash();
    CWalletTx wtx(pwalletMain, MakeTransactionRef(std::move(tx)));

    CDataStream ssMB(ParseHexV(request.params[1], "proof"), SER_NETWORK, PROTOCOL_VERSION);
    CMerkleBlock merkleBlock;
    ssMB >> merkleBlock;

    //Search partial merkle tree in proof for our transaction and index in valid block
    std::vector<uint256> vMatch;
    std::vector<unsigned int> vIndex;
    unsigned int txnIndex = 0;
    if (merkleBlock.txn.ExtractMatches(vMatch, vIndex) == merkleBlock.header.hashMerkleRoot) {
        LOCK(cs_main);

        if (!mapBlockIndex.count(merkleBlock.header.GetHash()) || !chainActive.Contains(mapBlockIndex[merkleBlock.header.GetHash()]))
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Block not found in chain");

        std::vector<uint256>::const_iterator it;
        if ((it = std::find(vMatch.begin(), vMatch.end(), hashTx)) == vMatch.end()) {
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Transaction given doesn't exist in proof");
        }

        txnIndex = vIndex[it - vMatch.begin()];
    } else {
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Something wrong with merkleblock");
    }

    wtx.nIndex = txnIndex;
    wtx.hashBlock = merkleBlock.header.GetHash();

    LOCK2(cs_main, pwalletMain->cs_wallet);

    if (pwalletMain->IsMine(tx)) {
        pwalletMain->AddToWallet(wtx, false);
        return NullUniValue;
    }

    throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "No addresses in wallet correspond to included transaction");
}

UniValue removeprunedfunds(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "removeprunedfunds \"txid\"\n"
            "\nDeletes the specified transaction from the wallet. Meant for use with pruned wallets and as a companion to importprunedfunds. This will effect wallet balances.\n"
            "\nArguments:\n"
            "1. \"txid\"           (string, required) The hex-encoded id of the transaction you are deleting\n"
            "\nExamples:\n" +
            HelpExampleCli("removeprunedfunds", "\"a8d0c0184dde994a09ec054286f1ce581bebf46446a512166eae7628734ea0a5\"") +
            "\nAs a JSON-RPC call\n" + HelpExampleRpc("removprunedfunds", "\"a8d0c0184dde994a09ec054286f1ce581bebf46446a512166eae7628734ea0a5\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    uint256 hash;
    hash.SetHex(request.params[0].get_str());
    std::vector<uint256> vHash;
    vHash.push_back(hash);
    std::vector<uint256> vHashOut;

    if (pwalletMain->ZapSelectTx(vHash, vHashOut) != DB_LOAD_OK) {
        throw JSONRPCError(RPC_INTERNAL_ERROR, "Could not properly delete the transaction.");
    }

    if (vHashOut.empty()) {
        throw JSONRPCError(RPC_INTERNAL_ERROR, "Transaction does not exist in wallet.");
    }

    return NullUniValue;
}

UniValue importpubkey(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 4)
        throw std::runtime_error(
            "importpubkey \"pubkey\" ( \"label\" rescan )\n"
            "\nAdds a public key (in hex) that can be watched as if it were in your wallet but cannot be used to spend.\n"
            "\nArguments:\n"
            "1. \"pubkey\"           (string, required) The hex-encoded public key\n"
            "2. \"label\"            (string, optional, default=\"\") An optional label\n"
            "3. rescan               (boolean, optional, default=true) Rescan the wallet for transactions\n"
            "\nNote: This call can take minutes to complete if rescan is true.\n"
            "\nExamples:\n"
            "\nImport a public key with rescan\n" +
            HelpExampleCli("importpubkey", "\"mypubkey\"") +
            "\nImport using a label without rescan\n" + HelpExampleCli("importpubkey", "\"mypubkey\" \"testing\" false") +
            "\nAs a JSON-RPC call\n" + HelpExampleRpc("importpubkey", "\"mypubkey\", \"testing\", false"));


    std::string strLabel = "";
    if (request.params.size() > 1)
        strLabel = request.params[1].get_str();

    // Whether to perform rescan after import
    bool fRescan = true;
    if (request.params.size() > 2)
        fRescan = request.params[2].get_bool();

    if (fRescan && fPruneMode)
        throw JSONRPCError(RPC_WALLET_ERROR, "Rescan is disabled in pruned mode");

    if (!IsHex(request.params[0].get_str()))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Pubkey must be a hex string");
    std::vector<unsigned char> data(ParseHex(request.params[0].get_str()));
    CPubKey pubKey(data.begin(), data.end());
    if (!pubKey.IsFullyValid())
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Pubkey is not a valid public key");

    LOCK2(cs_main, pwalletMain->cs_wallet);

    ImportAddress(CDynamicAddress(pubKey.GetID()), strLabel);
    ImportScript(GetScriptForRawPubKey(pubKey), strLabel, false);

    if (fRescan) {
        pwalletMain->ScanForWalletTransactions(chainActive.Genesis(), true);
        pwalletMain->ReacceptWalletTransactions();
    }

    return NullUniValue;
}


UniValue importwallet(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() > 2)
        throw std::runtime_error(
            "importwallet \"filename\"\n"
            "\nImports keys from a wallet dump file (see dumpwallet).\n"
            "\nArguments:\n"
            "1. \"filename\"    (string, required) The wallet file\n"
            "2. forcerescan     (boolean, optional, default=true) Rescan the wallet for transactions\n"
            "\nExamples:\n"
            "\nDump the wallet\n" +
            HelpExampleCli("dumpwallet", "\"test\"") +
            "\nImport the wallet\n" + HelpExampleCli("importwallet", "\"test\"") +
            "\nImport using the json rpc call\n" + HelpExampleRpc("importwallet", "\"test\""));

    if (fPruneMode)
        throw JSONRPCError(RPC_WALLET_ERROR, "Importing wallets is disabled in pruned mode");

    LOCK2(cs_main, pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();

    std::ifstream file;
    file.open(request.params[0].get_str().c_str(), std::ios::in | std::ios::ate);
    if (!file.is_open())
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Cannot open wallet dump file");

    bool neednotRescan = true;
    bool forcerescan = false;
    if (!request.params[1].isNull())
        forcerescan = request.params[1].get_bool();

    int64_t nTimeBegin = chainActive.Tip()->GetBlockTime();

    bool fGood = true;

    int64_t nFilesize = std::max((int64_t)1, (int64_t)file.tellg());
    file.seekg(0, file.beg);

    pwalletMain->ShowProgress(_("Importing..."), 0); // show progress dialog in GUI
    while (file.good()) {
        pwalletMain->ShowProgress("", std::max(1, std::min(99, (int)(((double)file.tellg() / (double)nFilesize) * 100))));
        std::string line;
        std::getline(file, line);
        if (line.empty() || line[0] == '#')
            continue;

        std::vector<std::string> vstr;
        boost::split(vstr, line, boost::is_any_of(" "));
        if (vstr.size() < 2)
            continue;
        CDynamicSecret vchSecret;
        if (!vchSecret.SetString(vstr[0]))
            continue;
        CKey key = vchSecret.GetKey();
        CPubKey pubkey = key.GetPubKey();
        assert(key.VerifyPubKey(pubkey));
        CKeyID keyid = pubkey.GetID();
        if (pwalletMain->HaveKey(keyid)) {
            LogPrintf("Skipping import of %s (key already present)\n", CDynamicAddress(keyid).ToString());
            continue;
        }
        neednotRescan=false;
        int64_t nTime = DecodeDumpTime(vstr[1]);
        std::string strLabel;
        bool fLabel = true;
        for (unsigned int nStr = 2; nStr < vstr.size(); nStr++) {
            if (boost::algorithm::starts_with(vstr[nStr], "#"))
                break;
            if (vstr[nStr] == "change=1")
                fLabel = false;
            if (vstr[nStr] == "reserve=1")
                fLabel = false;
            if (boost::algorithm::starts_with(vstr[nStr], "label=")) {
                strLabel = DecodeDumpString(vstr[nStr].substr(6));
                fLabel = true;
            }
        }
        LogPrintf("Importing %s...\n", CDynamicAddress(keyid).ToString());
        if (!pwalletMain->AddKeyPubKey(key, pubkey)) {
            fGood = false;
            continue;
        }
        pwalletMain->mapKeyMetadata[keyid].nCreateTime = nTime;
        if (fLabel)
            pwalletMain->SetAddressBook(keyid, strLabel, "receive");
        nTimeBegin = std::min(nTimeBegin, nTime);
    }
    file.close();
    pwalletMain->ShowProgress("", 100); // hide progress dialog in GUI
    if (forcerescan || !neednotRescan) {
    pwalletMain->UpdateTimeFirstKey(nTimeBegin);

    CBlockIndex* pindex = chainActive.FindEarliestAtLeast(nTimeBegin - 7200);

    LogPrintf("Rescanning last %i blocks\n", pindex ? chainActive.Height() - pindex->nHeight + 1 : 0);
    pwalletMain->ScanForWalletTransactions(pindex);
    pwalletMain->MarkDirty();
    }
   
    if (!fGood)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error adding some keys to wallet");

    return NullUniValue;
}

UniValue importelectrumwallet(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() < 1 || request.params.size() > 2)
        throw std::runtime_error(
            "importelectrumwallet \"filename\" index\n"
            "\nImports keys from an Electrum wallet export file (.csv or .json)\n"
            "\nArguments:\n"
            "1. \"filename\"    (string, required) The Electrum wallet export file, should be in csv or json format\n"
            "2. index         (numeric, optional, default=0) Rescan the wallet for transactions starting from this block index\n"
            "\nExamples:\n"
            "\nImport the wallet\n" +
            HelpExampleCli("importelectrumwallet", "\"test.csv\"") + HelpExampleCli("importelectrumwallet", "\"test.json\"") +
            "\nImport using the json rpc call\n" + HelpExampleRpc("importelectrumwallet", "\"test.csv\"") + HelpExampleRpc("importelectrumwallet", "\"test.json\""));

    if (fPruneMode)
        throw JSONRPCError(RPC_WALLET_ERROR, "Importing wallets is disabled in pruned mode");

    LOCK2(cs_main, pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();

    std::ifstream file;
    std::string strFileName = request.params[0].get_str();
    size_t nDotPos = strFileName.find_last_of(".");
    if (nDotPos == std::string::npos)
        throw JSONRPCError(RPC_INVALID_PARAMETER, "File has no extension, should be .json or .csv");

    std::string strFileExt = strFileName.substr(nDotPos + 1);
    if (strFileExt != "json" && strFileExt != "csv")
        throw JSONRPCError(RPC_INVALID_PARAMETER, "File has wrong extension, should be .json or .csv");

    file.open(strFileName.c_str(), std::ios::in | std::ios::ate);
    if (!file.is_open())
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Cannot open Electrum wallet export file");

    bool fGood = true;

    int64_t nFilesize = std::max((int64_t)1, (int64_t)file.tellg());
    file.seekg(0, file.beg);

    pwalletMain->ShowProgress(_("Importing..."), 0); // show progress dialog in GUI

    if (strFileExt == "csv") {
        while (file.good()) {
            pwalletMain->ShowProgress("", std::max(1, std::min(99, (int)(((double)file.tellg() / (double)nFilesize) * 100))));
            std::string line;
            std::getline(file, line);
            if (line.empty() || line == "address,private_key")
                continue;
            std::vector<std::string> vstr;
            boost::split(vstr, line, boost::is_any_of(","));
            if (vstr.size() < 2)
                continue;
            CDynamicSecret vchSecret;
            if (!vchSecret.SetString(vstr[1]))
                continue;
            CKey key = vchSecret.GetKey();
            CPubKey pubkey = key.GetPubKey();
            assert(key.VerifyPubKey(pubkey));
            CKeyID keyid = pubkey.GetID();
            if (pwalletMain->HaveKey(keyid)) {
                LogPrintf("Skipping import of %s (key already present)\n", CDynamicAddress(keyid).ToString());
                continue;
            }
            LogPrintf("Importing %s...\n", CDynamicAddress(keyid).ToString());
            if (!pwalletMain->AddKeyPubKey(key, pubkey)) {
                fGood = false;
                continue;
            }
        }
    } else {
        // json
        char* buffer = new char[nFilesize];
        file.read(buffer, nFilesize);
        UniValue data(UniValue::VOBJ);
        if (!data.read(buffer))
            throw JSONRPCError(RPC_TYPE_ERROR, "Cannot parse Electrum wallet export file");
        delete[] buffer;

        std::vector<std::string> vKeys = data.getKeys();

        for (size_t i = 0; i < data.size(); i++) {
            pwalletMain->ShowProgress("", std::max(1, std::min(99, int(i * 100 / data.size()))));
            if (!data[vKeys[i]].isStr())
                continue;
            CDynamicSecret vchSecret;
            if (!vchSecret.SetString(data[vKeys[i]].get_str()))
                continue;
            CKey key = vchSecret.GetKey();
            CPubKey pubkey = key.GetPubKey();
            assert(key.VerifyPubKey(pubkey));
            CKeyID keyid = pubkey.GetID();
            if (pwalletMain->HaveKey(keyid)) {
                LogPrintf("Skipping import of %s (key already present)\n", CDynamicAddress(keyid).ToString());
                continue;
            }
            LogPrintf("Importing %s...\n", CDynamicAddress(keyid).ToString());
            if (!pwalletMain->AddKeyPubKey(key, pubkey)) {
                fGood = false;
                continue;
            }
        }
    }
    file.close();
    pwalletMain->ShowProgress("", 100); // hide progress dialog in GUI

    // Whether to perform rescan after import
    int nStartHeight = 0;
    if (request.params.size() > 1)
        nStartHeight = request.params[1].get_int();
    if (chainActive.Height() < nStartHeight)
        nStartHeight = chainActive.Height();

    // Assume that electrum wallet was created at that block
    int nTimeBegin = chainActive[nStartHeight]->GetBlockTime();
    pwalletMain->UpdateTimeFirstKey(nTimeBegin);

    LogPrintf("Rescanning %i blocks\n", chainActive.Height() - nStartHeight + 1);
    pwalletMain->ScanForWalletTransactions(chainActive[nStartHeight], true);

    if (!fGood)
        throw JSONRPCError(RPC_WALLET_ERROR, "Error adding some keys to wallet");

    return NullUniValue;
}

UniValue importmnemonic(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp)) {
        return NullUniValue;
    }

    LOCK2(cs_main, pwalletMain->cs_wallet);
    UniValue entry(UniValue::VOBJ);
    if (request.fHelp || request.params.size() > 4)
        throw std::runtime_error(
            "importmnemonic \"mnemonic\"\n"
            "\nImports mnemonic\n"
            "\nArguments:\n"
            "1. \"mnemonic\"    (string, required) mnemonic delimited by the dash charactor (-)\n"
            "2. \"begin\"       (int, optional) begin\n"
            "3. \"end\"         (int, optional) end\n"
            "4. forcerescan               (boolean, optional, default=true) forcerescan the wallet for transactions\n"
            "\nExamples:\n"
            "\nImports mnemonic\n"
            + HelpExampleCli("importmnemonic", "\"inflict-witness-off-property-target-faint-gather-match-outdoor-weapon-wide-mix\"")
        );
    if (fPruneMode)
        throw JSONRPCError(RPC_WALLET_ERROR, "Importing wallets is disabled in pruned mode");
    
    std::string mnemonicstr = "";
    
    if (!request.params[0].isNull())
        mnemonicstr = request.params[0].get_str();
    
    uint32_t begin = 0, end = 100;
    
    if (!request.params[1].isNull())
        begin = (uint32_t)request.params[1].get_int();
    
    if (!request.params[2].isNull())
        end = (uint32_t)request.params[2].get_int();
    
    bool forcerescan = false;
    
    if(!request.params[3].isNull())
        forcerescan = request.params[3].get_bool();
    
    Mnemonic mnemonic(mnemonicstr);
    
    if(mnemonic.getMnemonic().size() <= 0){
        entry.push_back(Pair("error", "mnemonic size is error"));
        return entry;
    }
    
    int64_t nCreationTime = 1230912000;
    uint32_t nInternalChainCounter = begin;
    uint32_t nExternalChainCounter = begin;
    int skipcount = 0;
    pwalletMain->ShowProgress(_("Importing..."), 0); // show progress dialog in GUI
    bool intenal = false;
    unsigned char *seedkey = mnemonic.MnemonicToSeed();
    
    // for now we use a fixed keypath scheme of m/44‘/0'/0'/0/k
    CExtKey masterKey;             //hd master key
    CExtKey bip44Key;              //bip44 key m/44'
    CExtKey coinTypeKey;           //coin_type key m/44'/0'
    CExtKey accountKey;            //key at m/44'/0'/0'
    CExtKey chainChildKeyexternal;         //key at m/44'/0'/0'/0 (external) or m/44'/0'/0'/1 (internal)
    CExtKey chainChildKeyinternal;
    
    masterKey.SetMaster(seedkey, SEED_KEY_SIZE);
    
    // derive purpose m/44'
    masterKey.Derive(bip44Key, 0x80000000 + 44);
     // derive coin_type(dynamic) m/44'/0'
    bip44Key.Derive(coinTypeKey, 0x80000000);
    // derive account m/44'/0'/0'
    // use hardened derivation (child keys >= 0x80000000 are hardened after bip32)
    coinTypeKey.Derive(accountKey, 0x80000000);
    accountKey.Derive(chainChildKeyexternal, 0);
    accountKey.Derive(chainChildKeyinternal, 1);
    while(nInternalChainCounter <= end || nExternalChainCounter <= end){
        //pwalletMain->ShowProgress("", std::max(1, std::min(99, (int)(((double)file.tellg() / (double)nFilesize) * 100))));
        pwalletMain->ShowProgress("", std::max(1, std::min(99,  (int)(((double)(nInternalChainCounter+nExternalChainCounter) / (double)end)* (double)100 / (double)2))));
        
        if(nExternalChainCounter > end)
            intenal = true;
        
        CKey key;
        pwalletMain->DeriveNewChildKeyBIP44BychainChildKey((intenal?chainChildKeyinternal:chainChildKeyexternal),key,intenal,&nInternalChainCounter,&nExternalChainCounter);
        
        CPubKey pubkey = key.GetPubKey();
        assert(key.VerifyPubKey(pubkey));
        CKeyID keyid = pubkey.GetID();
        if (pwalletMain->HaveKey(keyid)) {
            LogPrintf("Skipping import of %s (key already present)\n", CDynamicAddress(keyid).ToString());
            skipcount++;
            continue;
        }
        std::string strLabel="";
        
        LogPrintf("Importing %s...\n", CDynamicAddress(keyid).ToString());
        if (!pwalletMain->AddKeyPubKey(key, pubkey)) {
            
        }
        pwalletMain->mapKeyMetadata[keyid].nCreateTime = nCreationTime;
        
        pwalletMain->SetAddressBook(keyid, strLabel, "receive");
        //MilliSleep(30);
    }
    int64_t nEndTime = GetTime();
    pwalletMain->ShowProgress("", 100); // hide progress dialog in GUI
    if(skipcount < ((int)end * 2 + 2) || forcerescan ){
        pwalletMain->UpdateTimeFirstKey(nCreationTime);
        pwalletMain->ScanForWalletTransactions(chainActive.Genesis(), true);
        pwalletMain->MarkDirty();
    }
     entry.push_back(Pair("nEndTime", nEndTime - nCreationTime));
    return entry;
}

UniValue dumpprivkey(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "dumpprivkey \"dynamicaddress\"\n"
            "\nReveals the private key corresponding to 'dynamicaddress'.\n"
            "Then the importprivkey can be used with this output\n"
            "\nArguments:\n"
            "1. \"dynamicaddress\"   (string, required) The dynamic address for the private key\n"
            "\nResult:\n"
            "\"key\"                (string) The private key\n"
            "\nExamples:\n" +
            HelpExampleCli("dumpprivkey", "\"myaddress\"") + HelpExampleCli("importprivkey", "\"mykey\"") + HelpExampleRpc("dumpprivkey", "\"myaddress\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();

    std::string strAddress = request.params[0].get_str();
    CDynamicAddress address;
    if (!address.SetString(strAddress))
        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid Dynamic address");
    CKeyID keyID;
    if (!address.GetKeyID(keyID))
        throw JSONRPCError(RPC_TYPE_ERROR, "Address does not refer to a key");
    CKey vchSecret;
    if (!pwalletMain->GetKey(keyID, vchSecret))
        throw JSONRPCError(RPC_WALLET_ERROR, "Private key for address " + strAddress + " is not known");
    return CDynamicSecret(vchSecret).ToString();
}

UniValue dumpbdapkeys(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;
    
    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "dumpbdapkeys \"account id\"\n"
            "\nReveals the private key corresponding to 'BDAP account'.\n"
            "Then the importbdapkeys can be used with this output\n"
            "\nArguments:\n"
            "1. \"account id\"      (string, required) The BDAP id for the private keys\n"
            "\nResult:\n"
            "\"wallet_address\"     (string) The wallet address\n"
            "\"wallet_privkey\"     (string) The wallet address private key\n"
            "\"link_address\"       (string) The link address\n"
            "\"link_privkey\"       (string) The link address private keys\n"
            "\"dht_publickey\"      (string) The DHT public key\n"
            "\"dht_privkey\"        (string) The DHT private key\n"
            "\nExamples:\n"
            + HelpExampleCli("dumpbdapkeys", "\"Alice\"")
            + HelpExampleRpc("dumpbdapkeys", "\"Alice\"")
        );

    LOCK2(cs_main, pwalletMain->cs_wallet);

    UniValue result(UniValue::VOBJ);

    EnsureWalletIsUnlocked();

    CharString vchObjectID = vchFromValue(request.params[0]);
    ToLowerCase(vchObjectID);

    CDomainEntry entry;
    entry.DomainComponent = vchDefaultDomainName;
    entry.OrganizationalUnit = vchDefaultPublicOU;
    entry.ObjectID = vchObjectID;
    if (!pDomainEntryDB->GetDomainEntryInfo(entry.vchFullObjectPath(), entry)) {
        throw JSONRPCError(RPC_TYPE_ERROR, "Can not find BDAP entry " + entry.GetFullObjectPath());
    }

    // Get wallet address private key from wallet db.
    CKeyID walletKeyID;
    CDynamicAddress address = entry.GetWalletAddress();
    if (!address.GetKeyID(walletKeyID))
        throw JSONRPCError(RPC_TYPE_ERROR, "Address does not refer to a key " + address.ToString());
    CKey vchWalletSecret;
    if (!pwalletMain->GetKey(walletKeyID, vchWalletSecret))
        throw JSONRPCError(RPC_WALLET_ERROR, "Private key for address " + address.ToString() + " is not known");
    std::string strWalletPrivKey = CDynamicSecret(vchWalletSecret).ToString();

    // Get link address private key from wallet db.
    CKeyID linkKeyID;
    CDynamicAddress linkAddress = entry.GetLinkAddress();
    if (!linkAddress.GetKeyID(linkKeyID))
        throw JSONRPCError(RPC_TYPE_ERROR, "Link address does not refer to a key" + linkAddress.ToString());
    CKey vchLinkSecret;
    if (!pwalletMain->GetKey(linkKeyID, vchLinkSecret))
        throw JSONRPCError(RPC_WALLET_ERROR, "Private key for address " + linkAddress.ToString() + " is not known");
    std::string strLinkPrivKey = CDynamicSecret(vchLinkSecret).ToString();

    // Get DHT private key from wallet db.
    CKeyEd25519 keyDHT;
    std::vector<unsigned char> vchDHTPubKey = vchFromString(entry.DHTPubKeyString());
    CKeyID dhtKeyID(Hash160(vchDHTPubKey.begin(), vchDHTPubKey.end()));
    if (pwalletMain && !pwalletMain->GetDHTKey(dhtKeyID, keyDHT)) {
        throw JSONRPCError(RPC_WALLET_ERROR, "Error getting ed25519 private key for " + entry.DHTPubKeyString());
    }
    std::string strDHTPrivKey = keyDHT.GetPrivSeedString();

    result.push_back(Pair("wallet_address", address.ToString()));
    result.push_back(Pair("wallet_privkey", strWalletPrivKey));
    result.push_back(Pair("link_address", linkAddress.ToString()));
    result.push_back(Pair("link_privkey", strLinkPrivKey));
    result.push_back(Pair("dht_publickey", entry.DHTPubKeyString()));
    result.push_back(Pair("dht_privkey", strDHTPrivKey));

    return result;
}

UniValue dumphdinfo(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 0)
        throw std::runtime_error(
            "dumphdinfo\n"
            "Returns an object containing sensitive private info about this HD wallet.\n"
            "\nResult:\n"
            "{\n"
            "  \"hdseed\": \"seed\",                    (string) The HD seed (bip32, in hex)\n"
            "  \"mnemonic\": \"words\",                 (string) The mnemonic for this HD wallet (bip39, english words) \n"
            "  \"mnemonicpassphrase\": \"passphrase\",  (string) The mnemonic passphrase for this HD wallet (bip39)\n"
            "}\n"
            "\nExamples:\n" +
            HelpExampleCli("dumphdinfo", "") + HelpExampleRpc("dumphdinfo", ""));

    LOCK(pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();

    CHDChain hdChainCurrent;
    if (!pwalletMain->GetHDChain(hdChainCurrent))
        throw JSONRPCError(RPC_WALLET_ERROR, "This wallet is not a HD wallet.");

    if (!pwalletMain->GetDecryptedHDChain(hdChainCurrent))
        throw JSONRPCError(RPC_INTERNAL_ERROR, "Cannot decrypt HD seed");

    SecureString ssMnemonic;
    SecureString ssMnemonicPassphrase;
    hdChainCurrent.GetMnemonic(ssMnemonic, ssMnemonicPassphrase);

    UniValue obj(UniValue::VOBJ);
    obj.push_back(Pair("hdseed", HexStr(hdChainCurrent.GetSeed())));
    obj.push_back(Pair("mnemonic", ssMnemonic.c_str()));
    obj.push_back(Pair("mnemonicpassphrase", ssMnemonicPassphrase.c_str()));

    return obj;
}

UniValue dumpwallet(const JSONRPCRequest& request)
{
    if (!EnsureWalletIsAvailable(request.fHelp))
        return NullUniValue;

    if (request.fHelp || request.params.size() != 1)
        throw std::runtime_error(
            "dumpwallet \"filename\"\n"
            "\nDumps all wallet keys in a human-readable format.\n"
            "\nArguments:\n"
            "1. \"filename\"    (string, required) The filename\n"
            "\nExamples:\n" +
            HelpExampleCli("dumpwallet", "\"test\"") + HelpExampleRpc("dumpwallet", "\"test\""));

    LOCK2(cs_main, pwalletMain->cs_wallet);

    EnsureWalletIsUnlocked();

    std::ofstream file;
    file.open(request.params[0].get_str().c_str());
    if (!file.is_open())
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Cannot open wallet dump file");

    std::map<CTxDestination, int64_t> mapKeyBirth;
    std::set<CKeyID> setKeyPool;
    pwalletMain->GetKeyBirthTimes(mapKeyBirth);
    pwalletMain->GetAllReserveKeys(setKeyPool);

    // sort time/key pairs
    std::vector<std::pair<int64_t, CKeyID> > vKeyBirth;
    for (const auto& entry : mapKeyBirth) {
        if (const CKeyID* keyID = boost::get<CKeyID>(&entry.first)) { // set and test
            vKeyBirth.push_back(std::make_pair(entry.second, *keyID));
        }
    }
    mapKeyBirth.clear();
    std::sort(vKeyBirth.begin(), vKeyBirth.end());

    // produce output
    file << strprintf("# Wallet dump created by Dynamic %s\n", CLIENT_BUILD);
    file << strprintf("# * Created on %s\n", EncodeDumpTime(GetTime()));
    file << strprintf("# * Best block at time of backup was %i (%s),\n", chainActive.Height(), chainActive.Tip()->GetBlockHash().ToString());
    file << strprintf("#   mined on %s\n", EncodeDumpTime(chainActive.Tip()->GetBlockTime()));
    file << "\n";

    UniValue obj(UniValue::VOBJ);
    obj.push_back(Pair("dynamicversion", CLIENT_BUILD));
    obj.push_back(Pair("lastblockheight", chainActive.Height()));
    obj.push_back(Pair("lastblockhash", chainActive.Tip()->GetBlockHash().ToString()));
    obj.push_back(Pair("lastblocktime", EncodeDumpTime(chainActive.Tip()->GetBlockTime())));

    // add the base58check encoded extended master if the wallet uses HD
    CHDChain hdChainCurrent;
    if (pwalletMain->GetHDChain(hdChainCurrent)) {
        if (!pwalletMain->GetDecryptedHDChain(hdChainCurrent))
            throw JSONRPCError(RPC_INTERNAL_ERROR, "Cannot decrypt HD chain");

        SecureString ssMnemonic;
        SecureString ssMnemonicPassphrase;
        hdChainCurrent.GetMnemonic(ssMnemonic, ssMnemonicPassphrase);
        file << "# mnemonic: " << ssMnemonic << "\n";
        file << "# mnemonic passphrase: " << ssMnemonicPassphrase << "\n\n";

        SecureVector vchSeed = hdChainCurrent.GetSeed();
        file << "# HD seed: " << HexStr(vchSeed) << "\n\n";

        CExtKey masterKey;
        masterKey.SetMaster(&vchSeed[0], vchSeed.size());

        CDynamicExtKey b58extkey;
        b58extkey.SetKey(masterKey);

        file << "# extended private masterkey: " << b58extkey.ToString() << "\n";

        CExtPubKey masterPubkey;
        masterPubkey = masterKey.Neutered();

        CDynamicExtPubKey b58extpubkey;
        b58extpubkey.SetKey(masterPubkey);
        file << "# extended public masterkey: " << b58extpubkey.ToString() << "\n\n";

        for (size_t i = 0; i < hdChainCurrent.CountAccounts(); ++i) {
            CHDAccount acc;
            if (hdChainCurrent.GetAccount(i, acc)) {
                file << "# external chain counter: " << acc.nExternalChainCounter << "\n";
                file << "# internal chain counter: " << acc.nInternalChainCounter << "\n\n";
            } else {
                file << "# WARNING: ACCOUNT " << i << " IS MISSING!"
                     << "\n\n";
            }
        }
    }

    for (std::vector<std::pair<int64_t, CKeyID> >::const_iterator it = vKeyBirth.begin(); it != vKeyBirth.end(); it++) {
        const CKeyID& keyid = it->second;
        std::string strTime = EncodeDumpTime(it->first);
        std::string strAddr = CDynamicAddress(keyid).ToString();
        CKey key;
        if (pwalletMain->GetKey(keyid, key)) {
            file << strprintf("%s %s ", CDynamicSecret(key).ToString(), strTime);
            if (pwalletMain->mapAddressBook.count(keyid)) {
                file << strprintf("label=%s", EncodeDumpString(pwalletMain->mapAddressBook[keyid].name));
            } else if (setKeyPool.count(keyid)) {
                file << "reserve=1";
            } else {
                file << "change=1";
            }
            file << strprintf(" # addr=%s%s\n", strAddr, (pwalletMain->mapHdPubKeys.count(keyid) ? " hdkeypath=" + pwalletMain->mapHdPubKeys[keyid].GetKeyPath() : ""));
        }
    }
    file << "\n";
    file << "# End of dump\n";
    file.close();

    std::string strWarning = strprintf(_("%s file contains all private keys from this wallet. Do not share it with anyone!"), request.params[0].get_str().c_str());
    obj.push_back(Pair("keys", int(vKeyBirth.size())));
    obj.push_back(Pair("file", request.params[0].get_str().c_str()));
    obj.push_back(Pair("warning", strWarning));
    return obj;
}

UniValue ProcessImport(const UniValue& data, const int64_t timestamp)
{
    try {
        bool success = false;

        // Required fields.
        const UniValue& scriptPubKey = data["scriptPubKey"];

        // Should have script or JSON with "address".
        if (!(scriptPubKey.getType() == UniValue::VOBJ && scriptPubKey.exists("address")) && !(scriptPubKey.getType() == UniValue::VSTR)) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Invalid scriptPubKey");
        }

        // Optional fields.
        const std::string& strRedeemScript = data.exists("redeemscript") ? data["redeemscript"].get_str() : "";
        const UniValue& pubKeys = data.exists("pubkeys") ? data["pubkeys"].get_array() : UniValue();
        const UniValue& keys = data.exists("keys") ? data["keys"].get_array() : UniValue();
        const bool& internal = data.exists("internal") ? data["internal"].get_bool() : false;
        const bool& watchOnly = data.exists("watchonly") ? data["watchonly"].get_bool() : false;
        const std::string& label = data.exists("label") && !internal ? data["label"].get_str() : "";

        bool isScript = scriptPubKey.getType() == UniValue::VSTR;
        bool isP2SH = strRedeemScript.length() > 0;
        const std::string& output = isScript ? scriptPubKey.get_str() : scriptPubKey["address"].get_str();

        // Parse the output.
        CScript script;
        CDynamicAddress address;

        if (!isScript) {
            address = CDynamicAddress(output);
            script = GetScriptForDestination(address.Get());
        } else {
            if (!IsHex(output)) {
                throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid scriptPubKey");
            }

            std::vector<unsigned char> vData(ParseHex(output));
            script = CScript(vData.begin(), vData.end());
        }

        // Watchonly and private keys
        if (watchOnly && keys.size()) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Incompatibility found between watchonly and keys");
        }

        // Internal + Label
        if (internal && data.exists("label")) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Incompatibility found between internal and label");
        }

        // Not having Internal + Script
        if (!internal && isScript) {
            throw JSONRPCError(RPC_INVALID_PARAMETER, "Internal must be set for hex scriptPubKey");
        }

        // Keys / PubKeys size check.
        if (!isP2SH && (keys.size() > 1 || pubKeys.size() > 1)) { // Address / scriptPubKey
            throw JSONRPCError(RPC_INVALID_PARAMETER, "More than private key given for one address");
        }

        // Invalid P2SH redeemScript
        if (isP2SH && !IsHex(strRedeemScript)) {
            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid redeem script");
        }

        // Process. //

        // P2SH
        if (isP2SH) {
            // Import redeem script.
            std::vector<unsigned char> vData(ParseHex(strRedeemScript));
            CScript redeemScript = CScript(vData.begin(), vData.end());

            // Invalid P2SH address
            if (!script.IsPayToScriptHash()) {
                throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid P2SH address / script");
            }

            pwalletMain->MarkDirty();

            if (!pwalletMain->HaveWatchOnly(redeemScript) && !pwalletMain->AddWatchOnly(redeemScript, timestamp)) {
                throw JSONRPCError(RPC_WALLET_ERROR, "Error adding address to wallet");
            }

            if (!pwalletMain->HaveCScript(redeemScript) && !pwalletMain->AddCScript(redeemScript)) {
                throw JSONRPCError(RPC_WALLET_ERROR, "Error adding p2sh redeemScript to wallet");
            }

            CDynamicAddress redeemAddress = CDynamicAddress(CScriptID(redeemScript));
            CScript redeemDestination = GetScriptForDestination(redeemAddress.Get());

            if (::IsMine(*pwalletMain, redeemDestination) == ISMINE_SPENDABLE) {
                throw JSONRPCError(RPC_WALLET_ERROR, "The wallet already contains the private key for this address or script");
            }

            pwalletMain->MarkDirty();

            if (!pwalletMain->HaveWatchOnly(redeemDestination) && !pwalletMain->AddWatchOnly(redeemDestination, timestamp)) {
                throw JSONRPCError(RPC_WALLET_ERROR, "Error adding address to wallet");
            }

            // add to address book or update label
            if (address.IsValid()) {
                pwalletMain->SetAddressBook(address.Get(), label, "receive");
            }

            // Import private keys.
            if (keys.size()) {
                for (size_t i = 0; i < keys.size(); i++) {
                    const std::string& privkey = keys[i].get_str();

                    CDynamicSecret vchSecret;
                    bool fGood = vchSecret.SetString(privkey);

                    if (!fGood) {
                        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid private key encoding");
                    }

                    CKey key = vchSecret.GetKey();

                    if (!key.IsValid()) {
                        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Private key outside allowed range");
                    }

                    CPubKey pubkey = key.GetPubKey();
                    assert(key.VerifyPubKey(pubkey));

                    CKeyID vchAddress = pubkey.GetID();
                    pwalletMain->MarkDirty();
                    pwalletMain->SetAddressBook(vchAddress, label, "receive");

                    if (pwalletMain->HaveKey(vchAddress)) {
                        throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Already have this key");
                    }

                    pwalletMain->mapKeyMetadata[vchAddress].nCreateTime = timestamp;

                    if (!pwalletMain->AddKeyPubKey(key, pubkey)) {
                        throw JSONRPCError(RPC_WALLET_ERROR, "Error adding key to wallet");
                    }

                    pwalletMain->UpdateTimeFirstKey(timestamp);
                }
            }

            success = true;
        } else {
            // Import public keys.
            if (pubKeys.size() && keys.size() == 0) {
                const std::string& strPubKey = pubKeys[0].get_str();

                if (!IsHex(strPubKey)) {
                    throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Pubkey must be a hex string");
                }

                std::vector<unsigned char> data(ParseHex(strPubKey));
                CPubKey pubKey(data.begin(), data.end());

                if (!pubKey.IsFullyValid()) {
                    throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Pubkey is not a valid public key");
                }

                CDynamicAddress pubKeyAddress = CDynamicAddress(pubKey.GetID());

                // Consistency check.
                if (!isScript && !(pubKeyAddress.Get() == address.Get())) {
                    throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Consistency check failed");
                }

                // Consistency check.
                if (isScript) {
                    CDynamicAddress scriptAddress;
                    CTxDestination destination;

                    if (ExtractDestination(script, destination)) {
                        scriptAddress = CDynamicAddress(destination);
                        if (!(scriptAddress.Get() == pubKeyAddress.Get())) {
                            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Consistency check failed");
                        }
                    }
                }

                CScript pubKeyScript = GetScriptForDestination(pubKeyAddress.Get());

                if (::IsMine(*pwalletMain, pubKeyScript) == ISMINE_SPENDABLE) {
                    throw JSONRPCError(RPC_WALLET_ERROR, "The wallet already contains the private key for this address or script");
                }

                pwalletMain->MarkDirty();

                if (!pwalletMain->HaveWatchOnly(pubKeyScript) && !pwalletMain->AddWatchOnly(pubKeyScript, timestamp)) {
                    throw JSONRPCError(RPC_WALLET_ERROR, "Error adding address to wallet");
                }

                // add to address book or update label
                if (pubKeyAddress.IsValid()) {
                    pwalletMain->SetAddressBook(pubKeyAddress.Get(), label, "receive");
                }

                // TODO Is this necessary?
                CScript scriptRawPubKey = GetScriptForRawPubKey(pubKey);

                if (::IsMine(*pwalletMain, scriptRawPubKey) == ISMINE_SPENDABLE) {
                    throw JSONRPCError(RPC_WALLET_ERROR, "The wallet already contains the private key for this address or script");
                }

                pwalletMain->MarkDirty();

                if (!pwalletMain->HaveWatchOnly(scriptRawPubKey) && !pwalletMain->AddWatchOnly(scriptRawPubKey, timestamp)) {
                    throw JSONRPCError(RPC_WALLET_ERROR, "Error adding address to wallet");
                }

                success = true;
            }

            // Import private keys.
            if (keys.size()) {
                const std::string& strPrivkey = keys[0].get_str();

                // Checks.
                CDynamicSecret vchSecret;
                bool fGood = vchSecret.SetString(strPrivkey);

                if (!fGood) {
                    throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Invalid private key encoding");
                }

                CKey key = vchSecret.GetKey();
                if (!key.IsValid()) {
                    throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Private key outside allowed range");
                }

                CPubKey pubKey = key.GetPubKey();
                assert(key.VerifyPubKey(pubKey));

                CDynamicAddress pubKeyAddress = CDynamicAddress(pubKey.GetID());

                // Consistency check.
                if (!isScript && !(pubKeyAddress.Get() == address.Get())) {
                    throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Consistency check failed");
                }

                // Consistency check.
                if (isScript) {
                    CDynamicAddress scriptAddress;
                    CTxDestination destination;

                    if (ExtractDestination(script, destination)) {
                        scriptAddress = CDynamicAddress(destination);
                        if (!(scriptAddress.Get() == pubKeyAddress.Get())) {
                            throw JSONRPCError(RPC_INVALID_ADDRESS_OR_KEY, "Consistency check failed");
                        }
                    }
                }

                CKeyID vchAddress = pubKey.GetID();
                pwalletMain->MarkDirty();
                pwalletMain->SetAddressBook(vchAddress, label, "receive");

                if (pwalletMain->HaveKey(vchAddress)) {
                    return false;
                }

                pwalletMain->mapKeyMetadata[vchAddress].nCreateTime = timestamp;

                if (!pwalletMain->AddKeyPubKey(key, pubKey)) {
                    throw JSONRPCError(RPC_WALLET_ERROR, "Error adding key to wallet");
                }

                pwalletMain->UpdateTimeFirstKey(timestamp);

                success = true;
            }

            // Import scriptPubKey only.
            if (pubKeys.size() == 0 && keys.size() == 0) {
                if (::IsMine(*pwalletMain, script) == ISMINE_SPENDABLE) {
                    throw JSONRPCError(RPC_WALLET_ERROR, "The wallet already contains the private key for this address or script");
                }

                pwalletMain->MarkDirty();

                if (!pwalletMain->HaveWatchOnly(script) && !pwalletMain->AddWatchOnly(script, timestamp)) {
                    throw JSONRPCError(RPC_WALLET_ERROR, "Error adding address to wallet");
                }

                if (scriptPubKey.getType() == UniValue::VOBJ) {
                    // add to address book or update label
                    if (address.IsValid()) {
                        pwalletMain->SetAddressBook(address.Get(), label, "receive");
                    }
                }

                success = true;
            }
        }

        UniValue result = UniValue(UniValue::VOBJ);
        result.pushKV("success", UniValue(success));
        return result;
    } catch (const UniValue& e) {
        UniValue result = UniValue(UniValue::VOBJ);
        result.pushKV("success", UniValue(false));
        result.pushKV("error", e);
        return result;
    } catch (...) {
        UniValue result = UniValue(UniValue::VOBJ);
        result.pushKV("success", UniValue(false));
        result.pushKV("error", JSONRPCError(RPC_MISC_ERROR, "Missing required fields"));
        return result;
    }
}

int64_t GetImportTimestamp(const UniValue& data, int64_t now)
{
    if (data.exists("timestamp")) {
        const UniValue& timestamp = data["timestamp"];
        if (timestamp.isNum()) {
            return timestamp.get_int64();
        } else if (timestamp.isStr() && timestamp.get_str() == "now") {
            return now;
        }
        throw JSONRPCError(RPC_TYPE_ERROR, strprintf("Expected number or \"now\" timestamp value for key. got type %s", uvTypeName(timestamp.type())));
    }
    throw JSONRPCError(RPC_TYPE_ERROR, "Missing required timestamp field for key");
}

UniValue importmulti(const JSONRPCRequest& mainRequest)
{
    // clang-format off
    if (mainRequest.fHelp || mainRequest.params.size() < 1 || mainRequest.params.size() > 2)
        throw std::runtime_error(
            "importmulti \"requests\" \"options\"\n\n"
            "Import addresses/scripts (with private or public keys, redeem script (P2SH)), rescanning all addresses in one-shot-only (rescan can be disabled via options).\n\n"
            "Arguments:\n"
            "1. requests     (array, required) Data to be imported\n"
            "  [     (array of json objects)\n"
            "    {\n"
            "      \"scriptPubKey\": \"<script>\" | { \"address\":\"<address>\" }, (string / json, required) Type of scriptPubKey (string for script, json for address)\n"
            "      \"timestamp\": timestamp | \"now\"                        , (integer / string, required) Creation time of the key in seconds since epoch (Jan 1 1970 GMT),\n"
            "                                                              or the string \"now\" to substitute the current synced blockchain time. The timestamp of the oldest\n"
            "                                                              key will determine how far back blockchain rescans need to begin for missing wallet transactions.\n"
            "                                                              \"now\" can be specified to bypass scanning, for keys which are known to never have been used, and\n"
            "                                                              0 can be specified to scan the entire blockchain. Blocks up to 2 hours before the earliest key\n"
            "                                                              creation time of all keys being imported by the importmulti call will be scanned.\n"
            "      \"redeemscript\": \"<script>\"                            , (string, optional) Allowed only if the scriptPubKey is a P2SH address or a P2SH scriptPubKey\n"
            "      \"pubkeys\": [\"<pubKey>\", ... ]                         , (array, optional) Array of strings giving pubkeys that must occur in the output or redeemscript\n"
            "      \"keys\": [\"<key>\", ... ]                               , (array, optional) Array of strings giving private keys whose corresponding public keys must occur in the output or redeemscript\n"
            "      \"internal\": <true>                                    , (boolean, optional, default: false) Stating whether matching outputs should be be treated as not incoming payments\n"
            "      \"watchonly\": <true>                                   , (boolean, optional, default: false) Stating whether matching outputs should be considered watched even when they're not spendable, only allowed if keys are empty\n"
            "      \"label\": <label>                                      , (string, optional, default: '') Label to assign to the address (aka account name, for now), only allowed with internal=false\n"
            "    }\n"
            "  ,...\n"
            "  ]\n"
            "2. options                 (json, optional)\n"
            "  {\n"
            "     \"rescan\": <false>,         (boolean, optional, default: true) Stating if should rescan the blockchain after all imports\n"
            "  }\n"
            "\nExamples:\n" +
            HelpExampleCli("importmulti", "'[{ \"scriptPubKey\": { \"address\": \"<my address>\" }, \"timestamp\":1455191478 }, "
                                          "{ \"scriptPubKey\": { \"address\": \"<my 2nd address>\" }, \"label\": \"example 2\", \"timestamp\": 1455191480 }]'") +
            HelpExampleCli("importmulti", "'[{ \"scriptPubKey\": { \"address\": \"<my address>\" }, \"timestamp\":1455191478 }]' '{ \"rescan\": false}'") +

            "\nResponse is an array with the same size as the input that has the execution result :\n"
            "  [{ \"success\": true } , { \"success\": false, \"error\": { \"code\": -1, \"message\": \"Internal Server Error\"} }, ... ]\n");

    // clang-format on
    if (!EnsureWalletIsAvailable(mainRequest.fHelp)) {
        return NullUniValue;
    }

    RPCTypeCheck(mainRequest.params, boost::assign::list_of(UniValue::VARR)(UniValue::VOBJ));

    const UniValue& requests = mainRequest.params[0];

    //Default options
    bool fRescan = true;

    if (mainRequest.params.size() > 1) {
        const UniValue& options = mainRequest.params[1];

        if (options.exists("rescan")) {
            fRescan = options["rescan"].get_bool();
        }
    }

    LOCK2(cs_main, pwalletMain->cs_wallet);
    EnsureWalletIsUnlocked();

    // Verify all timestamps are present before importing any keys.
    const int64_t now = chainActive.Tip() ? chainActive.Tip()->GetMedianTimePast() : 0;
    for (const UniValue& data : requests.getValues()) {
        GetImportTimestamp(data, now);
    }

    bool fRunScan = false;
    const int64_t minimumTimestamp = 1;
    int64_t nLowestTimestamp = 0;

    if (fRescan && chainActive.Tip()) {
        nLowestTimestamp = chainActive.Tip()->GetBlockTime();
    } else {
        fRescan = false;
    }

    UniValue response(UniValue::VARR);

    BOOST_FOREACH (const UniValue& data, requests.getValues()) {
        const int64_t timestamp = std::max(GetImportTimestamp(data, now), minimumTimestamp);
        const UniValue result = ProcessImport(data, timestamp);
        response.push_back(result);

        if (!fRescan) {
            continue;
        }

        // If at least one request was successful then allow rescan.
        if (result["success"].get_bool()) {
            fRunScan = true;
        }

        // Get the lowest timestamp.
        if (timestamp < nLowestTimestamp) {
            nLowestTimestamp = timestamp;
        }
    }

    if (fRescan && fRunScan && requests.size()) {
        CBlockIndex* pindex = nLowestTimestamp > minimumTimestamp ? chainActive.FindEarliestAtLeast(std::max<int64_t>(nLowestTimestamp - 7200, 0)) : chainActive.Genesis();
        CBlockIndex* scannedRange = nullptr;
        if (pindex) {
            scannedRange = pwalletMain->ScanForWalletTransactions(pindex, true);
            pwalletMain->ReacceptWalletTransactions();
        }

        if (!scannedRange || scannedRange->nHeight > pindex->nHeight) {
            std::vector<UniValue> results = response.getValues();
            response.clear();
            response.setArray();
            size_t i = 0;
            for (const UniValue& request : requests.getValues()) {
                // If key creation date is within the successfully scanned
                // range, or if the import result already has an error set, let
                // the result stand unmodified. Otherwise replace the result
                // with an error message.
                if (GetImportTimestamp(request, now) - 7200 >= scannedRange->GetBlockTimeMax() || results.at(i).exists("error")) {
                    response.push_back(results.at(i));
                } else {
                    UniValue result = UniValue(UniValue::VOBJ);
                    result.pushKV("success", UniValue(false));
                    result.pushKV("error", JSONRPCError(RPC_MISC_ERROR, strprintf("Failed to rescan before time %d, transactions may be missing.", scannedRange->GetBlockTimeMax())));
                    response.push_back(std::move(result));
                }
                ++i;
            }
        }
    }
    return response;
}
