// Copyright (c) 2009-2017 Satoshi Nakamoto
// Copyright (c) 2009-2017 The Bitcoin Developers
// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2017 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "chainparams.h"

#include "arith_uint256.h"
#include "chainparamsseeds.h"
#include "hash.h"
#include "consensus/merkle.h"
#include "streams.h"
#include "tinyformat.h"
#include "util.h"
#include "utilstrencodings.h"
#include "uint256.h"

#include <assert.h>

#include <boost/assign/list_of.hpp>

static CBlock CreateGenesisBlock(const char* pszTimestamp, const CScript& genesisOutputScript, const uint32_t nTime, const uint32_t nNonce, const uint32_t nBits, const int32_t nVersion, const CAmount& genesisReward)
{
    CMutableTransaction txNew;
    txNew.nVersion = 1;
    txNew.vin.resize(1);
    txNew.vout.resize(1);
    txNew.vin[0].scriptSig = CScript() << 1488879644 << CScriptNum(4) << std::vector<unsigned char>((const unsigned char*)pszTimestamp, (const unsigned char*)pszTimestamp + strlen(pszTimestamp));
    txNew.vout[0].nValue = genesisReward;
    txNew.vout[0].scriptPubKey = genesisOutputScript;

    CBlock genesis;
    genesis.nTime    = nTime;
    genesis.nBits    = nBits;
    genesis.nNonce   = nNonce;
    genesis.nVersion = nVersion;
    genesis.vtx.push_back(txNew);
    genesis.hashPrevBlock.SetNull();
    genesis.hashMerkleRoot = BlockMerkleRoot(genesis);
    return genesis;
}

static void MineGenesis(CBlockHeader& genesisBlock, const uint256& powLimit, bool noProduction)
{
    if(noProduction)
        genesisBlock.nTime = std::time(0);
    genesisBlock.nNonce = 0;

    printf("NOTE: Genesis nTime = %u \n", genesisBlock.nTime);
    printf("WARN: Genesis nNonce (BLANK!) = %u \n", genesisBlock.nNonce);

    arith_uint256 besthash;
    memset(&besthash,0xFF,32);
    arith_uint256 hashTarget = UintToArith256(powLimit);
    printf("Target: %s\n", hashTarget.GetHex().c_str());
    arith_uint256 newhash = UintToArith256(genesisBlock.GetHash());
    while (newhash > hashTarget) {
        genesisBlock.nNonce++;
        if (genesisBlock.nNonce == 0) {
            printf("NONCE WRAPPED, incrementing time\n");
            ++genesisBlock.nTime;
        }
        // If nothing found after trying for a while, print status
        if ((genesisBlock.nNonce & 0xfff) == 0)
            printf("nonce %08X: hash = %s (target = %s)\n",
                   genesisBlock.nNonce, newhash.ToString().c_str(),
                   hashTarget.ToString().c_str());

        if(newhash < besthash) {
            besthash = newhash;
            printf("New best: %s\n", newhash.GetHex().c_str());
        }
        newhash = UintToArith256(genesisBlock.GetHash());
    }
    printf("Genesis nTime = %u \n", genesisBlock.nTime);
    printf("Genesis nNonce = %u \n", genesisBlock.nNonce);
    printf("Genesis nBits: %08x\n", genesisBlock.nBits);
    printf("Genesis Hash = %s\n", newhash.ToString().c_str());
    printf("Genesis Hash Merkle Root = %s\n", genesisBlock.hashMerkleRoot.ToString().c_str());
}

/**
 * Build the genesis block. Note that the output of its generation
 * transaction cannot be spent since it did not originally exist in the
 * database.
 *
 * CBlock(hash=00000ffd590b14, ver=1, hashPrevBlock=00000000000000, hashMerkleRoot=e0028e, nTime=1390095618, nBits=1e0ffff0, nNonce=28917698, vtx=1)
 *   CTransaction(hash=e0028e, ver=1, vin.size=1, vout.size=1, nLockTime=0)
 *     CTxIn(COutPoint(000000, -1), coinbase 04ffff001d01044c5957697265642030392f4a616e2f3230313420546865204772616e64204578706572696d656e7420476f6573204c6976653a204f76657273746f636b2e636f6d204973204e6f7720416363657074696e6720426974636f696e73)
 *     CTxOut(nValue=50.00000000, scriptPubKey=0xA9037BAC7050C479B121CF)
 *   vMerkleTree: e0028e
 */
static CBlock CreateGenesisBlock(uint32_t nTime, uint32_t nNonce, uint32_t nBits, int32_t nVersion, const CAmount& genesisReward)
{
    const char* pszTimestamp = "NY Times 03/07/2017: Questions About Mac Security and Google Password Prompts";
    const CScript genesisOutputScript = CScript() << ParseHex("") << OP_CHECKSIG;
    return CreateGenesisBlock(pszTimestamp, genesisOutputScript, nTime, nNonce, nBits, nVersion, genesisReward);
}

/**
 * Main network
 */

class CMainParams : public CChainParams {
public:
    CMainParams() {
        strNetworkID = "main";
        consensus.nRewardsStart = 20545; // PoW Rewards begin on block 20546
        consensus.nDynodePaymentsStartBlock = 20545; // Dynode Payments begin on block 20546
        consensus.nInstantSendKeepLock = 24;
        consensus.nBudgetPaymentsStartBlock = 20546; // actual historical value
        consensus.nBudgetPaymentsCycleBlocks = 20545; //Blocks per month
        consensus.nBudgetPaymentsWindowBlocks = 100;
        consensus.nBudgetProposalEstablishingTime = 24 * 60 * 60;
        consensus.nSuperblockStartBlock = 20545;
        consensus.nSuperblockCycle = 20545; // 675 (Blocks per day) x 365.25 (Days per Year) / 12 = 20545
        consensus.nGovernanceMinQuorum = 10;
        consensus.nGovernanceFilterElements = 20000;
        consensus.nDynodeMinimumConfirmations = 15;
        consensus.nMajorityEnforceBlockUpgrade = 750;
        consensus.nMajorityRejectBlockOutdated = 950;
        consensus.nMajorityWindow = 1000;
        consensus.powLimit = uint256S("0000fffff0000000000000000000000000000000000000000000000000000000");
        consensus.nPowTargetTimespan = 12 * 60 * 60; // Dynamic: 12 hours
        consensus.nPowTargetSpacing = 2 * 64; // Dynamic: 128 seconds
        consensus.fPowAllowMinDifficultyBlocks = false;
        consensus.fPowNoRetargeting = false;
        consensus.nRuleChangeActivationThreshold = 321; // 95% of nMinerConfirmationWindow
        consensus.nMinerConfirmationWindow = 338; // nPowTargetTimespan / nPowTargetSpacing
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].bit = 28;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nStartTime = 1199145601; // January 1, 2008
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nTimeout = 1230767999; // December 31, 2008

        // Deployment of BIP68, BIP112, and BIP113.
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].bit = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nStartTime = 1488881400; // Mar 7th 2017 10:10:00
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nTimeout = 1520424000; // Mar 7th 2018 12:00:00

        /**
         * The message start string is designed to be unlikely to occur in normal data.
         * The characters are rarely used upper ASCII, not valid as UTF-8, and produce
         * a large 32-bit integer with any alignment.
         */
        pchMessageStart[0] = 0x2f;
        pchMessageStart[1] = 0x32;
        pchMessageStart[2] = 0x15;
        pchMessageStart[3] = 0x41;
        vAlertPubKey = ParseHex("04ae9821c0e83ed7b23a08513047ec23e3acf41be066239d48c7571032efc858b30af168516aaf3320f57c431cf697de8dfd00b0c86112c231dbef04de46b8a731");
        nDefaultPort = 31300;
        nMaxTipAge = 24 * 60 * 64;
        nPruneAfterHeight = 10000;
        startNewChain = false;

        genesis = CreateGenesisBlock(1488880318, 36545, UintToArith256(consensus.powLimit).GetCompact(), 1, (1 * COIN));
        if(startNewChain == true) { MineGenesis(genesis, consensus.powLimit, true); }

        consensus.hashGenesisBlock = genesis.GetHash();

        if(!startNewChain)
            assert(consensus.hashGenesisBlock == uint256S("0x0000a2fa14a8ea28a124fc358e9ae8bc5bc8df4ded0fa5cf25f05570c0c58153"));
            assert(genesis.hashMerkleRoot == uint256S("0xac6c83307e7d351571e51e150b95776ad845f976bbdc4b3e2c7b4a3991ef55ab"));

        vSeeds.push_back(CDNSSeedData("dnsseeder.io", "dyn.dnsseeder.io"));
        vSeeds.push_back(CDNSSeedData("dnsseeder.com", "dyn.dnsseeder.com"));
        vSeeds.push_back(CDNSSeedData("dnsseeder.host", "dyn.dnsseeder.host"));
        vSeeds.push_back(CDNSSeedData("dnsseeder.net", "dyn.dnsseeder.net"));

        // Dynamic addresses start with 'D'
        base58Prefixes[PUBKEY_ADDRESS] = std::vector<unsigned char>(1,30);
        // Dynamic script addresses start with '5'
        base58Prefixes[SCRIPT_ADDRESS] = std::vector<unsigned char>(1,10);
        // Dynamic private keys start with 'y'
        base58Prefixes[SECRET_KEY] =     std::vector<unsigned char>(1,140);
        // Dynamic BIP32 pubkeys start with 'xpub' (Bitcoin defaults)
        base58Prefixes[EXT_PUBLIC_KEY] = boost::assign::list_of(0x04)(0x88)(0xB2)(0x1E).convert_to_container<std::vector<unsigned char> >();
        // Dynamic BIP32 prvkeys start with 'xprv' (Bitcoin defaults)
        base58Prefixes[EXT_SECRET_KEY] = boost::assign::list_of(0x04)(0x88)(0xAD)(0xE4).convert_to_container<std::vector<unsigned char> >();
        // Dynamic BIP44 coin type is '5'
        base58Prefixes[EXT_COIN_TYPE]  = boost::assign::list_of(0x80)(0x00)(0x00)(0x05).convert_to_container<std::vector<unsigned char> >();

        vFixedSeeds = std::vector<SeedSpec6>(pnSeed6_main, pnSeed6_main + ARRAYLEN(pnSeed6_main));

        fMiningRequiresPeers = true;
        fDefaultConsistencyChecks = false;
        fRequireStandard = true;
        fMineBlocksOnDemand = false;
        fTestnetToBeDeprecatedFieldRPC = false;

        nPoolMaxTransactions = 3;
        nFulfilledRequestExpireTime = 60 * 60; // fulfilled requests expire in 1 hour
        strSporkPubKey = "04ae9821c0e83ed7b23a08513047ec23e3acf41be066239d48c7571032efc858b30af168516aaf3320f57c431cf697de8dfd00b0c86112c231dbef04de46b8a731";
        strDynodePaymentsPubKey = "04ae9821c0e83ed7b23a08513047ec23e3acf41be066239d48c7571032efc858b30af168516aaf3320f57c431cf697de8dfd00b0c86112c231dbef04de46b8a731";

        checkpointData = (CCheckpointData) {
            boost::assign::map_list_of
            (  0, uint256S("0x0000a2fa14a8ea28a124fc358e9ae8bc5bc8df4ded0fa5cf25f05570c0c58153"))
            ( 100, uint256S("0x00009f9b564e0ab99e1b3644097e30c350900d3bb214b94d0d30f704c1676852"))
            ( 1000, uint256S("0x0000234e58755834be654058945708a5ebebd500774cca8350bc54c141dddeaa"))
            ( 2000, uint256S("0x000099cf2da3442075c8092169ecabf824a0278500f37108e9e3f68c19ab2b7b"))
            ( 2310, uint256S("0x0000d404dbd5b5a188fa07b98b5c40069728158ba6232f1a1de14217692e0f59"))
            ( 3000, uint256S("0x0000ddb88030e6865b6bb53d0f42483e88c18b4a5ce61510f1a0f11e1ff8d269"))
            ( 4500, uint256S("0x0000aba4fd749aa3abadbdd9850f9388fa89b44bf430838492eb965ff51d7481")),
            1488880318, // * UNIX timestamp of last checkpoint block
            0,          // * total number of transactions between genesis and last checkpoint
            //   (the tx=... number in the SetBestChain debug.log lines)
            2000        // * estimated number of transactions per day after checkpoint
        };
    }
};
static CMainParams mainParams;

/**
 * Testnet (v3)
 */
class CTestNetParams : public CChainParams {
public:
    CTestNetParams() {
        strNetworkID = "test";
        consensus.nRewardsStart = 0; // Rewards starts on block 0
        consensus.nDynodePaymentsStartBlock = 0;
        consensus.nInstantSendKeepLock = 24;
        consensus.nBudgetPaymentsStartBlock = 200;
        consensus.nBudgetPaymentsCycleBlocks = 50;
        consensus.nBudgetPaymentsWindowBlocks = 10;
        consensus.nBudgetProposalEstablishingTime = 60 * 20;
        consensus.nSuperblockStartBlock = 0;
        consensus.nSuperblockCycle = 24; // Superblocks can be issued hourly on testnet
        consensus.nGovernanceMinQuorum = 1;
        consensus.nGovernanceFilterElements = 500;
        consensus.nDynodeMinimumConfirmations = 1;
        consensus.nMajorityEnforceBlockUpgrade = 510;
        consensus.nMajorityRejectBlockOutdated = 750;
        consensus.nMajorityWindow = 1000;
        consensus.powLimit = uint256S("000fffff00000000000000000000000000000000000000000000000000000000");
        consensus.nPowTargetTimespan = 12 * 60 * 60; // Dynamic: 12 hours
        consensus.nPowTargetSpacing = 2 * 64; // Dynamic: 64 seconds
        consensus.fPowAllowMinDifficultyBlocks = true;
        consensus.fPowNoRetargeting = false;
        consensus.nRuleChangeActivationThreshold = 254; // 75% of nMinerConfirmationWindow
        consensus.nMinerConfirmationWindow = 338; // nPowTargetTimespan / nPowTargetSpacing
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].bit = 28;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nStartTime = 1199145601; // January 1, 2008
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nTimeout = 1230767999; // December 31, 2008

        // Deployment of BIP68, BIP112, and BIP113.
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].bit = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nStartTime = 1488881400; // Mar 7th 2017 10:10:00
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nTimeout = 1520424000; // Mar 7th 2018 12:00:00

        pchMessageStart[0] = 0x2f;
        pchMessageStart[1] = 0x32;
        pchMessageStart[2] = 0x15;
        pchMessageStart[3] = 0x40;
        vAlertPubKey = ParseHex("");
        nDefaultPort = 31400;
        nMaxTipAge = 24 * 60 * 64;
        nPruneAfterHeight = 100;
        startNewChain = false;

        genesis = CreateGenesisBlock(1488880451, 8582, UintToArith256(consensus.powLimit).GetCompact(), 1, (1 * COIN));
        if(startNewChain == true) {
            MineGenesis(genesis, consensus.powLimit, true);
        }

        consensus.hashGenesisBlock = genesis.GetHash();

        if(!startNewChain)
            assert(consensus.hashGenesisBlock == uint256S("0x000e095576d948220036ce358d053cc95b3cf5aff141da4c49a3c5854f2d991b"));
            assert(genesis.hashMerkleRoot == uint256S("0xac6c83307e7d351571e51e150b95776ad845f976bbdc4b3e2c7b4a3991ef55ab"));

        vFixedSeeds.clear();
        vSeeds.clear();
        //vSeeds.push_back(CDNSSeedData("",  ""));
        //vSeeds.push_back(CDNSSeedData("", ""));

        // Testnet Dynamic addresses start with 'y'
        base58Prefixes[PUBKEY_ADDRESS] = std::vector<unsigned char>(1,30);
        // Testnet Dynamic script addresses start with '8' or '9'
        base58Prefixes[SCRIPT_ADDRESS] = std::vector<unsigned char>(1,10);
        // Testnet private keys start with '9' or 'c' (Bitcoin defaults)
        base58Prefixes[SECRET_KEY] =     std::vector<unsigned char>(1,158);
        // Testnet Dynamic BIP32 pubkeys start with 'tpub' (Bitcoin defaults)
        base58Prefixes[EXT_PUBLIC_KEY] = boost::assign::list_of(0x04)(0x35)(0x87)(0xCF).convert_to_container<std::vector<unsigned char> >();
        // Testnet Dynamic BIP32 prvkeys start with 'tprv' (Bitcoin defaults)
        base58Prefixes[EXT_SECRET_KEY] = boost::assign::list_of(0x04)(0x35)(0x83)(0x94).convert_to_container<std::vector<unsigned char> >();
        // Testnet Dynamic BIP44 coin type is '1' (All coin's testnet default)
        base58Prefixes[EXT_COIN_TYPE]  = boost::assign::list_of(0x80)(0x00)(0x00)(0x01).convert_to_container<std::vector<unsigned char> >();

        vFixedSeeds = std::vector<SeedSpec6>(pnSeed6_test, pnSeed6_test + ARRAYLEN(pnSeed6_test));

        fMiningRequiresPeers = true;
        fDefaultConsistencyChecks = false;
        fRequireStandard = false;
        fMineBlocksOnDemand = false;
        fTestnetToBeDeprecatedFieldRPC = true;

        nPoolMaxTransactions = 3;
        nFulfilledRequestExpireTime = 5 * 60; // fulfilled requests expire in 5 minutes
        strSporkPubKey = "";
        strDynodePaymentsPubKey = "";

        checkpointData = (CCheckpointData) {
            boost::assign::map_list_of
            (  0, uint256S("0x000e095576d948220036ce358d053cc95b3cf5aff141da4c49a3c5854f2d991b")),
            1488880451, // * UNIX timestamp of last checkpoint block
            0,    // * total number of transactions between genesis and last checkpoint
            //   (the tx=... number in the SetBestChain debug.log lines)
            1000        // * estimated number of transactions per day after checkpoint
        };
    }
};
static CTestNetParams testNetParams;

/**
 * Regression test
 */
class CRegTestParams : public CChainParams {
public:
    CRegTestParams() {
        strNetworkID = "regtest";
        consensus.nRewardsStart = 0; // Rewards starts on block 0
        consensus.nDynodePaymentsStartBlock = 0;
        consensus.nInstantSendKeepLock = 24;
        consensus.nBudgetPaymentsStartBlock = 1000;
        consensus.nBudgetPaymentsCycleBlocks = 50;
        consensus.nBudgetPaymentsWindowBlocks = 10;
        consensus.nBudgetProposalEstablishingTime = 60 * 20;
        consensus.nSuperblockStartBlock = 0;
        consensus.nSuperblockCycle = 10;
        consensus.nGovernanceMinQuorum = 1;
        consensus.nGovernanceFilterElements = 100;
        consensus.nDynodeMinimumConfirmations = 1;
        consensus.nMajorityEnforceBlockUpgrade = 750;
        consensus.nMajorityRejectBlockOutdated = 950;
        consensus.nMajorityWindow = 1000;
        consensus.powLimit = uint256S("00fffff000000000000000000000000000000000000000000000000000000000");
        consensus.nPowTargetTimespan = 12 * 60 * 60; // Dynamic: 12 hours
        consensus.nPowTargetSpacing = 2 * 64; // Dynamic: 64 seconds
        consensus.fPowAllowMinDifficultyBlocks = true;
        consensus.fPowNoRetargeting = true;
        consensus.nRuleChangeActivationThreshold = 254; // 75% of nMinerConfirmationWindow
        consensus.nMinerConfirmationWindow = 338; // Faster than normal for regtest (144 instead of 2016)
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].bit = 28;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nStartTime = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nTimeout = 999999999999ULL;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].bit = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nStartTime = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nTimeout = 999999999999ULL;

        pchMessageStart[0] = 0x2f;
        pchMessageStart[1] = 0x32;
        pchMessageStart[2] = 0x15;
        pchMessageStart[3] = 0x3f;
        nMaxTipAge = 24 * 60 * 64;
        nDefaultPort = 31500;
        nPruneAfterHeight = 100;
        startNewChain = false;

        genesis = CreateGenesisBlock(1488880510, 671, UintToArith256(consensus.powLimit).GetCompact(), 1, (1 * COIN));
        if(startNewChain == true) {
            MineGenesis(genesis, consensus.powLimit, true);
        }

        consensus.hashGenesisBlock = genesis.GetHash();

        if(!startNewChain)
            assert(consensus.hashGenesisBlock == uint256S("0x00665d6b289ad23d9cfd63cd0326b6118640b49125ab214dd96588d9eb63ed6e"));
            assert(genesis.hashMerkleRoot == uint256S("0xac6c83307e7d351571e51e150b95776ad845f976bbdc4b3e2c7b4a3991ef55ab"));

        vFixedSeeds.clear(); //! Regtest mode doesn't have any fixed seeds.
        vSeeds.clear();  //! Regtest mode doesn't have any DNS seeds.

        fMiningRequiresPeers = false;
        fDefaultConsistencyChecks = true;
        fRequireStandard = false;
        fMineBlocksOnDemand = true;
        fTestnetToBeDeprecatedFieldRPC = false;

        nFulfilledRequestExpireTime = 5 * 60; // fulfilled requests expire in 5 minutes
        checkpointData = (CCheckpointData) {
            boost::assign::map_list_of
            (  0, uint256S("0x00665d6b289ad23d9cfd63cd0326b6118640b49125ab214dd96588d9eb63ed6e")),
            1488880510, // * UNIX timestamp of last checkpoint block
            0,    // * total number of transactions between genesis and last checkpoint
            //   (the tx=... number in the SetBestChain debug.log lines)
            500        // * estimated number of transactions per day after checkpoint
        };
        // Regtest Dynamic addresses start with 'y'
        base58Prefixes[PUBKEY_ADDRESS] = std::vector<unsigned char>(1,140);
        // Regtest Dynamic script addresses start with '8' or '9'
        base58Prefixes[SCRIPT_ADDRESS] = std::vector<unsigned char>(1,19);
        // Regtest private keys start with '9' or 'c' (Bitcoin defaults)
        base58Prefixes[SECRET_KEY] =     std::vector<unsigned char>(1,239);
        // Regtest Dynamic BIP32 pubkeys start with 'tpub' (Bitcoin defaults)
        base58Prefixes[EXT_PUBLIC_KEY] = boost::assign::list_of(0x04)(0x35)(0x87)(0xCF).convert_to_container<std::vector<unsigned char> >();
        // Regtest Dynamic BIP32 prvkeys start with 'tprv' (Bitcoin defaults)
        base58Prefixes[EXT_SECRET_KEY] = boost::assign::list_of(0x04)(0x35)(0x83)(0x94).convert_to_container<std::vector<unsigned char> >();
        // Regtest Dynamic BIP44 coin type is '1' (All coin's testnet default)
        base58Prefixes[EXT_COIN_TYPE]  = boost::assign::list_of(0x80)(0x00)(0x00)(0x01).convert_to_container<std::vector<unsigned char> >();
    }
};
static CRegTestParams regTestParams;

static CChainParams *pCurrentParams = 0;

const CChainParams &Params() {
    assert(pCurrentParams);
    return *pCurrentParams;
}

CChainParams& Params(const std::string& chain)
{
    if (chain == CBaseChainParams::MAIN)
        return mainParams;
    else if (chain == CBaseChainParams::TESTNET)
        return testNetParams;
    else if (chain == CBaseChainParams::REGTEST)
        return regTestParams;
    else
        throw std::runtime_error(strprintf("%s: Unknown chain %s.", __func__, chain));
}

void SelectParams(const std::string& network)
{
    SelectBaseParams(network);
    pCurrentParams = &Params(network);
}
