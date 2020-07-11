// Copyright (c) 2009-2019 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "chainparams.h"

#include "arith_uint256.h"
#include "consensus/merkle.h"
#include "hash.h"
#include "streams.h"

#include "tinyformat.h"
#include "util.h"
#include "utilstrencodings.h"

#include "uint256.h"

#include <assert.h>

#include <boost/assign/list_of.hpp>

#include "chainparamsseeds.h"

const arith_uint256 maxUint = UintToArith256(uint256S("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"));

static CBlock CreateGenesisBlock(const char* pszTimestamp, const CScript& genesisOutputScript, const uint32_t nTime, const uint32_t nNonce, const uint32_t nBits, const int32_t nVersion, const CAmount& genesisReward)
{
    CMutableTransaction txNew;
    txNew.nVersion = 1;
    txNew.vin.resize(1);
    txNew.vout.resize(1);
    txNew.vin[0].scriptSig = CScript() << 1512926956 << CScriptNum(4) << std::vector<unsigned char>((const unsigned char*)pszTimestamp, (const unsigned char*)pszTimestamp + strlen(pszTimestamp));
    txNew.vout[0].nValue = genesisReward;
    txNew.vout[0].scriptPubKey = genesisOutputScript;

    CBlock genesis;
    genesis.nTime = nTime;
    genesis.nBits = nBits;
    genesis.nNonce = nNonce;
    genesis.nVersion = nVersion;
    genesis.vtx.push_back(MakeTransactionRef(std::move(txNew)));
    genesis.hashPrevBlock.SetNull();
    genesis.hashMerkleRoot = BlockMerkleRoot(genesis);
    return genesis;
}

static void MineGenesis(CBlockHeader& genesisBlock, const uint256& powLimit, bool noProduction)
{
    if (noProduction)
        genesisBlock.nTime = std::time(0);
    genesisBlock.nNonce = 0;

    printf("NOTE: Genesis nTime = %u \n", genesisBlock.nTime);
    printf("WARN: Genesis nNonce (BLANK!) = %u \n", genesisBlock.nNonce);

    arith_uint256 besthash = maxUint;
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

        if (newhash < besthash) {
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
    const char* pszTimestamp = "NY Times Monday 18th Dec 2017: Google Thinks I’m Dead";
    const CScript genesisOutputScript = CScript() << ParseHex("") << OP_CHECKSIG;
    return CreateGenesisBlock(pszTimestamp, genesisOutputScript, nTime, nNonce, nBits, nVersion, genesisReward);
}

/**
 * Main network
 */

class CMainParams : public CChainParams
{
public:
    CMainParams()
    {
        strNetworkID = "main";
        consensus.nRewardsStart = 5137;               // PoW Rewards begin on block 5137
        consensus.nDynodePaymentsStartBlock = 10273;  // Dynode Payments begin on block 10273
        consensus.nMinCountDynodesPaymentStart = 500; // Dynode Payments begin once 500 Dynodes exist or more.

        consensus.nInstantSendConfirmationsRequired = 11;
        consensus.nInstantSendKeepLock = 24;

        consensus.nBudgetPaymentsStartBlock = 2055;   // actual historical value
        consensus.nBudgetPaymentsCycleBlocks = 20545; //Blocks per month
        consensus.nBudgetPaymentsWindowBlocks = 100;
        consensus.nBudgetProposalEstablishingTime = 24 * 60 * 60;

        consensus.nSuperblockStartBlock = 2055;
        consensus.nSuperblockStartHash = uint256S("0000008d283128ffecb10803a3317348908fd23bc9dceaba26f6d520a387de28");
        consensus.nSuperblockCycle = 20545; // 675 (Blocks per day) x 365.25 (Days per Year) / 12 = 20545

        consensus.nGovernanceMinQuorum = 10;
        consensus.nGovernanceFilterElements = 20000;

        consensus.nDynodeMinimumConfirmations = 15;

        consensus.nMajorityEnforceBlockUpgrade = 750;
        consensus.nMajorityRejectBlockOutdated = 950;
        consensus.nMajorityWindow = 1000;

        consensus.powLimit = uint256S("00000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPowTargetTimespan = 30 * 64; // Dynamic: 1920 seconds
        consensus.nPowTargetSpacing = DEFAULT_AVERAGE_POW_BLOCK_TIME;
        consensus.nUpdateDiffAlgoHeight = 10; // Dynamic: Algorithm fork block
        consensus.nPowAveragingWindow = 5;
        consensus.nPowMaxAdjustUp = 32;
        consensus.nPowMaxAdjustDown = 48;
        assert(maxUint / UintToArith256(consensus.powLimit) >= consensus.nPowAveragingWindow);
        consensus.fPowAllowMinDifficultyBlocks = false;
        consensus.fPowNoRetargeting = false;
        consensus.nRuleChangeActivationThreshold = 321; // 95% of nMinerConfirmationWindow
        consensus.nMinerConfirmationWindow = 30;        // nPowTargetTimespan / nPowTargetSpacing

        consensus.posLimit = uint256S("0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPosTargetSpacing = DEFAULT_AVERAGE_POS_BLOCK_TIME;
        consensus.nTargetPosTimespan = 30 * 60;  // 30 minutes
        consensus.nStakeMinDepth = 675; // 675 blocks = 1 day
        consensus.nStakeMinAge = 1 * 60 * 60 * 24; // 1 day minimum stake age
        consensus.nMaxClockDrift = 15 * 60;    // 15 minutes to elapse before new modifier is computed

        consensus.nMaxReorganizationDepth = 100;

        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].bit = 28;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nStartTime = 1199145601; // January 1, 2008
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nTimeout = 1230767999;   // December 31, 2008

        // Deployment of BIP68, BIP112, and BIP113.
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].bit = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nStartTime = 1513591800; // Dec 18th 2017 10:10:00
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nTimeout = 1545134400;   // Dec 18th 2018 12:00:00

        // Deployment of BIP147
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].bit = 2;
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nStartTime = 1533945600; // Aug 11th, 2018
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nTimeout = 1565481600; // Aug 11th, 2019
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nWindowSize = 4032;
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nThreshold = 3226; // 80% of 4032

        // Deployment of InstantSend autolocks
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].bit = 4;
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nStartTime = 1533945600; // Aug 11th, 2018
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nTimeout = 1565481600; // Aug 11th, 2019
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nWindowSize = 4032;
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nThreshold = 3226; // 80% of 4032

        // The best chain should have at least this much work.
        consensus.nMinimumChainWork = 215493;

        // By default assume that the signatures in ancestors of this block are valid.
        consensus.defaultAssumeValid = uint256S("0x00000000261e02ae2148505c0265b5f57c34a26b0958a21c1b2bb5a3d8c746ea"); // 215493

        /**
         * The message start string is designed to be unlikely to occur in normal data.
         * The characters are rarely used upper ASCII, not valid as UTF-8, and produce
         * a large 32-bit integer with any alignment.
         */
        pchMessageStart[0] = 0x5e;
        pchMessageStart[1] = 0x61;
        pchMessageStart[2] = 0x74;
        pchMessageStart[3] = 0x80;
        vAlertPubKey = ParseHex("04bf1391ff0c61a5d9a02cd2e997b707ced89bb48514e26d89f2464c98295ffef3f587263c94e6024d4e455802ad73e1e9694f3e482ff6e074736cb2327f9cd3e7");
        nDefaultPort = DEFAULT_P2P_PORT;
        nPruneAfterHeight = 20545;
        startNewChain = false;

        genesis = CreateGenesisBlock(1513619300, 626614, UintToArith256(consensus.powLimit).GetCompact(), 1, (1 * COIN));
        if (startNewChain == true) {
            MineGenesis(genesis, consensus.powLimit, true);
        }

        consensus.hashGenesisBlock = genesis.GetHash();

        if (!startNewChain) {
            assert(consensus.hashGenesisBlock == uint256S("0x00000e140b0c3028f898431890e9dea79ae6ca537ac9362c65b45325db712de2"));
            assert(genesis.hashMerkleRoot == uint256S("0xfa0e753db5a853ebbc52594eb62fa8219155547b426fba8789fa96dbf07e6ed5"));
        }

        vSeeds.push_back(CDNSSeedData("dnsseeder.network", "dyn-mainnet01.dnsseeder.network"));
        vSeeds.push_back(CDNSSeedData("dnsseeder.network", "dyn-mainnet02.dnsseeder.network"));
        vSeeds.push_back(CDNSSeedData("dnsseeder.network", "dyn-mainnet03.dnsseeder.network"));

        // Dynamic addresses start with 'D'
        base58Prefixes[PUBKEY_ADDRESS] = std::vector<unsigned char>(1, 30);
        // Dynamic script addresses start with '5'
        base58Prefixes[SCRIPT_ADDRESS] = std::vector<unsigned char>(1, 10);
        // Dynamic private keys start with 'y'
        base58Prefixes[SECRET_KEY] = std::vector<unsigned char>(1, 140);
        // Dynamic BIP32 pubkeys start with 'xpub' (Bitcoin defaults)
        base58Prefixes[EXT_PUBLIC_KEY] = boost::assign::list_of(0x04)(0x88)(0xB2)(0x1E).convert_to_container<std::vector<unsigned char> >();
        // Dynamic BIP32 prvkeys start with 'xprv' (Bitcoin defaults)
        base58Prefixes[EXT_SECRET_KEY] = boost::assign::list_of(0x04)(0x88)(0xAD)(0xE4).convert_to_container<std::vector<unsigned char> >();
        // Dynamic Stealth Address start with 'L'
        base58Prefixes[STEALTH_ADDRESS] = {0x0F};
        // Dynamic BIP44 coin type is '5'
        nExtCoinType = 5;

        vFixedSeeds = std::vector<SeedSpec6>(pnSeed6_main, pnSeed6_main + ARRAYLEN(pnSeed6_main));

        fMiningRequiresPeers = true;
        fDefaultConsistencyChecks = false;
        fRequireStandard = true;
        fRequireRoutableExternalIP = true;
        fMineBlocksOnDemand = false;
        fAllowMultipleAddressesFromGroup = false;
        fAllowMultiplePorts = false;

        nPoolMaxTransactions = 3;
        nFulfilledRequestExpireTime = 60 * 60; // fulfilled requests expire in 1 hour

        vSporkAddresses = {"DDDax6fjzoCqHj9nwTgNdAQsucFBJUJ3Jk"};
        nMinSporkKeys = 1;

        checkpointData = (CCheckpointData){
            boost::assign::map_list_of(0, uint256S("0x00000e140b0c3028f898431890e9dea79ae6ca537ac9362c65b45325db712de2"))(200, uint256S("0x000000f7f9132cefc7af54b131bb25bf33686af87987f60ed68ee00841d3f12b"))(1000, uint256S("0x0000009fc6bc247441a334333a5b24c81d0d606df8c0d8c2fd373c1241bc2036"))(4000, uint256S("0x00000013fceb3082d6c812b372baa18682cfb4ffbbc6a55073e602c8a2679de5"))(10000, uint256S("0x000000043989ffa9fc3fb37663e32b81f8da490d9d38808cd8455ca5996415f4"))(40000, uint256S("0x0000000385a212537b0048c47d5cbce3fc6a12f16d8b9afcd13c129c9abc768f"))(80000, uint256S("0x0000000094c0ca21a4a4b8ad76ab76a3751759627d3be47d885389672818d5a8"))(100000, uint256S("0x000000006403817b5efdb846e0dacf5959dbb65439531bf8ab0aa0c7c41837f1"))};

        chainTxData = ChainTxData{
            0,  // * UNIX timestamp of last known number of transactions
            0,  // * total number of transactions between genesis and that timestamp
                //   (the tx=... number in the SetBestChain debug.log lines)
            0.1 // * estimated number of transactions per second after that timestamp
        };
    }
};
static CMainParams mainParams;

/**
 * Testnet (v3)
 */
class CTestNetParams : public CChainParams
{
public:
    CTestNetParams()
    {
        strNetworkID = "test";

        consensus.nRewardsStart = 0; // Rewards starts on block 0
        consensus.nDynodePaymentsStartBlock = 0;
        consensus.nMinCountDynodesPaymentStart = 1; // Dynode Payments begin once 1 Dynode exists or more.

        consensus.nInstantSendConfirmationsRequired = 11;
        consensus.nInstantSendKeepLock = 24;

        consensus.nBudgetPaymentsStartBlock = 200;
        consensus.nBudgetPaymentsCycleBlocks = 50;
        consensus.nBudgetPaymentsWindowBlocks = 10;
        consensus.nBudgetProposalEstablishingTime = 60 * 20;

        consensus.nSuperblockStartBlock = 0;
        consensus.nSuperblockStartHash = uint256(); // do not check this on testnet
        consensus.nSuperblockCycle = 24;            // Superblocks can be issued hourly on testnet

        consensus.nGovernanceMinQuorum = 1;
        consensus.nGovernanceFilterElements = 500;

        consensus.nDynodeMinimumConfirmations = 1;

        consensus.nMajorityEnforceBlockUpgrade = 510;
        consensus.nMajorityRejectBlockOutdated = 750;
        consensus.nMajorityWindow = 1000;

        consensus.powLimit = uint256S("00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPowAveragingWindow = 5;
        consensus.nPowMaxAdjustUp = 32;
        consensus.nPowMaxAdjustDown = 48;
        consensus.nPowTargetTimespan = 30 * 64; // Dynamic: 1920 seconds
        consensus.nPowTargetSpacing = DEFAULT_AVERAGE_POW_BLOCK_TIME;
        consensus.nUpdateDiffAlgoHeight = 10; // Dynamic: Algorithm fork block
        assert(maxUint / UintToArith256(consensus.powLimit) >= consensus.nPowAveragingWindow);
        consensus.fPowAllowMinDifficultyBlocks = true;
        consensus.fPowNoRetargeting = false;
        consensus.nRuleChangeActivationThreshold = 254; // 75% of nMinerConfirmationWindow
        consensus.nMinerConfirmationWindow = 30;        // nPowTargetTimespan / nPowTargetSpacing

        consensus.posLimit = uint256S("0fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPosTargetSpacing = DEFAULT_AVERAGE_POS_BLOCK_TIME;
        consensus.nTargetPosTimespan = 30 * 60;  // 30 minutes
        consensus.nStakeMinDepth = 20;
        consensus.nStakeMinAge = 1 * 60 * 60;     // 1 hour minimum stake age
        consensus.nMaxClockDrift = 15 * 60;    // 15 minutes to elapse before new modifier is computed

        consensus.nMaxReorganizationDepth = 100;

        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].bit = 28;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nStartTime = 1199145601; // January 1, 2008
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nTimeout = 1230767999;   // December 31, 2008

        // Deployment of BIP68, BIP112, and BIP113.
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].bit = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nStartTime = 1513591800; // Dec 18th 2017 10:10:00
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nTimeout = 1545134400;   // Dec 18th 2018 12:00:00

        // Deployment of BIP147
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].bit = 2;
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nStartTime = 1517792400; // Feb 5th, 2018
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nTimeout = 1549328400; // Feb 5th, 2019
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nWindowSize = 100;
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nThreshold = 50; // 50% of 100

        // Deployment of InstantSend autolocks
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].bit = 4;
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nStartTime = 1532476800; // Jul 25th, 2018
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nTimeout = 1564012800; // Jul 25th, 2019
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nWindowSize = 100;
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nThreshold = 50; // 50% of 100

        // The best chain should have at least this much work.
        consensus.nMinimumChainWork = 210; // 210

        // By default assume that the signatures in ancestors of this block are valid.
        consensus.defaultAssumeValid = uint256S("0x00001a32bc1d6887d29d3847e21fcfb1026e14369df048a7f0666acdd9ccdf0d"); // 210

        pchMessageStart[0] = 0x2f;
        pchMessageStart[1] = 0x32;
        pchMessageStart[2] = 0x15;
        pchMessageStart[3] = 0x40;
        vAlertPubKey = ParseHex("04b375643a0b5fe3a9882b412be4046272528ad24576c72a933b44935a8491d52e243e905b1dd6947094984fc8e9f42d17cc1058033036d7940d9078851641a445");
        nDefaultPort = DEFAULT_P2P_PORT + 100;
        nPruneAfterHeight = 100;
        startNewChain = false;

        genesis = CreateGenesisBlock(1515641597, 747, UintToArith256(consensus.powLimit).GetCompact(), 1, (1 * COIN));
        if (startNewChain == true) {
            MineGenesis(genesis, consensus.powLimit, true);
        }

        consensus.hashGenesisBlock = genesis.GetHash();

        if (!startNewChain) {
            assert(consensus.hashGenesisBlock == uint256S("0x00ff3a06390940bc3fffb7948cc6d0ede8fde544a5fa9eeeafbc4ac65d21f087"));
            assert(genesis.hashMerkleRoot == uint256S("0xfa0e753db5a853ebbc52594eb62fa8219155547b426fba8789fa96dbf07e6ed5"));
        }
        vFixedSeeds.clear();
        vSeeds.clear();
        //vSeeds.push_back(CDNSSeedData("",  ""));
        //vSeeds.push_back(CDNSSeedData("", ""));

        // Testnet Dynamic addresses start with 'y'
        base58Prefixes[PUBKEY_ADDRESS] = std::vector<unsigned char>(1, 30);
        // Testnet Dynamic script addresses start with '8' or '9'
        base58Prefixes[SCRIPT_ADDRESS] = std::vector<unsigned char>(1, 10);
        // Testnet private keys start with '9' or 'c' (Bitcoin defaults)
        base58Prefixes[SECRET_KEY] = std::vector<unsigned char>(1, 158);
        // Testnet Dynamic BIP32 pubkeys start with 'tpub' (Bitcoin defaults)
        base58Prefixes[EXT_PUBLIC_KEY] = boost::assign::list_of(0x04)(0x35)(0x87)(0xCF).convert_to_container<std::vector<unsigned char> >();
        // Testnet Dynamic BIP32 prvkeys start with 'tprv' (Bitcoin defaults)
        base58Prefixes[EXT_SECRET_KEY] = boost::assign::list_of(0x04)(0x35)(0x83)(0x94).convert_to_container<std::vector<unsigned char> >();
        // Dynamic Stealth Address start with 'T'
        base58Prefixes[STEALTH_ADDRESS] = {0x15};
        // Testnet Dynamic BIP44 coin type is '1' (All coin's testnet default)
        nExtCoinType = 1;

        vFixedSeeds = std::vector<SeedSpec6>(pnSeed6_test, pnSeed6_test + ARRAYLEN(pnSeed6_test));

        fMiningRequiresPeers = false;
        fDefaultConsistencyChecks = false;
        fRequireStandard = false;
        fRequireRoutableExternalIP = true;
        fMineBlocksOnDemand = false;
        fAllowMultipleAddressesFromGroup = false;
        fAllowMultiplePorts = false;

        nPoolMaxTransactions = 3;
        nFulfilledRequestExpireTime = 5 * 60; // fulfilled requests expire in 5 minutes
        vSporkAddresses = {"DBUPr7TYK8auydiK22QYXP1mHDJ71h2G7N"};
        nMinSporkKeys = 1;

        checkpointData = (CCheckpointData){
            boost::assign::map_list_of
            (0, uint256S("0x00ff3a06390940bc3fffb7948cc6d0ede8fde544a5fa9eeeafbc4ac65d21f087"))
            (101, uint256S("0x00000ad8c5b094f7d6e63c658f8b29dfb2fbe49bb395bd41751d70e73e1f765b"))
            (702, uint256S("0x000004ca904b66049dbb93c7f129dec1c9e42213d42a5983a4b6ff7f4f88a912"))
        };

        chainTxData = ChainTxData{
            0,  // * UNIX timestamp of last known number of transactions
            0,  // * total number of transactions between genesis and that timestamp
                //   (the tx=... number in the SetBestChain debug.log lines)
            0.1 // * estimated number of transactions per second after that timestamp
        };
    }
};
static CTestNetParams testNetParams;

/**
 * Regression test
 */
class CRegTestParams : public CChainParams
{
public:
    CRegTestParams()
    {
        strNetworkID = "regtest";
        consensus.nRewardsStart = 0; // Rewards starts on block 0
        consensus.nDynodePaymentsStartBlock = 0;
        consensus.nMinCountDynodesPaymentStart = 1; // Dynode Payments begin once 1 Dynode exists or more.

        consensus.nInstantSendConfirmationsRequired = 11;
        consensus.nInstantSendKeepLock = 24;

        consensus.nBudgetPaymentsStartBlock = 1000;
        consensus.nBudgetPaymentsCycleBlocks = 50;
        consensus.nBudgetPaymentsWindowBlocks = 10;
        consensus.nBudgetProposalEstablishingTime = 60 * 20;

        consensus.nSuperblockStartBlock = 0;
        consensus.nSuperblockStartHash = uint256(); // do not check this on regtest
        consensus.nSuperblockCycle = 10;

        consensus.nGovernanceMinQuorum = 1;
        consensus.nGovernanceFilterElements = 100;

        consensus.nDynodeMinimumConfirmations = 1;

        consensus.nMajorityEnforceBlockUpgrade = 750;
        consensus.nMajorityRejectBlockOutdated = 950;
        consensus.nMajorityWindow = 1000;

        consensus.powLimit = uint256S("000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPowAveragingWindow = 5;
        consensus.nPowMaxAdjustUp = 32;
        consensus.nPowMaxAdjustDown = 48;
        consensus.nPowTargetTimespan = 30 * 64; // Dynamic: 1920 seconds
        consensus.nPowTargetSpacing = DEFAULT_AVERAGE_POW_BLOCK_TIME;
        consensus.nUpdateDiffAlgoHeight = 10; // Dynamic: Algorithm fork block
        assert(maxUint / UintToArith256(consensus.powLimit) >= consensus.nPowAveragingWindow);
        consensus.fPowAllowMinDifficultyBlocks = true;
        consensus.fPowNoRetargeting = true;
        consensus.nRuleChangeActivationThreshold = 254; // 75% of nMinerConfirmationWindow
        consensus.nMinerConfirmationWindow = 30;        // nPowTargetTimespan / nPowTargetSpacing

        consensus.posLimit = uint256S("00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPosTargetSpacing = DEFAULT_AVERAGE_POS_BLOCK_TIME;
        consensus.nTargetPosTimespan = 30 * 60;  // 30 minutes
        consensus.nStakeMinDepth = 20;
        consensus.nStakeMinAge = 1 * 60 * 60;     // 1 hour minimum stake age
        consensus.nMaxClockDrift = 15 * 60;    // 15 minutes to elapse before new modifier is computed

        consensus.nMaxReorganizationDepth = 100;

        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].bit = 28;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nStartTime = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nTimeout = 999999999999ULL;

        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].bit = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nStartTime = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nTimeout = 999999999999ULL;

        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].bit = 2;
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nStartTime = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_BIP147].nTimeout = 999999999999ULL;

        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].bit = 4;
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nStartTime = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_ISAUTOLOCKS].nTimeout = 999999999999ULL;

        // The best chain should have at least this much work.
        consensus.nMinimumChainWork = 0;

        // By default assume that the signatures in ancestors of this block are valid.
        consensus.defaultAssumeValid = uint256S("0x");

        pchMessageStart[0] = 0x2f;
        pchMessageStart[1] = 0x32;
        pchMessageStart[2] = 0x15;
        pchMessageStart[3] = 0x3f;
        vAlertPubKey = ParseHex("04e8118b469667861157f3b2b28056ae92581ce61ce2db80d04a701f5ec5391b751e6136bafdcca7b8d0b564a5afce213e8069bdd1d17131f61d116b73dbf7e2d6");
        nDefaultPort = DEFAULT_P2P_PORT + 200;
        nPruneAfterHeight = 100;
        startNewChain = false;

        genesis = CreateGenesisBlock(1513619951, 1754, UintToArith256(consensus.powLimit).GetCompact(), 1, (1 * COIN));
        if (startNewChain == true) {
            MineGenesis(genesis, consensus.powLimit, true);
        }

        consensus.hashGenesisBlock = genesis.GetHash();

        if (!startNewChain) {
            assert(consensus.hashGenesisBlock == uint256S("0x000ab751d858e116043e741d097311f2382e600c219483cfda8f25c7f369cc2c"));
            assert(genesis.hashMerkleRoot == uint256S("0xfa0e753db5a853ebbc52594eb62fa8219155547b426fba8789fa96dbf07e6ed5"));
        }

        vFixedSeeds.clear(); //! Regtest mode doesn't have any fixed seeds.
        vSeeds.clear();      //! Regtest mode doesn't have any DNS seeds.

        fMiningRequiresPeers = false;
        fDefaultConsistencyChecks = true;
        fRequireStandard = false;
        fRequireRoutableExternalIP = true;
        fMineBlocksOnDemand = true;
        fAllowMultipleAddressesFromGroup = false;
        fAllowMultiplePorts = false;

        nFulfilledRequestExpireTime = 5 * 60; // fulfilled requests expire in 5 minutes

        vSporkAddresses = {"ygUqnUfyRnRfBUks3EBc937tgmYBwQYE2S"}; //private key: cT21Wm3oozS7HpP9K9g1SDxdgr2vw9TBPSTxjxeArLjYxGastsf9
        nMinSporkKeys = 1;

        checkpointData = (CCheckpointData){
            boost::assign::map_list_of(0, uint256S("0x000ab751d858e116043e741d097311f2382e600c219483cfda8f25c7f369cc2c"))};

        chainTxData = ChainTxData{
            0,  // * UNIX timestamp of last known number of transactions
            0,  // * total number of transactions between genesis and that timestamp
                //   (the tx=... number in the SetBestChain debug.log lines)
            0.1 // * estimated number of transactions per second after that timestamp
        };

        // Regtest Dynamic addresses start with 'y'
        base58Prefixes[PUBKEY_ADDRESS] = std::vector<unsigned char>(1, 140);
        // Regtest Dynamic script addresses start with '8' or '9'
        base58Prefixes[SCRIPT_ADDRESS] = std::vector<unsigned char>(1, 19);
        // Regtest private keys start with '9' or 'c' (Bitcoin defaults)
        base58Prefixes[SECRET_KEY] = std::vector<unsigned char>(1, 239);
        // Regtest Dynamic BIP32 pubkeys start with 'tpub' (Bitcoin defaults)
        base58Prefixes[EXT_PUBLIC_KEY] = boost::assign::list_of(0x04)(0x35)(0x87)(0xCF).convert_to_container<std::vector<unsigned char> >();
        // Regtest Dynamic BIP32 prvkeys start with 'tprv' (Bitcoin defaults)
        base58Prefixes[EXT_SECRET_KEY] = boost::assign::list_of(0x04)(0x35)(0x83)(0x94).convert_to_container<std::vector<unsigned char> >();
        // Dynamic Stealth Address start with 'R'
        base58Prefixes[STEALTH_ADDRESS] = {0x13};
        // Regtest Dynamic BIP44 coin type is '1' (All coin's testnet default)
        nExtCoinType = 1;
    }
    void UpdateBIP9Parameters(Consensus::DeploymentPos d, int64_t nStartTime, int64_t nTimeout)
    {
        consensus.vDeployments[d].nStartTime = nStartTime;
        consensus.vDeployments[d].nTimeout = nTimeout;
    }
};
static CRegTestParams regTestParams;


/**
 * Privatenet
 */
class CPrivateNetParams : public CChainParams
{
public:
    CPrivateNetParams()
    {
        strNetworkID = "privatenet";

        consensus.nRewardsStart = 0; // Rewards starts on block 0
        consensus.nDynodePaymentsStartBlock = 0;
        consensus.nMinCountDynodesPaymentStart = 1; // Dynode Payments begin once 1 Dynode exists or more.

        consensus.nInstantSendConfirmationsRequired = 11;
        consensus.nInstantSendKeepLock = 24;

        consensus.nBudgetPaymentsStartBlock = 200;
        consensus.nBudgetPaymentsCycleBlocks = 50;
        consensus.nBudgetPaymentsWindowBlocks = 10;
        consensus.nBudgetProposalEstablishingTime = 60 * 20;

        consensus.nSuperblockStartBlock = 0;
        consensus.nSuperblockStartHash = uint256(); // do not check this on testnet
        consensus.nSuperblockCycle = 24;            // Superblocks can be issued hourly on testnet

        consensus.nGovernanceMinQuorum = 1;
        consensus.nGovernanceFilterElements = 500;

        consensus.nDynodeMinimumConfirmations = 1;

        consensus.nMajorityEnforceBlockUpgrade = 510;
        consensus.nMajorityRejectBlockOutdated = 750;
        consensus.nMajorityWindow = 1000;

        consensus.powLimit = uint256S("0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPowAveragingWindow = 5;
        consensus.nPowMaxAdjustUp = 32;
        consensus.nPowMaxAdjustDown = 48;
        consensus.nPowTargetTimespan = 30 * 64; // Dynamic: 1920 seconds
        consensus.nPowTargetSpacing = DEFAULT_AVERAGE_POW_BLOCK_TIME;
        consensus.nUpdateDiffAlgoHeight = 10; // Dynamic: Algorithm fork block
        assert(maxUint / UintToArith256(consensus.powLimit) >= consensus.nPowAveragingWindow);
        consensus.fPowAllowMinDifficultyBlocks = true;
        consensus.fPowNoRetargeting = false;
        consensus.nRuleChangeActivationThreshold = 254; // 75% of nMinerConfirmationWindow
        consensus.nMinerConfirmationWindow = 30;        // nPowTargetTimespan / nPowTargetSpacing

        consensus.posLimit = uint256S("000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        consensus.nPosTargetSpacing = DEFAULT_AVERAGE_POS_BLOCK_TIME;
        consensus.nTargetPosTimespan = 30 * 60;  // 30 minutes
        consensus.nStakeMinDepth = 20;
        consensus.nStakeMinAge = 1 * 60 * 60; // 1 hour
        consensus.nMaxClockDrift = 15 * 60;    // 15 minutes to elapse before new modifier is computed

        consensus.nMaxReorganizationDepth = 100;

        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].bit = 28;
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nStartTime = 1199145601; // January 1, 2008
        consensus.vDeployments[Consensus::DEPLOYMENT_TESTDUMMY].nTimeout = 1230767999;   // December 31, 2008

        // Deployment of BIP68, BIP112, and BIP113.
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].bit = 0;
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nStartTime = 1513591800; // Dec 18th 2017 10:10:00
        consensus.vDeployments[Consensus::DEPLOYMENT_CSV].nTimeout = 1545134400;   // Dec 18th 2018 12:00:00

        // The best chain should have at least this much work.
        consensus.nMinimumChainWork = 210; // 210

        // By default assume that the signatures in ancestors of this block are valid.
        consensus.defaultAssumeValid = uint256S("0x00001a32bc1d6887d29d3847e21fcfb1026e14369df048a7f0666acdd9ccdf0d"); // 210

        pchMessageStart[0] = 0x2f;
        pchMessageStart[1] = 0x32;
        pchMessageStart[2] = 0x15;
        pchMessageStart[3] = 0x40;
        // To import alert key:  importprivkey 6Jjb9DG1cr71VWiwxg97zVEyZUBhFzzGhqE7GY9DrbYYM6gVgxS
        vAlertPubKey = ParseHex("043d9e8440ea8fe66b0c2639f0a0931c9d7c41132ec9ee04cdf5d9e88ada2c2df52d93a0c1983958d3aea56df9fb3d1a61ca4eb6f72c27456fc313be80cdc70032");
        nDefaultPort = DEFAULT_P2P_PORT + 300; // 33600 
        nPruneAfterHeight = 100;
        startNewChain = false;

        genesis = CreateGenesisBlock(1559867972, 60883, UintToArith256(consensus.powLimit).GetCompact(), 1, (1 * COIN));
        if (startNewChain == true) {
            MineGenesis(genesis, consensus.powLimit, true);
        }

        consensus.hashGenesisBlock = genesis.GetHash();

        if (!startNewChain) {
            assert(consensus.hashGenesisBlock == uint256S("0x000055a9348d53bed51996102ad11d129207e85dc197d01a5a69d5fd10af0e8a"));
            assert(genesis.hashMerkleRoot == uint256S("0xfa0e753db5a853ebbc52594eb62fa8219155547b426fba8789fa96dbf07e6ed5"));
        }
        vFixedSeeds.clear();
        vSeeds.clear();
        //vSeeds.push_back(CDNSSeedData("",  ""));
        //vSeeds.push_back(CDNSSeedData("", ""));

        // Privatenet Dynamic addresses start with 'y'
        base58Prefixes[PUBKEY_ADDRESS] = std::vector<unsigned char>(1, 30);
        // Privatenet Dynamic script addresses start with '8' or '9'
        base58Prefixes[SCRIPT_ADDRESS] = std::vector<unsigned char>(1, 10);
        // Privatenet private keys start with '9' or 'c' (Bitcoin defaults)
        base58Prefixes[SECRET_KEY] = std::vector<unsigned char>(1, 158);
        // Privatenet Dynamic BIP32 pubkeys start with 'tpub' (Bitcoin defaults)
        base58Prefixes[EXT_PUBLIC_KEY] = boost::assign::list_of(0x04)(0x35)(0x87)(0xCF).convert_to_container<std::vector<unsigned char> >();
        // Privatenet Dynamic BIP32 prvkeys start with 'tprv' (Bitcoin defaults)
        base58Prefixes[EXT_SECRET_KEY] = boost::assign::list_of(0x04)(0x35)(0x83)(0x94).convert_to_container<std::vector<unsigned char> >();
        // Privatenet Stealth Address start with 'P'
        base58Prefixes[STEALTH_ADDRESS] = {0x12};
        // Privatenet Dynamic BIP44 coin type is '1' (All coin's testnet default)
        nExtCoinType = 1;

        vFixedSeeds = std::vector<SeedSpec6>(pnSeed6_privatenet, pnSeed6_privatenet + ARRAYLEN(pnSeed6_privatenet));

        fMiningRequiresPeers = false;
        fDefaultConsistencyChecks = false;
        fRequireStandard = false;
        fRequireRoutableExternalIP = true;
        fMineBlocksOnDemand = false;
        fAllowMultipleAddressesFromGroup = false;
        fAllowMultiplePorts = false;

        nPoolMaxTransactions = 3;
        nFulfilledRequestExpireTime = 5 * 60; // fulfilled requests expire in 5 minutes
        // To import spork key (D777Y4eMXrf1NgDSY1Q7kjoZuVso1ed7HL): importprivkey QWUVh41RrhjhnF813U5XLU4S8qYjvDQ5L1n53jC7Qawr8bBCQfFh
        vSporkAddresses = {"D777Y4eMXrf1NgDSY1Q7kjoZuVso1ed7HL"};
        nMinSporkKeys = 1;

        checkpointData = (CCheckpointData){
            boost::assign::map_list_of(0, uint256S("0x00ff3a06390940bc3fffb7948cc6d0ede8fde544a5fa9eeeafbc4ac65d21f087"))};

        chainTxData = ChainTxData{
            0,  // * UNIX timestamp of last known number of transactions
            0,  // * total number of transactions between genesis and that timestamp
                //   (the tx=... number in the SetBestChain debug.log lines)
            0.1 // * estimated number of transactions per second after that timestamp
        };
    }
};
static CPrivateNetParams privateNetParams;

static CChainParams* pCurrentParams = 0;

const CChainParams& Params()
{
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
    else if (chain == CBaseChainParams::PRIVATENET)
        return privateNetParams;
    else
        throw std::runtime_error(strprintf("%s: Unknown chain %s.", __func__, chain));
}

void SelectParams(const std::string& network)
{
    SelectBaseParams(network);
    pCurrentParams = &Params(network);
}

void UpdateRegtestBIP9Parameters(Consensus::DeploymentPos d, int64_t nStartTime, int64_t nTimeout)
{
    regTestParams.UpdateBIP9Parameters(d, nStartTime, nTimeout);
}
