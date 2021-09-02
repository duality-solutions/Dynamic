// Copyright (c) 2014-2017 The Dash Core Developers
// Copyright (c) 2016-2021 Duality Blockchain Solutions Developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "governance-validators.h"
#include "univalue.h"
#include "utilstrencodings.h"

#include "data/proposals_valid.json.h"
#include "data/proposals_invalid.json.h"

#include "test/test_dynamic.h"

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <fstream>
#include <string>

#include <univalue.h>

#ifdef ENABLE_GOVERNANCE_TESTS
extern UniValue read_json(const std::string& jsondata);

BOOST_FIXTURE_TEST_SUITE(governance_validators_tests, BasicTestingSetup)

std::string CreateEncodedProposalObject(const UniValue& objJSON)
{
    UniValue innerArray(UniValue::VARR);
    innerArray.push_back(UniValue("proposal"));
    innerArray.push_back(objJSON);

    UniValue outerArray(UniValue::VARR);
    outerArray.push_back(innerArray);
    
    std::string strData = outerArray.write();
    std::string strHex = HexStr(strData);
    return strHex;
}

BOOST_AUTO_TEST_CASE(valid_proposals_test)
{
    // all proposals are valid but expired
    UniValue tests = read_json(std::string(json_tests::proposals_valid, json_tests::proposals_valid + sizeof(json_tests::proposals_valid)));

    BOOST_CHECK_MESSAGE(tests.size(), "Empty `tests`");
    for(size_t i = 0; i < tests.size(); ++i) {
        const UniValue& objProposal = tests[i];

        // legacy format
        std::string strHexData1 = CreateEncodedProposalObject(objProposal);
        CProposalValidator validator1(strHexData1);
        BOOST_CHECK_MESSAGE(validator1.Validate(false), validator1.GetErrorMessages());
        BOOST_CHECK_MESSAGE(!validator1.Validate(), validator1.GetErrorMessages());

        // new format
        std::string strHexData2 = HexStr(objProposal.write());
        CProposalValidator validator2(strHexData2);
        BOOST_CHECK_MESSAGE(validator2.Validate(false), validator2.GetErrorMessages());
        BOOST_CHECK_MESSAGE(!validator2.Validate(), validator2.GetErrorMessages());
    }
}

BOOST_AUTO_TEST_CASE(invalid_proposals_test)
{
    // all proposals are invalid regardless of being expired or not
    // (i.e. we don't even check for expiration here)
    UniValue tests = read_json(std::string(json_tests::proposals_invalid, json_tests::proposals_invalid + sizeof(json_tests::proposals_invalid)));

    BOOST_CHECK_MESSAGE(tests.size(), "Empty `tests`");
    for(size_t i = 0; i < tests.size(); ++i) {
        const UniValue& objProposal = tests[i];

        // legacy format
        std::string strHexData1 = CreateEncodedProposalObject(objProposal);
        CProposalValidator validator1(strHexData1);
        BOOST_CHECK_MESSAGE(!validator1.Validate(false), validator1.GetErrorMessages());

        // new format
        std::string strHexData2 = HexStr(objProposal.write());
        CProposalValidator validator2(strHexData2);
        BOOST_CHECK_MESSAGE(!validator2.Validate(false), validator2.GetErrorMessages());
    }
}

BOOST_AUTO_TEST_SUITE_END()

#endif // ENABLE_GOVERNANCE_TESTS
