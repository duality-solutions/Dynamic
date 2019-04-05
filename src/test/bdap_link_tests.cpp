// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "key.h"

#include "base58.h"
#include "script/script.h"
#include "uint256.h"
#include "util.h"
#include "utilstrencodings.h"
#include "test/test_dynamic.h"
#include "bdap/linking.h"


#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <boost/test/unit_test.hpp>


//bdap_link_tests

BOOST_FIXTURE_TEST_SUITE(bdap_link_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(bdap_link_test1)
{
    CKey key;
    key.MakeNewKey(true);
    std::string strTestFQDN = "test@public.bdap.io";
    std::vector<unsigned char> vchSignature;
    CreateSignatureProof(key, strTestFQDN, vchSignature);
    CPubKey pubKey = key.GetPubKey();
    CKeyID keyID = pubKey.GetID();
    CDynamicAddress address = CDynamicAddress(keyID);
    BOOST_CHECK(SignatureProofIsValid(address, strTestFQDN, vchSignature) == true);

    std::cout << "Exit: bdap_link_test1\n";

}

BOOST_AUTO_TEST_SUITE_END()