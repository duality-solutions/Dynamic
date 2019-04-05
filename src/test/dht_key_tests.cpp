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
#include "dht/ed25519.h"


#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <boost/test/unit_test.hpp>


//dht_key_tests

BOOST_FIXTURE_TEST_SUITE(dht_key_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(dht_key_getlinksharedpubkey_test)
{
    CKeyEd25519 dhtKey1;
    CKeyEd25519 dhtKey2;
    std::vector<unsigned char> vchSharedPubKey1 = GetLinkSharedPubKey(dhtKey1, dhtKey2.GetPubKey());
    std::vector<unsigned char> vchSharedPubKey2 = GetLinkSharedPubKey(dhtKey2, dhtKey1.GetPubKey());
    BOOST_CHECK(vchSharedPubKey1 == vchSharedPubKey2);
    std::cout << "Exit: dht_key_getlinksharedpubkey_test\n";

}

BOOST_AUTO_TEST_SUITE_END()