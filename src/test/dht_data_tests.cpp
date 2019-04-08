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
#include "bdap/utils.h"
#include "dht/ed25519.h"
#include "dht/datarecord.h"
#include "dht/dataheader.h"
#include "dht/datachunk.h"

#include <string>
#include <stdint.h>
#include <string.h>
#include <vector>

#include <boost/test/unit_test.hpp>

//dht_data_tests

BOOST_FIXTURE_TEST_SUITE(dht_data_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(dht_data_test1)
{

    int nExpireTime = GetTime() + 86400; // 1 Day

    uint16_t nTotalSlots = 32;

    CKeyEd25519 key; //key1
    CKeyEd25519 key2; //key2
    CKeyEd25519 key3; //key3 - this key will not be added
    std::string strOperationType = "oauth"; 
    std::vector<std::vector<unsigned char>> vvchPubKeys;
    std::vector<unsigned char> vchValue = vchFromString("GET /authorize?response_type=code&client_id=s6BhdRkqt3&state=xyz &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb HTTP/1.1"); //= vchFromValue(); //?????

    std::cout << "Enter: dht_data_tests\n";

    vvchPubKeys.push_back(key.GetPubKeyBytes()); //add public keys for key1
    vvchPubKeys.push_back(key2.GetPubKeyBytes()); //add public keys for key2

    //store record containing public keys for key1 and key2
    CDataRecord DataRecord(strOperationType, nTotalSlots, vvchPubKeys, vchValue, 1, nExpireTime, DHT::DataFormat::BinaryBlob);

    CRecordHeader header = DataRecord.GetHeader();
    std::vector<CDataChunk> vChunks = DataRecord.GetChunks();

    CDataRecord getRecord(strOperationType, nTotalSlots, header, vChunks, key.GetPrivSeedBytes()); //retrieve and decrypt record using key1
    CDataRecord getRecord2(strOperationType, nTotalSlots, header, vChunks, key2.GetPrivSeedBytes()); //retrieve and decrypt record using key2
    CDataRecord getRecord3(strOperationType, nTotalSlots, header, vChunks, key3.GetPrivSeedBytes()); //retrieve and decrypt record using key3

    BOOST_CHECK(DataRecord.Value() == getRecord.Value()); //compare value using key1 to original value
    BOOST_CHECK(DataRecord.Value() == getRecord2.Value()); //compare value using key2 to original value
    BOOST_CHECK((DataRecord.Value() == getRecord3.Value()) == false); //compare value using key3 to original value. expect this to be false

    std::cout << "Exit: dht_data_tests\n";

}

BOOST_AUTO_TEST_SUITE_END()