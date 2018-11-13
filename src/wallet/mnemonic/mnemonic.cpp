// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "mnemonic.h"

#include "util.h"

#include <iostream>
#include <cstring>

#include <openssl/ecdsa.h>
#include <openssl/rand.h>
#include <openssl/obj_mac.h>
#include <openssl/evp.h>

void Mnemonic::setMnemonic(string &mnemoics){
    SplitStringToVector(mnemoics,data,"-");

    //check mnemonic size
    if((data.size() % 3) != 0 || (data.size() / 3) < 4 || (data.size() / 3) > 8)
        data.clear();
       
}

unsigned char *Mnemonic::MnemonicToSeed(){
    unsigned char *out;
    const char *mnemoichar;
    string mnemoics;
    //const char mnemonic[78] = {'i','n','f','l','i','c','t',' ','w','i','t','n','e','s','s',' ','o','f','f',' ','p','r','o','p','e','r','t','y',' ','t','a','r','g','e','t',' ','f','a','i','n','t',' ','g','a','t','h','e','r',' ','m','a','t','c','h',' ','o','u','t','d','o','o','r',' ','w','e','a','p','o','n',' ','w','i','d','e',' ','m','i','x'};
    unsigned char salt_value[] = {'m','n','e','m','o','n','i','c'};

    for(string str : data){
        mnemoics.append(str);
        mnemoics.append(" ");
    }

    mnemoics.erase((mnemoics.length()-1),1);
    mnemoichar = mnemoics.c_str();

    out = (unsigned char *) malloc(sizeof(unsigned char) * SEED_KEY_SIZE);

    LogPrintf("!!!!size%d,mnemoichar:%s\n",mnemoics.length(),mnemoichar);
    
    LogPrintf("\n");
    if(PKCS5_PBKDF2_HMAC(mnemoichar, mnemoics.length(), salt_value, sizeof(salt_value), PBKDF2_ROUNDS,EVP_sha512() ,SEED_KEY_SIZE, out) == 0 )
    {
        LogPrintf("PKCS5_PBKDF2_HMAC_SHA1 failed\n");
        free(out);
        return nullptr;
    }
    
    return out;
}

vector<unsigned char> Mnemonic::toVector(){
    vector<unsigned char> seedVector;
    unsigned char *out = MnemonicToSeed();
    for(int i = 0; i < SEED_KEY_SIZE; i++){
        seedVector.push_back(out[i]);
    }
    return seedVector;
}

void SplitStringToVector(const string& s, vector<string>& v, const string& c)  
{  
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        string str = s.substr(pos1, pos2-pos1);
        if(!str.empty())
            v.push_back(s.substr(pos1, pos2-pos1));
         
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));    
}
