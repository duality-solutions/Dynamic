// Copyright (c) 2016-2018 Duality Blockchain Solutions Developers
// Copyright (c) 2009-2018 The Bitcoin Developers
// Copyright (c) 2009-2018 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_MNEMONIC_H
#define DYNAMIC_MNEMONIC_H

#include <vector>
#include <string> 

#define SEED_KEY_SIZE 64
#define PBKDF2_ROUNDS 2048
using namespace std;

class Mnemonic {
    private:
        vector<string> data;
    public:
        void setMnemonic(string &mnemoics);

        vector<string> getMnemonic(){return data;};
        vector<unsigned char> toVector();
        unsigned char *MnemonicToSeed();

        Mnemonic(){
        };

        Mnemonic(string mnemoics){
            setMnemonic(mnemoics);
        };

        ~Mnemonic(){

        }
        
};

void SplitStringToVector(const string& source, vector<string>& v, const string& c);  



#endif // DYNAMIC_MNEMONIC_H