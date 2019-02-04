// Copyright (c) 2018 Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_BDAP_UTILS_H
#define DYNAMIC_BDAP_UTILS_H

#include "bdap/bdap.h"
#include "amount.h"
#include "primitives/transaction.h"

#include <string>
#include <vector>

class CCoinsViewCache;
class CDynamicAddress;
struct CRecipient;
class CScript;
class CTxOut;
class CTransaction;
class uint256;
class UniValue;
namespace Consensus {
    struct Params;
}



std::string BDAPFromOp(const int op);
bool IsBDAPDataOutput(const CTxOut& out);
int GetBDAPDataOutput(const CTransactionRef& tx);
bool GetBDAPData(const CTransactionRef& tx, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash, int& nOut);
bool GetBDAPData(const CScript& scriptPubKey, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash);
bool GetBDAPData(const CTxOut& out, std::vector<unsigned char>& vchData, std::vector<unsigned char>& vchHash);
std::string stringFromVch(const CharString& vch);
std::vector<unsigned char> vchFromValue(const UniValue& value);
std::vector<unsigned char> vchFromString(const std::string& str);
void CreateRecipient(const CScript& scriptPubKey, CRecipient& recipient);
void ToLowerCase(CharString& vchValue);
void ToLowerCase(std::string& strValue);
CAmount GetDataFee(const CScript& scriptPubKey);
CAmount GetBDAPFee(const CScript& scriptPubKey);
bool DecodeBDAPTx(const CTransactionRef& tx, int& op1, int& op2, std::vector<std::vector<unsigned char> >& vvch);
bool FindBDAPInTx(const CCoinsViewCache &inputs, const CTransaction& tx, std::vector<std::vector<unsigned char> >& vvch);
int GetBDAPOpType(const CScript& script);
int GetBDAPOpType(const CTxOut& out);
std::string GetBDAPOpTypeString(const int& op1, const int& op2);
bool GetBDAPOpScript(const CTransactionRef& tx, CScript& scriptBDAPOp, vchCharString& vvchOpParameters, int& op1, int& op2);
bool GetBDAPOpScript(const CTransactionRef& tx, CScript& scriptBDAPOp);
bool GetBDAPDataScript(const CTransaction& tx, CScript& scriptBDAPData);
bool GetBDAPDataScript(const CTransactionRef& ptx, CScript& scriptBDAPData);
bool IsBDAPOperationOutput(const CTxOut& out);
int GetBDAPOperationOutIndex(const CTransactionRef& tx);
int GetBDAPOperationOutIndex(int nHeight, const uint256& txHash);
bool GetBDAPTransaction(int nHeight, const uint256& hash, CTransactionRef &txOut, const Consensus::Params& consensusParams);
CDynamicAddress GetScriptAddress(const CScript& pubScript);
int GetBDAPOpCodeFromOutput(const CTxOut& out);
bool ExtractOpTypeValue(const CScript& script, std::string& strOpType, std::vector<unsigned char>& vchValue);
bool GetScriptOpTypeValue(const std::vector<CRecipient>& vecSend, CScript& bdapOpScript, std::string& strOpType, std::vector<unsigned char>& vchValue);
bool GetTransactionOpTypeValue(const CTransaction& tx, CScript& bdapOpScript, std::string& strOpType, std::vector<unsigned char>& vchValue);
int GetLinkVersionFromData(const std::vector<unsigned char>& vchData);
bool GetPreviousTxRefById(const uint256& prevTxId, CTransactionRef& prevTx);

#endif // DYNAMIC_BDAP_UTILS_H