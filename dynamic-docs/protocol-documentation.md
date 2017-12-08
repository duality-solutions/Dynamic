Protocol Documentation - 1.4.0.0
=====================================

This document describes the protocol extensions for all additional functionality build into the Dynamic protocol. This doesn't include any of the Bitcoin protocol, which has been left intact in the Dynamic project. For more information about the core protocol, please see https://en.bitcoin.it/w/index.php?title#Protocol_documentation&action#edit

## Common Structures

### Simple types

uint256  => char[32]

CScript => uchar[]

### COutPoint

Bitcoin Outpoint https://bitcoin.org/en/glossary/outpoint

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 32 | hash | uint256 | Hash of transactional output which is being referenced
| 4 | n | uint32_t | Index of transaction which is being referenced


### CTxIn

Bitcoin Input https://bitcoin.org/en/glossary/input

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 36 | prevout | COutPoint | The previous output from an existing transaction, in the form of an unspent output
| 1+ | script length | var_int | The length of the signature script
| ? | script | CScript | The script which is validated for this input to be spent
| 4 | nSequence | uint_32t | Transaction version as defined by the sender. Intended for "replacement" of transactions when information is updated before inclusion into a block.

### CTxOut

Bitcoin Output https://bitcoin.org/en/glossary/output

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 8 | nValue | int64_t | Transfered value
| ? | scriptPubKey | CScript | The script for indicating what conditions must be fulfilled for this output to be further spent

### CPubKey

Bitcoin Public Key https://bitcoin.org/en/glossary/public-key

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 33-65 | vch | char[] | The public portion of a keypair which can be used to verify signatures made with the private portion of the keypair.

## Message Types

### DNANNOUNCE - "dnb"

CDynodeBroadcast

Whenever a Dynode comes online or a client is syncing, they will send this message which describes the Dynode entry and how to validate messages from it.

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 41 | vin | CTxIn | The unspent output which is holding 1000 Dynamic
| # | addr | CService | IPv4 address of the Dynode
| 33-65 | pubKeyCollateralAddress | CPubKey | CPubKey of the main 1000 Dynamic unspent output
| 33-65 | pubKeyDynode | CPubKey | CPubKey of the secondary signing key (For all other messaging other than announce message)
| 71-73 | sig | char[] | Signature of this message (verifiable via pubKeyCollateralAddress)
| 8 | sigTime | int64_t | Time which the signature was created
| 4 | nProtocolVersion | int | The protocol version of the Dynode
| # | lastPing | CDynodePing | The last known ping of the Dynode
| 8 | nLastPsq | int64_t | The last time the Dynode sent a PSQ message (for mixing)

### DNPING - "dnp"

CDynodePing

Every few minutes, Dynodes ping the network with a message that propagates the whole network.

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 41 | vin | CTxIn | The unspent output of the Dynode which is signing the message
| 32 | blockHash | uint256 | Current chaintip blockhash minus 12
| 8 | sigTime | int64_t | Signature time for this ping
| 71-73 | vchSig | char[] | Signature of this message by Dynode (verifiable via pubKeyDynode)

### DynodePAYMENTVOTE - "dnw"

CDynodePaymentVote

When a new block is found on the network, a Dynode quorum will be determined and those 10 selected Dynodes will issue a Dynode payment vote message to pick the next winning node.

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 41 | vinDynode | CTxIn | The unspent output of the Dynode which is signing the message
| 4 | nBlockHeight | int | The blockheight which the payee should be paid
| ? | payeeAddress | CScript | The address to pay to
| 71-73 | sig | char[] | Signature of the Dynode which is signing the message

### PSTX - "pstx"

CPrivatesendBroadcastTx

Dynodes can broadcast subsidised transactions without fees for the sake of security in mixing. This is done via the PSTX message.

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| # | tx | CTransaction | The transaction
| 41 | vin | CTxIn | Dynode unspent output
| 71-73 | vchSig | char[] | Signature of this message by Dynode (verifiable via pubKeyDynode)
| 8 | sigTime | int64_t | Time this message was signed

### PSTATUSUPDATE - "pssu"

Mixing pool status update

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 4 | nMsgSessionID | int | Session ID
| 4 | nMsgState | int | Current state of mixing process
| 4 | nMsgEntriesCount | int | Number of entries in the mixing pool
| 4 | nMsgStatusUpdate | int | Update state and/or signal if entry was accepted or not
| 4 | nMsgMessageID | int | ID of the typical Dynode reply message

### PSQUEUE - "psq"

CPrivatesendQueue

Asks users to sign final mixing tx message.

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 4 | nDenom | int | Which denomination is allowed in this mixing session
| 41 | vin | CTxIn | unspend output from Dynode which is hosting this session
| 4 | nTime | int | the time this PSQ was created
| 4 | fReady | int | if the mixing pool is ready to be executed
| 71-73 | vchSig | char[] | Signature of this message by Dynode (verifiable via pubKeyDynode)

### PSACCEPT - "psa"

Response to PSQ message which allows the user to join a mixing pool

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 4 | nDenom | int | denomination that will be exclusively used when submitting inputs into the pool
| 41+ | txCollateral | int | collateral tx that will be charged if this client acts maliciousely

### PSVIN - "psi"

CPrivatesendEntry

When queue is ready user is expected to send his entry to start actual mixing

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| ? | vecTxPSIn | CTxPSIn[] | vector of users inputs (CTxPSIn serialization is equal to CTxIn serialization)
| 8 | nAmount | int64_t | depreciated, can be removed with future protocol bump
| ? | txCollateral | CTransaction | Collateral transaction which is used to prevent misbehavior and also to charge fees randomly
| ? | vecTxPSOut | CTxPSOut[] | vector of user outputs (CTxPSOut serialization is equal to CTxOut serialization)

### PSSIGNFINALTX - "pss"

User's signed inputs for a group transaction in a mixing session

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| # | inputs | CTxIn[] | signed inputs for mixing session


### TXLOCKREQUEST - "is"

CTxLockRequest

Transaction Lock Request, serialization is the same as for CTransaction.

### TXLOCKVOTE - "txlvote"

CTxLockVote

Transaction Lock Vote

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 32 | txHash | uint256 | txid of the transaction to lock
| 36 | outpoint | COutPoint | The utxo to lock in this transaction
| 36 | outpointDynode | COutPoint | The utxo of the dynode which is signing the vote
| 71-73 | vchDynodeSignature | char[] | Signature of this message by dynode (verifiable via pubKeyDynode)

### DNGOVERNANCEOBJECT - "govobj"

Governance Object

A proposal, contract or setting.

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 32 | nHashParent | uint256 | Parent object, 0 is root
| 4 | nRevision | int | Object revision in the system
| 8 | nTime | int64_t | Time which this object was created
| 32 | nCollateralHash | uint256 | Hash of the collateral fee transaction
| 0-16384 | strData | string | Data field - can be used for anything
| 4 | nObjectType | int | ????
| 41 | vinDynode | CTxIn | Unspent output for the Dynode which is signing this object
| 71-73 | vchSig | char[] | Signature of the Dynode

### DNGOVERNANCEOBJECTVOTE - "govobjvote"

Governance Vote

Dynodes use governance voting in response to new proposals, contracts, settings or finalized budgets.

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 41+ | vinDynode | CTxIn | Unspent output for the Dynode which is voting
| 32 | nParentHash | uint256 | Object which we're voting on (proposal, contract, setting or final budget)
| 4 | nVoteOutcome | int | ???
| 4 | nVoteSignal | int | ???
| 8 | nTime | int64_t | Time which the vote was created
| 71-73 | vchSig | char[] | Signature of the Dynode

### SPORK - "spork"

Spork

Spork

| Field Size | Field Name | Data type | Description |
| ---------- | ----------- | --------- | -------- |
| 4 | nSporkID | int | 
| 8 | nValue | int64_t | 
| 8 | nTimeSigned | int64_t | 
| 66* | vchSig | char[] | Unclear if 66 is the correct size, but this is what it appears to be in most cases

#### Defined Sporks (per src/sporks.h)
 
| Spork ID | Number | Name | Description | 
| ---------- | ---------- | ----------- | ----------- |
| 10001 | 2 | INSTANTSEND_ENABLED | Turns on and off InstantSend network wide
| 10002 | 3 | INSTANTSEND_BLOCK_FILTERING | Turns on and off InstantSend block filtering
| 10004 | 5 | INSTANTSEND_MAX_VALUE | Controls the max value for an InstantSend transaction (currently 2000 Dynamic)
| 10007 | 8 | DYNODE_PAYMENT_ENFORCEMENT | Requires dynodes to be paid by miners when blocks are processed
| 10008 | 9 | SUPERBLOCKS_ENABLED | Superblocks are enabled (the 10% comes to fund the dynamic treasury)
| 10009 | 10 | DYNODE_PAY_UPDATED_NODES | Only current protocol version dynode's will be paid (not older nodes)
| 10011 | 12 | RECONSIDER_BLOCKS |
| 10012 | 13 | OLD_SUPERBLOCK_FLAG |
| 10013 | 14 | REQUIRE_SENTINEL_FLAG | Only dynode's running sentinel will be paid 

## Undocumented messages

### DYNODEPAYMENTBLOCK - "dnwb"

Dynode Payment Block

*NOTE: Per src/protocol.cpp, there is no message for this (only inventory)*

### DNVERIFY - "dnv"

Dynode Verify

### PSFINALTX - "psf"

Privatesend Final Transaction

### PSCOMPLETE - "psc"

PrivateSend Complete

### TXLOCKREQUEST - "ix"

Tx Lock Request

### DNGOVERNANCESYNC - "govsync"

Governance Sync

### PSEG - "Pseg"

Dynode List/Entry Sync

Get Dynode list or specific entry

### SYNCSTATUSCOUNT - "ssc"

Sync Status Count

### DYNODEPAYMENTSYNC - "dnget"

Dynode Payment Sync