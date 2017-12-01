**Dynamic v1.5.0.0**
* Fix Network Time Protocol (NTP)
* Introduce, OP_MINT, OP_REWARD_DYNODE and OP_REWARD_MINING opcode for Fluid Protocol
* Add string generation/parsing system to generate tokens for Fluid Protocol
* Set authentication keys for token generation to statically-defined addresses
* Update CBlockIndex and CChain models for storing Fluid Protocol derived variables
* Allow opcodes to carry token instruction and to detect tokens
* Implement derivation of token data into datasets
* Derive parameters (One-Time Reward, Dynode & PoW Reward) from datasets
* Implement token-history indexing and prevent replay attacks
* Change statically-defined addresses to identity-derived addresses (dynamic)
* Introduce RPC Calls maketoken, getrawpubkey, burndynamic, sendfluidtransaction, signtoken, consenttoken, verifyquorum, fluidcommandshistory, getfluidmasters

**Dynamic v1.4.0.0**

* Securely erase potentially sensitive keys/values
* Fix issue where config was created at launch but not read
* [BUILD] quiet annoying warnings without adding new ones
* [BUILD]Fix warning deprecated in OSX Sierra
* Improve EncodeBase58/DecodeBase58 performance.
* Use hardware timestamps in RNG seeding
* Add OSX keystroke to clear RPCConsole
* Update DB_CORRUPT message
* HD Wallet
* Repair Traffic Graph
* Scammer Warning and Translations
* Amend DEFAULT_CHECKBLOCKS
* Do not shadow upper local variable 'send', prevent -Wshadow compiler warning
* Convert last boost::scoped_ptr to std::unique_ptr
* Qt: fix UI bug that could result in paying unexpected fee
* Fix Locks and Do not add random inbound peers to addrman
* Use std::thread::hardwarencurrency, instead of Boost, to determine available cores
* Sync icon now opens modaloverlay.ui
* Fix Memleak and Enforce Fix
* Sort dynamic.qrc
* Sort MakeFiles
* Remove namespace std;/Repair Tests
* Fix Signal/Slot/Strings
* Implement modaloverlay
* Qt: Sort transactions by date
* Kill insecure_random and associated global state
* Only send one GetAddr response per connection.
* Refactor: Removed begin/end_ptr functions.
* LevelDB 1.19
* Increase context.threads to 4
* Fix races
* Fix calculation of number of bound sockets to use
* Fix unlocked access to vNodes.size()
* Move GetAccountBalance from rpcwallet.cpp into CWallet::GetAccountBalance
* UpdateTip: log only one line at most per block
* VerifyDB: don't check blocks that have been pruned 
* qt: askpassphrasedialog: Clear pass fields on accept
* net: Avoid duplicate getheaders requests.
* Check non-null pindex before potentially referencing
* mapNextTx: use pointer as key, simplify value
* Implement indirectmap.h and update memusage.h
* Add/Repair LOCK's
* Fix parameter naming inconsistencies 
* Clicking on the lock icon will open the passphrase dialog
* Fix bip32_tests.cpp
* Update Argon2d, hash.cpp/h
* Repair SLOT issue in rpcconsole.cpp
* Fix incorrect psTx usages
* Fix torcontrol.cpp unused private field warning
* Update Encryption(crypter.cpp/h)
* Remove old HD wallet code
* Move InitLoadWallet to init.cpp
* Revert Tick Changes/Fix UI Thread Issue
* Sentinel/Dynode Fixes
* Remove unused functions/cleanup code
* Reduce Keypool to 1000
* Optimise Reindex
* Bump Governance/InstantSend/PrivateSend/Core Proto/Versions


**Dynamic v1.3.0.2**
* [Sync] Fix issue with headers first sync
* [Sync] [Consensus] Shift Fork Logic to its own file
* [Qt] Add CheckForks in the Qt Project File
* [Fork] Silence usage of pindex compeletely
* [Sync]Timeouts/DB/Headers/Limits
* Reduce nDefaultDbCache to 512MiB
* Bump Proto and ONLY connect to 1.3.0.1 (Proto 70200)
* Bump Governance/Core Proto/Versions


**Dynamic v1.3.0.1**
* Bump Protocols to lock out nodes at or below v1.2 to prevent any forks


**Dynamic v1.3.0.0**	
* c++11:Backport from bitcoin-core: don't throw from the reverselock destructor
* InitError instead of throw on failure
* Hard Fork at block 300,000 for Delta difficulty retarget algorithm


**Dynamic v1.2.0.0**	
* Make RelayWalletTransaction attempt to AcceptToMemoryPool
* Update tests for Byteswap
* Ensure is in valid range
* Make strWalletFile const
* Avoid ugly exception in log on unknown inv type
* libconsensus: Add input validation of flags/missing flags & make catch() args const
* Add LockedPool
* Add getmemoryinfo
* Add benchmark for lockedpool allocation/deallocation
* trivial: fix bloom filter init to isEmpty = true
* Lockedpool fixes
* Add include utility to miner.cpp
* Don't return the address of a P2SH of a P2SH
* Implement (begin|end)_ptr in C++11 and add deprecation comment
* add include stdlib.h to random.cpp
* Generate auth cookie in hex instead of base64
* Do not shadow variable (multiple files)
* dynamic-cli: More detailed error reporting
* Add High TX Fee Warning
* C++11: s/boost::scoped_ptr/std::unique_ptr/
* Do not shadow variables in networking code
* Remove shadowed variables in Qt files
* Do not shadow LOCK's criticalblock variable for LOCK inside LOCK
* Do not shadow variables in src/wallet
* Berkeley DB v6 compatibility fix
* Reduce default number of blocks to check at startup
* Fix random segfault when closing Choose data directory dialog
* Fix several node initialization issues
* Add FEATURE_HD
* Improve handling of unconnecting headers
* Fix DoS vulnerability in mempool acceptance
* Bump default db cache to 300MiB
* Fix a bug where the SplashScreen will not be hidden during startup
* Stop trimming when mapTx is empty
* Evict orphans which are included or precluded by accepted blocks
* Reduce unnecessary hashing in signrawtransaction
* Watchdog check removed until Sentinel is updated/compatible fully
* Bump protocol versions to 70000
* Added IPv4 seed nodes to chainparamsseeds.h


**Dynamic v1.1.0.0**			
* Inline with BTC 0.12		
* HD Wallet Code Improvements		
* Remove/Replace Boost usage for c++11		
* Do not shadow member variables in httpserver		
* Update dbwrapper_tests.cpp		
* Access WorkQueue::running only within the cs lock		
* Use BIP32_HARDENED_KEY_LIMIT		
* Update NULL to use nullptr in GetWork & GetBlockTemplate		
* Few changes to governance rpc		
* Safety check in CInstantSend::SyncTransaction		
* Full path in 'failed to load cache' warnings		
* Refactor privateSendSigner		
* Net Fixes/DNS Seed Fix		
* Don't add non-current watchdogs to seen map		
* [RPC] remove the option of having multiple timer interfaces		
* Fix memory leak in httprpc.cpp		
* Make KEY_SIZE a compile-time constant
