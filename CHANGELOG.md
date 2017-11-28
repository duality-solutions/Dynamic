**Dynamic v1.5.0.0**
* Fix Network Time Protocol (NTP)
* Update secp256k1
* Remove block 300,000 fork data
* New Hash Settings
* Amend CPU Core Count
* Revert/Update and Strip Argon2d code
* Update LevelDB to 1.20
* Add Dynode checks to prevent payments until 500 are active
* Reduce nPowTargetTimespan to 1920 seconds
* Reduce nMinerConfirmationWindow to 30 blocks
* [Qt] Reduce a significant cs_main lock freeze 
* remove InstantSend votes for failed lock attemts after some timeout
* Fix dnp relay bug
* fix trafficgraphdatatests for qt4
* Fix edge case for IS (skip inputs that are too large)
* allow up to 40 chars in proposal name
* Multiple Fixes/Implement connman broadly
* Add more logging for MN votes and MNs missing votes
* Remove bogus assert on number of oubound connections.
* update nCollateralMinConfBlockHash for local (hot) dynode on dn start
* Fix sync reset on lack of activity
* fix nLastWatchdogVoteTime updates
* Fix bug: nCachedBlockHeight was not updated on start
* Fix compilation with qt < 5.2
* RPC help formatting updates
* Relay govobj and govvote to every compatible peer, not only to the one with the same version
* remove send addresses from listreceivedbyaddress output
* Remove cs_main from ThreadMnbRequestConnections
* do not calculate stuff that are not going to be visible in simple PSUI anyway & fix fSkipUnconfirmed
* Keep track of wallet UTXOs and use them for PS balances and rounds calculations
* speedup MakeCollateralAmounts by skiping denominated inputs early
* Reduce min relay tx fee
* more vin -> outpoint in dynode rpc output
* Move some (spamy) CDynodeSync log messages to new log category
* Eliminate g_connman use in InstantSend module.
* Remove some recursive locks
* Fix dynode score/rank calculations (#1620)
* InstandSend overhaul & TXMempool Fixes
* fix TrafficGraphData bandwidth calculation
* Fix losing keys on PrivateSend
* Refactor masternode management
* Multiple Selection for peer and ban tables
* qt: Fixing division by zero in time remaining
* [qt] sync-overlay: Don't show progress twice
* qt: Plug many memory leaks
* [GUI] Backport Bitcoin Qt/Gui changes up to 0.14.x
* Fix Unlocked Access to vNodes
* Fix Sync
* Fix empty tooltip during sync under specific conditions
* fix SPORK_5_INSTANTSEND_MAX_VALUE validation in CWallet::CreateTransaction
* Eliminate g_connman use in spork module. 
* Use connman passed to ThreadSendAlert() instead of g_connman global.
* Fix duplicate headers download in initial sync
* fix off-by-1 in CSuperblock::GetPaymentsLimit
* fix number of blocks to wait after successful mixing tx
* Backport Bitcoin PR#7868: net: Split DNS resolving functionality out of net structures
* net: require lookup functions to specify all arguments to make it clear where DNS resolves are happening
* net: manually resolve dns seed sources
* net: resolve outside of storage structures
* net: disable resolving from storage structures
* net: No longer send local address in addrMe
* safe version of GetDynodeByRank
* Do not add random inbound peers to addrman.
* Partially backport Bitcoin PR#9626: Clean up a few CConnman cs_vNodes/CNode things
* Delete some unused (and broken) functions in CConnman
* Ensure cs_vNodes is held when using the return value from FindNode
* Use GetAdjustedTime instead of GetTime when dealing with network-wide timestamps
* slightly refactor CPSNotificationInterface
* drop masternode index
* drop pCurrentBlockIndex and use cached block height instead (nCachedBlockHeight)
* add/use GetUTXO[Coins/Confirmations] helpers instead of GetInputAge[InstantSend]
* net: Consistently use GetTimeMicros() for inactivity checks 
* Fix DynodeRateCheck
* Always good to initialise
* Necessary split of main.h to validation.cpp/net_processing.cpp
* Relay tx in sendrawtransaction according to its inv.type
* Fix : Reject invalid instantsend transaction
* fix instantsendtoaddress param convertion
* Fix potential deadlock in CInstantSend::UpdateLockedTransaction (#1571) 
* limit UpdatedBlockTip in IBD
* Pass reference when calling HasPayeeWithVotes
* Sync overhaul
* Make sure mixing messages are relayed/accepted properly
* backport 9008: Remove assert(nMaxInbound > 0)
* Backport Bitcoin PR#8049: Expose information on whether transaction relay is enabled in (#1545)
* fix potential deadlock in CDynodeMan::CheckDnbAndUpdateDynodeList
* fix potential deadlock in CGovernanceManager::ProcessVote
* add 6 to strAllowedChars
* Backport Bitcoin PR#8085: p2p: Begin encapsulation
* change invalid version string constant
* Added feeler connections increasing good addrs in the tried table.
* Backport Bitcoin PR#8113: Rework addnode behaviour (#1525) 
* Fix vulnerability with mapDynodeOrphanObjects
* Remove bad chain alert partition check
* Fix potential deadlocks in InstantSend
* fix CDSNotificationInterface::UpdatedBlockTip signature to match the one in CValidationInterface
* fix a bug in CommitFinalTransaction
* fixed potential deadlock in CSuperblockManager::IsSuperblockTriggered
* Fix issues with mapSeenGovernanceObjects
* Backport Bitcoin PR#8084: Add recently accepted blocks and txn to AttemptToEvictConnection
* Backport Bitcoin PR#7906: net: prerequisites for p2p encapsulation changes
* fix race that could fail to persist a ban
* Remove non-determinism which is breaking net_tests
* Implement BIP69 outside of CTxIn/CTxOut 
* fix MakeCollateralAmounts
* Removal of Unused Files and CleanUp
* Further fixes to PrivateSend
* New rpc call 'dynodelist info'
* Backport Bitcoin PR#7749: Enforce expected outbound services
* Backport Bitcoin PR#7696: Fix de-serialization bug where AddrMan is corrupted after exception
* Fixed issues with propagation of governance objects and update governance
* Backport Bitcoin PR#7458 : [Net] peers.dat, banlist.dat recreated when missing
* Backport Bitcoin PR#7350: Banlist updates
* Replace watchdogs with ping 
* Update timedata.h
* Trivial Fixes
* Eliminate unnecessary call to CheckBlock 
* PrivateSend: dont waste keys from keypool on failure in CreateDenominated
* Refactor PS and fix minor build issues preventing Travis-CI from completing previously
* Fix Governance Test File
* Increase test coverage for addrman and addrinfo
* Backport Bitcoin PRs #6589, #7180 and remaining part of #7181
* Don't try to create empty datadir before the real path is known
* Documentation: Add spork message / details to protocol-documentation
* Validate proposals on prepare and submit

**Dynamic v1.4.0.0
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
