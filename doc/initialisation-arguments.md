![DYN logo](https://github.com/duality-solutions/Logos/blob/master/Duality%20Logos/Dynamic/PNG/128x128.png)

DYNAMIC INITIALISATION ARGUMENTS
================================


OPTIONS
-------
* "-version" ("Print version and exit")
* "-alerts" ("Receive and display P2P network alerts (default: true)")
* "-alertnotify=[cmd]" ("Execute command when a relevant alert is received or we see a really long fork")
* "-blocknotify=[cmd]" ("Execute command when the best block changes")
* "-blocksonly" ("Whether to operate in a blocks only mode (default: false)")
* "-checkblocks=[n]" ("How many blocks to check at startup (default: 0, 0 = all)")
* "-checklevel=[n]" ("How thorough the block verification of -checkblocks is (0-4)")
* "-conf=[file]" ("Specify configuration file")
* "-daemon" ("Run in the background as a daemon and accept commands")
* "-datadir=[dir]" ("Specify data directory")
* "-dbcache=[n]" ("Set database cache size in megabytes")
* "-loadblock=[file]" ("Imports blocks from external blk000??.dat file on startup")
* "-maxorphantx=[n]" ("Keep at most [n] unconnectable transactions in memory")
* "-maxmempool=[n]" ("Keep the transaction memory pool below [n] megabytes")
* "-mempoolexpiry=[n]" ("Do not keep transactions in the mempool longer than [n] hours")
* "-par=[n]" ("Set the number of script verification threads (0 = auto)")
* "-pid=[file]" ("Specify pid file")
* "-prune=[n]" ("Reduce storage requirements by pruning (deleting) old blocks. This mode is incompatible with -txindex and -rescan. "
            "Warning: Reverting this setting requires re-downloading the entire blockchain. "
            "(default: 0 = disable pruning blocks")
* "-reindex-chainstate" ("Rebuild chain state from the currently indexed blocks")
* "-reindex" ("Rebuild chain state and block index from the blk*.dat files on disk")
* "-sysperms" ("Create new files with system default permissions, instead of umask 077 (only effective with disabled wallet functionality)")
* "-txindex" ("Maintain a full transaction index, used by the getrawtransaction rpc call")
* "-addressindex" ("Maintain a full address index, used to query for the balance, txids and unspent outputs for addresses")
* "-timestampindex" ("Maintain a timestamp index for block hashes, used to query blocks hashes by a range of timestamps")
* "-spentindex" ("Maintain a full spent index, used to query the spending txid and input index for an outpoint")

CONNECTION OPTIONS
-----------------
* "-addnode=[ip]" ("Add a node to connect to and attempt to keep the connection open")
* "-banscore=[n]" ("Threshold for disconnecting misbehaving peers")
* "-bantime=[n]" ("Number of seconds to keep misbehaving peers from reconnecting")
* "-bind=[addr]" ("Bind to given address and always listen on it. Use [host]:port notation for IPv6")
* "-connect=[ip]" ("Connect only to the specified node(s)")
* "-discover" ("Discover own IP addresses (default: 1 when listening and no -externalip or -proxy)")
* "-dns" ("Allow DNS lookups for -addnode, -seednode and -connect")
* "-dnsseed" ("Query for peer addresses via DNS lookup, if low on addresses (default: 1 unless -connect)")
* "-externalip=[ip]" ("Specify your own public address")
* "-forcednsseed" ("Always query for peer addresses via DNS lookup")
* "-listen" ("Accept connections from outside (default: 1 if no -proxy or -connect)")
* "-listenonion" ("Automatically create Tor hidden service")
* "-maxconnections=[n]" ("Maintain at most [n] connections to peers (temporary service connections excluded)")
* "-maxreceivebuffer=[n]" ("Maximum per-connection receive buffer, [n]*1000 bytes")
* "-maxsendbuffer=[n]" ("Maximum per-connection send buffer, [n]*1000 bytes")
* "-onion=[ip:port]" ("Use separate SOCKS5 proxy to reach peers via Tor hidden services")
* "-onlynet=[net]" ("Only connect to nodes in network [net] (ipv4, ipv6 or onion)")
* "-permitbaremultisig" ("Relay non-P2SH multisig")
* "-peerbloomfilters" ("Support filtering of blocks and transaction with bloom filters")
* "-enforcenodebloom" ("Enforce minimum protocol version to limit use of bloom filters")
* "-port=[port]" ("Listen for connections on [port]")
* "-proxy=[ip:port]" ("Connect through SOCKS5 proxy")
* "-proxyrandomize" ("Randomize credentials for every proxy connection. This enables Tor stream isolation")
* "-seednode=[ip]" ("Connect to a node to retrieve peer addresses, and disconnect")
* "-timeout=[n]" ("Specify connection timeout in milliseconds (minimum: 1)")
* "-torcontrol=[ip]:[port]" ("Tor control port to use if onion listening enabled")
* "-torpassword=[pass]" ("Tor control port password (default: empty)")
* "-upnp" ("Use UPnP to map the listening port (default: 1 when listening and no -proxy)")
* "-whitebind=[addr]" ("Bind to given address and whitelist peers connecting to it. Use [host]:port notation for IPv6")
* "-whitelist=[netmask]" ("Whitelist peers connecting from the given netmask or IP address. Can be specified multiple times.") +
        " " + ("Whitelisted peers cannot be DoS banned and their transactions are always relayed, even if they are already in the mempool, useful e.g. for a gateway")
* "-whitelistrelay" ("Accept relayed transactions received from whitelisted peers even when not relaying transactions")
* "-whitelistforcerelay" ("Force relay of transactions from whitelisted peers even they violate local relay policy")
* "-maxuploadtarget=[n]" ("Tries to keep outbound traffic under the given target (in MiB per 24h), 0 = no limit")


WALLET OPTIONS
--------------
* "-disablewallet"("Do not load the wallet and disable wallet RPC calls")
* "-keypool=[n]" ("Set key pool size to [n]")
* "-fallbackfee=[amt]" ("A fee rate (in kB) that will be used when fee estimation has insufficient data"),
* "-mintxfee=[amt]" ("Fees (in kB) smaller than this are considered zero fee for transaction creation"),
* "-paytxfee=[amt]" ("Fee (in kB) to add to transactions you send"),
* "-rescan" ("Rescan the block chain for missing wallet transactions on startup")
* "-salvagewallet" ("Attempt to recover private keys from a corrupt wallet.dat on startup")
* "-spendzeroconfchange" ("Spend unconfirmed change when sending transactions")
* "-txconfirmtarget=[n]" ("If paytxfee is not set, include enough fee so transactions begin confirmation on average within n blocks")
* "-maxtxfee=[amt]" ("Maximum total fees to use in a single wallet transaction setting this too low may abort large transactions"),
* "-usehd" ("Use hierarchical deterministic key generation (HD) after bip32. Only has effect during wallet creation/first start")
* "-mnemonic" ("User defined mnemonic for HD wallet (bip39). Only has effect during wallet creation/first start (default: randomly generated)")
* "-mnemonicpassphrase" ("User defined memonic passphrase for HD wallet (bip39). Only has effect during wallet creation/first start (default: randomly generated)")
* "-hdseed" ("User defined seed for HD wallet (should be in hex). Only has effect during wallet creation/first start (default: randomly generated)")
* "-upgradewallet" ("Upgrade wallet to latest format on startup")
* "-wallet=[file]" ("Specify wallet file (within data directory)")
* "-walletbroadcast" ("Make the wallet broadcast transactions")
* "-walletnotify=[cmd]" ("Execute command when a wallet transaction changes (string in cmd is replaced by TxID)")
* "-zapwallettxes=[mode]" ("Delete all wallet transactions and only recover those parts of the blockchain through -rescan on startup") +
        " " + ("(1 = keep tx meta data e.g. account owner and payment request information, 2 = drop tx meta data)")
* "-createwalletbackups=[n]" ("Number of automatic wallet backups")
* "-walletbackupsdir=[dir]" ("Specify full path to directory for automatic wallet backups (must exist)")
* "-keepass" ("Use KeePass 2 integration using KeePassHttp plugin")
* "-keepassport=[port]" ("Connect to KeePassHttp on port [port]")
* "-keepasskey=[key]" ("KeePassHttp key for AES encrypted communication with KeePass")
* "-keepassid=[name]" ("KeePassHttp id for the established association")
* "-keepassname=[name]" ("Name to construct url for KeePass entry that stores the wallet passphrase")
* "-windowtitle=[name]" ("Wallet window title")


ZEROMQ NOTIFICATION OPTIONS
---------------------------
* "-zmqpubhashblock=[address]" ("Enable publish hash block in [address]")
* "-zmqpubhashtx=[address]" ("Enable publish hash transaction in [address]")
* "-zmqpubhashtxlock=[address]" ("Enable publish hash transaction (locked via InstantSend) in [address]")
* "-zmqpubrawblock=[address]" ("Enable publish raw block in [address]")
* "-zmqpubrawtx=[address]" ("Enable publish raw transaction in [address]")
* "-zmqpubrawtxlock=[address]" ("Enable publish raw transaction (locked via InstantSend) in [address]")


DEBUGGING TESTING OPTIONS
-------------------------
* "-uacomment=[cmt]" ("Append comment to the user agent string")
* "-checkblockindex" ("Do a full consistency check for mapBlockIndex, setBlockIndexCandidates, chainActive and mapBlocksUnlinked occasionally. Also sets -checkmempool")
* * "-checkmempool=[n]" ("Run checks every [n] transactions")
* "-checkpoints" ("Disable expensive verification for known chain history")
* "-dblogsize=[n]" ("Flush wallet database activity from memory to disk log every [n] megabytes")
* "-disablesafemode" ("Disable safemode, override a real safe mode event")
* "-testsafemode" ("Force safe mode")
* "-dropmessagestest=[n]" ("Randomly drop 1 of every [n] network messages")
* "-fuzzmessagestest=[n]" ("Randomly fuzz 1 of every [n] network messages")
* "-flushwallet" ("Run a thread to flush wallet periodically")
* "-stopafterblockimport" ("Stop running after importing blocks from disk")
* "-limitancestorcount=[n]" ("Do not accept transactions if number of in-mempool ancestors is [n] or more")
* "-limitancestorsize=[n]" ("Do not accept transactions whose size with all in-mempool ancestors exceeds [n] kilobytes")
* "-limitdescendantcount=[n]" ("Do not accept transactions if any ancestor would have [n] or more in-mempool descendants")
* "-limitdescendantsize=[n]" ("Do not accept transactions if any ancestor would have more than [n] kilobytes of in-mempool descendants.")
    
debug Categories are "addrman, alert, bench, coindb, db, http, libevent, lock, mempool, mempoolrej, net, proxy, prune, qt, rand, reindex, rpc, selectcoins, tor, zmq, Dynamic (or specifically: privatesend, instantsend, dynode, spork, keepass, dnpayments, gobject)"
    
* "-debug=[category]" ("Output debugging information (supplying [category] is optional)") +
        ("If [category] is not supplied or if [category] = 1, output all debugging information.") + ("[category] can be:")
* "-nodebug" ("Turn off debugging messages, same as -debug=0")
* "-gen" ("Generate coins")
* "-genproclimit-cpu=[n]" ("Set the number of threads for coin generation if enabled (-1 = all cores)")
* "-help-debug" ("Show all debugging options (usage: --help -help-debug)")
* "-logips" ("Include IP addresses in debug output")
* "-logtimestamps" ("Prepend debug output with timestamp")
* "-logtimemicros" ("Add microsecond precision to debug timestamps")
* "-logthreadnames" ("Add thread names to debug messages")
* "-mocktime=[n]", ("Replace actual time with [n] seconds since epoch")
* "-limitfreerelay=[n]" ("Continuously rate-limit free transactions to [n]*1000 bytes per minute")
* "-relaypriority" ("Require high priority for relaying free or low-fee transactions")
* "-maxsigcachesize=[n]" ("Limit size of signature cache to [n] MiB")
* "-minrelaytxfee=[amt]" ("Fees (in kB) smaller than this are considered zero fee for relaying, mining and transaction creation"),
* "-printtoconsole" ("Send trace/debug info to console instead of debug.log file")
* "-printtodebuglog" ("Send trace/debug info to debug.log file")
* "-printpriority" ("Log transaction priority and fee per kB when mining blocks")
* "-privdb" ("Sets the DB_PRIVATE flag in the wallet db environment")
* "-shrinkdebugfile" ("Shrink debug.log file on client startup (default: 1 when no -debug)")
* "-litemode=[n]" ("Disable all Dynamic specific functionality (Dynodes, PrivateSend, InstantSend, Governance) (0-1)")

DYNODE OPTIONS
--------------
* "-dynode=[n]" ("Enable the client to act as a Dynode (0-1)")
* "-dnconf=[file]" ("Specify Dynode configuration file", "dynode.conf")
* "-dnconflock=[n]" ("Lock Dynodes from Dynode configuration file")
* "-dynodepairingkey=[n]" ("Set the Dynode private key")

PRIVATESEND OPTIONS
-------------------
* "-enableprivatesend=[n]" ("Enable use of automated PrivateSend for funds stored in this wallet (0-1)")
* "-privatesendmultisession=[n]" ("Enable multiple PrivateSend mixing sessions per block, experimental (0-1)")
* "-privatesendrounds=[n]" ("Use N separate Dynodes for each denominated input to mix funds (2-16)")
* "-privatesendamount=[n]" ("Keep N DYN anonymized")
* "-liquidityprovider=[n]" ("Provide liquidity to PrivateSend by infrequently mixing coins on a continual basis (0-100, 1=very frequent, high fees, 100=very infrequent, low fees)")

INSTANTSEND OPTIONS
-------------------
* "-enableinstantsend=[n]" ("Enable InstantSend, show confirmations for locked transactions (0-1)")
* "-instantsenddepth=[n]" ("Show N confirmations for a successfully locked transaction (0-9999)")
* "-instantsendnotify=[cmd]" ("Execute command when a wallet InstantSend transaction is successfully locked")


NODE RELAY OPTIONS
------------------
* "-acceptnonstdtxn" ("Relay and mine \"non-standard\" transactions", "testnet/regtest only ")
* "-bytespersigop" ("Minimum bytes per sigop in transactions we relay and mine")
* "-datacarrier" ("Relay and mine data carrier transactions")
* "-datacarriersize" ("Maximum size of data in data carrier transactions we relay and mine")
* "-mempoolreplacement" ("Enable transaction replacement in the memory pool")

BLOCK CREATION OPTIONS
----------------------
* "-blockminsize=[n]" ("Set minimum block size in bytes")
* "-blockmaxsize=[n]" ("Set maximum block size in bytes")
* "-blockprioritysize=[n]" ("Set maximum size of high-priority/low-fee transactions in bytes")
* "-blockversion=[n]" ("Override block version to test forking scenarios")

RPC SERVER OPTIONS
------------------
* "-server" ("Accept command line and JSON-RPC commands")
* "-rest" ("Accept public REST requests")
* "-rpcbind=[addr]" ("Bind to given address to listen for JSON-RPC connections. Use [host]:port notation for IPv6. This option can be specified multiple times (default: bind to all interfaces)")
* "-rpccookiefile=[loc]" ("Location of the auth cookie (default: data dir)")
* "-rpcuser=[user]" ("Username for JSON-RPC connections")
* "-rpcpassword=[pw]" ("Password for JSON-RPC connections")
* "-rpcauth=[userpw]" ("Username and hashed password for JSON-RPC connections. The field [userpw] comes in the format: [USERNAME]:[SALT]$[HASH]. A canonical python script is included in share/rpcuser. This option can be specified multiple times")
* "-rpcport=[port]" ("Listen for JSON-RPC connections on [port]")
* "-rpcallowip=[ip]" ("Allow JSON-RPC connections from specified source. Valid for [ip] are a single IP (e.g. 1.2.3.4), a network/netmask (e.g. 1.2.3.4/255.255.255.0) or a network/CIDR (e.g. 1.2.3.4/24). This option can be specified multiple times")
* "-rpcthreads=[n]" ("Set the number of threads to service RPC calls")
* "-rpcworkqueue=[n]" ("Set the depth of the work queue to service RPC calls")
* "-rpcservertimeout=[n]" ("Timeout during HTTP requests")
    
