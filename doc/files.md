Used in 1.4.0.0
---------------------
* wallet.dat: personal wallet (BDB) with keys and transactions
* peers.dat: peer IP address database (custom format);
* blocks/blk000??.dat: block data (custom, 128 MiB per file);
* blocks/rev000??.dat; block undo data (custom);
* blocks/index/*; block index (LevelDB);
* chainstate/*; block chain state database (LevelDB);
* database/*: BDB database environment;
