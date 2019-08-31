# BDAP - Blockchain Directory Access Protocol

## Abstract

[BDAP (Blockchain Directory Access Protocol)](https://duality.solutions/bdap) gives programmable access control and direct communication with users on the network, adding a layer of resource hierarchy and providing a distributed database with account linking, making it possible to develop core information systems using the blockchain technology of [Dynamic](https://github.com/duality-solutions/dynamic) and [Sequence](https://github.com/duality-solutions/sequence). This design allows user controlled nodes to securely connect, privately share data without a third-party intermediary, and to scale the database up indefinitely.

BDAP enables the creation of applications across the spectrum of industry, drastically reducing the costs involved with core information systems. Removing the requirement for the majority of administration roles and the need for trusted third parties. Rendering current centralized database systems such as [LDAP (Lightweight Directory Access Protocol)](https://en.wikipedia.org/wiki/Lightweight_Directory_Access_Protocol) a thing of the past.

## Technical Information

[BDAP](https://duality.solutions/bdap) is used to create and amend entries on a [libtorrent](https://github.com/arvidn/libtorrent) based DHT (Distributed Hash Table), and utilizes the blockchain of [Dynamic](https://github.com/duality-solutions/dynamic) to provide decentralization and security. 

### BDAP Development Kit (BDK)

Developers can build their own BDAP based dApps (decentralised applications) by using the BDAP Development Kit (BDK).

### BDAP Domain Entry Database

Needs content.

### BDAP Data Auditing

Needs content.

### BDAP Entry Creation

Needs content.

### BDAP Entry Updating

Needs content.

### BDAP Entry Linking 

Entry linking is a type of DAP binding operation and is used to manage domain entry link requests. When linking entries, we use stealth addresses so the linkage requests remain private. Link requests (class CLinkRequest) are stored, serialized and encrypted in a [BDAP](https://duality.solutions/bdap) ```OP_RETURN``` transaction. The link request recipient can decrypt the [BDAP](https://duality.solutions/bdap) ```OP_RETURN``` transaction and get the needed information (class CLinkAccept) to accept the link request. It is used to bootstrap the linkage relationship with a new set of public keys.

##### class CLinkRequest

```
CharString RequestorFullObjectPath; // Requestor's BDAP object path

CharString RecipientFullObjectPath; // Recipient's BDAP object path

CharString RequestorPubKey; // ed25519 public key new/unique for this link

CharString SharedPubKey; // ed25519 shared public key(RequestorPubKey + Recipient's BDAP DHT PubKey)

CharString LinkMessage // Link message to recipient

CharString SignatureProof; // Requestor's BDAP account ownership proof by signing the recipient's object path with their wallet public key
```

##### class CLinkAccept

```
CharString RequestorFullObjectPath; // Requestor's BDAP object path

CharString RecipientFullObjectPath; // Recipient's BDAP object path
    
uint256 txLinkRequestHash; // transaction hash for the link request.

CharString RecipientPubKey; // ed25519 public key new/unique for this link

CharString SharedPubKey; // ed25519 shared public key using the requestor and recipient keys

CharString SignatureProof; // Acceptor's BDAP account ownership proof by signing the requestor's object path with their wallet public key.
```

### BDAP Sidechaining

Needs content.

### BDAP RPC Calls

* **adduser** - adds a public name entry to blockchain directory:```adduser "userid" "common name" "registration days"```

* **getusers** - list all BDAP public users:```getusers "records per page" "page returned"```

* **getgroups** - list all BDAP public groups:```getgroups "records per page" "page returned"```

* **getuserinfo** - list BDAP entry:```getuserinfo "public name"```

* **updateuser** - update an existing public name blockchain directory entry:```updateuser "userid" "common name" "registration days"```

* **updategroup** - update an existing public group blockchain directory entry:```updategroup "groupid" "common name" "registration days"```

* **deleteuser** - delete an existing public name blockchain directory entry:```deleteuser "userid"```

* **deletegroup** - delete an existing public name blockchain directory entry:```deletegroup "groupid"```

* **addgroup** - add public group entry to blockchain directory:```addgroup "groupid" "common name"```

* **getgroupinfo** - list BDAP entry:```getgroupinfo "groupid"```

* **mybdapaccounts** - returns your BDAP accounts: ```mybdapaccounts```

* **link** - link commands are request, accept, pending, complete, and delete:```link "operation" "common name" "registration days"```

### DHT RPC Calls

* **getmutable** - gets mutable data from the DHT:```getmutable "pubkey" "operation"```

* **putmutable** - saves mutable data in the DHT:```putmutable "dht value" "operation" "pubkey" "privkey"```

* **dhtinfo** - gets DHT network stats and info:```dhtinfo```

* **dhtdb** - gets the local DHT cache database contents:```dhtdb```

* **putbdapdata** - saves mutable data in the DHT for a BDAP entry:```putbdapdata "bdap id" "dht value" "operation"```

* **getbdapdata** - gets the mutable data from the DHT for a BDAP entry:```getbdapdata "bdap id" "operation"```

* **dhtputmessages** - gets all DHT put messages in memory:```dhtputmessages```

* **dhtgetmessages** - gets all DHT get messages in memory:```dhtgetmessages```

### BDAP Code

All of the code for [BDAP](https://duality.solutions/bdap) can be found in the [/src/bdap/](https://github.com/duality-solutions/Dynamic/tree/master/src/bdap) directory of [Dynamic](https://github.com/duality-solutions/dynamic).

### DHT Code

All of the code for the DHT can be found in the [/src/dht/](https://github.com/duality-solutions/Dynamic/tree/master/src/dht) directory of [Dynamic](https://github.com/duality-solutions/dynamic).
