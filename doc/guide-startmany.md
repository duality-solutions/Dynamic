#start-many Setup Guide

## Two Options for Setting up your Wallet
There are many ways to setup a wallet to support start-many. This guide will walk through two of them.

1. [Importing an existing wallet (recommended if you are consolidating wallets).](#option1)
2. [Sending 1,000 DYN to new wallet addresses.](#option2)

## <a name="option1"></a>Option 1. Importing an existing wallet

This is the way to go if you are consolidating multiple wallets into one that supports start-many. 

### From your single-instance Dynode Wallet

Open your QT Wallet and go to console (from the menu select Tools => Debug Console)

Dump the private key from your Dynode's pulic key.

```
walletpassphrase [your_wallet_passphrase] 600
dumpprivkey [sn_public_key]
```

Copy the resulting priviate key. You'll use it in the next step.

### From your multi-instance Dynode Wallet

Open your QT Wallet and go to console (from the menu select Tools => Debug Console)

Import the private key from the step above.

```
walletpassphrase [your_wallet_passphrase] 600
importprivkey [single_instance_private_key]
```

The wallet will re-scan and you will see your available balance increase by the amount that was in the imported wallet.

[Skip Option 2. and go to Create dynode.conf file](#dynodeconf)

## <a name="option2"></a>Option 2. Starting with a new wallet

[If you used Option 1 above, then you can skip down to Create dynode.conf file.](#dynodeconf)

### Create New Wallet Addresses

1. Open the QT Wallet.
2. Click the Receive tab.
3. Fill in the form to request a payment.
    * Label: sn01
    * Amount: 1000 (optional)
    * Click *Request payment*
5. Click the *Copy Address* button

Create a new wallet address for each Dynode.

Close your QT Wallet.

### Send 1,000 DYN to New Addresses

Just like setting up a standard SN. Send exactly 1,000 DYN to each new address created above.

### Create New Dynode Private Keys

Open your QT Wallet and go to console (from the menu select Tools => Debug Console)

Issue the following:

```dynode genkey```

*Note: A Dynode private key will need to be created for each Dynode you run. You should not use the same Dynode private key for multiple Dynodes.*

Close your QT Wallet.

## <a name="dynodeconf"></a>Create dynode.conf file

Remember... this is local. Make sure your QT is not running.

Create the dynode.conf file in the same directory as your wallet.dat.

Copy the Dynode private key and correspondig collateral output transaction that holds the 1K DYNAMIC.

The Dynode private key may be an existing key from [Option 1](#option1), or a newly generated key from [Option 2](#option2). 

*Please note, the Dynode priviate key is not the same as a wallet private key. Never put your wallet private key in the dynode.conf file. That is equivalent to putting your 1,000 DYN on the remote server and defeats the purpose of a hot/cold setup.*

### Get the collateral output

Open your QT Wallet and go to console (from the menu select Tools => Debug Console)

Issue the following:

```dynode outputs```

Make note of the hash (which is your collaterla_output) and index.

### Enter your Dynode details into your dynode.conf file
[From the dynamic github repo](https://github.com/duality-solutions/dynamic/blob/master/doc/dynode_conf.md)

The new dynode.conf format consists of a space seperated text file. Each line consisting of an alias, IP address followed by port, Dynode private key, collateral output transaction id and collateral output index. 
(!!! Currently not implemented: "donation address and donation percentage (the latter two are optional and should be in format "address:percentage")." !!!)

```
alias ipaddress:port dynode_private_key collateral_output collateral_output_index (!!! see above "donationin_address:donation_percentage" !!!)
```



Example:

```
sn01 127.0.0.1:31300 93HaYBVUCYjEMeeH1Y4sBGLALQZE1Yc1K64xiqgX37tGBDQL8Xg 2bcd3c84c84f87eaa86e4e56834c92927a07f9e18718810b92e0d0324456a67c 0
sn02 127.0.0.2:31300 93WaAb3htPJEV8E9aQcN23Jt97bPex7YvWfgMDTUdWJvzmrMqey aa9f1034d973377a5e733272c3d0eced1de22555ad45d6b24abadff8087948d4 0 (!!! see above "7gnwGHt17heGpG9Crfeh4KGpYNFugPhJdh:25" !!!)
```

## What about the dynamic.conf file?

If you are using a dynode.conf file you no longer need the dynamic.conf file. The exception is if you need custom settings (thanks oblox). 

## Update dynamic.conf on server

If you generated a new Dynode private key, you will need to update the remote dynamic.conf files.

Shut down the daemon and then edit the file.

```sudo nano .dynamic/dynamic.conf```

### Edit the dynodeprivkey
If you generated a new Dynode private key, you will need to update the dynodeprivkey value in your remote dynamic.conf file.

## Start your Dynodes

### Remote

If your remote server is not running, start your remote daemon as you normally would. 

I usually confirm that remote is on the correct block by issuing:

```dynamicd getinfo```

And compare with the official explorer at http://explorer.dynamicpay.io/chain/Dynamic

### Local

Finally... time to start from local.

#### Open up your QT Wallet

From the menu select Tools => Debug Console

If you want to review your dynode.conf setting before starting the Dynodes, issue the following in the Debug Console:

```dynode list-conf```

Give it the eye-ball test. If satisfied, you can start your nodes one of two ways.

1. dynode start-alias [alias_from_dynode.conf]. Example ```dynode start-alias sn01```
2. dynode start-many
