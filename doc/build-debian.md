# Debian 9 (Stretch) build guide for Dynamic

**NOTE: Lines ending in \ are part of a shell command that spreads over multiple 
lines, these can be copypasted as a whole (as opposed to line-by-line).**

## Update the system

```
apt-get update
apt-get -y upgrade
apt-get -y dist-upgrade
```

## Essential build packages
```
apt-get install build-essential autoconf automake libtool pkg-config git
```

Reboot, autoconf for Dynamic might complain about pkg-config otherwise.

```
shutdown -r now
```

## Libraries needed for Dynamic
```
apt-get -y install libminiupnpc-dev libdb++-dev libboost-system-dev \
          libboost-filesystem-dev libboost-program-options-dev \
          libboost-thread-dev libboost-chrono-dev libssl1.0-dev libevent-dev
```
          
## Dynamic build

```
useradd -m -d /home/dynamic -s /bin/bash dynamic
su - dynamic
git clone https://github.com/duality-solutions/Dynamic src
cd src
./autogen.sh
./configure --disable-tests --disable-gui-tests --disable-bench \
              --disable-zmq --enable-experimental-asm --with-incompatible-bdb \
              --with-gui=no
```

**NOTE: BUILDING Dynamic WITH LESS THAN 2GB OF RAM DOES NOT WORK!!!**

On systems with very little RAM (2GB) this CXXFLAGS cheat should fix
memory-related build issues (i.e. Cannot allocate and the likes).

```
CXXFLAGS="--param ggc-min-expand=1 --param ggc-min-heapsize=32768" make
```

On systems with enough RAM (i.e. 4GB and up) just execute make:

```
make
```

## Local installation

On Debian as well as Ubuntu, ~/bin usually is in the PATH per default.

```
mkdir ~/bin
cp src/dynamic-cli src/dynamic-tx src/dynamicd ~/bin
strip ~/bin/dynamic*
```

## Base config for Dynamic wallet
```
mkdir ~/.dynamic
echo "txindex=1" >> ~/.dynamic/dynamic.conf
echo "listen=1" >> ~/.dynamic/dynamic.conf
echo "rpcuser=dynamic" >> ~/.dynamic/dynamic.conf
echo "rpcpassword=`head -c 32 /dev/urandom | base64`" >> ~/.dynamic/dynamic.conf
chmod 0700 ~/.dynamic
chmod 0600 ~/.dynamic/dynamic.conf
```

## Start dynamicd

Watch debug.log to see if theres any errors. Log display can be exited by pressing
`ctrl-c`.

```
~/bin/dynamicd -daemon
tail -f ~/.dynamic/debug.log
```

Check with http://dyn.blocksandchain.com/ that the block height is the same
as it is reported by the output of the below command. It is labeled 'blocks'
in the getinfo-output. If the numbers match, the installation is completed.

```
dynamic-cli getinfo
{
  "version": 2040000,
  "protocolversion": 71000,
  "walletversion": 204000,
  "balance": 0.00000000,
  "privatesend_balance": 0.00000000,
  "blocks": 193619,
  "timeoffset": 0,
  "connections": 16,
  "proxy": "127.0.0.1:9050",
  "difficulty": 1.129563238994795,
  "testnet": false,
  "keypoololdest": 1538953839,
  "keypoolsize": 1999,
  "paytxfee": 0.00000000,
  "relayfee": 0.00001000,
  "errors": ""
}

```
