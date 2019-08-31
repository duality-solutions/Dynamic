Release Process
====================

* update translations (ping wumpus, Diapolo or tcatm on IRC)
* see https://github.com/duality-solutions/dynamic/blob/master/doc/translation_process.md#syncing-with-transifex

* * *

###update (commit) version in sources

	contrib/verifysfbinaries/verify.sh
	doc/README*
	share/setup.nsi
	src/clientversion.h (change CLIENT_VERSION_IS_RELEASE to true)

###tag version in git

	git tag -s v(new version, e.g. 2.4.0.0)

###write release notes. git shortlog helps a lot, for example:

	git shortlog --no-merges v(current version, e.g. 2.3.5.0)..v(new version, e.g. 2.4.0.0)

* * *

###update gitian

 In order to take advantage of the new caching features in gitian, be sure to update to a recent version (e9741525c or higher is recommended)

###perform gitian builds

 From a directory containing the dynamic source, gitian-builder and gitian.sigs

	export SIGNER=(your gitian key, ie bluematt, sipa, etc)
	export VERSION=(new version, e.g. 2.4.0.0)
	pushd ./dynamic
	git checkout v${VERSION}
	popd
	pushd ./gitian-builder

###fetch and build inputs: (first time, or when dependency versions change)
 
	mkdir -p inputs

 Register and download the Apple SDK: (see OSX Readme for details)
 
 https://developer.apple.com/downloads/download.action?path=Developer_Tools/xcode_4.6.3/xcode4630916281a.dmg
 
 Using a Mac, create a tarball for the 10.7 SDK and copy it to the inputs directory:
 
	tar -C /Volumes/Xcode/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/ -czf MacOSX10.7.sdk.tar.gz MacOSX10.7.sdk

###Optional: Seed the Gitian sources cache

  By default, gitian will fetch source files as needed. For offline builds, they can be fetched ahead of time:

	make -C ../dynamic/depends download SOURCES_PATH=`pwd`/cache/common

  Only missing files will be fetched, so this is safe to re-run for each build.

###Build Dynamic for Linux, Windows, and OS X:

	./bin/gbuild --commit dynamic=v${VERSION} ../dynamic/contrib/gitian-descriptors/gitian-linux.yml
	./bin/gsign --signer $SIGNER --release ${VERSION}-linux --destination ../gitian.sigs/ ../dynamic/contrib/gitian-descriptors/gitian-linux.yml
	mv build/out/dynamic-*.tar.gz build/out/src/dynamic-*.tar.gz ../
	./bin/gbuild --commit dynamic=v${VERSION} ../dynamic/contrib/gitian-descriptors/gitian-win.yml
	./bin/gsign --signer $SIGNER --release ${VERSION}-win --destination ../gitian.sigs/ ../dynamic/contrib/gitian-descriptors/gitian-win.yml
	mv build/out/dynamic-*.zip build/out/dynamic-*.exe ../
	./bin/gbuild --commit dynamic=v${VERSION} ../dynamic/contrib/gitian-descriptors/gitian-osx.yml
	./bin/gsign --signer $SIGNER --release ${VERSION}-osx-unsigned --destination ../gitian.sigs/ ../dynamic/contrib/gitian-descriptors/gitian-osx.yml
	mv build/out/dynamic-*-unsigned.tar.gz inputs/dynamic-osx-unsigned.tar.gz
	mv build/out/dynamic-*.tar.gz build/out/dynamic-*.dmg ../
	popd
  Build output expected:

  1. source tarball (dynamic-${VERSION}.tar.gz)
  2. linux 32-bit and 64-bit binaries dist tarballs (dynamic-${VERSION}-linux[32|64].tar.gz)
  3. windows 32-bit and 64-bit installers and dist zips (dynamic-${VERSION}-win[32|64]-setup.exe, dynamic-${VERSION}-win[32|64].zip)
  4. OSX unsigned installer (dynamic-${VERSION}-osx-unsigned.dmg)
  5. Gitian signatures (in gitian.sigs/${VERSION}-<linux|win|osx-unsigned>/(your gitian key)/

###Next steps:

Commit your signature to gitian.sigs:

	pushd gitian.sigs
	git add ${VERSION}-linux/${SIGNER}
	git add ${VERSION}-win/${SIGNER}
	git add ${VERSION}-osx-unsigned/${SIGNER}
	git commit -a
	git push  # Assuming you can push to the gitian.sigs tree
	popd

  Wait for OSX detached signature:
	Once the OSX build has 3 matching signatures, Evan(?) ***TODO*** will sign it with the apple App-Store key.
	He will then upload a detached signature to be combined with the unsigned app to create a signed binary.

  Create the signed OSX binary:

	pushd ./gitian-builder
	# Fetch the signature as instructed by Evan
	cp signature.tar.gz inputs/
	./bin/gbuild -i ../dynamic/contrib/gitian-descriptors/gitian-osx-signer.yml
	./bin/gsign --signer $SIGNER --release ${VERSION}-osx-signed --destination ../gitian.sigs/ ../dynamic/contrib/gitian-descriptors/gitian-osx-signer.yml
	mv build/out/dynamic-osx-signed.dmg ../dynamic-${VERSION}-osx.dmg
	popd

Commit your signature for the signed OSX binary:

	pushd gitian.sigs
	git add ${VERSION}-osx-signed/${SIGNER}
	git commit -a
	git push  # Assuming you can push to the gitian.sigs tree
	popd

-------------------------------------------------------------------------

### After 3 or more people have gitian-built and their results match:

- Perform code-signing.

    - Code-sign Windows -setup.exe (in a Windows virtual machine using signtool)

  Note: only Evan has the code-signing keys currently.

- Create `SHA256SUMS.asc` for the builds, and GPG-sign it:
```bash
sha256sum * > SHA256SUMS
gpg --digest-algo sha256 --clearsign SHA256SUMS # outputs SHA256SUMS.asc
rm SHA256SUMS
```
(the digest algorithm is forced to sha256 to avoid confusion of the `Hash:` header that GPG adds with the SHA256 used for the files)

- Upload zips and installers, as well as `SHA256SUMS.asc` from last step, to the dynamic.org server
  into `/var/www/bin/dynamic-${VERSION}`

- Update dynamicpay.io version ***TODO***

  - First, check to see if the dynamicpay.io maintainers have prepared a
    release: https://github.com/duality-solutions/dynamic/labels/Releases

      - If they have, it will have previously failed their Travis CI
        checks because the final release files weren't uploaded.
        Trigger a Travis CI rebuild---if it passes, merge.

  - If they have not prepared a release, follow the Dynamic.org release
    instructions: https://github.com/duality-solutions/dynamic#release-notes

  - After the pull request is merged, the website will automatically show the newest version within 15 minutes, as well
    as update the OS download links. Ping @saivann/@harding (saivann/harding on Freenode) in case anything goes wrong

- Add release notes for the new version to the directory `doc/release-notes` in git master

- Celebrate
