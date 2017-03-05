#!/bin/bash
# create multiresolution windows icon
#mainnet
ICON_SRC=../../src/qt/res/icons/dynamic.png
ICON_DST=../../src/qt/res/icons/dynamic.ico
convert ${ICON_SRC} -resize 16x16 dynamic-16.png
convert ${ICON_SRC} -resize 32x32 dynamic-32.png
convert ${ICON_SRC} -resize 48x48 dynamic-48.png
convert dynamic-16.png dynamic-32.png dynamic-48.png ${ICON_DST}
#testnet
ICON_SRC=../../src/qt/res/icons/dynamic_testnet.png
ICON_DST=../../src/qt/res/icons/dynamic_testnet.ico
convert ${ICON_SRC} -resize 16x16 dynamic-16.png
convert ${ICON_SRC} -resize 32x32 dynamic-32.png
convert ${ICON_SRC} -resize 48x48 dynamic-48.png
convert dynamic-16.png dynamic-32.png dynamic-48.png ${ICON_DST}
rm dynamic-16.png dynamic-32.png dynamic-48.png
