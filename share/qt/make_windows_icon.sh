#!/bin/bash
# create multiresolution windows icon
#mainnet
ICON_SRC=../../src/qt/res/icons/darksilk.png
ICON_DST=../../src/qt/res/icons/darksilk.ico
convert ${ICON_SRC} -resize 16x16 darksilk-16.png
convert ${ICON_SRC} -resize 32x32 darksilk-32.png
convert ${ICON_SRC} -resize 48x48 darksilk-48.png
convert darksilk-16.png darksilk-32.png darksilk-48.png ${ICON_DST}
#testnet
ICON_SRC=../../src/qt/res/icons/darksilk_testnet.png
ICON_DST=../../src/qt/res/icons/darksilk_testnet.ico
convert ${ICON_SRC} -resize 16x16 darksilk-16.png
convert ${ICON_SRC} -resize 32x32 darksilk-32.png
convert ${ICON_SRC} -resize 48x48 darksilk-48.png
convert darksilk-16.png darksilk-32.png darksilk-48.png ${ICON_DST}
rm darksilk-16.png darksilk-32.png darksilk-48.png
