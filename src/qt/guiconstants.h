// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_GUICONSTANTS_H
#define DYNAMIC_QT_GUICONSTANTS_H

/* Milliseconds between model updates */
static const int MODEL_UPDATE_DELAY = 250;

/* AskPassphraseDialog -- Maximum passphrase length */
static const int MAX_PASSPHRASE_SIZE = 1024;

/* DynamicGUI -- Size of icons in status bar */
static const int STATUSBAR_ICONSIZE = 16;

static const bool DEFAULT_SPLASHSCREEN = true;

/* Invalid field background style */
#define STYLE_INVALID "background:#FF8080"

/* Transaction list -- unconfirmed transaction */
#define COLOR_UNCONFIRMED QColor(128, 128, 128)
/* Transaction list -- negative amount */
#define COLOR_NEGATIVE QColor(255, 0, 0)
/* Transaction list -- bare address (without label) */
#define COLOR_BAREADDRESS QColor(140, 140, 140)
/* Transaction list -- TX status decoration - open until date */
#define COLOR_TX_STATUS_OPENUNTILDATE QColor(64, 64, 255)
/* Transaction list -- TX status decoration - offline */
#define COLOR_TX_STATUS_OFFLINE QColor(192, 192, 192)
/* Transaction list -- TX status decoration - danger, tx needs attention */
#define COLOR_TX_STATUS_DANGER QColor(200, 100, 100)
/* Transaction list -- TX status decoration - default color */
#define COLOR_BLACK QColor(0, 0, 0)
/* Transaction list -- TX status decoration - Locked by InstantSend (Dark Blue) */
#define COLOR_TX_STATUS_LOCKED QColor(13, 81, 140)
/* Transaction list -- TX status decoration - Fluid Transaction (Light Blue) */
#define COLOR_FLUID_TX QColor(11, 129, 158)
/* Transaction list -- TX status decoration - Dynode Reward (Purple)*/
#define COLOR_DYNODE_REWARD QColor(150, 20, 150)
/* Transaction list -- TX status decoration - Generated (Gold) */
#define COLOR_GENERATED QColor(156, 123, 19)
/* Transaction list -- TX status decoration - stake (Green) */
#define COLOR_STAKE QColor(102, 128, 14)
/* Transaction list -- TX status decoration - orphan (Light Gray) */
#define COLOR_ORPHAN QColor(211, 211, 211)

/* Tooltips longer than this (in characters) are converted into rich text,
   so that they can be word-wrapped.
 */
static const int TOOLTIP_WRAP_THRESHOLD = 80;

/* Maximum allowed URI length */
static const int MAX_URI_LENGTH = 255;

/* QRCodeDialog -- size of exported QR Code image */
#define QR_IMAGE_SIZE 300

/* Number of frames in spinner animation */
#define SPINNER_FRAMES 36

#define QAPP_ORG_NAME "Duality Solutions"
#define QAPP_ORG_DOMAIN "duality.solutions"
#define QAPP_APP_NAME_DEFAULT "Dynamic-Qt"
#define QAPP_APP_NAME_TESTNET "Dynamic-Qt-testnet"

#endif // DYNAMIC_QT_GUICONSTANTS_H
