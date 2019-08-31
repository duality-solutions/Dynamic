// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_MACDOCKICONHANDLER_H
#define DYNAMIC_QT_MACDOCKICONHANDLER_H

#include <QMainWindow>
#include <QObject>

QT_BEGIN_NAMESPACE
class QIcon;
class QMenu;
class QWidget;
QT_END_NAMESPACE

/** Macintosh-specific dock icon handler.
 */
class MacDockIconHandler : public QObject
{
    Q_OBJECT

public:
    ~MacDockIconHandler();

    QMenu* dockMenu();
    void setIcon(const QIcon& icon);
    void setMainWindow(QMainWindow* window);
    static MacDockIconHandler* instance();
    static void cleanup();
    void handleDockIconClickEvent();

Q_SIGNALS:
    void dockIconClicked();

private:
    MacDockIconHandler();

    QWidget* m_dummyWidget;
    QMenu* m_dockMenu;
    QMainWindow* mainWindow;
};

#endif // DYNAMIC_QT_MACDOCKICONHANDLER_H
