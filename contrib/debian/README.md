
Debian
====================
This directory contains files used to package dynamicd/dynamic-qt
for Debian-based Linux systems. If you compile dynamicd/dynamic-qt yourself, there are some useful files here.

## dynamic: URI support ##


dynamic-qt.desktop  (Gnome / Open Desktop)
To install:

	sudo desktop-file-install dynamic-qt.desktop
	sudo update-desktop-database

If you build yourself, you will either need to modify the paths in
the .desktop file or copy or symlink your dynamic-qt binary to `/usr/bin`
and the `../../share/pixmaps/dynamic128.png` to `/usr/share/pixmaps`

dynamic-qt.protocol (KDE)

