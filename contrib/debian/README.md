
Debian
====================
This directory contains files used to package darksilkd/darksilk-qt
for Debian-based Linux systems. If you compile darksilkd/darksilk-qt yourself, there are some useful files here.

## darksilk: URI support ##


darksilk-qt.desktop  (Gnome / Open Desktop)
To install:

	sudo desktop-file-install darksilk-qt.desktop
	sudo update-desktop-database

If you build yourself, you will either need to modify the paths in
the .desktop file or copy or symlink your darksilk-qt binary to `/usr/bin`
and the `../../share/pixmaps/darksilk128.png` to `/usr/share/pixmaps`

darksilk-qt.protocol (KDE)

