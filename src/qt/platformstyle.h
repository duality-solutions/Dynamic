// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_PLATFORMSTYLE_H
#define DYNAMIC_QT_PLATFORMSTYLE_H

#include <QIcon>
#include <QPixmap>
#include <QString>

/* Coin network-specific GUI style information */
class PlatformStyle
{
public:
    /** Get style associated with provided platform name, or 0 if not known */
    static const PlatformStyle* instantiate(const QString& platformId);

    const QString& getName() const { return name; }

    bool getImagesOnButtons() const { return imagesOnButtons; }
    bool getUseExtraSpacing() const { return useExtraSpacing; }

    QColor TextColor() const;
    QColor SingleColor() const;
    QColor LightBlueColor() const;
    QColor DarkBlueColor() const;
    QColor LightOrangeColor() const;
    QColor DarkOrangeColor() const;

    /** Colorize an image (given filename) with the icon color */
    QImage SingleColorImage(const QString& filename) const;

    /** Colorize an icon (given filename) with the icon color */
    QIcon SingleColorIcon(const QString& filename) const;

    /** Colorize an icon (given object) with the icon color */
    QIcon SingleColorIcon(const QIcon& icon) const;

    /** Colorize an icon (given object) with the (given color) */
    QIcon SingleColorIcon(const QIcon& icon, const QColor& color) const;

    /** Set icon with two states on and off */
    QIcon SingleColorIconOnOff(const QString& filenameOn, const QString& filenameOff) const;

    /** Colorize an icon (given filename) with the text color */
    QIcon TextColorIcon(const QString& filename) const;

    /** Colorize an icon (given object) with the text color */
    QIcon TextColorIcon(const QIcon& icon) const;

    /** Colorize an icon (given filename) with the color dark orange */
    QIcon OrangeColorIcon(const QString& filename) const;

    /** Colorize an icon (given object) with the color dark orange */
    QIcon OrangeColorIcon(const QIcon& icon) const;

/* ASSET START */
    QColor MainBackGroundColor() const;
    QColor TopWidgetBackGroundColor() const;
    QColor WidgetBackGroundColor() const;
    QColor SendEntriesBackGroundColor() const;
    QColor AssetTxColor() const;
/* ASSET END */

private:
    PlatformStyle(const QString& name, bool imagesOnButtons, bool colorizeIcons, bool useExtraSpacing);

    QString name;
    bool imagesOnButtons;
    bool colorizeIcons;
    bool useExtraSpacing;
    QColor singleColor;
    QColor textColor;
    /* ... more to come later */
};

#endif // DYNAMIC_QT_PLATFORMSTYLE_H
