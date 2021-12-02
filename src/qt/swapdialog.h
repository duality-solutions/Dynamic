// Copyright (c) 2021-present Duality Blockchain Solutions Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_QT_SWAPDIALOG_H
#define DYNAMIC_QT_SWAPDIALOG_H

#include <QDialog>
#include <QAbstractButton>

namespace Ui
{
class SwapDialog;
}

class SwapDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SwapDialog(QWidget* parent);
    ~SwapDialog();

    QString getSwapAddress();

protected Q_SLOTS:
    void accept();
    void reject();

private Q_SLOTS:
    void buttonBoxClicked(QAbstractButton*);

private:
    Ui::SwapDialog* ui;
};

#endif // DYNAMIC_QT_SWAPDIALOG_H
