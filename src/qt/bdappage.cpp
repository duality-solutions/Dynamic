#include "bdappage.h"
#include "ui_bdappage.h"

#include "guiutil.h"
#include "walletmodel.h"

#include <stdio.h>

BdapPage::BdapPage(const PlatformStyle* platformStyle, QWidget* parent) : QWidget(parent),
                                                                            ui(new Ui::BdapPage)
{
    ui->setupUi(this);

}

BdapPage::~BdapPage()
{
    delete ui;
}

void BdapPage::setModel(WalletModel* model)
{
    this->model = model;
}








