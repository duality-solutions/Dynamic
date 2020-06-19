// Copyright (c) 2016-2019 Duality Blockchain Solutions Developers
// Copyright (c) 2014-2019 The Dash Core Developers
// Copyright (c) 2009-2019 The Bitcoin Developers
// Copyright (c) 2009-2019 Satoshi Nakamoto
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef DYNAMIC_COINCONTROL_H
#define DYNAMIC_COINCONTROL_H

#include "policy/feerate.h"
#include "policy/fees.h"
#include "primitives/transaction.h"
#include "wallet/wallet.h"

/** Coin Control Features. */
class CCoinControl
{
public:
    CTxDestination destChange;

/* ASSET START */

    //! If set, all asset change will be sent to this address, if not destChange will be used
    CTxDestination assetDestChange;
/* ASSET END */

    bool fUsePrivateSend;
    bool fUseInstantSend;
    //! If false, allows unselected inputs, but requires all selected inputs be used
    bool fAllowOtherInputs;
    //! Includes watch only addresses which match the ISMINE_WATCH_SOLVABLE criteria
    bool fAllowWatchOnly;
    //! Override estimated feerate
    bool fOverrideFeeRate;
    //! Override the default payTxFee if set
    boost::optional<CFeeRate> m_feerate;
    //! Override the default confirmation target if set
    boost::optional<unsigned int> m_confirm_target;
    //! Signal BIP-125 replace by fee.
    bool signalRbf;
    //! Fee estimation mode to control arguments to estimateSmartFee
    FeeEstimateMode m_fee_mode;

    /** ASSET START */
    //! Name of the asset that is selected, used when sending assets with coincontrol
    std::string strAssetSelected;
    /** ASSET END */

    CCoinControl()
    {
        SetNull();
    }

    void SetNull()
    {
        destChange = CNoDestination();
        assetDestChange = CNoDestination();
        fUseInstantSend = false;
        fUsePrivateSend = true;
        fAllowOtherInputs = false;
        fAllowWatchOnly = false;
        setSelected.clear();
        m_feerate.reset();
        fOverrideFeeRate = false;
        m_confirm_target.reset();
        signalRbf = fWalletRbf;
        m_fee_mode = FeeEstimateMode::UNSET;
        strAssetSelected = "";
        setAssetsSelected.clear();
/* ASSET START */
        strAssetSelected = "";
        setAssetsSelected.clear();
/* ASSET END */
    }

    bool HasSelected() const
    {
        return (setSelected.size() > 0);
    }

    bool HasAssetSelected() const
    {
        return (setAssetsSelected.size() > 0);
    }

    bool IsSelected(const COutPoint& output) const
    {
        return (setSelected.count(output) > 0);
    }

    void Select(const COutPoint& output)
    {
        setSelected.insert(output);
    }

/* ASSET START */
    void SelectAsset(const COutPoint& output)
    {
        setAssetsSelected.insert(output);
    }
/* ASSET END */

    void UnSelect(const COutPoint& output)
    {
        setSelected.erase(output);
    }

/* ASSET START */
   void UnSelectAsset(const COutPoint& output)
    {
        setAssetsSelected.erase(output);
        if (!setSelected.size())
            strAssetSelected = "";
    }
/* ASSET END */

    void UnSelectAll()
    {
        setSelected.clear();
    }

    void ListSelected(std::vector<COutPoint>& vOutpoints) const
    {
        vOutpoints.assign(setSelected.begin(), setSelected.end());
    }

/* ASSET START */
    void ListSelectedAssets(std::vector<COutPoint>& vOutpoints) const
    {
        vOutpoints.assign(setAssetsSelected.begin(), setAssetsSelected.end());
    }
/* ASSET END */
    
private:
    std::set<COutPoint> setSelected;
/* ASSET START */
    std::set<COutPoint> setAssetsSelected;
/* ASSET END */
};

#endif // DYNAMIC_COINCONTROL_H
