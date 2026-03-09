"""
Fast numpy BDT tree calibration, analytical bond pricing, and delivery option valuation.

Uses Arrow-Debreu state prices + Newton's method for O(n^2) calibration,
and vectorized textbook bond pricing at terminal nodes.
"""

import numpy as np


def calibrate_bdt(discounts, sigma, dt):
    """
    Calibrate a BDT binomial tree using Arrow-Debreu state prices.

    Parameters
    ----------
    discounts : array-like
        Discount factors D[k] = D(0, (k+1)*dt) for k = 0, ..., n-1.
    sigma : float or array-like
        Log-normal volatility. Scalar for flat vol, or array of length n.
    dt : float
        Time step in years.

    Returns
    -------
    rate_tree : ndarray, shape (n+1, n+1)
        Continuously compounded short rates. Upper-triangular (rate_tree[i,k]
        is state i at step k, for i <= k). Lower triangle is NaN.
    z_top : ndarray, shape (n,)
        Log-rate at the top node of each column (z_top[k] for column k).
    ad_prices : ndarray, shape (n+1, n+1)
        Arrow-Debreu state prices. ad_prices[i,k] = price of $1 in state i at step k.
    """
    discounts = np.asarray(discounts, dtype=float)
    n = len(discounts)

    if np.isscalar(sigma):
        sigma = np.full(n, sigma)
    else:
        sigma = np.asarray(sigma, dtype=float)

    rate_tree = np.full((n + 1, n + 1), np.nan)
    ad_prices = np.full((n + 1, n + 1), np.nan)
    z_top = np.zeros(n)

    # Step 0: single node
    r0 = -np.log(discounts[0]) / dt
    rate_tree[0, 0] = r0
    ad_prices[0, 0] = 1.0

    if n == 1:
        z_top[0] = np.log(100 * r0)
        return rate_tree, z_top, ad_prices

    # Column-by-column calibration
    for k in range(1, n):
        sig_k = sigma[k]
        spacing = 2 * sig_k * np.sqrt(dt)

        # Propagate Arrow-Debreu prices from column k-1 to column k
        # ad_new[i] for i = 0, ..., k  (k+1 states)
        ad_new = np.zeros(k + 1)
        for i in range(k + 1):
            # Contribution from state i-1 (up move) if it exists
            if i > 0:
                disc_prev = np.exp(-rate_tree[i - 1, k - 1] * dt)
                ad_new[i] += 0.5 * disc_prev * ad_prices[i - 1, k - 1]
            # Contribution from state i (down move) if it exists
            if i < k:
                disc_prev = np.exp(-rate_tree[i, k - 1] * dt)
                ad_new[i] += 0.5 * disc_prev * ad_prices[i, k - 1]

        ad_prices[:k + 1, k] = ad_new

        # Newton solve for z0 (top node log-rate)
        # Target: sum_i AD[i] * exp(-r_i * dt) = D[k]
        # where r_i = exp(z0 - i * spacing) / 100
        target = discounts[k]
        z0 = np.log(100 * rate_tree[0, k - 1])  # initial guess

        for _ in range(20):
            states = np.arange(k + 1)
            r_vec = np.exp(z0 - states * spacing) / 100
            disc_vec = np.exp(-r_vec * dt)
            f_val = np.sum(ad_new * disc_vec) - target
            # Derivative: df/dz0 = sum_i AD[i] * exp(-r_i*dt) * (-r_i*dt)
            df_val = np.sum(ad_new * disc_vec * (-r_vec * dt))
            if abs(df_val) < 1e-30:
                break
            step = f_val / df_val
            z0 -= step
            if abs(step) < 1e-14:
                break

        z_top[k] = z0
        states = np.arange(k + 1)
        rate_tree[:k + 1, k] = np.exp(z0 - states * spacing) / 100

    # Fill z_top[0]
    z_top[0] = np.log(100 * r0)

    return rate_tree, z_top, ad_prices


def price_bonds_at_nodes(rates_cc, cpn_arr, T_rem_arr, freq, face, accr_frac_arr):
    """
    Vectorized analytical bond pricing at multiple rate states.

    Uses the textbook formula with continuously compounded rates converted
    to semi-annual yields: y = freq * (exp(r/freq) - 1).

    Parameters
    ----------
    rates_cc : ndarray, shape (S,)
        Continuously compounded short rates at each state.
    cpn_arr : ndarray, shape (J,)
        Annual coupon rates (decimal) for each bond.
    T_rem_arr : ndarray, shape (J,)
        Remaining time to maturity (years) for each bond.
    freq : int
        Coupon frequency (2 for semi-annual).
    face : float
        Face value.
    accr_frac_arr : ndarray, shape (J,)
        Accrued fraction of current coupon period for each bond.

    Returns
    -------
    dirty_prices : ndarray, shape (S, J)
        Dirty prices at each (state, bond) pair.
    clean_prices : ndarray, shape (S, J)
        Clean prices (dirty minus accrued interest).
    """
    rates_cc = np.asarray(rates_cc, dtype=float)
    cpn_arr = np.asarray(cpn_arr, dtype=float)
    T_rem_arr = np.asarray(T_rem_arr, dtype=float)
    accr_frac_arr = np.asarray(accr_frac_arr, dtype=float)

    S = len(rates_cc)
    J = len(cpn_arr)

    # Convert cc rates to semi-annual yield: y = freq * (exp(r/freq) - 1)
    y = freq * (np.exp(rates_cc[:, None] / freq) - 1)  # (S, 1)

    # Per-period quantities
    cpn_n = cpn_arr[None, :] / freq  # (1, J)
    y_n = y / freq                    # (S, 1) -> broadcasts to (S, J)
    N = T_rem_arr[None, :] * freq     # (1, J) number of remaining coupon periods

    # Textbook formula: P = face * [c/y * (1 - (1+y)^-N) + (1+y)^-N] * (1+y)^accr
    # Handle y_n ~ 0 carefully
    with np.errstate(divide='ignore', invalid='ignore'):
        disc_factor = (1 + y_n) ** (-N)  # (S, J)
        annuity = np.where(
            np.abs(y_n) < 1e-12,
            N * face,
            face * cpn_n / y_n * (1 - disc_factor)
        )
        dirty = (annuity + face * disc_factor) * (1 + y_n) ** accr_frac_arr[None, :]

    # Accrued interest
    ai = face * cpn_n * accr_frac_arr[None, :]  # (S, J)
    clean = dirty - ai

    return dirty, clean


def backward_induction_no_discount(payoff, n_steps):
    """
    Backward induction with no discounting (futures/forward measure).

    Parameters
    ----------
    payoff : ndarray, shape (n_steps+1,)
        Terminal payoff at each state.
    n_steps : int
        Number of tree steps.

    Returns
    -------
    tree : ndarray, shape (n_steps+1, n_steps+1)
        Full value tree. tree[i, k] is the value at state i, step k.
        Lower triangle is NaN.
    """
    tree = np.full((n_steps + 1, n_steps + 1), np.nan)
    tree[:n_steps + 1, n_steps] = payoff

    for k in range(n_steps - 1, -1, -1):
        for i in range(k + 1):
            tree[i, k] = 0.5 * (tree[i, k + 1] + tree[i + 1, k + 1])

    return tree


def price_bonds_on_tree(rate_tree, n_delivery, cpn_arr, T_rem_arr, dt,
                        freq=2, face=100):
    """
    Price bonds via backward induction on the BDT rate tree.

    The tree must extend to (or beyond) each bond's maturity. Each bond's
    coupon cash flows are placed on the tree and backward-inducted using the
    short rates at each node for discounting. This correctly incorporates
    the model-implied term structure at every node.

    Parameters
    ----------
    rate_tree : ndarray, shape (n+1, n+1)
        Calibrated BDT rate tree (continuously compounded short rates).
        Must cover at least max(n_delivery + T_rem_arr * steps_per_year) columns.
    n_delivery : int
        Tree step index for the delivery date.
    cpn_arr : ndarray, shape (J,)
        Annual coupon rates (decimal) for each bond.
    T_rem_arr : ndarray, shape (J,)
        Remaining time to maturity at delivery (years) for each bond.
    dt : float
        Time step in years.
    freq : int
        Coupon frequency (2 for semi-annual).
    face : float
        Face value.

    Returns
    -------
    clean_at_delivery : ndarray, shape (S, J)
        Clean prices at each delivery-date state for each bond.
        S = n_delivery + 1 states.
    dirty_at_delivery : ndarray, shape (S, J)
        Dirty prices at each delivery-date state.
    """
    cpn_arr = np.asarray(cpn_arr, dtype=float)
    T_rem_arr = np.asarray(T_rem_arr, dtype=float)
    J = len(cpn_arr)
    S = n_delivery + 1
    n_tree = rate_tree.shape[1] - 1  # max step in tree

    clean_at_delivery = np.full((S, J), np.nan)
    dirty_at_delivery = np.full((S, J), np.nan)

    cpn_period = 1.0 / freq  # years between coupons

    for j in range(J):
        cpn = cpn_arr[j]
        T_rem = T_rem_arr[j]
        cpn_payment = face * cpn / freq

        # Number of tree steps from delivery to this bond's maturity
        n_bond_steps = int(round(T_rem / dt))
        n_total = n_delivery + n_bond_steps  # total tree steps needed

        if n_total > n_tree:
            # Tree isn't long enough for this bond — skip or raise
            continue

        # Build coupon schedule: which tree steps get coupon payments?
        # Coupons fall every cpn_period years = cpn_period/dt steps from maturity
        cpn_step_interval = int(round(cpn_period / dt))
        # Steps (from delivery) where coupons fall
        coupon_steps_from_del = set()
        step = n_bond_steps  # maturity step (from delivery)
        while step > 0:
            coupon_steps_from_del.add(step)
            step -= cpn_step_interval
        # Convert to absolute tree steps
        coupon_abs_steps = {n_delivery + s for s in coupon_steps_from_del}

        # Backward induction from maturity to delivery
        # At maturity: value = face + last coupon
        mat_col = n_total
        n_states_mat = mat_col + 1
        values = np.full(n_states_mat, face + cpn_payment)

        for k in range(mat_col - 1, n_delivery - 1, -1):
            n_states_k = k + 1
            disc = np.exp(-rate_tree[:n_states_k, k] * dt)
            values = disc * 0.5 * (values[:n_states_k] + values[1:n_states_k + 1])
            if k in coupon_abs_steps:
                values += cpn_payment

        # values now has S = n_delivery+1 entries = bond dirty prices at delivery
        dirty_at_delivery[:, j] = values[:S]

        # Accrued interest at delivery
        # Find fraction of current coupon period elapsed at delivery
        # Steps from delivery to next coupon
        steps_to_next_cpn = min(
            (s for s in coupon_steps_from_del if s > 0), default=cpn_step_interval
        )
        frac_elapsed = 1.0 - (steps_to_next_cpn * dt) / cpn_period
        frac_elapsed = max(0.0, min(1.0, frac_elapsed))
        ai = face * cpn / freq * frac_elapsed
        clean_at_delivery[:, j] = dirty_at_delivery[:, j] - ai

    return clean_at_delivery, dirty_at_delivery


def futures_delivery_option(rate_tree, cpn_arr, T_rem_arr, cf_arr, n_delivery,
                            dt, freq=2, face=100, accr_frac_arr=None):
    """
    Compute futures price and delivery option value via BDT tree.

    Parameters
    ----------
    rate_tree : ndarray, shape (n+1, n+1)
        Calibrated BDT rate tree from calibrate_bdt.
    cpn_arr : ndarray, shape (J,)
        Annual coupon rates (decimal) for each deliverable bond.
    T_rem_arr : ndarray, shape (J,)
        Remaining time to maturity at delivery (years).
    cf_arr : ndarray, shape (J,)
        Conversion factors for each bond.
    n_delivery : int
        Tree step index for delivery date.
    dt : float
        Time step in years.
    freq : int
        Coupon frequency (2 for semi-annual).
    face : float
        Face value.
    accr_frac_arr : ndarray or None
        Accrued fraction for each bond at delivery. Defaults to 0.

    Returns
    -------
    result : dict with keys:
        'futures_price' : float - tree-implied futures price (with CTD switching)
        'futures_no_switch' : float - futures price using fixed CTD (median state)
        'option_value' : float - delivery option in price
        'option_value_32' : float - delivery option in 32nds
        'ctd_by_state' : ndarray, shape (S,) - CTD index at each terminal state
        'futures_tree' : ndarray - full backward-induction tree
        'terminal_adj_prices' : ndarray, shape (S, J) - adjusted prices at terminal
        'terminal_clean_prices' : ndarray, shape (S, J) - clean prices at terminal
        'terminal_rates' : ndarray, shape (S,) - short rates at terminal states
    """
    cpn_arr = np.asarray(cpn_arr, dtype=float)
    T_rem_arr = np.asarray(T_rem_arr, dtype=float)
    cf_arr = np.asarray(cf_arr, dtype=float)
    J = len(cpn_arr)
    S = n_delivery + 1  # number of terminal states

    if accr_frac_arr is None:
        accr_frac_arr = np.zeros(J)
    else:
        accr_frac_arr = np.asarray(accr_frac_arr, dtype=float)

    # Terminal short rates
    terminal_rates = rate_tree[:S, n_delivery]

    # Price all bonds at all terminal states
    _, clean_prices = price_bonds_at_nodes(
        terminal_rates, cpn_arr, T_rem_arr, freq, face, accr_frac_arr
    )

    # Adjusted prices = clean / CF
    adj_prices = clean_prices / cf_arr[None, :]

    # CTD at each state = bond with lowest adjusted price
    ctd_by_state = np.argmin(adj_prices, axis=1)

    # Futures payoff = min adjusted price at each state
    futures_payoff = np.min(adj_prices, axis=1)

    # Backward induction (no discounting)
    futures_tree = backward_induction_no_discount(futures_payoff, n_delivery)
    futures_price = futures_tree[0, 0]

    # No-switch benchmark: use CTD from median state
    median_state = S // 2
    fixed_ctd_idx = ctd_by_state[median_state]
    no_switch_payoff = adj_prices[:, fixed_ctd_idx]
    no_switch_tree = backward_induction_no_discount(no_switch_payoff, n_delivery)
    futures_no_switch = no_switch_tree[0, 0]

    option_value = futures_no_switch - futures_price
    option_value_32 = option_value * 32

    return {
        'futures_price': futures_price,
        'futures_no_switch': futures_no_switch,
        'option_value': option_value,
        'option_value_32': option_value_32,
        'ctd_by_state': ctd_by_state,
        'futures_tree': futures_tree,
        'terminal_adj_prices': adj_prices,
        'terminal_clean_prices': clean_prices,
        'terminal_rates': terminal_rates,
        'fixed_ctd_idx': fixed_ctd_idx,
    }


def futures_delivery_option_tree(rate_tree, cpn_arr, T_rem_arr, cf_arr,
                                  n_delivery, dt, freq=2, face=100):
    """
    Compute futures price and delivery option using full tree-based bond pricing.

    Unlike futures_delivery_option (which uses a flat-yield shortcut at terminal
    nodes), this function backward-inducts each bond's cash flows through the
    rate tree from maturity to delivery. The tree must extend to the longest
    bond maturity.

    Parameters
    ----------
    rate_tree : ndarray, shape (n+1, n+1)
        Calibrated BDT rate tree. Must extend to at least
        n_delivery + max(T_rem_arr) / dt columns.
    cpn_arr, T_rem_arr, cf_arr : ndarray, shape (J,)
        Bond coupons (decimal), remaining maturities at delivery, conversion factors.
    n_delivery : int
        Tree step for delivery date.
    dt : float
        Time step.
    freq : int
        Coupon frequency (default 2).
    face : float
        Face value (default 100).

    Returns
    -------
    dict : Same keys as futures_delivery_option.
    """
    cpn_arr = np.asarray(cpn_arr, dtype=float)
    T_rem_arr = np.asarray(T_rem_arr, dtype=float)
    cf_arr = np.asarray(cf_arr, dtype=float)
    S = n_delivery + 1

    # Price bonds via backward induction through the tree
    clean_prices, dirty_prices = price_bonds_on_tree(
        rate_tree, n_delivery, cpn_arr, T_rem_arr, dt, freq, face
    )

    # Check for any bonds that couldn't be priced (tree too short)
    valid = ~np.isnan(clean_prices[0, :])
    if not valid.all():
        import warnings
        n_skip = (~valid).sum()
        warnings.warn(f'{n_skip} bonds skipped (tree too short for their maturity)')
        # Filter to valid bonds only
        cpn_arr = cpn_arr[valid]
        T_rem_arr = T_rem_arr[valid]
        cf_arr = cf_arr[valid]
        clean_prices = clean_prices[:, valid]
        dirty_prices = dirty_prices[:, valid]

    # Adjusted prices = clean / CF
    adj_prices = clean_prices / cf_arr[None, :]

    # CTD at each state
    ctd_by_state = np.argmin(adj_prices, axis=1)

    # Futures payoff = min adjusted price
    futures_payoff = np.min(adj_prices, axis=1)

    # Backward induction (no discounting — forward measure)
    futures_tree = backward_induction_no_discount(futures_payoff, n_delivery)
    futures_price = futures_tree[0, 0]

    # No-switch benchmark
    median_state = S // 2
    fixed_ctd_idx = ctd_by_state[median_state]
    no_switch_payoff = adj_prices[:, fixed_ctd_idx]
    no_switch_tree = backward_induction_no_discount(no_switch_payoff, n_delivery)
    futures_no_switch = no_switch_tree[0, 0]

    option_value = futures_no_switch - futures_price
    option_value_32 = option_value * 32

    terminal_rates = rate_tree[:S, n_delivery]

    return {
        'futures_price': futures_price,
        'futures_no_switch': futures_no_switch,
        'option_value': option_value,
        'option_value_32': option_value_32,
        'ctd_by_state': ctd_by_state,
        'futures_tree': futures_tree,
        'terminal_adj_prices': adj_prices,
        'terminal_clean_prices': clean_prices,
        'terminal_rates': terminal_rates,
        'fixed_ctd_idx': fixed_ctd_idx,
    }
