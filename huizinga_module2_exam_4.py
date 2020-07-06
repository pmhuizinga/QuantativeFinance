"""
Question 4. Liquidity-Adjusted VaR (LVaR) while not considered in the lecture, is effectively VaR plus
VaR of the bid/ask spread itself. Use the positive value of the Standard Normal Factor in the formula,
LVaR = VaR + ∆Liquidity
= Portfolio Value × −µ + Factor × σ + 1 2(µSpread + Factor × σSpread)
Compute the proportions attributed to VaR and liquidity adjustment for the following cases:
(a) Consider a portfolio of USD 16 million composed of shares in a technology company. Daily mean
and volatility of its returns are 1% and 3%, respectively. Bid-ask spread also varies with time, its
daily mean and volatility are 35 bps and 150 bps. Compute 99%/1D LVaR and attributions to it,

"""
from scipy.stats import norm

def calculate_Var(w, p_u=0.0, p_s=0.0, s_u=0.0, s_s=0.0, conf=.99):
    factor = norm.ppf(conf)
    relative_var_pct = factor * p_s
    relative_var_abs = relative_var_pct * w
    absolute_var_pct = -p_u + relative_var_pct
    absolute_var_abs = absolute_var_pct * w
    lvar_pct = -p_u + relative_var_pct + 0.5 * (s_u + factor * s_s)
    lvar_abs = lvar_pct * w
    print('relative Var %: {}'.format(round(relative_var_pct * 100, 2)))
    print('relative Var $: {}'.format(round(relative_var_abs, 2)))
    print('absolute Var %: {}'.format(round(absolute_var_pct * 100, 2)))
    print('absolute Var $: {}'.format(round(absolute_var_abs, 2)))
    print('Lvar %: {}'.format(round(lvar_pct * 100, 2)))
    print('Lvar $: {}'.format(round(lvar_abs, 2)))
    print('-' * 80)


# 4a
p_val = 16e6
p_mean = 0.01
p_std = 0.03
ba_spread_mean = 0.0035
ba_spread_std = 0.015
confidence = .99
calculate_Var(p_val, p_mean, p_std, ba_spread_mean, ba_spread_std, confidence)

# 4b
p_val = 40e6
p_mean = 0
p_std = 0.03
ba_spread_std = 0
ba_spread_mean = 0.0015
confidence = .99
calculate_Var(p_val, p_mean, p_std, ba_spread_mean, ba_spread_std, confidence)

ba_spread_mean = 0.0125
calculate_Var(p_val, p_mean, p_std, ba_spread_mean, ba_spread_std, confidence)

