import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
from test_neg_xgb import NegatedXGB
from walkforward import walkforward_full

res = walkforward_full(NegatedXGB, gauge_fix=False, verbose=False)
print(f"Negated XGB Gross Sharpe: {res.get('gross_sharpe', 0):.6f}")
print(f"Negated XGB IC: {res.get('ic', 0):.6f}")
