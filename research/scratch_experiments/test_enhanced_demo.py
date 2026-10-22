import sys
sys.path.insert(0, r"C:\Users\Sagar Jain\Documents\ChatGPT\Trading Bot\comp6_scientist_release")
from enhanced_demo import EnhancedDemoPredictor
from walkforward import walkforward_full

res = walkforward_full(EnhancedDemoPredictor, gauge_fix=False, verbose=False)
print(f"Enhanced Demo Full Sharpe: {res['sharpe']:.6f}")

# Test with different spans
for span in [72, 120, 240, 360]:
    class T(EnhancedDemoPredictor):
        def __init__(self):
            super().__init__()
            self.span = span
    res = walkforward_full(T, gauge_fix=False, verbose=False)
    print(f"  span={span}: {res['sharpe']:.6f}")
