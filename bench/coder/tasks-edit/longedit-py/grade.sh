#!/bin/bash
set -e
python3 - <<'PY'
import sys, re
sys.path.insert(0, "src")
import fruitstand as f
src = open("src/fruitstand.py").read()
assert not hasattr(f, "apply_tax") and "apply_tax" not in src, "apply_tax still present"
assert f.add_sales_tax(100) == 108.0
for name in ("bananas", "cherries", "mangoes"):
    p = getattr(f, f"price_for_{name}")
    assert p(11) == round(p(1) * 11, 2), f"{name} discounted at 11"
    assert p(12) < round(p(1) * 12, 2), f"{name} not discounted at 12"
for name in ("apples", "grapes", "pineapples"):
    p = getattr(f, f"price_for_{name}")
    assert p(10) < round(p(1) * 10, 2), f"{name} lost its 10-unit discount"
k = f.price_for_kiwis
assert k(4) == k(4, 0)
assert abs(k(4, discount=0.25) - k(4) * 0.75) < 0.011, "kiwi discount"
assert f.taxed_grapes(3) == f.add_sales_tax(f.price_for_grapes(3))
order = {"apples": 2, "pears": 3}
assert f.summary(order) == f.add_sales_tax(f.price_for_apples(2) + f.price_for_pears(3)), "summary"
assert len(re.findall(r"^def price_for_", src, re.M)) == 30
PY
