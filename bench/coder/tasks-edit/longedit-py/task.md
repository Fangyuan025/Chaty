Make these changes to src/fruitstand.py. Keep every other function exactly as it is.

1. The bulk discount starts at 10 units for every fruit. For bananas, cherries and mangoes only, it should start at 12 units instead.
2. Give `price_for_kiwis` a second parameter `discount` with default 0. It is a fraction (0.25 = 25% off) applied after the bulk discount and before rounding.
3. Rename `apply_tax` to `add_sales_tax`, everywhere it is defined or used.
4. `summary(order)` leaves one fruit out of the total. Fix it.
