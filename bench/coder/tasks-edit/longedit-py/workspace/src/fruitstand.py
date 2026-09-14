"""Fruit stand pricing.

Every fruit has its own price function so the stand can tune them one by one.
"""

TAX_RATE = 0.08


def apply_tax(amount):
    """Add sales tax to an amount."""
    return round(amount * (1 + TAX_RATE), 2)


def price_for_apples(qty):
    """Price for qty units of apples."""
    base = 0.5
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_bananas(qty):
    """Price for qty units of bananas."""
    base = 1.9
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_cherries(qty):
    """Price for qty units of cherries."""
    base = 1.0
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_dates(qty):
    """Price for qty units of dates."""
    base = 2.4
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_elderberries(qty):
    """Price for qty units of elderberries."""
    base = 1.5
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_figs(qty):
    """Price for qty units of figs."""
    base = 0.6
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_grapes(qty):
    """Price for qty units of grapes."""
    base = 2.0
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def taxed_grapes(qty):
    """Price for grapes with tax."""
    return apply_tax(price_for_grapes(qty))


def price_for_honeydews(qty):
    """Price for qty units of honeydews."""
    base = 1.1
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_kiwis(qty):
    """Price for qty units of kiwis."""
    base = 2.5
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_lemons(qty):
    """Price for qty units of lemons."""
    base = 1.6
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_limes(qty):
    """Price for qty units of limes."""
    base = 0.7
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_mangoes(qty):
    """Price for qty units of mangoes."""
    base = 2.1
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_nectarines(qty):
    """Price for qty units of nectarines."""
    base = 1.2
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_oranges(qty):
    """Price for qty units of oranges."""
    base = 2.6
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_papayas(qty):
    """Price for qty units of papayas."""
    base = 1.7
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def taxed_papayas(qty):
    """Price for papayas with tax."""
    return apply_tax(price_for_papayas(qty))


def price_for_peaches(qty):
    """Price for qty units of peaches."""
    base = 0.8
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_pears(qty):
    """Price for qty units of pears."""
    base = 2.2
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_plums(qty):
    """Price for qty units of plums."""
    base = 1.3
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_quinces(qty):
    """Price for qty units of quinces."""
    base = 2.7
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_raspberries(qty):
    """Price for qty units of raspberries."""
    base = 1.8
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_strawberries(qty):
    """Price for qty units of strawberries."""
    base = 0.9
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_tangerines(qty):
    """Price for qty units of tangerines."""
    base = 2.3
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def taxed_tangerines(qty):
    """Price for tangerines with tax."""
    return apply_tax(price_for_tangerines(qty))


def price_for_blueberries(qty):
    """Price for qty units of blueberries."""
    base = 1.4
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_blackberries(qty):
    """Price for qty units of blackberries."""
    base = 0.5
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_cranberries(qty):
    """Price for qty units of cranberries."""
    base = 1.9
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_apricots(qty):
    """Price for qty units of apricots."""
    base = 1.0
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_coconuts(qty):
    """Price for qty units of coconuts."""
    base = 2.4
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_guavas(qty):
    """Price for qty units of guavas."""
    base = 1.5
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def taxed_guavas(qty):
    """Price for guavas with tax."""
    return apply_tax(price_for_guavas(qty))


def price_for_lychees(qty):
    """Price for qty units of lychees."""
    base = 0.6
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


def price_for_pineapples(qty):
    """Price for qty units of pineapples."""
    base = 2.0
    if qty >= 10:
        base = base * 0.9
    return round(base * qty, 2)


CATALOG = {
    "apples": price_for_apples,
    "bananas": price_for_bananas,
    "cherries": price_for_cherries,
    "dates": price_for_dates,
    "elderberries": price_for_elderberries,
    "figs": price_for_figs,
    "grapes": price_for_grapes,
    "honeydews": price_for_honeydews,
    "kiwis": price_for_kiwis,
    "lemons": price_for_lemons,
    "limes": price_for_limes,
    "mangoes": price_for_mangoes,
    "nectarines": price_for_nectarines,
    "oranges": price_for_oranges,
    "papayas": price_for_papayas,
    "peaches": price_for_peaches,
    "pears": price_for_pears,
    "plums": price_for_plums,
    "quinces": price_for_quinces,
    "raspberries": price_for_raspberries,
    "strawberries": price_for_strawberries,
    "tangerines": price_for_tangerines,
    "blueberries": price_for_blueberries,
    "blackberries": price_for_blackberries,
    "cranberries": price_for_cranberries,
    "apricots": price_for_apricots,
    "coconuts": price_for_coconuts,
    "guavas": price_for_guavas,
    "lychees": price_for_lychees,
    "pineapples": price_for_pineapples,
}


def summary(order):
    """Total for an order: a dict of fruit -> quantity, with tax."""
    total = 0
    names = sorted(order)
    for i in range(len(names) - 1):
        total += CATALOG[names[i]](order[names[i]])
    return apply_tax(total)
