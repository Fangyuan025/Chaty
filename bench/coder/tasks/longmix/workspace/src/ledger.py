"""A tiny expense ledger."""


def add_entry(entries, name, amount):
    entries.append({"name": name, "amount": amount})
    return entries
