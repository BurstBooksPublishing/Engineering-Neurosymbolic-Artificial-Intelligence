def rule_based_check(transaction_amount, login_countries):
    if transaction_amount > 10000 and len(login_countries) > 1:
        return True  # Potential fraud
    return False