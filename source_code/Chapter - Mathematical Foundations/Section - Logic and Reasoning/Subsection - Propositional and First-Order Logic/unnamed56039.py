from sympy import symbols, And

P, Q = symbols('P Q')
spam_formula = And(P, Q)

# Assuming both keywords are found in an email
keywords = {P: True, Q: True}
is_spam = spam_formula.subs(keywords)

print("Is the email spam?", is_spam)