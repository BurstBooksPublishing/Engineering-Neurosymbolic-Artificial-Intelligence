from pyswip import Prolog

prolog = Prolog()

prolog.assertz("disease(flu) :- symptom(fever), symptom(cough), test(positive_flu_test)")
prolog.assertz("disease(cold) :- symptom(cough), test(negative_flu_test)")

def check_disease(symptoms, test_results):
    for symptom in symptoms:
        prolog.assertz(f"symptom({symptom})")
    for result in test_results:
        prolog.assertz(f"test({result})")
    
    diseases = list(prolog.query("disease(D)"))
    return [disease['D'] for disease in diseases]