import numpy as np
import experta as exp
from sklearn.neural_network import MLPClassifier

# Modified neural network to output probabilities
neural_network = MLPClassifier(probability=True)
neural_network.fit(patient_data, ['Flu'])

# Getting probability of flu
probabilities = neural_network.predict_proba(np.array([[1, 1]]))
flu_probability = probabilities[0][0]  # Probability of flu

class MedicalDiagnosisWithConfidence(exp.KnowledgeEngine):
    @exp.Rule(exp.Fact(symptom='fever'), 
              exp.Fact(symptom='cough'), 
              exp.Fact(flu_probability=exp.TEST(lambda p: p > 0.8)))
    def diagnose_flu_with_confidence(self):
        print("High confidence in flu diagnosis based on symptoms and neural network analysis.")

# Create and run the knowledge engine
engine = MedicalDiagnosisWithConfidence()
engine.reset()
engine.declare(exp.Fact(symptom='fever'))
engine.declare(exp.Fact(symptom='cough'))
engine.declare(exp.Fact(flu_probability=flu_probability))
engine.run()