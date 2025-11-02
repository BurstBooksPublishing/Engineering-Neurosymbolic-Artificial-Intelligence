import spacy
from spacy.matcher import Matcher

# Load a pre-trained NLP model
nlp = spacy.load('en_core_web_sm')

# Example statements
statement1 = "All birds can fly."
statement2 = "Penguins can fly."

# Process statements with the NLP model
doc1 = nlp(statement1)
doc2 = nlp(statement2)

# Initialize a Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Define a pattern for exceptions in natural logic
pattern = [{"LOWER": "penguins"}]

# Add the pattern to the matcher
matcher.add("EXCEPTION_PATTERN", [pattern])

# Function to check if an exception applies
def check_exception(doc):
    matches = matcher(doc)
    return len(matches) > 0

# Apply natural logic to determine implication
def natural_logic_inference(doc1, doc2):
    if check_exception(doc2):
        return "No implication (exception found)"
    else:
        return "Implication holds"

# Test the natural logic inference
result = natural_logic_inference(doc1, doc2)
print(result)