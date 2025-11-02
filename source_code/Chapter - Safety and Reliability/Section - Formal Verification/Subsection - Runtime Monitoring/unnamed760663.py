from flask import Flask, render_template
import random

app = Flask(__name__)

@app.route("/")
def dashboard():
    # Simulated data update
    data = random.randint(1, 200)
    processed = neural_process(data)
    decision = symbolic_reasoning(processed["processed_data"])
    return render_template('dashboard.html', decision=decision)

if __name__ == "__main__":
    app.run(debug=True)