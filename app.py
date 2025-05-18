from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load saved model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cgpa = float(request.form['cgpa'])
        prediction = model.predict([[cgpa]])[0]
        return render_template("index.html", prediction=round(prediction, 2), cgpa=cgpa)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
