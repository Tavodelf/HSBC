from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pgmpy.inference import VariableElimination

# Cargar el modelo
with open('modelo_bayesiano_credit_risk.pkl', 'rb') as f:
    final_model, mejor_config, features = pickle.load(f)

inference = VariableElimination(final_model)

app = Flask(__name__)

@app.route('/')
def home():
    return "Servidor funcionando."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    new_data = pd.DataFrame([data])
    
    def predecir_probabilidad(row):
        evidence = {k: row[k] for k in features if pd.notnull(row[k])}
        probas = inference.predict_probability(evidence=evidence, show_progress=False)
        return probas['credit_risk_score'].values  
    
    y_probs = new_data.apply(predecir_probabilidad, axis=1).values[0]
    
    credito = recomendar_credito(y_probs, data['income'])

    return jsonify({
        'credit_risk_score': float(y_probs),
        'credito_recomendado': float(credito)
    })

def recomendar_credito(y_probs, ingreso):
    if y_probs < 0.10:
        credito_base = 20000
    elif y_probs < 0.30:
        credito_base = 10000
    elif y_probs < 0.50:
        credito_base = 5000
    elif y_probs < 0.70:
        credito_base = 2000
    elif y_probs < 0.90:
        credito_base = 1000
    else:
        credito_base = 0

    credito_maximo_por_ingreso = ingreso * 0.3
    credito_recomendado = min(credito_base, credito_maximo_por_ingreso)

    return credito_recomendado

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)

