from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pgmpy.inference import VariableElimination
import os

# Cargar el modelo
with open('modelo_bayesiano_credit_risk.pkl', 'rb') as f:
    final_model, mejor_config, features = pickle.load(f)

# Inicializar inferencia
inference = VariableElimination(final_model)

# Crear la app Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Servidor funcionando."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Convertir input en DataFrame
    new_data = pd.DataFrame([data])
    
    # Función para predecir la probabilidad
    def predecir_probabilidad(row):
        evidence = {k: row[k] for k in features if pd.notnull(row[k])}
        probas = inference.query(variables=['credit_risk_score'], evidence=evidence, show_progress=False)
        return probas.values[1]  # CORREGIDO: accedemos directo a probas.values[1]
    
    # Aplicar predicción
    y_probs = new_data.apply(predecir_probabilidad, axis=1).values[0]
    
    # Calcular crédito recomendado
    credito = recomendar_credito(y_probs, data['income'])

    return jsonify({
        'credit_risk_score': float(y_probs),
        'credito_recomendado': float(credito)
    })

# Lógica de recomendación de crédito
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

# Ejecutar app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
