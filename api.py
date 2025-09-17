from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from src.patient_summary import PatientSummary
from src.biomarker_analysis import BiomarkerAnalysis
from src.regional_analysis import RegionalAnalysis

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize modules and load data
try:
    patient_module = PatientSummary()
    patient_module.load_data("data/patient_data.csv")

    biomarker_module = BiomarkerAnalysis()
    biomarker_module.load_data("data/biomarker_data.csv", "data/reference_ranges.csv")

    regional_module = RegionalAnalysis()
    regional_module.load_data("data/regional_data.csv")
    
    # Temporarily comment out MedGemma model loading to resolve CORS issue
    # medgemma_tokenizer = AutoTokenizer.from_pretrained("google/medgemma-27b")
    # medgemma_model = AutoModelForCausalLM.from_pretrained("google/medgemma-27b")
    medgemma_tokenizer = None
    medgemma_model = None

    # Load DDI library
    with open('data/ddi_library.json', 'r') as f:
        ddi_library = json.load(f)

except Exception as e:
    print(f"Error loading data or models: {str(e)}")
    # Handle error appropriately
    patient_module = None
    biomarker_module = None
    regional_module = None
    medgemma_tokenizer = None
    medgemma_model = None
    ddi_library = []

@app.route('/')
def home():
    return "Predictive Health System API is running."

@app.route('/patient_summary/<patient_id>', methods=['GET'])
def get_patient_summary(patient_id):
    if not patient_module:
        return jsonify({"error": "Patient module not initialized"}), 500
    
    try:
        timeline = patient_module.get_visit_timeline(patient_id)
        if not timeline:
            return jsonify({"error": "No data found for this patient ID"}), 404
        
        recurring_illnesses = patient_module.get_recurring_illnesses(patient_id)
        
        summary_data = {
            "visit_timeline": timeline,
            "recurring_illnesses": recurring_illnesses
        }
        
        return jsonify(summary_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/biomarker_analysis/<patient_id>', methods=['GET'])
def get_biomarker_analysis(patient_id):
    if not biomarker_module:
        return jsonify({"error": "Biomarker module not initialized"}), 500

    try:
        analysis = biomarker_module.analyze_biomarkers(patient_id)
        if not analysis:
            return jsonify({"error": "No biomarker data found for this patient ID"}), 404
            
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/regional_trends/<region>', methods=['GET'])
def get_regional_trends(region):
    if not regional_module:
        return jsonify({"error": "Regional module not initialized"}), 500

    try:
        analysis = regional_module.analyze_regional_patterns(region)
        if not analysis:
            return jsonify({"error": "No data available for selected region"}), 404
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ddi_check', methods=['POST'])
def ddi_check():
    data = request.get_json()
    if not data or 'current_medications' not in data or 'new_medications' not in data:
        return jsonify({"error": "Invalid request body"}), 400

    current_meds = [med.lower() for med in data['current_medications']]
    new_meds = [med.lower() for med in data['new_medications']]
    interactions = []

    all_meds = current_meds + new_meds

    for i in range(len(all_meds)):
        for j in range(i + 1, len(all_meds)):
            med1 = all_meds[i]
            med2 = all_meds[j]
            for interaction in ddi_library:
                drugA = interaction['drugA'].lower()
                drugB = interaction['drugB'].lower()
                if (med1 == drugA and med2 == drugB) or \
                   (med1 == drugB and med2 == drugA):
                    interactions.append(interaction)

    return jsonify({"interactions": interactions})

# @app.route('/medgemma_analyze', methods=['POST'])
# def medgemma_analyze():
#     if not medgemma_model or not medgemma_tokenizer:
#         return jsonify({"error": "MedGemma model not initialized"}), 500
#
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({"error": "No text provided for analysis"}), 400
#
#     try:
#         input_text = data['text']
#         # This is a mocked response. In a real scenario, you would process the output 
#         # from the model to extract the medications in a structured format.
#         mock_structured_response = {
#             "medications": [
#                 {"name": "Atorvastatin", "dosage": "10mg", "frequency": "daily"}
#             ],
#             "symptoms": ["headache", "fever"],
#             "diagnosis": "Suspected viral infection"
#         }
#         return jsonify(mock_structured_response)
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
