from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from deep_translator import GoogleTranslator  # ✅ replaced googletrans with deep-translator

# Load saved model and encoder
model = joblib.load("artifacts/model.joblib")
label_encoder = joblib.load("artifacts/label_encoder.joblib")

app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Translator instance
def translate_text(text: str, target_lang: str):
    try:
        if target_lang != "en":
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return translated
    except Exception:
        return text
    return text

# Input format
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    language: str = "en"

# Fertilizer suggestions
def fertilizer_suggestion(N, P, K):
    suggestions = []
    if N < 50:
        suggestions.append("Apply Urea for Nitrogen")
    if P < 50:
        suggestions.append("Apply Single Super Phosphate (SSP) for Phosphorus")
    if K < 50:
        suggestions.append("Apply Muriate of Potash (MOP) for Potassium")
    if not suggestions:
        suggestions.append("Soil nutrients are balanced, no extra fertilizer needed")
    return suggestions

# Reason for crop recommendation
def reason_for_crop(crop, temperature, humidity, rainfall):
    if crop.lower() in ["rice"]:
        return "High rainfall and humidity favor rice growth in your soil."
    elif crop.lower() in ["maize", "corn"]:
        return "Your Nitrogen and temperature are suitable for maize."
    elif crop.lower() in ["mungbean"]:
        return "Mungbean improves soil fertility by fixing nitrogen naturally."
    else:
        return f"{crop} is well-suited based on your soil and climate conditions."

@app.get("/")
def home():
    return {"message": "🌱 AI Crop Recommendation API is running!"}

@app.post("/predict")
def predict_crop(data: CropInput):
    X = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    prediction = model.predict(X)[0]
    crop = label_encoder.inverse_transform([prediction])[0]

    fertilizers = fertilizer_suggestion(data.N, data.P, data.K)
    reason = reason_for_crop(crop, data.temperature, data.humidity, data.rainfall)

    output_text = f"Recommended Crop: {crop}\nReason: {reason}\nFertilizers: {', '.join(fertilizers)}"
    output_text = translate_text(output_text, data.language)  # ✅ uses deep-translator safely

    return {
        "crop": crop,
        "reason": reason,
        "fertilizers": fertilizers,
        "output_text": output_text
    }
