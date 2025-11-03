from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from googletrans import Translator
import os

# Load saved model and encoder
model = joblib.load("artifacts/model.joblib")
label_encoder = joblib.load("artifacts/label_encoder.joblib")

app = FastAPI()

# ✅ Allow both local and Render frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        
        "*",  # Allows all origins (safe for testing)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = Translator()

# 📥 Define input format
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    language: str = "en"

# 🌿 Fertilizer suggestions logic
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

# 🌾 Reason for crop recommendation
def reason_for_crop(crop, temperature, humidity, rainfall):
    if crop.lower() in ["rice"]:
        return "High rainfall and humidity favor rice growth in your soil."
    elif crop.lower() in ["maize", "corn"]:
        return "Your Nitrogen and temperature are suitable for maize."
    elif crop.lower() in ["mungbean"]:
        return "Mungbean improves soil fertility by fixing nitrogen naturally."
    else:
        return f"{crop} is well-suited based on your soil and climate conditions."

# 🏠 Root route to test if API is live
@app.get("/")
def home():
    return {"message": "🌱 AI Crop Recommendation API is running!"}

# 🔮 Prediction endpoint
@app.post("/predict")
def predict_crop(data: CropInput):
    X = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    prediction = model.predict(X)[0]
    crop = label_encoder.inverse_transform([prediction])[0]

    fertilizers = fertilizer_suggestion(data.N, data.P, data.K)
    reason = reason_for_crop(crop, data.temperature, data.humidity, data.rainfall)

    output_text = f"Recommended Crop: {crop}\nReason: {reason}\nFertilizers: {', '.join(fertilizers)}"
    if data.language != "en":
        translated = translator.translate(output_text, dest=data.language)
        output_text = translated.text

    return {
        "crop": crop,
        "reason": reason,
        "fertilizers": fertilizers,
        "output_text": output_text
    }

# 🚀 Important for Render deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
