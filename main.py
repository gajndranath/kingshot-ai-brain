from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from agent import AIController

app = FastAPI(title="Kingshot AI Brain", version="1.0.0")
ai_controller = AIController()

class TranslationRequest(BaseModel):
    text: str

class CoachingRequest(BaseModel):
    question: str
    player_stats: Optional[dict] = None

class BattleAnalysisRequest(BaseModel):
    image_url: str

class ScanNapRequest(BaseModel):
    image_url: str
    safe_tags: list

class ScanDonationRequest(BaseModel):
    image_url: str

@app.get("/")
def read_root():
    return {"status": "AI Brain is running online"}

@app.post("/api/translate")
async def translate_text(req: TranslationRequest):
    try:
        result = await ai_controller.translate_text(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/coach")
async def coach_player(req: CoachingRequest):
    try:
        response = await ai_controller.generate_coaching_advice(req.question, req.player_stats)
        return {"advice": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-battle")
async def analyze_battle(req: BattleAnalysisRequest):
    try:
        analysis = await ai_controller.analyze_battle_screenshot(req.image_url)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan-nap")
async def scan_nap(req: ScanNapRequest):
    try:
        result = await ai_controller.scan_nap_violation(req.image_url, req.safe_tags)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan-donation")
async def scan_donation(req: ScanDonationRequest):
    try:
        result = await ai_controller.scan_tech_donation(req.image_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trivia")
async def get_trivia():
    try:
        trivia = await ai_controller.generate_daily_trivia()
        return trivia
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
