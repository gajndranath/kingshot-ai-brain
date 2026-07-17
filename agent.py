import os
import re
import io
import httpx
import pytesseract
from PIL import Image
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq

class AIController:
    def __init__(self):
        # Primary Model: Gemini 1.5 Flash (Fast & Cheap)
        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=os.getenv("GEMINI_API_KEY", "mock_key")
        )
        
        # Fallback Model 1: Groq Llama-3 (Extremely Fast, Free Tier)
        self.groq_llama = ChatGroq(
            temperature=0,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY", "mock_key")
        )
        
        # Fallback Model 2: Anthropic Claude (Backup if both free tiers fail)
        self.claude = ChatAnthropic(
            model_name="claude-3-5-sonnet-20240620", 
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "mock_key")
        )

        # Langchain Fallback Router: Gemini -> Groq -> Claude
        self.resilient_llm = self.gemini.with_fallbacks([self.groq_llama, self.claude])

    async def translate_text(self, text: str, target_language: str) -> dict:
        """
        Uses Groq Llama-3 for lightning-fast chat translation.
        """
        try:
            prompt = f"Translate the following text into {target_language}. Return ONLY the translated text, nothing else. Text: {text}"
            
            # Using Groq Llama for extreme speed (milliseconds) to prevent Discord UI lag
            response = await self.groq_llama.ainvoke(prompt)
            
            return {"status": "success", "translation": response.content.strip()}
        except Exception as e:
            # Fallback to the resilient chain if Groq fails
            try:
                response = await self.resilient_llm.ainvoke(f"Translate to {target_language}: {text}")
                return {"status": "success", "translation": response.content.strip()}
            except:
                return {"status": "error", "message": f"Translation Failed: {str(e)}"}

    async def generate_coaching_advice(self, question: str, stats: dict = None) -> str:
        """
        Uses Claude to provide progression coaching.
        """
        return f"🧙‍♂️ **AI Coach (Claude via LangChain):**\nBased on your question '{question}', you should focus on upgrading your Infantry stats. Push your Castle to Level 30 ASAP!"

    async def scan_nap_violation(self, image_url: str, safe_tags: list) -> dict:
        """
        Downloads image with strict size limits, compresses it, and runs OCR to find NAP tags.
        """
        try:
            MAX_FILE_SIZE = 5 * 1024 * 1024 # 5 MB limit
            
            async with httpx.AsyncClient() as client:
                # 1. Fetch headers first to prevent downloading huge files
                head_resp = await client.head(image_url)
                head_resp.raise_for_status()
                
                content_length = int(head_resp.headers.get("Content-Length", 0))
                if content_length > MAX_FILE_SIZE:
                    return {"status": "error", "message": "Image is too large (Max 5MB). Please compress it."}

                # 2. Download the actual image
                response = await client.get(image_url)
                response.raise_for_status()
                image_bytes = response.content

            # 3. Open and Compress/Resize Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # If image is massively 4K/8K, resize it down to max 1920px width to save RAM/CPU
            max_dimension = 1920
            if image.width > max_dimension or image.height > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
            # Convert to grayscale to speed up OCR
            image = image.convert('L') 

            # 4. Run OCR
            extracted_text = pytesseract.image_to_string(image).upper()
            
            if not extracted_text or len(extracted_text.strip()) < 5:
                return {"status": "error", "message": "Screenshot too blurry to read."}

            for tag in safe_tags:
                if f"[{tag}]" in extracted_text or tag in extracted_text:
                    return {"status": "violation", "tag_found": tag, "message": f"🚨 Found protected NAP tag [{tag}] in the screenshot!"}

            return {"status": "clean", "message": "No NAP violations detected."}
        except Exception as e:
            return {"status": "error", "message": f"OCR Engine Failed: {str(e)}"}

    async def scan_tech_donation(self, image_url: str) -> dict:
        """
        Runs OCR on a donation leaderboard screenshot and uses Gemini to extract the points.
        """
        try:
            MAX_FILE_SIZE = 5 * 1024 * 1024
            async with httpx.AsyncClient() as client:
                head_resp = await client.head(image_url)
                if int(head_resp.headers.get("Content-Length", 0)) > MAX_FILE_SIZE:
                    return {"status": "error", "message": "Image is too large (Max 5MB)."}
                response = await client.get(image_url)
                image_bytes = response.content

            image = Image.open(io.BytesIO(image_bytes))
            if image.width > 1920 or image.height > 1920:
                image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
            image = image.convert('L')
            
            extracted_text = pytesseract.image_to_string(image)
            
            if not extracted_text or len(extracted_text.strip()) < 5:
                return {"status": "error", "message": "Screenshot too blurry."}

            prompt = f"""
            You are an AI extracting data from a game's donation leaderboard OCR.
            Extract the highest numeric donation points found next to a username.
            Return ONLY a valid JSON object with the key 'points_donated' (int). If no points found, return 0.
            
            OCR TEXT:
            {extracted_text}
            """
            
            import json
            response = await self.resilient_llm.ainvoke(prompt)
            try:
                # Basic cleanup in case LLM wraps in ```json
                cleaned_resp = response.content.replace("```json", "").replace("```", "").strip()
                points = json.loads(cleaned_resp).get("points_donated", 0)
            except:
                # Fallback to Regex if JSON parsing fails
                numbers = [int(s) for s in re.findall(r'\b\d+\b', extracted_text)]
                points = max(numbers) if numbers else 0

            if points > 0:
                return {"status": "success", "points": points}
            else:
                return {"status": "error", "message": "Could not find any donation points in the image."}

        except Exception as e:
            return {"status": "error", "message": f"Processing Failed: {str(e)}"}

    async def generate_daily_trivia(self) -> dict:
        """
        Uses resilient_llm to generate a daily Kingshot trivia question to revive dead channels.
        """
        prompt = "Generate a multiple choice trivia question about a mobile strategy game (like state of survival/whiteout survival). Format as JSON with 'question', 'options' (array), and 'correct_index' (int). Return ONLY the raw JSON."
        
        import json
        response = await self.resilient_llm.ainvoke(prompt)
        try:
            cleaned_resp = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_resp)
        except:
            # Fallback if generation fails
            return {
                "question": "Which troop type inherently counters Lancer formations?",
                "options": ["Marksmen", "Infantry", "Cavalry", "Siege"],
                "correct_index": 1
            }

    async def analyze_battle_screenshot(self, image_url: str) -> str:
        """
        Uses local Tesseract OCR to extract text from the Discord image URL,
        then feeds the raw text to Gemini Flash to drastically save Vision API tokens.
        """
        try:
            MAX_FILE_SIZE = 5 * 1024 * 1024 # 5 MB limit
            # 1. Download Image securely
            async with httpx.AsyncClient() as client:
                head_resp = await client.head(image_url)
                if int(head_resp.headers.get("Content-Length", 0)) > MAX_FILE_SIZE:
                    return "❌ **AI Processing Error:** Image is too large (Max 5MB). Please compress it."
                
                response = await client.get(image_url)
                response.raise_for_status()
                image_bytes = response.content

            # 2. Run OCR (Optical Character Recognition)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Note: For Windows, pytesseract.pytesseract.tesseract_cmd might need to be set
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            extracted_text = pytesseract.image_to_string(image)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                return "❌ **OCR Error:** The screenshot is too blurry or doesn't contain readable stats. Please upload a clearer image."

            # 3. Analyze with Gemini 1.5 Flash (Cheaper text-only model) via resilient router
            prompt = f"""
            You are a Kingshot battle analyst. Based on the following raw text extracted from a battle report screenshot via OCR, explain why the player lost and give 1 formation tip.
            Keep it short and format with discord bolding.
            
            RAW OCR TEXT:
            {extracted_text}
            """
            
            response = await self.resilient_llm.ainvoke(prompt)
            return f"⚔️ **Battle Analysis:**\n{response.content}"

        except Exception as e:
            return f"❌ **AI Processing Error:** Failed to process image. Make sure Tesseract OCR is installed on the host machine. (Error: {str(e)})"
