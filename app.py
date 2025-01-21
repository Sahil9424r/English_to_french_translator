from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import MarianMTModel, MarianTokenizer
import os
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the tokenizer and model using the provided files
MODEL_DIR = "./saved_translational_model"  # Path to the directory with 8 files

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory '{MODEL_DIR}' does not exist.")

# Load the tokenizer and model
try:
    tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
    model = MarianMTModel.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the correct device
except Exception as e:
    raise RuntimeError(f"Error loading the model or tokenizer: {e}")

# Templates directory (no need for StaticFiles anymore)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Render the HTML form for translation.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate")
async def translate_text(text: str = Form(...)):
    """
    Translate English text to French.
    """
    try:
        # Preprocess and tokenize
        text = text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # Tokenize the input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        # Move tensors to model's device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate the French translation
        outputs = model.generate(**inputs)
        french_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the translation
        return {"english_text": text, "french_translation": french_translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point to run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
