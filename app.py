from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates #ui
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

#initialize FastAPI app
app = FastAPI(title = "Text Summarizer", description = "Text Summariztion using T5", version = "1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

#load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

#device
if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model.to(device)

#Templating
templates = Jinja2Templates(directory="templates")

#input schema => format
class DialogueInput(BaseModel):
  dialogue: str

def clean_data(text):

  text = re.sub(r"\r\n"," ", text)#lines
  text = re.sub(r"\s+"," ", text)#spaces
  text = re.sub(r"<.*?>"," ", text)#html tags
  text = text.strip().lower()
  return text

def summarize_dialogue(dialogue : str) -> str:
  dialogue = clean_data(dialogue)

  #tokenize
  inputs = tokenizer(
      dialogue,
      padding="max_length",
      max_length = 512,
      truncation = True,
      return_tensors = "pt"
  ).to(device)#pt=pytorch tensors

  #generate summary => token ids
  model.to(device)
  targets = model.generate(
      input_ids =  inputs["input_ids"],
      attention_mask = inputs["attention_mask"],
      max_length = 150,
      num_beams =4, #4 different summary compared select best one
      early_stopping = True #as best found stop producing
  )

  #convert token ids to text summary => decoding
  summary = tokenizer.decode(targets[0], skip_special_tokens = True)#remove EOS,SEPERATORS
  return summary

#api endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
  summary = summarize_dialogue(dialogue_input.dialogue)
  return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )