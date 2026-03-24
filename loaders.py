import os
from pathlib import Path
import pypdf
import pytesseract
from PIL import Image
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Initialize Models Globally ---
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading BLIP Image Captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --- Helper: Visual Captioning ---
def caption_image(image):
    """Generates a text description of a PIL Image."""
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# --- Text & PDF Loaders ---
def load_text_files(folder):
    texts = []
    for p in Path(folder).rglob("*.txt"):
        texts.append(p.read_text(errors="ignore"))
    return texts

def load_pdfs(folder):
    texts = []
    for p in Path(folder).rglob("*.pdf"):
        reader = pypdf.PdfReader(str(p))
        content = []
        for page in reader.pages:
            content.append(page.extract_text() or "")
        texts.append("\n".join(content))
    return texts

# --- Image Loader (OCR + Caption) ---
def image_to_text(img_path):
    img = Image.open(img_path).convert('RGB')
    
    # 1. OCR for extracted text
    ocr_text = pytesseract.image_to_string(img).strip()
    
    # 2. BLIP for visual context
    visual_caption = caption_image(img)
    
    combined_text = f"--- Image Document ---\nVisual Description: {visual_caption}\n"
    if ocr_text:
        combined_text += f"Extracted Text (OCR): {ocr_text}\n"
        
    return combined_text

def load_images(folder):
    texts = []
    for p in Path(folder).rglob("*.png"):
        texts.append(image_to_text(str(p)))
    for p in Path(folder).rglob("*.jpg"):
        texts.append(image_to_text(str(p)))
    return texts

# --- Video Loader (Audio + Frame Captions) ---
def extract_and_caption_frames(video_path, frame_interval_seconds=5):
    """Extracts frames using OpenCV and captions them with BLIP."""
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    success, image = vidcap.read()
    count = 0
    captions = []
    
    while success:
        # Extract 1 frame per 'frame_interval_seconds'
        if count % int(fps * frame_interval_seconds) == 0:
            # Convert OpenCV BGR to standard RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)
            
            caption = caption_image(pil_img)
            timestamp = int(count / fps)
            captions.append(f"[Frame at {timestamp}s]: {caption}")
            
        success, image = vidcap.read()
        count += 1
        
    vidcap.release()
    return "\n".join(captions)

def video_to_text(video_path):
    # 1. Audio Transcription
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path)
    
    transcript_result = whisper_model.transcribe(audio_path)
    audio_text = transcript_result["text"]
    
    clip.close()
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    # 2. Visual Frame Captioning
    visual_text = extract_and_caption_frames(video_path, frame_interval_seconds=5)
    
    # Combine
    combined_text = (
        f"--- Video Document ---\n"
        f"Audio Transcription:\n{audio_text}\n\n"
        f"Visual Event Timeline:\n{visual_text}\n"
    )
    return combined_text

def load_videos(folder):
    texts = []
    for p in Path(folder).rglob("*.mp4"):
        texts.append(video_to_text(str(p)))
    return texts

# --- Master Builder ---
def build_corpus(base_folder):
    collection = []
    collection += load_text_files(base_folder)
    collection += load_pdfs(base_folder)
    collection += load_images(base_folder)
    collection += load_videos(base_folder)
    return collection