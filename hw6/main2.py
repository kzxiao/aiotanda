import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline

def translate_chinese_to_english(text, model_name="Helsinki-NLP/opus-mt-zh-en"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def generate_image(prompt, model_name="CompVis/stable-diffusion-v1-4", output_path="generated_image.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]  # Get the first generated image
    image.save(output_path)
    return output_path

if __name__ == "__main__":
    text = '一隻小狗在大安森林公園湖邊戲水' # @param {type:"string"}
    chinese_text = f'{text}'  # Replace with your input in Chinese
    english_text = translate_chinese_to_english(chinese_text)
    print(f"Translated Text: {english_text}")

    print("Generating image from prompt...")
    image_path = generate_image(english_text)
    print(f"Image saved at: {image_path}")

    img = Image.open(image_path)
    img