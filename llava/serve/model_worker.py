"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread

import cv2
import numpy as np
import base64
'''
import re
import os
import requests
from openai import OpenAI

class PersonalInfoMasker:
    def _init_(self):
        # self.client = OpenAI(api_key="")
        # self.api_key = ""
        self._input_text = ""
        self._masked_text = ""

    # Getter for input_text
    def get_input_text(self):
        return self._input_text
    
    # Setter for input_text
    def set_input_text(self, input_text):
        self._input_text = input_text

    # Getter for masked_text
    def get_masked_text(self):
        return self._masked_text

    # Function to mask personal information using regex
    def mask_personal_info(self, text):
        # Define patterns for PII (Names, Age, Emails, Phone Numbers)
        name_pattern = r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b"  # Simple pattern for names (First Last)
        age_pattern = r"\b(?:[1-9]|[1-9][0-9])\s*(?:years old|year-old|y/o)\b"  # Pattern for age (XX years old)
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email pattern
        phone_pattern = r"\b(?:\+?(\d{1,3})?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"  # Phone number pattern

        # Masking personal information with special characters
        masked_text = re.sub(name_pattern, '', text)  # Mask names
        masked_text = re.sub(age_pattern, '## years old', masked_text)  # Mask ages
        masked_text = re.sub(email_pattern, '@.*', masked_text)  # Mask email addresses
        masked_text = re.sub(phone_pattern, '--', masked_text)  # Mask phone numbers

        return masked_text

    # Function to process text with GPT and mask PII
    def process_text(self):
        if not self._input_text:
            raise ValueError("Input text is empty. Set input text before processing.")

        # # Use GPT-3.5 to assist with PII detection
        # response = self.client.chat.completions.create(
        #     model="gpt-3.5-turbo",
            # messages=[
            #     {"role": "system", "content": "You are both data privacy and medical assistant. Your job is to carefully examine text to obfuscate only the patient's personal information in medical reports leaving behind the information like the name of medicals tests, follow-up treatments, etc."},
            #     {"role": "user", "content": f"I have a medical document containing sensitive information related to a patient. Your task is to review the entire text and carefully identify any personally identifiable information (PII) such as the patient's name, address, phone number, date of birth, Social Security number, or any other details that could be used to identify the patient. Replace all instances of PII with '[MASKED]' to protect the patient's privacy, but leave essential medical information such as test names, procedures, medications, and medical conditions untouched. Here is the text:\n\n{self._input_text}"},
            # ]
        #     )
        api_key = os.getenv("OPENAI_API_KEY")   
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-4",
            "messages": [
                {
                "role": "system", "content": "You are a private information obfuscator, you simply put *s in place of sensitive info only like names, phone numbers, email address.",
                "role": "user", "content": f"Anonymize this text:\n\n{self._input_text}",
                }
                ],
        }

        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        
        # Get the GPT-3.5 modified content
        # gpt_masked_content = response_json
        # Get the GPT-4 modified content
        gpt_masked_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Further mask personal information using regular expressions
        # self._masked_text = self.mask_personal_info(gpt_masked_content)
        choices = response_json['choices'][0]
        return str(choices['message']['content'])

    def process_text(self):
        if not self._input_text:
            raise ValueError("Input text is empty. Set input text before processing.")

    # Mask personal information using regex
        preprocessed_text = self.mask_personal_info(self._input_text)

    # Send the preprocessed text to GPT for further obfuscation (if needed) and content generation
        api_key = os.getenv("OPENAI_API_KEY")   
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
        data = {
             "model": "gpt-4",
             "messages": [
                 {
                 "role": "system", 
                 "content": "You are a private information obfuscator, you simply put *s in place of sensitive info only like names, phone numbers, email address."
                 },
                 {
                 "role": "user", 
                 "content": f"Anonymize this text:\n\n{preprocessed_text}"
                 }
                 ],
        }

        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()

    # Get the GPT-4 modified content
        gpt_masked_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')

    # Further mask any remaining personal information using regex
        self._masked_text = self.mask_personal_info(gpt_masked_content)
        return self._masked_text
'''

import spacy
import requests
import os

class PersonalInfoMasker:
    def __init__(self):
        # Load the SpaCy model for NER
        self.nlp = spacy.load("en_core_web_sm")
        self._input_text = ""

    # Getter for input_text
    def get_input_text(self):
        return self._input_text
    
    # Setter for input_text
    def set_input_text(self, input_text):
        self._input_text = input_text

    # Function to mask personal information using SpaCy NER
    def mask_personal_info(self, text):
        doc = self.nlp(text)
        masked_text = text
        # Replace entities with [MASKED]
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "DATE", "GPE", "ORG", "EMAIL", "PHONE"]:
                masked_text = masked_text.replace(ent.text, "[MASKED]")
        return masked_text

    # Function to process text with GPT and mask PII
    def process_text(self):
        if not self._input_text:
            raise ValueError("Input text is empty. Set input text before processing.")

        # Apply NER-based masking first
        preprocessed_text = self.mask_personal_info(self._input_text)

        # Prepare API request to OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in anonymizing personal data. Replace any remaining personal information such as names, phone numbers, email addresses, and other sensitive details with '[MASKED]' in the following text."
                },
                {
                    "role": "user",
                    "content": f"Here is the text with potentially sensitive information:\n\n{preprocessed_text}"
                }
            ],
        }

        # Send request to OpenAI API
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        gpt_masked_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')

        return gpt_masked_content

class ImageObfuscator:
    def __init__(self, pixel_size=2):
        self._image = None
        self._obfuscated_image = None
        self._pixel_size = pixel_size

    def set_image(self, image):
        if isinstance(image, str):
            # Assume the string is a base64 encoded image
            self._image = self._decode_base64_image(image)
        elif isinstance(image, np.ndarray):
            self._image = image
        else:
            raise ValueError("Input must be a string (base64 encoded image) or a numpy array")

    def _decode_base64_image(self, base64_string):
        # Remove the data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode the image")
        
        return image

    def get_obfuscated_image(self):
        return self._obfuscated_image

    def obfuscate(self):
        if self._image is None:
            raise ValueError("No image set")

        height, width = self._image.shape[:2]

        # Ensure pixel size doesn't result in zero or negative resize dimensions
        if self._pixel_size <= 0:
            raise ValueError("Pixel size must be greater than 0")

        # Ensure width and height divided by pixel size are at least 1
        new_width = max(1, width // self._pixel_size)
        new_height = max(1, height // self._pixel_size)

        # Resize the image down
        small = cv2.resize(self._image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Resize it back to the original dimensions using nearest neighbor interpolation
        self._obfuscated_image = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    def get_obfuscated_image_base64(self):
        if self._obfuscated_image is None:
            raise ValueError("No obfuscated image available")
        
        _, buffer = cv2.imencode('.png', self._obfuscated_image)
        return base64.b64encode(buffer).decode('utf-8')

def obfuscate_for_llava(image, pixel_size=2):
    """
    Obfuscate an input image for use with LLaVA-Med.
    
    :param image: string (base64 encoded image) or numpy array representing the image
    :param pixel_size: int, size of pixelation effect
    :return: numpy array of the obfuscated image
    """
    obfuscator = ImageObfuscator(pixel_size)
    obfuscator.set_image(image)
    obfuscator.obfuscate()
    return obfuscator.get_obfuscated_image()

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = 'llava' in self.model_name.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        import os
        import cv2
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt

        # Print original prompt
        print("Original Prompt: ", ori_prompt)
        
        text_obfs = PersonalInfoMasker()
        text_obfs.set_input_text(ori_prompt)
        ori_prompt = text_obfs.process_text()
        
        # Print obfuscated text
        print("Obfuscated Prompt: ", ori_prompt)

        # Handle images if provided
        images = params.get("images", None)
        if images is not None:
            # Create directory if it doesn't exist
            image_dir = "images"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            
            for idx, image in enumerate(images):
                # Save the original image
                original_image_path = os.path.join(image_dir, f"original_image_{idx}.png")

                # Check if the image is a NumPy array
                if isinstance(image, np.ndarray):
                    # Ensure the original image is in uint8 format
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    
                    # Save the original image
                    cv2.imwrite(original_image_path, image)
                    print(f"Original image saved at: {original_image_path}")
                else:
                    print(f"Image {idx} is not a valid NumPy array. Skipping...")

                # Obfuscate the image
                obfuscated_image = obfuscate_for_llava(image)

                # Check if obfuscated_image is a NumPy array and save
                if isinstance(obfuscated_image, np.ndarray):
                    # Ensure the obfuscated image is in uint8 format
                    if obfuscated_image.dtype != np.uint8:
                        obfuscated_image = (obfuscated_image * 255).astype(np.uint8)

                    # Save the obfuscated image
                    obfuscated_image_path = os.path.join(image_dir, f"obfuscated_image_{idx}.png")
                    cv2.imwrite(obfuscated_image_path, obfuscated_image)
                    print(f"Obfuscated image saved at: {obfuscated_image_path}")
                else:
                    print(f"Obfuscated image {idx} is not a valid NumPy array. Skipping...")

            # Print a message after obfuscation
            print("Images have been obfuscated and saved.")
        
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")