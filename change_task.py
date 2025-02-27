import time
import argparse
import re
import requests
import subprocess
import os
import glob
import logging
import warnings
import torch
import base64
from typing import List, Optional
import py_compile
import json


class LLMWrapper:
    """
    Sends a request to your local or remote LLM for plan generation or repairs.
    Adjust 'api_url' and the request payload for your actual server.
    """
    def __init__(self, temperature: float = 0.7,
                 model_name: str = "llama3.1:8b-instruct-fp16",
                 api_url: str = "http://localhost:8888"):
        self.temperature = temperature
        self.model_name = model_name
        self.api_url = api_url

    def send_request(self, prompt: str, stream=False) -> str:
        """
        Example POST to /api/generate.
        Adjust to your real endpoint or logic.
        """
        try:
            #print(f"[LLM] Sending request to model '{self.model_name}' with prompt:\n{prompt}\n")
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": stream,
                },
                timeout=999
            )
            if response.status_code != 200:
                print(f"[LLM] Error: status {response.status_code}, {response.text}")
                return ""
            data = response.json()
            llm_text = data.get('response', '')
            #print(f"[LLM] Received text:\n{llm_text}\n")
            return llm_text
        except Exception as e:
            print(f"[LLM] Exception: {e}")
            return ""

###############################################################################
#                          Repair Task (Optional)                         #
###############################################################################
def repair_task(task: str, scene_description: str, scene: str) -> str:
    """
    Attempt to fix code using a smaller or the same LLM.
    Return the fixed code or empty string if fails.
    """
    grammar_ebnf = """
<program> ::= { <block-statement> [';'] | <statement> ';' }
<statement> ::= <variable-assign> | <function-call> | <return>
<block-statement> ::= <loop> | <conditional>
<loop> ::= <int> '{' <program> '}'
<function-call> ::= <function-name> ['(' <argument> ')']
<variable-assign> ::= <variable> '=' <function-call>
<conditional> ::= '?' <condition> '{' <program> '}'
<condition> ::= <operand> <comparator> <operand> { '&' <condition> | '|' <condition> }
<comparator> ::= '>' | '<' | '==' | '!='
<function-name> ::= <alpha> { <alpha> }
<argument> ::= <value> { ',' <value> }
<return> ::= '->' <value>
<operand> ::= <value> | <function-call>
<value> ::= <literal-value> | <variable>
<variable> ::= '_' <int>
<literal-value> ::= <int> | <float> | <string> | <bool>
""".strip()

    skill_definitions = """
'scan': 'scan(object_name: str)',
'scan_abstract': 'scan_abstract(question: str)',
'orienting': 'orienting(object_name: str)',
'approach': 'approach()',
'goto': 'goto(object_name: str)',
'move_forward': 'move_forward(distance: int)',
'move_backward': 'move_backward(distance: int)',
'move_left': 'move_left(distance: int)',
'move_right': 'move_right(distance: int)',
'move_up': 'move_up(distance: int)',
'move_down': 'move_down(distance: int)',
'turn_cw': 'turn_cw(degrees: int)',
'turn_ccw': 'turn_ccw(degrees: int)',
'move_in_circle': 'move_in_circle(cw: bool)',
'delay': 'delay(seconds: int)',
is_visible': 'is_visible(object_name: str)',
'object_x': 'object_x(object_name: str)',
'object_y': 'object_y(object_name: str)',
'object_width': 'object_width(object_name: str)',
'object_height': 'object_height(object_name: str)',
'object_dis': 'object_dis(object_name: str)',
'probe': 'probe(question: str)',
'log': 'log(text: str)',
'take_picture': 'take_picture()',
're_plan': 're_plan()'
"""

    # We'll use the same (or smaller) LLM:
    small_model = LLMWrapper(
        temperature=0.7,
        model_name="llama3.1:8b-instruct-fp16",
        api_url="http://localhost:8888"
    )

    prompt_text = f"""
### **Prompt for Task Refinement with Scene Awareness**  

The input consists of:  

1) **Scene (YOLO Output):**  
{scene}

2) **Scene Description (VLM Generated):**  
{scene_description}
  

3) **Original Task:**  
{task}

### **Instructions**  
- Adjust the task to be more relevant to the scene while keeping some of the original intent if it is relevant but have some creativity.  
- Ensure the task remains realistic given what the drone can do.  
- Remove ambiguity and make the task more actionable.  
- Avoid redundant or illogical objectives.  
- Maintain a **clear, concise, and structured** format.  

### **REQUIRED FORMAT**  
<task>  
</task>  

If the task is completely unreasonable or unsafe, modify it into a **logical alternative** that fits the scene. Do **not** provide explanations—only return the corrected task.
"""

    #print("[Repair] Attempting with small LLM. Prompt:\n", prompt_text)
    raw_llm_response = small_model.send_request(prompt_text)

    # 2) The model might put something like:
    # <task>
    # ml(60)
    # </task>
    #
    # or it might ignore instructions. We'll parse out code between <code> ... </code>.
    #print(f"{'+' * 10}\n{raw_llm_response}\n{'+' * 10}\n")
    code_snippet = extract_task(raw_llm_response)

    # If the code block was empty or missing, fallback to the entire response (strip).
    if not code_snippet.strip():
        code_snippet = raw_llm_response.strip()

    # 3) If it's still too chatty, do additional cleanup:
    code_snippet = extract_task(code_snippet)
    #print(f"{'+' * 10 }\n{code_snippet}\n{'+' * 10 }\n")
    #print(code_snippet)
    return code_snippet


def extract_task(text: str) -> str:

    patterns = [
        re.compile(r"<refined_task>(.*?)</refined_task>", re.DOTALL),
        re.compile(r"\*\*Refined Task:\*\*\s*(.*?)$", re.DOTALL),
        re.compile(r"\*\*Refined Task\*\*:\s*(.*?)$", re.DOTALL),
        re.compile(r"\*\*<refined_task>\*\*\s*(.*?)$", re.DOTALL),
        re.compile(r"\(refined_task\)(.*?)$", re.DOTALL),
        re.compile(r" ## Refined Task:(.*?)\*\*", re.DOTALL),
        re.compile(r"/task>(.*?)/refined_task>", re.DOTALL),
        re.compile(r"<task>(.*?)</task>", re.DOTALL)
    ]

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            print("***")
            print(pattern)
            return match.group(1).strip()
    print(f"{'+' * 10}\n{text}\n{'+' * 10}\n")

    return text

def main():
    parser = argparse.ArgumentParser(description="Configure model variables via command-line arguments.")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000", help="YOLO server or detection server")
    parser.add_argument("--vlm_api_url", type=str, default="http://localhost:8889", help="VLM API URL")
    parser.add_argument("--vlm_model_name", type=str, default="llava",
                        help="VLM model name (unused here but kept for reference)")
    parser.add_argument("--llm_model_name", type=str, default="llama3.1:8b-instruct-fp16", help="LLM model name")
    parser.add_argument("--llm_api_url", type=str, default="http://localhost:8888", help="LLM API URL")
    args = parser.parse_args()

    LLM_MODEL_NAME = args.llm_model_name
    YOLO_API_URL = args.api_url
    OLLAMA_API_URL = args.llm_api_url
    VLM_API_URL = args.vlm_api_url
    VLM_MODEL_NAME = args.vlm_model_name
    llm = LLMWrapper(temperature=0.7, model_name=LLM_MODEL_NAME, api_url=OLLAMA_API_URL)

    print(f"  YOLO_API_URL:   {YOLO_API_URL}")
    print(f"  VLM_API_URL:    {VLM_API_URL}")
    print(f"  VLM_MODEL_NAME: {VLM_MODEL_NAME}")

    input_file = "/home/liu/bap/minispec/python_version/minispec_GPT_refined"
    #vlm = VLMWrapper(remote_url=f"{YOLO_API_URL}/vlm", model_name=VLM_MODEL_NAME, temperature=0.2)

    # Load JSON data from a file
    with open(f"{input_file}.json", "r") as file:
        data = json.load(file)

    task_index = 0
    #images_dir = '/home/liu/Downloads/VisDrone2019-DET-train/images'
    #image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))

    # Modify the data
    for item in data:
        orignal_task = item["Task"]
        scene_description = item["Scene_description"]
        scene = item["Scene"]
        repaired_task = repair_task(orignal_task, scene_description, scene)
        #vlm_response = vlm.send_request(image_files[image_index])
        task_index += 1
        # Change Scene values (example: replace underscores with spaces)
        item["Task"] = repaired_task

        # Add a new field
        #item["Scene_description"] = vlm_response

    # Save the modified data back to the file
    with open(f"{input_file}_modified_task.json", "w") as file:
        json.dump(data, file, indent=2)

    print("JSON file updated successfully.")

if __name__ == "__main__":
    main()
