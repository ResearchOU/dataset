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

# Optional import for ollama, kept from original code
import ollama

###############################################################################
#                                Setup Logging                                #
###############################################################################
warnings.simplefilter("ignore", category=FutureWarning)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
torch.set_printoptions(profile="default")

###############################################################################
#                            Timer Helper & Utilities                         #
###############################################################################
def timer_step(step_name: str, last_time: float) -> float:
    """Prints the elapsed time since 'last_time' and returns the current time."""
    current_time = time.time()
    elapsed = current_time - last_time
    print(f"[TIMER] {step_name} took: {elapsed:.6f} seconds")
    return current_time

def log_to_file(prompt: str, response: str, log_file: str = "missions_log.txt"):
    """
    Append the prompt and response to a log file.
    """
    with open(log_file, "a") as file:
        file.write("Mission Log:\n")
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Response: {response}\n")
        file.write("-" * 40 + "\n")

def check_syntax(filepath: str) -> bool:
    """
    Check whether a Python file has valid syntax.
    Returns True if syntax is valid, False otherwise.
    Does NOT actually execute the file.
    """
    try:
        py_compile.compile(filepath, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"Syntax error in '{filepath}': {e}")
        return False

def encode_image_to_base64(image_path):
    """
    Encode an image to a Base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def get_newest_file(directory: str) -> str:
    """
    Get the newest file from the specified directory.
    """
    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        print("No files found in the directory.")
        return None
    newest_file = max(files, key=os.path.getmtime)
    return newest_file

def detect_one_image(image_path: str, server_url: str) -> list:
    """
    Sends a single image to the YOLO server at /detect,
    returns the 'detections' list from the JSON response.

    :param image_path: path to a local image (JPG, PNG, etc.)
    :param server_url: base URL of your YOLO server, e.g. "http://localhost:5000"
    :return: list of detections (dicts) or [] if error
    """
    detect_endpoint = f"{server_url}/detect"
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return []

    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            resp = requests.post(detect_endpoint, files=files, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            # 'detections' in the JSON
            return data.get("detections", [])
        else:
            print(f"[ERROR] YOLO /detect returned status {resp.status_code}")
            return []
    except Exception as e:
        print(f"[ERROR] YOLO detection failed: {e}")
        return []

def get_latest_detection(api_url: str):
    """
    Fetch only the latest detection's 'detections' array from the given API URL.
    """
    try:
        response = requests.get(f"{api_url}/detections/latest")
        if response.status_code == 200:
            latest_detection = response.json()
            detections = latest_detection.get('detections', [])
            return detections
        elif response.status_code == 404:
            return []
        else:
            print(f"Error: Received unexpected status code {response.status_code}")
            print("Response:", response.text)
            return "error"
    except Exception as e:
        print(f"Failed to fetch the latest detection: {e}")

###############################################################################
#                              VLM Wrapper                                    #
###############################################################################
class VLMWrapper:
    def __init__(self,
                 remote_url: str,
                 temperature: float = 0.2,
                 default_prompt: str = "Describe briefly what is going on in this camera image"):
        """
        :param remote_url: Full URL of the remote server endpoint
        :param temperature: Temperature for generation
        :param default_prompt: Default prompt to be used if none is provided
        """
        self.remote_url = remote_url
        self.temperature = temperature
        self.default_prompt = default_prompt

    def send_request(self, img: str = None, prompt: str = None) -> str:
        """
        Sends a POST request with the image (base64-encoded) to the remote server.
        """
        if prompt is None:
            prompt = self.default_prompt

        if not img:
            # If no image path provided, try to get the newest one
            uploads_directory = "/home/liu/bap/uploads"
            image = get_newest_file(uploads_directory)
            print(f"**/n" * 5)

        if not img:
            print("No image file found to send to the VLM.")
            return "No image file found."

        print(f"Using image file: {img}")
        encoded_image = encode_image_to_base64(img)
        if not encoded_image:
            print("Failed to encode image.")
            return "Image encoding failed."

        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "image_base64": encoded_image
        }

        print(f"Sending request to remote: {self.remote_url}")
        try:
            response = requests.post(self.remote_url, json=payload)
        except Exception as e:
            print(f"Could not connect to remote server: {e}")
            return f"Connection error: {e}"

        if response.status_code != 200:
            print(f"Error: remote server responded with status code {response.status_code}")
            print(f"Response content: {response.text}")
            return response.text

        try:
            data = response.json()
            response_text = data.get('response', 'No response found in JSON.')
        except ValueError:
            print("Invalid JSON returned by remote server.")
            return "Invalid JSON returned by remote server."

        print(f"Response from remote server:\n{response_text}")
        return response_text

###############################################################################
#                              LLM Wrapper                                    #
###############################################################################
class LLMWrapper:
    def __init__(self, temperature, model_name: str, api_url: str):
        self.temperature = temperature
        self.model_name = model_name
        self.api_url = api_url  # e.g. "http://localhost:8888"

    def send_request(self, prompt: str, stream=False) -> str:
        """
        Send a request to the local LLM API and return the text response.
        """
        try:
            print(f"Sending request to model '{self.model_name}' with prompt:\n{prompt}\n")

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
                print(f"Error: API responded with status code {response.status_code}")
                print(f"Response content: {response.text}")
                return f"ERROR - status code {response.status_code}"

            data = response.json()
            text = data.get('response', 'No response found')

            print(f"Response from model:\n{text}")
            return text

        except Exception as e:
            print(f"Request failed: {e}")
            return ""

###############################################################################
#                   High-Level Function to Fetch MiniSpec                     #
###############################################################################
def fetch_minispec_from_llm(llm: LLMWrapper,
                            vlm: VLMWrapper,
                            img: str,
                            short_prompt: str,
                            big_prompt: str):
    """
        Calls the LLM (and optionally the VLM) to get Python code directly.
        Returns:
            (python_code_str, vlm_response, llm_time, vlm_time)
        """
    print("Fetching MiniSpec plan from LLM...")
    llm_response = ""
    vlm_response = ""
    llm_time = 0.0
    vlm_time = 0.0

    try:
        # 1) LLM call
        llm_start = time.time()
        print("Sending request to LLM...")
        raw_llm_response = llm.send_request(big_prompt)
        llm_end = time.time()
        llm_time = llm_end - llm_start

        # Clean up the LLM response
        cleaned_response = re.sub(r"<think>.*?</think>\n?", "", raw_llm_response, flags=re.DOTALL)
        # Optional: strip lines like "response: blah"
        cleaned_response = re.sub(r"(?m)^response:.*", "", cleaned_response)
        cleaned_response = re.sub(r"(?m)^Response:.*", "", cleaned_response)

        # If there's a line "response: something", extract it
        match = re.search(r"response:\s*(.*)", raw_llm_response, re.IGNORECASE)
        if match:
            cleaned_response = match.group(1).strip()
        python_code = cleaned_response.strip()
        # 2) VLM call
        vlm_start = time.time()
        print("Sending request to VLM...")
        vlm_response = "Hello, I did not give a response"#vlm.send_request(img)
        vlm_end = time.time()
        vlm_time = vlm_end - vlm_start

        print("LLM's response (raw):", python_code)
        print("LLM's response (clean):", cleaned_response)
        print("VLM's response:", vlm_response)

        # Log for debugging
        log_to_file(short_prompt, raw_llm_response)

        return cleaned_response.strip(), vlm_response, llm_time, vlm_time

    except Exception as e:
        print(f"Error fetching MiniSpec plan: {e}")
        return "", "", llm_time, vlm_time

###############################################################################
#                    Orchestrator: Process a Single Prompt                    #
###############################################################################
def process_prompt(llm: LLMWrapper,
                   vlm: VLMWrapper,
                   prompt: str,
                   image_path: str,
                   detection_api_url: str,
                   index: int,
                   prompt_type: str) -> dict:
    """
    For a single prompt:
      1) Possibly gather scene info or YOLO detection
      2) Build a "big prompt" instructing the LLM to produce Python code
      3) Fetch the Python code
      4) Write that code directly into a Python file (inside the try block)
      5) Return timing and success status
    """
    timings = {}

    # 1) Get scene detection
    step_start = time.time()
    scene_desc = detect_one_image(image_path, detection_api_url)
    #scene_desc = # get_latest_detection(detection_api_url)
    #print(f"Detection {scene_desc}")
    timings["get_latest_detection"] = time.time() - step_start

    # 2) Build the "big prompt"
    step_start = time.time()

    big_prompt = (
        "You are a robot pilot, and you should follow the user's instructions to generate a Python mission plan "
        "to fulfill the given task or provide advice if the input is unclear or unreasonable.\n\n"
        "Your response should carefully consider the system's capabilities, the scene description, and the task description. "
        "The drone is already airborne and connected when executing your plan.\n\n"
        "### System Capabilities\n"
        "The system includes low-level and high-level skills:\n\n"
        "- **High-level skills** (preferred when possible):\n"
        "  - scan(object_name: str): Rotate to find an object.\n"
        "  - scan_abstract(question: str): Rotate to find an abstract object.\n"
        "  - orienting(object_name: str): Adjust orientation to center an object.\n"
        "  - approach(): Move forward toward an object.\n"
        "  - goto(object_name: str): Orient and approach an object.\n\n"
        "- **Low-level skills**:\n"
        "  - move_forward(distance: int), move_backward(distance: int), move_left(distance: int), move_right(distance: int),\n"
        "  - move_up(distance: int), move_down(distance: int), turn_cw(degrees: int), turn_ccw(degrees: int),\n"
        "  - move_in_circle(cw: bool), delay(seconds: int), is_visible(object_name: str),\n"
        "  - object_x(object_name: str), object_y(object_name: str), object_width(object_name: str), object_height(object_name: str), object_dis(object_name: str),\n"
        "  - probe(question: str), log(text: str), take_picture(), re_plan()\n\n"
        "### Scene Description\n"
        f"{scene_desc}\n\n"
        "### Task Description\n"
        f"[A] {prompt}\n\n"
        "### Response Instructions:\n"
        "- Only return the mission logic **without a try statement or any additional wrapping**.\n"
        "- Use **high-level skills** when applicable, and **fallback to low-level skills** only if necessary.\n"
        "- If the task is unclear or unsafe, **log a message instead of executing an action**.\n"
        "- **Do not include explanations or extra formatting**—just return the Python logic.\n\n"
        "### Example Expected Response:\n"
        "if is_visible('target'):\n"
        "    goto('target')\n"
        "    take_picture()\n"
        "else:\n"
        "    scan('target')\n"
        "Now, **generate only the Python mission logic** without explanations or additional wrapping."
    )

    timings["build_prompt"] = time.time() - step_start

    # 3) Call fetch_minispec_from_llm
    step_start = time.time()
    python_code, vlm_answer, llm_time, vlm_time = fetch_minispec_from_llm(
        llm=llm,
        vlm=vlm,
        img=image_path,
        short_prompt=prompt,
        big_prompt=big_prompt
    )
    timings["fetch_pythoncode_overall"] = time.time() - step_start
    timings["llm_time"] = llm_time
    timings["vlm_time"] = vlm_time

    step_start = time.time()

    # Create a folder if needed
    os.makedirs("generated_plans_python_direct", exist_ok=True)
    output_filename = f"generated_plans_python_direct/mission_{prompt_type}_{index}.py"

    # You can adapt your standard template but skip the MiniSpec parser. For example:
    with open(output_filename, 'w') as f:
        f.write('import time\n')
        f.write('import asyncio\n')
        f.write('import datetime\n')
        f.write('import json\n')
        f.write('import os\n')
        f.write('import requests\n\n')
        f.write('# Suppose we have these imported from your drone library\n')
        f.write('from functions_gps import *\n\n')

        f.write('START_SAVING_URL = "http://localhost:5000/start_saving"\n')
        f.write('STOP_SAVING_URL = "http://localhost:5000/stop_saving"\n\n')

        f.write('async def main():\n')
        f.write('    mission_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\n')
        f.write(f'    mission_description = "mission_{prompt_type}_{index}"\n')
        f.write('    mission_directory = f"mission_{mission_description}_{mission_timestamp}"\n')
        f.write('    original_code = "LLM direct python code (no minispec)"\n')
        f.write('    start_data = {\n')
        f.write('        "mission_directory": mission_directory,\n')
        f.write('        "original_code": original_code,\n')
        f.write('        "translated_code": "Direct Python from LLM"\n')
        f.write('    }\n')
        f.write('    requests.post(START_SAVING_URL, json=start_data)\n\n')

        f.write('    await connect_drone()\n')
        f.write('    await ensure_armed_and_taken_off()\n\n')

        f.write('    try:\n')
        # Insert the code from the LLM here
        for line in python_code.split('\n'):
            f.write(f"        {line}\n")
        f.write('    except Exception as e:\n')
        f.write('        print("Caught exception in plan execution:", e)\n\n')
        f.write('    await land_drone()\n')
        f.write('    requests.post(STOP_SAVING_URL)\n')
        f.write('    print("STOP_SAVING_URL")\n\n')

        f.write('if __name__ == "__main__":\n')
        f.write('    result = asyncio.run(main())\n')
        f.write('    print("end")\n')
        f.write('    time.sleep(5)\n')
        f.write('    os._exit(0)\n')

    # Optional syntax check:
    syntax_ok = check_syntax(output_filename)
    if syntax_ok:
        print(f"[INFO] '{output_filename}' has valid Python syntax.")
    else:
        print(f"[ERROR] '{output_filename}' has syntax issues.")

    timings["write_output_time"] = time.time() - step_start


    return {
        "prompt_index": index,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "minispec_code": python_code,
        "vlm_answer": vlm_answer,
        #"parse_success": parse_success,
        #"write_success": write_success,
        "timings": timings,
        "Syntax_check": syntax_ok
    }

###############################################################################
#                                   MAIN                                      #
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Configure model variables via command-line arguments.")
    parser.add_argument("--llm_model_name", type=str, default="llama3.1:8b-instruct-fp16", help="LLM model name")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000", help="YOLO server or detection server")
    parser.add_argument("--vlm_api_url", type=str, default="http://172.83.13.213:8889", help="VLM API URL")
    parser.add_argument("--vlm_model_name", type=str, default="llama3.2-vision", help="VLM model name (unused here but kept for reference)")
    parser.add_argument("--llm_api_url", type=str, default="http://localhost:8888", help="LLM API URL")
    args = parser.parse_args()

    LLM_MODEL_NAME = args.llm_model_name
    YOLO_API_URL = args.api_url
    VLM_API_URL = args.vlm_api_url
    VLM_MODEL_NAME = args.vlm_model_name
    OLLAMA_API_URL = args.llm_api_url

    print("[INFO] Starting the program with settings:")
    print(f"  LLM_MODEL_NAME: {LLM_MODEL_NAME}")
    print(f"  YOLO_API_URL:   {YOLO_API_URL}")
    print(f"  VLM_API_URL:    {VLM_API_URL}")
    print(f"  VLM_MODEL_NAME: {VLM_MODEL_NAME}")
    print(f"  OLLAMA_API_URL: {OLLAMA_API_URL}")

    # Create the LLM/VLM wrappers
    llm = LLMWrapper(temperature=0.7, model_name=LLM_MODEL_NAME, api_url=OLLAMA_API_URL)
    vlm = VLMWrapper(remote_url=f"{VLM_API_URL}/vlm", temperature=0.2)

    simple_prompts = [
        # 1-15: move_forward with different distances
        "Move forward 10 cm",
        "Move forward 20 cm",
        "Move forward 30 cm",
        "Move forward 40 cm",
        "Move forward 50 cm",
        "Move forward 60 cm",
        "Move forward 70 cm",
        "Move forward 80 cm",
        "Move forward 90 cm",
        "Move forward 100 cm",
        "Move forward 110 cm",
        "Move forward 120 cm",
        "Move forward 150 cm",
        "Move forward 180 cm",
        "Move forward 200 cm",

        # 16-23: move_backward
        "Move backward 10 cm",
        "Move backward 20 cm",
        "Move backward 30 cm",
        "Move backward 40 cm",
        "Move backward 50 cm",
        "Move backward 60 cm",
        "Move backward 80 cm",
        "Move backward 100 cm",

        # 24-29: move_left
        "Move left 10 cm",
        "Move left 20 cm",
        "Move left 30 cm",
        "Move left 40 cm",
        "Move left 50 cm",
        "Move left 60 cm",

        # 30-35: move_right
        "Move right 10 cm",
        "Move right 20 cm",
        "Move right 30 cm",
        "Move right 40 cm",
        "Move right 50 cm",
        "Move right 60 cm",

        # 36-41: move_up
        "Move up 5 cm",
        "Move up 10 cm",
        "Move up 15 cm",
        "Move up 20 cm",
        "Move up 25 cm",
        "Move up 30 cm",

        # 42-47: move_down
        "Move down 5 cm",
        "Move down 10 cm",
        "Move down 15 cm",
        "Move down 20 cm",
        "Move down 25 cm",
        "Move down 30 cm",

        # 48-53: turn_cw
        "Turn clockwise 30 degrees",
        "Turn clockwise 45 degrees",
        "Turn clockwise 60 degrees",
        "Turn clockwise 90 degrees",
        "Turn clockwise 120 degrees",
        "Turn clockwise 180 degrees",

        # 54-59: turn_ccw
        "Turn counterclockwise 30 degrees",
        "Turn counterclockwise 45 degrees",
        "Turn counterclockwise 60 degrees",
        "Turn counterclockwise 90 degrees",
        "Turn counterclockwise 120 degrees",
        "Turn counterclockwise 180 degrees",

        # 60-65: approach (high-level 'a')
        "Approach forward",
        "Approach now",
        "Approach carefully",
        "Approach the target",
        "Approach with caution",
        "Approach promptly",

        # 66-70: goto (high-level 'g')
        "Go to 'chair'",
        "Go to 'table'",
        "Go to 'person'",
        "Go to 'door'",
        "Go to 'bottle'",

        # 71-75: scan (abbr 's')
        "Scan for 'chair'",
        "Scan for 'person'",
        "Scan for 'bottle'",
        "Scan for 'book'",
        "Scan for 'apple'",

        # 76-80: scan_abstract (abbr 'sa')
        "Scan abstractly for 'any visible object?'",
        "Scan abstractly for 'largest item in view?'",
        "Scan abstractly for 'any electronics here?'",
        "Scan abstractly for 'a friend?'",
        "Scan abstractly for 'any moving object?'",

        # 81-85: take_picture (abbr 'tp')
        "Take a picture",
        "Take a snapshot",
        "Take a photo",
        "Take a camera shot",
        "Capture an image",

        # 86-90: delay (abbr 'd')
        "Wait for 1000 ms",
        "Wait for 2000 ms",
        "Wait for 3000 ms",
        "Wait for 500 ms",
        "Wait for 4000 ms",

        # 91-95: log (abbr 'l')
        "Log 'Starting operation'",
        "Log 'Mission in progress'",
        "Log 'Target acquired'",
        "Log 'Unable to find object'",
        "Log 'Task completed'",

        # 96-100: probe (abbr 'p')
        "Probe 'What is in front of me?'",
        "Probe 'Do we see a table?'",
        "Probe 'Is the environment safe?'",
        "Probe 'Any obstacles detected?'",
        "Probe 'Is the person friendly?'"
    ]

    medium_prompts = [
        # 1
        "Turn clockwise 90 degrees, then move forward 100 cm",
        # 2
        "Move up 20 cm, then scan for 'table'",
        # 3
        "Scan for 'chair', then approach it",
        # 4
        "Move forward 50 cm, then turn counterclockwise 45 degrees",
        # 5
        "If you see 'bottle', approach it",
        # 6
        "Go to 'chair', then take a picture",
        # 7
        "Approach forward, then log 'Approach complete'",
        # 8
        "Scan abstractly for 'any fruit?', then go to it",
        # 9
        "Move left 30 cm, then move forward 30 cm",
        # 10
        "Turn counterclockwise 180 degrees, then log 'Turned around'",

        # 11
        "If you see 'person', go to 'person'",
        # 12
        "Scan for 'laptop', then if visible, log 'Laptop found'",
        # 13
        "Move forward 60 cm, then take a picture",
        # 14
        "Turn clockwise 45 degrees, then move backward 20 cm",
        # 15
        "Scan for 'apple', approach if found",
        # 16
        "Move up 10 cm, then move forward 10 cm",
        # 17
        "Turn clockwise 90 degrees, then approach forward",
        # 18
        "Go to 'bottle', then wait for 2000 ms",
        # 19
        "Probe 'Is the path clear?', then approach",
        # 20
        "Scan abstractly for 'Is there a door?', then if found, approach",

        # 21
        "Move down 10 cm, then turn counterclockwise 90 degrees",
        # 22
        "If 'chair' is visible, approach forward",
        # 23
        "Log 'Starting medium task', then move right 20 cm",
        # 24
        "Take a picture, then log 'Snapshot taken'",
        # 25
        "Scan for 'person', then if visible, go to 'person'",
        # 26
        "If you see 'cat', log 'Cat detected'",
        # 27
        "Turn clockwise 180 degrees, then move forward 20 cm",
        # 28
        "Go to 'table', then take a picture",
        # 29
        "Scan for 'book', then log 'book found' if visible",
        # 30
        "Approach forward, then approach again",

        # 31
        "Move left 10 cm, then move down 5 cm",
        # 32
        "If 'bottle' is visible, log 'We have a bottle'",
        # 33
        "Probe 'What objects are in front?', then log 'Probing done'",
        # 34
        "Scan abstractly for 'any obstacle?', then move forward 30 cm",
        # 35
        "Move backward 20 cm, then approach forward",
        # 36
        "Turn counterclockwise 90 degrees, log 'Turn done'",
        # 37
        "If 'apple' is visible, go to 'apple'",
        # 38
        "Scan for 'chair', if found, take a picture",
        # 39
        "Delay 1000 ms, then log 'Resuming mission'",
        # 40
        "Move up 30 cm, then move forward 30 cm",

        # 41
        "Scan abstractly for 'any friend here?', then approach if found",
        # 42
        "Turn clockwise 45 degrees, then turn clockwise 45 degrees again",
        # 43
        "If 'table' is visible, move forward 50 cm",
        # 44
        "Log 'Checking environment', then scan for 'person'",
        # 45
        "Move right 50 cm, then turn counterclockwise 90 degrees",
        # 46
        "Probe 'Is it safe to proceed?', then if you see 'door', approach",
        # 47
        "Scan for 'chair', if not visible, log 'No chair found'",
        # 48
        "Go to 'apple', then take a picture",
        # 49
        "Move forward 40 cm, then move backward 40 cm",
        # 50
        "If 'bottle' is visible, turn clockwise 90 degrees",

        # 51
        "Scan abstractly for 'a place to land?', then approach if found",
        # 52
        "Move left 10 cm, then move left 10 cm again",
        # 53
        "Log 'Midway update', then wait for 3000 ms",
        # 54
        "If 'laptop' is visible, approach; else log 'Laptop not seen'",
        # 55
        "Probe 'Any obstacles behind us?', then turn clockwise 180 degrees",
        # 56
        "Move up 10 cm, then move down 10 cm",
        # 57
        "If 'chair' is visible, go to 'chair'; else scan for 'chair'",
        # 58
        "Scan for 'bottle', take a picture if found",
        # 59
        "Wait for 2000 ms, then move forward 100 cm",
        # 60
        "Turn counterclockwise 90 degrees, then approach forward",

        # 61
        "Move forward 20 cm, then log 'Forward done'",
        # 62
        "If 'person' is visible, log 'Hello person'",
        # 63
        "Approach forward, then take a picture",
        # 64
        "Scan abstractly for 'food?', if found, approach",
        # 65
        "Move right 20 cm, then move forward 20 cm",
        # 66
        "If 'dog' is visible, log 'Dog found'; else log 'No dog'",
        # 67
        "Go to 'chair', then delay 1000 ms",
        # 68
        "Turn clockwise 90 degrees, then turn clockwise 90 degrees again",
        # 69
        "Probe 'Are we near the target?', then log 'Probing complete'",
        # 70
        "Scan for 'apple', if visible, go to 'apple'",

        # 71
        "If 'book' is visible, take a picture",
        # 72
        "Move up 10 cm, move forward 10 cm, then move down 10 cm",
        # 73
        "Log 'Checking top view', then move up 20 cm",
        # 74
        "If 'person' is visible, approach; else scan for 'person'",
        # 75
        "Turn counterclockwise 45 degrees, then take a picture",
        # 76
        "Move forward 80 cm, then approach forward",
        # 77
        "If 'door' is visible, go to 'door'",
        # 78
        "Scan abstractly for 'any beverage?', approach if found",
        # 79
        "Delay 500 ms, then log 'Short wait done'",
        # 80
        "Move backward 30 cm, then turn clockwise 90 degrees",

        # 81
        "If 'chair' is visible, log 'Chair in sight'",
        # 82
        "Scan for 'laptop', approach if found",
        # 83
        "Log 'Mid-mission check', then probe 'Are there obstacles?'",
        # 84
        "Turn clockwise 90 degrees, then move right 20 cm",
        # 85
        "If 'cat' is visible, approach forward",
        # 86
        "Approach forward, then approach forward again",
        # 87
        "Scan abstractly for 'tools?', then log 'No tools' if not found",
        # 88
        "If 'bottle' is visible, move forward 20 cm",
        # 89
        "Probe 'Is the area clear?', then approach",
        # 90
        "Scan for 'phone', then take a picture",

        # 91
        "Move forward 10 cm, then move forward 10 cm again",
        # 92
        "If 'apple' is visible, log 'Apple found'; else log 'No apple'",
        # 93
        "Turn clockwise 270 degrees, then approach",
        # 94
        "Move left 40 cm, then move forward 50 cm",
        # 95
        "Delay 2000 ms, then log 'Continuing'",
        # 96
        "Scan abstractly for 'a friend?', take a picture if found",
        # 97
        "If 'laptop' is visible, go to 'laptop'",
        # 98
        "Move down 10 cm, then log 'Moving down done'",
        # 99
        "Turn counterclockwise 270 degrees, then approach forward",
        # 100
        "Scan for 'apple', if not visible, log 'Apple not found'"
    ]

    complex_prompts = [
        # 1
        "Turn in 30-degree increments until you see 'apple', then go to 'apple'",
        # 2
        "Scan for 'chair'; if not found, turn clockwise 45 degrees and scan again until found",
        # 3
        "Move forward 50 cm, then if 'table' is visible, approach it; else turn around",
        # 4
        "Loop turning 45 degrees four times, each time scanning for 'person'; if found, approach forward",
        # 5
        "If 'bottle' is not visible, scan for 'bottle', otherwise log 'Bottle already in view'",
        # 6
        "Move up 20 cm, approach forward, then move down 20 cm, and log 'Done overhead pass'",
        # 7
        "Scan abstractly for 'any clue?', if found, log 'Clue found'; else do a second scan",
        # 8
        "Probe 'Should I go left or right?', then turn accordingly",
        # 9
        "If 'door' is visible, approach; if not, move forward 50 cm, then turn clockwise 90 degrees",
        # 10
        "Perform a 360-degree rotation in steps of 60 degrees, scanning at each step",

        # 11
        "If 'chair' is visible, goto 'chair', else scan for 'chair' up to 3 times",
        # 12
        "Move forward 100 cm, turn clockwise 180 degrees, and move forward 100 cm again",
        # 13
        "Scan for 'person'; if found, take a picture, else log 'No person found' and turn around",
        # 14
        "Try approaching forward 3 times, each time turning 15 degrees right afterwards",
        # 15
        "Loop 6 times scanning for 'apple'; if discovered, approach it and break",
        # 16
        "If 'laptop' is visible, approach; if 'chair' is visible, approach; else turn 180 degrees",
        # 17
        "Turn clockwise 30 degrees, scan for 'bottle', if still not visible, turn again",
        # 18
        "Move left 20 cm, then if 'apple' is visible, goto 'apple'; else probe 'Where is apple?'",
        # 19
        "Delay 2000 ms, then move forward 50 cm, then log 'Step completed'",
        # 20
        "Scan abstractly for 'food item?'; if found, approach, else turn counterclockwise 90 degrees",

        # 21
        "Loop 8 times: move forward 10 cm, then turn clockwise 45 degrees",
        # 22
        "If 'table' is visible, goto 'table' and take a picture; otherwise approach forward",
        # 23
        "Scan for 'bottle', if found log 'Bottle found', else approach forward",
        # 24
        "Turn counterclockwise 90 degrees, move forward 50 cm, then scan abstractly for 'target?'",
        # 25
        "Repeat scanning for 'chair' every 45 degrees until a full 360 is done",
        # 26
        "If 'bottle' is visible, approach; else if 'apple' is visible, approach",
        # 27
        "Move forward 30 cm, check if 'door' is visible; if yes, go to 'door'; if no, turn 90 deg",
        # 28
        "Probe 'Any obstacles?', if yes, move up 20 cm, else approach forward",
        # 29
        "Scan for 'person', if not found, keep turning clockwise 30 degrees in a loop until found",
        # 30
        "Approach forward, take a picture, approach forward again",

        # 31
        "Move upward 10 cm if 'chair' is visible, else move downward 10 cm",
        # 32
        "Turn clockwise 90 degrees, if 'apple' is visible, goto 'apple', else log 'No apple'",
        # 33
        "Delay 1000 ms, log 'Step done', then approach forward",
        # 34
        "Scan abstractly for 'largest object?', then if found, goto it",
        # 35
        "Move forward 20 cm, turn clockwise 90 degrees, move forward 20 cm, take a picture",
        # 36
        "Loop 5 times: approach forward, turn clockwise 45 degrees",
        # 37
        "If 'book' is visible, log 'Book found'; else scan for 'book' up to 2 times",
        # 38
        "Probe 'Is the path blocked?' then move forward if safe, else move up 10 cm",
        # 39
        "Move right 20 cm, then if 'bottle' is visible, approach; else log 'No bottle found'",
        # 40
        "Loop turning 30 degrees until 'chair' is visible, then goto 'chair'",

        # 41
        "Scan abstractly for 'any tool?' if found, approach, else approach forward anyway",
        # 42
        "If 'laptop' is visible, take a picture, then goto 'laptop'; else turn 180 deg",
        # 43
        "Move forward 40 cm, then move backward 40 cm, repeat 2 times",
        # 44
        "Scan for 'person', if not found, re_plan with 'rp' command",
        # 45
        "Turn clockwise 90 degrees, move forward 50 cm, if 'door' is visible, approach",
        # 46
        "Loop 3 times: move forward 20 cm, log 'Forward move', turn clockwise 30 deg",
        # 47
        "If 'bottle' is visible, go to 'bottle', else if 'chair' is visible, go to 'chair', else scan",
        # 48
        "Scan abstractly for 'food?', probe 'Which food?' if found",
        # 49
        "Move up 10 cm, approach forward, move down 10 cm",
        # 50
        "Check if 'apple' is visible; if yes, approach; otherwise scan abstractly for 'apple'",

        # 51
        "Turn counterclockwise 45 degrees, move forward 25 cm, then turn clockwise 45 degrees",
        # 52
        "Loop turning 60 degrees 6 times, each time scanning for 'table'",
        # 53
        "Move forward 30 cm, if 'bottle' is visible, log 'bottle', else log 'no bottle'",
        # 54
        "Scan for 'person', if found, take_picture, else wait 2000 ms",
        # 55
        "If 'door' is visible, approach forward 2 times, else turn around",
        # 56
        "Probe 'What objects do we see?', then log the result",
        # 57
        "Loop 4 times: turn clockwise 90 degrees, scan for 'chair'",
        # 58
        "Move left 20 cm, move forward 20 cm, move right 20 cm",
        # 59
        "If 'cat' is visible, approach, else probe 'Where is cat?'",
        # 60
        "Delay 3000 ms, then approach forward, then log 'Approach complete'",

        # 61
        "Scan abstractly for 'obstacle?'; if found, move up 20 cm, else approach",
        # 62
        "Move backward 40 cm, if 'table' is still visible, approach",
        # 63
        "Turn clockwise 45 degrees, if 'bottle' is visible, goto 'bottle'; else scan again",
        # 64
        "Loop scanning for 'chair' every 30 degrees up to 360",
        # 65
        "Probe 'Is the scene clear?' then approach forward if yes",
        # 66
        "If 'laptop' is visible, approach, else if 'apple' is visible, approach",
        # 67
        "Move forward 60 cm, turn counterclockwise 180 degrees, move forward 60 cm",
        # 68
        "Scan for 'bottle'; if found, log 'Found bottle'; else turn 90 deg and repeat",
        # 69
        "If 'person' is visible, take a picture, then log 'Picture taken'",
        # 70
        "Loop turning 15 degrees 12 times, each time approach forward",

        # 71
        "Turn clockwise 90 degrees, move forward 100 cm, turn counterclockwise 90 degrees",
        # 72
        "If 'book' is visible, goto 'book', else log 'No book'",
        # 73
        "Scan abstractly for 'any computer?'; if found, approach, else re_plan",
        # 74
        "Move forward 50 cm, if 'apple' is visible, approach it, else log 'No apple'",
        # 75
        "Delay 1000 ms, turn clockwise 360 degrees in increments of 60 degrees, scanning at each step",
        # 76
        "If 'door' is visible, approach forward, else approach forward anyway",
        # 77
        "Loop 3 times: approach forward, take a picture",
        # 78
        "Scan for 'laptop', if not found, approach forward and scan again",
        # 79
        "Probe 'Is there a phone here?' then log the answer",
        # 80
        "If 'chair' is visible, goto 'chair', else if 'table' is visible, goto 'table'",

        # 81
        "Move up 10 cm, turn clockwise 90 degrees, move down 10 cm",
        # 82
        "Delay 500 ms, approach forward, then log 'Done approach'",
        # 83
        "If 'apple' is visible, approach; if 'bottle' is visible, approach",
        # 84
        "Loop turning 45 degrees 8 times, scanning abstractly for 'any large object?'",
        # 85
        "Scan abstractly for 'the tallest object?', goto it if found",
        # 86
        "Move forward 30 cm, then if 'door' is visible, take a picture, else approach forward again",
        # 87
        "If 'person' is visible, log 'Hello Person'; else scan for 'person'",
        # 88
        "Turn clockwise 270 degrees, move forward 20 cm, then turn clockwise 90 degrees",
        # 89
        "Loop 2 times: move left 10 cm, move right 10 cm",
        # 90
        "Scan for 'apple'; if found, log 'Apple spotted'; else re_plan",

        # 91
        "Probe 'Any living beings here?', then if 'person' is found, approach",
        # 92
        "If 'chair' is not visible, scan for 'chair', else approach it",
        # 93
        "Move forward 30 cm, move backward 30 cm, then log 'Oscillation done'",
        # 94
        "Loop turning 60 degrees until 'table' is visible or we complete 360 degrees",
        # 95
        "Delay 4000 ms, then approach forward, then take a picture",
        # 96
        "If 'bottle' is visible, approach forward, else move up 10 cm",
        # 97
        "Scan abstractly for 'pet?', if found, approach, else log 'No pet found'",
        # 98
        "Move down 10 cm, turn counterclockwise 45 degrees, move forward 10 cm",
        # 99
        "Loop 4 times: turn clockwise 90 degrees, if 'apple' is visible, goto 'apple' and break",
        # 100
        "If 'chair' is visible, approach, else if 'person' is visible, approach, else log 'No target'"
    ]

    # We’ll store the timing results for each prompt
    all_results = []
    image_index = 0
    images_dir = '/home/liu/Downloads/VisDrone2019-DET-val/images'
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))  # matches .jpg, .png, etc.

    # Process simple prompts
    for i, prompt in enumerate(simple_prompts, start=1):
        image_index += 1
        print(f"\n--- [Simple Prompt {i}] {prompt} ---")
        if i < 5:
            result = process_prompt(
                llm=llm,
                vlm=vlm,
                prompt=prompt,
                image_path=image_files[image_index],
                detection_api_url=YOLO_API_URL,
                index=i + (image_index * 1000),
                prompt_type="simple"
            )
            all_results.append(result)

    # Process medium prompts
    for i, prompt in enumerate(medium_prompts, start=1):
        image_index += 1
        print(f"\n--- [Medium Prompt {i}] {prompt} ---")
        if i < 5:
            result = process_prompt(
                llm=llm,
                vlm=vlm,
                prompt=prompt,
                image_path=image_files[image_index],
                detection_api_url=YOLO_API_URL,
                index=i + (image_index * 1000),
                prompt_type="medium"
            )
            all_results.append(result)

    # Process complex prompts
    for i, prompt in enumerate(complex_prompts, start=1):
        image_index += 1
        print(f"\n--- [Complex Prompt {i}] {prompt} ---")
        if i < 5:
            result = process_prompt(
                llm=llm,
                vlm=vlm,
                prompt=prompt,
                image_path=image_files[image_index],
                detection_api_url=YOLO_API_URL,
                index=i + (image_index * 1000),
                prompt_type="complex"
            )
            all_results.append(result)

    """print("\n\n==================== SUMMARY OF RESULTS ====================")
    for entry in all_results:
        t = entry["timings"]
        print(
            f"Prompt Type: {entry['prompt_type']}, "
            f"Index: {entry['prompt_index']}, "
            f"Prompt: {entry['prompt']}\n"
            f"  parse_success: {entry['parse_success']}, "
            f"write_success: {entry['write_success']}, "
            f"Syntax_ok: {entry['Syntax_check']}\n"
            f"  Timings => get_detection: {t['get_latest_detection']:.3f}s, "
            f"build_prompt: {t['build_prompt']:.3f}s, "
            f"fetch_minispec (overall): {t['fetch_minispec_overall']:.3f}s\n"
            f"    * LLM time: {t['llm_time']:.3f}s, "
            f"    * VLM time: {t['vlm_time']:.3f}s\n"
            f"  translate_time: {t['translate_time']:.3f}s, "
            f"write_output_time: {t['write_output_time']:.3f}s\n"
        )"""

        # ================== Save results to JSON ===================
        # We'll write them to a file called "timing_results_dataset.json"
    with open("timing_results_dataset.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("[INFO] Timing results saved to 'timing_results_dataset.json'.")
    print("[INFO] Done!")

if __name__ == "__main__":
    main()

