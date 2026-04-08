import boto3
import json
import os
import re
from dotenv import load_dotenv

# 1. Force load the .env file from the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "..", ".env")
load_dotenv(dotenv_path=env_path, override=True)

def llm(prompt_text: str, context_data: dict = None) -> dict:
    """
    Sends a prompt and optional context to Amazon Nova via Bedrock.
    Forces the output to be strictly parsed as JSON.
    """
    
    # 2. Get keys and CRITICALLY .strip() them to remove invisible spaces
    key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    region = os.getenv("AWS_REGION", "us-east-1").strip()
    
    # Using Amazon Nova Pro (supports on-demand throughput)
    # This model is available and doesn't require inference profiles
    model_id = "amazon.nova-pro-v1:0"

    if not key or not secret:
        return {"error": "Missing AWS Credentials in .env file"}

    # 3. Create an explicit Session
    try:
        session = boto3.Session(
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=region
        )
        client = session.client("bedrock-runtime")
    except Exception as e:
        return {"error": "Failed to create AWS Session", "details": str(e)}

    # 4. Format Prompt (Engineered to heavily discourage Markdown block output)
    json_instruction = "IMPORTANT: Respond ONLY with a valid JSON object. Do not include markdown formatting like ```json or any conversational text."
    
    if context_data:
        full_prompt = f"DATA CONTEXT:\n{json.dumps(context_data, indent=2)}\n\nINSTRUCTION:\n{prompt_text}\n\n{json_instruction}"
    else:
        full_prompt = f"{prompt_text}\n\n{json_instruction}"

    system_prompt = "You are an expert data processor. Your output must be exclusively raw, valid JSON."

    # 5. Call LLM
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": full_prompt}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.1}
        )
        
        raw_text = response["output"]["message"]["content"][0]["text"].strip()
        
        # 6. Advanced JSON extraction
        # This regex looks for the first { or [ and the last } or ] to capture the whole JSON block
        json_match = re.search(r'(\{.*\}|\[.*\])', raw_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # Fallback if regex somehow misses but text is valid JSON
        return json.loads(raw_text)
        
    except json.JSONDecodeError as e:
        return {
            "error": "Failed to parse LLM output as JSON", 
            "raw_response": raw_text, 
            "details": str(e)
        }
    except Exception as e:
        return {"error": "LLM Invocation Failed", "details": str(e)}

if __name__ == "__main__":
    print("Testing modular LLM connection...")
    
    # Test 1: Prompt Only
    print("\n--- Test 1: Prompt Only ---")
    res1 = llm(prompt_text="Return a JSON with key 'status' and value 'success'.")
    print(json.dumps(res1, indent=2))
    
    # Test 2: Prompt + Context Data
    print("\n--- Test 2: Prompt + Context Data ---")
    sample_context = {
        "patient_id": "EMER-992", 
        "vitals": {"heart_rate": 110, "blood_pressure": "140/90"},
        "symptoms": ["severe headache", "nausea"]
    }
    res2 = llm(
        prompt_text="Analyze the patient data. Return a JSON with 'risk_level' (High, Medium, Low) and 'recommended_action'.", 
        context_data=sample_context
    )
    print(json.dumps(res2, indent=2))