import os
import json
import shutil
import argparse
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dotenv import load_dotenv

# Import specific libraries for LLMs and PDF reading
import fitz  # PyMuPDF
import openai
import google.generativeai as genai
import requests

# Load environment variables from the .env file
load_dotenv()

# --- 1. MAIN CONFIGURATION ---
# Everything is controlled from here
APP_CONFIG = {
    "EBOOK_ROOT_FOLDER": "/path/to/your/ebooks",
    "LLM_PROVIDER": "Ollama",
    "API_KEYS": {
        "OPENAI": os.getenv("OPENAI_API_KEY"),
        "GOOGLE": os.getenv("GOOGLE_API_KEY"),
    },
    "OPENAI_CONFIG": {
        "MODEL": "gpt-4o-mini"
    },
    "GEMINI_CONFIG": {
        "MODEL": "gemini-1.5-flash"
    },
    "OLLAMA_CONFIG": {
        "BASE_URL": "http://localhost:11434",
        "MODEL": "llama3",
    },
    "PROCESSING_CONFIG": {
        "CATEGORY_DEPTH": 3,
        "FLEXIBLE_MODE": True,
        "IS_DRY_RUN": True,
        "NEEDS_REVIEW": True,
        "MAX_TEXT_CHUNK_LENGTH": 20000,
    },
    "CATEGORY_STRUCTURE": {
        "Development": [
            "Web Development", "Mobile Development", "DevOps & CI/CD",
            "AI & Machine Learning", "Game Development", "Database Development", "API Design"
        ],
        "Programming Languages": [
            "Python", "JavaScript & TypeScript", "Java & JVM", "C# & .NET",
            "C & C++", "Go", "Rust", "SQL", "Shell Scripting"
        ],
        "Infrastructure & SysAdmin": [
            "Cloud Services", "Networking", "Linux & Unix", "Windows Server",
            "Containers & Orchestration", "Virtualization"
        ],
        "Security": [
            "Network Security", "Application Security (AppSec)",
            "Penetration Testing & Ethical Hacking", "Cryptography", "Cloud Security",
            "Security Management & Compliance"
        ],
        "Data Disciplines": [
            "Data Analysis", "Data Engineering",
            "Big Data Technologies", "Data Visualization"
        ],
        "Architecture & Methodology": [
            "Software Architecture", "Project Management & Agile",
            "Clean Code & Best Practices", "Theoretical Computer Science"
        ],
    },
}

# --- 2. LLM CLIENTS (Abstraction and Implementation) ---
# ... (LLM client classes remain the same) ...
class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients."""
    @abstractmethod
    def get_analysis(self, prompt: str) -> dict | None:
        raise NotImplementedError

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key, model_name):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def get_analysis(self, prompt: str) -> dict | None:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert librarian categorizing technical books. You must reply with a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"ERROR with OpenAI API: {e}")
            return None

class GeminiClient(BaseLLMClient):
    def __init__(self, api_key, model_name):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def get_analysis(self, prompt: str) -> dict | None:
        try:
            full_prompt = f"{prompt}\n\nImportant: Your response must be a valid JSON object, and nothing else."
            response = self.model.generate_content(full_prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except Exception as e:
            print(f"ERROR with Gemini API: {e}")
            return None

class OllamaClient(BaseLLMClient):
    def __init__(self, base_url, model):
        self.url = f"{base_url}/api/generate"
        self.model = model

    def get_analysis(self, prompt: str) -> dict | None:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1}
            }
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            return json.loads(response.json()['response'])
        except requests.exceptions.RequestException as e:
            error_message = f"ERROR with Ollama API: {e}"
            if e.response is not None:
                error_message += f"\nResponse Body: {e.response.text}"
            print(error_message)
            return None
        except Exception as e:
            print(f"ERROR processing Ollama response: {e}")
            return None

# --- 3. CORE FUNCTIONALITY (BookProcessor) ---
# ... (BookProcessor class remains the same) ...
class BookProcessor:
    def __init__(self, config, llm_client):
        self.config = config
        self.client = llm_client
        self.log = []

    def _build_prompt(self, book_summary: str) -> str:
        cfg = self.config["PROCESSING_CONFIG"]
        depth = cfg["CATEGORY_DEPTH"]
        mode = "Flexible" if cfg["FLEXIBLE_MODE"] else "Strict"
        category_text = json.dumps(self.config["CATEGORY_STRUCTURE"], indent=2)
        prompt = f"""
        You are an expert librarian analyzing a summary of a technical book.
        **Task:**
        Analyze the summary below and return a JSON object with the following keys: "path", "summary", and "keywords".
        **Rules for 'path':**
        1. 'path' must be a list of strings.
        2. The list depth must be exactly {depth} levels.
        3. The first level must be a main category from the list below.
        4. Use the existing subcategories from the list whenever possible.
        5. Mode: **{mode}**.
           - **Strict mode**: You MUST use the categories from the list.
           - **Flexible mode**: If no existing subcategory is a good match, you may propose a new one. Prefix new categories with "NEW: ". Example: "NEW: Quantum Computing".
        **Rules for 'summary' and 'keywords':**
        1. 'summary': Create a concise summary of the book in 5-10 sentences, based on the text.
        2. 'keywords': Generate a list of 10 relevant, technical keywords.
        **Available Categories:**
        {category_text}
        **Book summary to analyze:**
        ---
        {book_summary}
        ---
        """
        return prompt

    def _extract_text_chunks(self, pdf_path: str) -> list[str]:
        chunks = []
        doc = None
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n\n"
            max_len = self.config["PROCESSING_CONFIG"]["MAX_TEXT_CHUNK_LENGTH"]
            for i in range(0, len(full_text), max_len):
                chunks.append(full_text[i:i + max_len])
            return chunks[:5] 
        except Exception as e:
            self.log.append(f"ERROR: Could not read text from {pdf_path}: {e}")
            return []
        finally:
            if doc:
                doc.close()

    def _get_map_reduce_summary(self, chunks: list[str]) -> str:
        self.log.append("INFO: Using simplified 'Map-Reduce' method (joining text chunks).")
        return " ".join(chunks)

    def _handle_review(self, metadata: dict) -> dict | None:
        print("\n--- MANUAL REVIEW ---")
        print(f"File: {metadata['original_filename']}")
        print(f"Suggested Path: {metadata['path']}")
        print(f"Summary: {metadata['summary']}")
        while True:
            choice = input("Approve (a), edit (e), or skip (s)? ").lower()
            if choice == 'a':
                metadata['review_status'] = 'manually_approved'
                return metadata
            elif choice == 's':
                self.log.append(f"INFO: User skipped file {metadata['original_filename']}.")
                return None
            elif choice == 'e':
                new_path_str = input(f"Enter new path, separated by '/': ")
                edited_path = [p.strip() for p in new_path_str.split('/')]
                expected_depth = self.config["PROCESSING_CONFIG"]["CATEGORY_DEPTH"]
                if len(edited_path) != expected_depth:
                    print(f"ERROR: Path depth is incorrect. Expected {expected_depth} levels, but got {len(edited_path)}. Please try again.")
                    continue
                metadata['path'] = edited_path
                metadata['review_status'] = 'manually_changed'
                print(f"New path set to: {metadata['path']}")
                return metadata
            else:
                print("Invalid choice. Please try again.")

    def _organize_files(self, pdf_path: str, metadata: dict):
        cfg = self.config["PROCESSING_CONFIG"]
        root_folder = self.config["EBOOK_ROOT_FOLDER"]
        path_elements = metadata["path"]
        final_path_elements = []
        for element in path_elements:
            if element.startswith("NEW: "):
                clean_element = element.replace("NEW: ", "").strip()
                final_path_elements.append(clean_element)
                self.log.append(f"INFO: New category '{clean_element}' was dynamically created.")
            else:
                final_path_elements.append(element)
        target_folder = os.path.join(root_folder, *final_path_elements)
        original_pdf_filename = os.path.basename(pdf_path)
        original_json_filename = os.path.splitext(original_pdf_filename)[0] + ".json"
        current_pdf_name = original_pdf_filename
        current_json_name = original_json_filename
        counter = 1
        while os.path.exists(os.path.join(target_folder, current_pdf_name)) or \
              os.path.exists(os.path.join(target_folder, current_json_name)):
            base_pdf, ext_pdf = os.path.splitext(original_pdf_filename)
            current_pdf_name = f"{base_pdf}_copy{counter}{ext_pdf}"
            base_json, ext_json = os.path.splitext(original_json_filename)
            current_json_name = f"{base_json}_copy{counter}{ext_json}"
            counter += 1
        if current_pdf_name != original_pdf_filename:
            self.log.append(f"INFO: PDF '{original_pdf_filename}' will be saved as '{current_pdf_name}' due to conflict.")
        if current_json_name != original_json_filename:
            self.log.append(f"INFO: JSON '{original_json_filename}' will be saved as '{current_json_name}' due to conflict.")
        log_action = "DRY RUN:" if cfg["IS_DRY_RUN"] else "ACTION:"
        if current_pdf_name != original_pdf_filename:
            log_message = f"{log_action} Target for '{original_pdf_filename}' (as '{current_pdf_name}') -> '{target_folder}'"
        else:
            log_message = f"{log_action} Target for '{current_pdf_name}' -> '{target_folder}'"
        self.log.append(log_message)
        if not cfg["IS_DRY_RUN"]:
            os.makedirs(target_folder, exist_ok=True)
            with open(os.path.join(target_folder, current_json_name), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            shutil.move(pdf_path, os.path.join(target_folder, current_pdf_name))

    def process_all_books(self):
        root_folder = self.config["EBOOK_ROOT_FOLDER"]
        for current_path, _, files in os.walk(root_folder):
            if any(cat in current_path for cat in self.config["CATEGORY_STRUCTURE"]):
                if current_path != root_folder:
                    continue
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(current_path, filename)
                    json_path = os.path.splitext(pdf_path)[0] + ".json"
                    if os.path.exists(json_path):
                        continue
                    self.log.append(f"--- Processing new file: {filename} ---")
                    chunks = self._extract_text_chunks(pdf_path)
                    if not chunks:
                        continue
                    summary_of_chunks = self._get_map_reduce_summary(chunks)
                    final_prompt = self._build_prompt(summary_of_chunks)
                    analysis = self.client.get_analysis(final_prompt)
                    if not analysis or not isinstance(analysis, dict):
                        self.log.append(f"ERROR: Received invalid or empty analysis for {filename}. Skipping.")
                        print(f"DEBUG: Invalid analysis received: {analysis}")
                        continue
                    required_keys = ['path', 'summary', 'keywords']
                    if not all(key in analysis for key in required_keys):
                        self.log.append(f"ERROR: LLM response for {filename} was missing one or more required keys ('path', 'summary', 'keywords'). Skipping.")
                        print(f"DEBUG: Malformed analysis received: {analysis}")
                        continue
                    metadata = {
                        "original_filename": filename,
                        "processed_date_utc": datetime.now(timezone.utc).isoformat(),
                        "llm_provider": self.config["LLM_PROVIDER"],
                        **analysis
                    }
                    final_metadata = metadata
                    if self.config["PROCESSING_CONFIG"]["NEEDS_REVIEW"]:
                        final_metadata = self._handle_review(metadata)
                    else:
                        final_metadata['review_status'] = 'auto_approved'
                    if final_metadata:
                        self._organize_files(pdf_path, final_metadata)
                    time.sleep(2)
    def print_log(self):
        print("\n" + "="*50)
        print("PROCESSING COMPLETE - LOG")
        print("="*50)
        for entry in self.log:
            print(entry)
        print("="*50)

# --- 4. MAIN ENTRY POINT ---

if __name__ == "__main__":
    # Store original APP_CONFIG values for comparison
    original_ebook_folder = APP_CONFIG["EBOOK_ROOT_FOLDER"]
    original_llm_provider = APP_CONFIG["LLM_PROVIDER"]
    original_model = ""
    if original_llm_provider == "OpenAI":
        original_model = APP_CONFIG["OPENAI_CONFIG"]["MODEL"]
    elif original_llm_provider == "Gemini":
        original_model = APP_CONFIG["GEMINI_CONFIG"]["MODEL"]
    elif original_llm_provider == "Ollama":
        original_model = APP_CONFIG["OLLAMA_CONFIG"]["MODEL"]
    original_category_depth = APP_CONFIG["PROCESSING_CONFIG"]["CATEGORY_DEPTH"]
    original_flexible_mode = APP_CONFIG["PROCESSING_CONFIG"]["FLEXIBLE_MODE"]
    original_is_dry_run = APP_CONFIG["PROCESSING_CONFIG"]["IS_DRY_RUN"]
    original_needs_review = APP_CONFIG["PROCESSING_CONFIG"]["NEEDS_REVIEW"]


    parser = argparse.ArgumentParser(description="Organize ebook files based on metadata and command-line configurations.")
    parser.add_argument("--ebook_folder", default=APP_CONFIG["EBOOK_ROOT_FOLDER"], help="Source directory containing ebook files. Overrides EBOOK_ROOT_FOLDER in APP_CONFIG.")
    parser.add_argument("--llm_provider", choices=["OpenAI", "Gemini", "Ollama"], default=APP_CONFIG["LLM_PROVIDER"], help="LLM provider to use. Overrides LLM_PROVIDER in APP_CONFIG.")
    parser.add_argument("--model", help="LLM model to use. Overrides model in APP_CONFIG for the selected provider.")
    parser.add_argument("--category_depth", type=int, default=APP_CONFIG["PROCESSING_CONFIG"]["CATEGORY_DEPTH"], help="Number of category levels for subfolders. Overrides CATEGORY_DEPTH in APP_CONFIG.")
    parser.add_argument("--flexible_mode", action=argparse.BooleanOptionalAction, default=APP_CONFIG["PROCESSING_CONFIG"]["FLEXIBLE_MODE"], help="Allow LLM to create new categories (e.g. --flexible_mode / --no-flexible_mode). Overrides FLEXIBLE_MODE in APP_CONFIG.")
    parser.add_argument("--is_dry_run", action=argparse.BooleanOptionalAction, default=APP_CONFIG["PROCESSING_CONFIG"]["IS_DRY_RUN"], help="Simulate without moving files (e.g. --is_dry_run / --no-is_dry_run). Overrides IS_DRY_RUN in APP_CONFIG.")
    parser.add_argument("--needs_review", action=argparse.BooleanOptionalAction, default=APP_CONFIG["PROCESSING_CONFIG"]["NEEDS_REVIEW"], help="Ask for manual approval for each book (e.g. --needs_review / --no-needs_review). Overrides NEEDS_REVIEW in APP_CONFIG.")

    # TEST 8: No Arguments (Defaults)
    print("--- TEST 8: No Arguments (Defaults) ---")
    test_args_list = [] # No arguments
    args = parser.parse_args(test_args_list)

    APP_CONFIG["EBOOK_ROOT_FOLDER"] = args.ebook_folder
    APP_CONFIG["LLM_PROVIDER"] = args.llm_provider
    APP_CONFIG["PROCESSING_CONFIG"]["CATEGORY_DEPTH"] = args.category_depth
    APP_CONFIG["PROCESSING_CONFIG"]["FLEXIBLE_MODE"] = args.flexible_mode
    APP_CONFIG["PROCESSING_CONFIG"]["IS_DRY_RUN"] = args.is_dry_run
    APP_CONFIG["PROCESSING_CONFIG"]["NEEDS_REVIEW"] = args.needs_review

    current_model_in_config = ""
    if args.llm_provider == "OpenAI":
        # If args.model is None, it means no override was given, so use the default from initial APP_CONFIG
        APP_CONFIG["OPENAI_CONFIG"]["MODEL"] = args.model if args.model is not None else original_model
        current_model_in_config = APP_CONFIG["OPENAI_CONFIG"]["MODEL"]
    elif args.llm_provider == "Gemini":
        APP_CONFIG["GEMINI_CONFIG"]["MODEL"] = args.model if args.model is not None else original_model
        current_model_in_config = APP_CONFIG["GEMINI_CONFIG"]["MODEL"]
    elif args.llm_provider == "Ollama":
        APP_CONFIG["OLLAMA_CONFIG"]["MODEL"] = args.model if args.model is not None else original_model
        current_model_in_config = APP_CONFIG["OLLAMA_CONFIG"]["MODEL"]

    print(f"APP_CONFIG['LLM_PROVIDER'] (Expected: {original_llm_provider}): {APP_CONFIG['LLM_PROVIDER']}")
    print(f"Model for {APP_CONFIG['LLM_PROVIDER']} (Expected: {original_model}): {current_model_in_config}")
    print(f"APP_CONFIG['EBOOK_ROOT_FOLDER'] (Expected: {original_ebook_folder}): {APP_CONFIG['EBOOK_ROOT_FOLDER']}")
    print(f"APP_CONFIG['PROCESSING_CONFIG']['CATEGORY_DEPTH'] (Expected: {original_category_depth}): {APP_CONFIG['PROCESSING_CONFIG']['CATEGORY_DEPTH']}")
    print(f"APP_CONFIG['PROCESSING_CONFIG']['FLEXIBLE_MODE'] (Expected: {original_flexible_mode}): {APP_CONFIG['PROCESSING_CONFIG']['FLEXIBLE_MODE']}")
    print(f"APP_CONFIG['PROCESSING_CONFIG']['IS_DRY_RUN'] (Expected: {original_is_dry_run}): {APP_CONFIG['PROCESSING_CONFIG']['IS_DRY_RUN']}")
    print(f"APP_CONFIG['PROCESSING_CONFIG']['NEEDS_REVIEW'] (Expected: {original_needs_review}): {APP_CONFIG['PROCESSING_CONFIG']['NEEDS_REVIEW']}")

    print("--- TEST 8 COMPLETE ---")
    exit()
