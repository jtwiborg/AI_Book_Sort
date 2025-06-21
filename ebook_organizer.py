import os
import json
import shutil
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
    "EBOOK_ROOT_FOLDER": "/path/to/your/ebooks",  # CHANGE THIS
    "LLM_PROVIDER": "Ollama",  # Choose between "OpenAI", "Gemini", "Ollama"
    "API_KEYS": {
        "OPENAI": os.getenv("OPENAI_API_KEY"),
        "GOOGLE": os.getenv("GOOGLE_API_KEY"),
    },
    "OLLAMA_CONFIG": {
        "BASE_URL": "http://localhost:11434",
        "MODEL": "llama3", # The model you have downloaded in Ollama
    },
    "PROCESSING_CONFIG": {
        "CATEGORY_DEPTH": 3,  # How many levels of folders (2 or 3)
        "FLEXIBLE_MODE": True,  # True: Allow LLM to create new categories
        "IS_DRY_RUN": True,  # True: Simulate without moving files. Recommended for the first run!
        "NEEDS_REVIEW": True, # True: Ask for manual approval for each book
    },
    # Predefined structure that the LLM will use as a starting point
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

class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients."""
    @abstractmethod
    def get_analysis(self, prompt: str) -> dict | None:
        raise NotImplementedError

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def get_analysis(self, prompt: str) -> dict | None:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # A good and cost-effective choice
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
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def get_analysis(self, prompt: str) -> dict | None:
        try:
            # Gemini requires JSON instructions to be part of the prompt itself
            full_prompt = f"{prompt}\n\nImportant: Your response must be a valid JSON object, and nothing else."
            response = self.model.generate_content(full_prompt)
            # Remove markdown formatting from Gemini's response
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
            # Ollama returns a JSON string inside another JSON...
            return json.loads(response.json()['response'])
        except Exception as e:
            print(f"ERROR with Ollama API: {e}")
            return None

# --- 3. CORE FUNCTIONALITY (BookProcessor) ---

class BookProcessor:
    def __init__(self, config, llm_client):
        self.config = config
        self.client = llm_client
        self.log = []

    def _build_prompt(self, book_summary: str) -> str:
        """Builds the complex prompt based on configuration."""
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
        """Extracts text from PDF and splits it into chunks to avoid token limits."""
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n\n"
            doc.close()

            # Simple chunking based on length for this example
            max_len = 20000 # Approx. characters per chunk
            for i in range(0, len(full_text), max_len):
                chunks.append(full_text[i:i + max_len])
            
            # Limit the number of chunks to save API costs in this example
            return chunks[:5] 
        except Exception as e:
            self.log.append(f"ERROR: Could not read text from {pdf_path}: {e}")
            return []

    def _get_map_reduce_summary(self, chunks: list[str]) -> str:
        """Creates a summary of summaries (Map-Reduce)."""
        # For this example, we just join the chunks.
        # A full-fledged app would send each chunk for summarization first.
        # This is to save time and API calls in this sketch.
        self.log.append("INFO: Using simplified 'Map-Reduce' method (joining text chunks).")
        return " ".join(chunks)

    def _handle_review(self, metadata: dict) -> dict | None:
        """Handles manual review of the categorization."""
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
                metadata['path'] = [p.strip() for p in new_path_str.split('/')]
                metadata['review_status'] = 'manually_changed'
                print(f"New path set to: {metadata['path']}")
                return metadata
            else:
                print("Invalid choice. Please try again.")

    def _organize_files(self, pdf_path: str, metadata: dict):
        """Creates directories and moves files based on metadata."""
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
        
        pdf_filename = os.path.basename(pdf_path)
        json_filename = os.path.splitext(pdf_filename)[0] + ".json"
        
        log_action = "DRY RUN:" if cfg["IS_DRY_RUN"] else "ACTION:"
        self.log.append(f"{log_action} Target for '{pdf_filename}' -> '{target_folder}'")

        if not cfg["IS_DRY_RUN"]:
            os.makedirs(target_folder, exist_ok=True)
            with open(os.path.join(target_folder, json_filename), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            shutil.move(pdf_path, os.path.join(target_folder, pdf_filename))

    # REPLACE the existing process_all_books method with this one

def process_all_books(self):
    """Main method to find and process all ebooks."""
    root_folder = self.config["EBOOK_ROOT_FOLDER"]
    
    for current_path, _, files in os.walk(root_folder):
        # Avoid re-processing already categorized folders
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
                
                # 1. Extract text
                chunks = self._extract_text_chunks(pdf_path)
                if not chunks:
                    continue
                
                # 2. Create summary (Map-Reduce)
                summary_of_chunks = self._get_map_reduce_summary(chunks)
                
                # 3. Build prompt and get analysis from LLM
                final_prompt = self._build_prompt(summary_of_chunks)
                analysis = self.client.get_analysis(final_prompt)
                
                # --- START OF THE FIX ---
                # Add validation to ensure the LLM response is valid before proceeding.
                if not analysis or not isinstance(analysis, dict):
                    self.log.append(f"ERROR: Received invalid or empty analysis for {filename}. Skipping.")
                    print(f"DEBUG: Invalid analysis received: {analysis}")
                    continue

                # Check for all required keys.
                required_keys = ['path', 'summary', 'keywords']
                if not all(key in analysis for key in required_keys):
                    self.log.append(f"ERROR: LLM response for {filename} was missing one or more required keys ('path', 'summary', 'keywords'). Skipping.")
                    print(f"DEBUG: Malformed analysis received: {analysis}")
                    continue
                # --- END OF THE FIX ---

                # 4. Prepare metadata
                metadata = {
                    "original_filename": filename,
                    "processed_date_utc": datetime.now(timezone.utc).isoformat(),
                    "llm_provider": self.config["LLM_PROVIDER"],
                    **analysis
                }
                
                # 5. Manual review (if enabled)
                final_metadata = metadata
                if self.config["PROCESSING_CONFIG"]["NEEDS_REVIEW"]:
                    final_metadata = self._handle_review(metadata)
                else:
                    final_metadata['review_status'] = 'auto_approved'

                if final_metadata:
                    # 6. Organize the files
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
    # Check if the root folder exists
    if not os.path.isdir(APP_CONFIG["EBOOK_ROOT_FOLDER"]):
        print(f"ERROR: Directory '{APP_CONFIG['EBOOK_ROOT_FOLDER']}' not found.")
        print("Please update EBOOK_ROOT_FOLDER in APP_CONFIG.")
    else:
        # Select and instantiate the correct LLM client
        provider = APP_CONFIG["LLM_PROVIDER"]
        llm_client = None

        if provider == "OpenAI":
            llm_client = OpenAIClient(APP_CONFIG["API_KEYS"]["OPENAI"])
        elif provider == "Gemini":
            llm_client = GeminiClient(APP_CONFIG["API_KEYS"]["GOOGLE"])
        elif provider == "Ollama":
            cfg = APP_CONFIG["OLLAMA_CONFIG"]
            llm_client = OllamaClient(cfg["BASE_URL"], cfg["MODEL"])
        
        if not llm_client:
            print(f"ERROR: Unknown or unconfigured LLM Provider '{provider}'.")
        else:
            print("="*50)
            print("E-BOOK ORGANIZER")
            print(f"Mode: {'DRY RUN' if APP_CONFIG['PROCESSING_CONFIG']['IS_DRY_RUN'] else 'LIVE'}")
            print(f"LLM Provider: {provider}")
            print("="*50)
            
            processor = BookProcessor(APP_CONFIG, llm_client)
            processor.process_all_books()
            processor.print_log()
