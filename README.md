# AI-Powered E-Book Organizer

A Python script that automatically organizes a collection of PDF e-books into a clean, hierarchical folder structure using Large Language Models (LLMs). It analyzes the content of each book to determine its category, generates a summary and keywords, and stores this information in a searchable metadata file.

## The Idea

Many of us have large collections of e-books, often in a single, messy folder. Finding a specific book or books on a certain topic can be difficult. This program solves that problem by acting as an intelligent librarian. It reads through each PDF, understands its core topics, and then automatically files it away in a logical subfolder, creating a structured and easily navigable library.

## Key Features

  - **Intelligent Categorization**: Uses LLMs (OpenAI, Gemini, or a local Ollama instance) to understand book content, not just filenames.
  - **Hierarchical Structure**: Automatically creates nested folders (e.g., `.../Development/Web Development/React/`).
  - **Configurable Depth**: Choose to organize into 2 or 3 levels of subcategories.
  - **Dynamic Categories**: A "Flexible Mode" allows the LLM to intelligently create new, specific subcategories if a book doesn't fit existing ones.
  - **Rich Metadata Generation**: For each book, it creates a `.json` file containing:
      - A detailed summary.
      - A list of relevant keywords.
      - The category path.
      - Processing information (date, LLM used).
  - **Multi-LLM Support**: Easily switch between different AI providers.
  - **Safe Operation Modes**:
      - **Dry Run**: Simulate the entire process without moving any files to see what the script *would* do.
      - **Manual Review**: A command-line prompt to approve, edit, or skip the LLM's suggested categorization for each book.
  - **Recursive Processing**: Scans all subdirectories of your main e-book folder for new PDFs to process.

## How It Works

The script follows a robust "Map-Reduce" inspired workflow for each PDF file:

1.  **Scan**: The script recursively scans the root e-book folder for any `.pdf` files that do not have a corresponding `.json` metadata file, marking them as "unprocessed".
2.  **Extract & Chunk**: It opens a PDF and extracts its text content using the `PyMuPDF` library. The text is divided into smaller "chunks" to be manageable for the LLM's context window.
3.  **Summarize (Simplified Map-Reduce)**: The text chunks are combined to form a comprehensive overview of the book's content. (A full implementation would summarize each chunk individually before combining them).
4.  **Analyze (LLM Call)**: This combined summary is sent to the selected LLM with a detailed prompt. The prompt instructs the LLM to act as an expert librarian and return a structured JSON object containing the category path, a detailed summary, and keywords.
5.  **Review (Optional)**: If `NEEDS_REVIEW` is enabled, the script presents the LLM's findings to the user, who can approve, edit, or skip the file.
6.  **Organize**: The script creates the required hierarchical folder structure (e.g., `./Security/Application Security (AppSec)/`). It then moves the original `.pdf` file and the newly created `.json` metadata file into the target directory.

## Setup and Installation

1.  **Prerequisites**:

      - Python 3.8 or newer.
      - If using Ollama, ensure it is running and you have downloaded a model (e.g., `ollama run llama3`).

2.  **Download the Script**:

      - Save the code as `ebook_organizer.py`.

3.  **Install Dependencies**:

      - Open your terminal or command prompt in the same directory as the script and run:
        ```bash
        pip install openai google-generativeai pymupdf requests python-dotenv
        ```

4.  **Set Up API Keys**:

      - Create a file named `.env` in the same directory.
      - Add your API keys to this file. The script will load them automatically.
        ```
        OPENAI_API_KEY="your_openai_key_here"
        GOOGLE_API_KEY="your_google_gemini_key_here"
        ```

## Configuration

All settings are controlled from the `APP_CONFIG` dictionary at the top of the `ebook_organizer.py` script.

```python
APP_CONFIG = {
    "EBOOK_ROOT_FOLDER": "/path/to/your/ebooks",
    "LLM_PROVIDER": "Ollama",
    # ... other settings
}
```

| Key | Description |
| :--- | :--- |
| `EBOOK_ROOT_FOLDER` | **(Required)** The absolute path to the main folder containing your e-books. |
| `LLM_PROVIDER` | The AI provider to use. Options: `"OpenAI"`, `"Gemini"`, `"Ollama"`. |
| `OLLAMA_CONFIG` | Contains the URL and `MODEL` name for your local Ollama instance. |
| `CATEGORY_DEPTH` | The number of subfolders to create. Options: `2` or `3`. |
| `FLEXIBLE_MODE` | `True`: Allows the LLM to create new subcategories. `False`: Restricts the LLM to only use categories from `CATEGORY_STRUCTURE`. |
| `IS_DRY_RUN` | `True`: Prints actions to the console without moving files or creating folders. **Highly recommended for the first run.** `False`: Executes all file operations. |
| `NEEDS_REVIEW` | `True`: Prompts you to confirm the categorization for each book. `False`: Approves all categorizations automatically. |
| `CATEGORY_STRUCTURE`| The predefined list of categories and subcategories that guides the LLM. You can edit this to suit your needs. |

## Usage

Once you have configured `APP_CONFIG`, run the script from your terminal:

```bash
python ebook_organizer.py
```

The script will begin scanning for files and processing them according to your configuration.

## Code Documentation

The script is built around a few key classes and methods that handle the logic.

### Main Classes

#### `BookProcessor`

This is the main orchestrator class that manages the entire workflow from finding files to organizing them.

  - `process_all_books()`: The main entry method. It uses `os.walk()` to find all unprocessed PDFs and then calls the processing logic for each one.
  - `_build_prompt()`: Dynamically constructs the detailed prompt sent to the LLM based on the settings in `APP_CONFIG` (e.g., category depth, flexible mode).
  - `_extract_text_chunks()`: Opens a PDF file and extracts its text content.
  - `_get_map_reduce_summary()`: Prepares the extracted text to be sent to the LLM.
  - `_handle_review()`: Manages the interactive command-line prompt for manual approval of categories.
  - `_organize_files()`: Handles the creation of directories and the moving of the PDF and JSON files, correctly interpreting `NEW:` prefixes for dynamic categories.

#### LLM Clients (`BaseLLMClient`, `OpenAIClient`, `GeminiClient`, `OllamaClient`)

These classes handle all communication with the different AI services.

  - `BaseLLMClient`: An abstract base class that defines a common interface (`get_analysis()`) that all clients must implement. This makes it easy to switch between providers.
  - **Specific Clients**: Each client (`OpenAIClient`, etc.) implements the `get_analysis()` method, formatting the request and parsing the response according to its specific API requirements.

## Example Workflow

1.  **Initial State**: You have a file at `/path/to/your/ebooks/unsorted/grokking_algorithms.pdf`.
2.  **Configuration**: You set `IS_DRY_RUN` to `False` and `CATEGORY_DEPTH` to `3`.
3.  **Execution**: You run `python ebook_organizer.py`.
4.  **Process**:
      - The script finds the PDF.
      - It extracts the text and sends it to the LLM.
      - The LLM determines the book is about fundamental algorithms and data structures. It returns a path like `["Architecture & Methodology", "Theoretical Computer Science", "Algorithms & Data Structures"]`.
      - It also returns a summary and keywords.
5.  **Final State**: The script creates the following structure and moves the files:
    ```
    /path/to/your/ebooks/
    └── Architecture & Methodology/
        └── Theoretical Computer Science/
            └── Algorithms & Data Structures/
                ├── grokking_algorithms.pdf
                └── grokking_algorithms.json
    ```
