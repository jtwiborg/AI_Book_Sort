# Contributing to AI-Powered E-Book Organizer

First off, thank you for considering contributing to this project! Your help is greatly appreciated.

## How to set up a development environment

1.  **Prerequisites**:
    *   Ensure you have Python 3.8 or newer installed.
    *   If you plan to use a local LLM with Ollama, make sure Ollama is running and you have downloaded a model (e.g., `ollama run llama3`). Refer to the [Ollama website](https://ollama.com/) for setup instructions.

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/jtwiborg/AI_Book_Sort.git
    cd AI_Book_Sort
    ```

3.  **Install Dependencies**:
    *   Open your terminal or command prompt in the project directory and run:
        ```bash
        pip install openai google-generativeai pymupdf requests python-dotenv EbookLib beautifulsoup4
        ```
    *   It's recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        pip install openai google-generativeai pymupdf requests python-dotenv EbookLib beautifulsoup4
        ```

4.  **Set Up API Keys**:
    *   Create a file named `.env` in the root project directory.
    *   Add your API keys to this file if you plan to use OpenAI or Google Gemini. The script will load them automatically.
        ```env
        OPENAI_API_KEY="your_openai_key_here"
        GOOGLE_API_KEY="your_google_gemini_key_here"
        ```
    *   You can obtain API keys from [OpenAI](https://platform.openai.com/api-keys) and [Google AI Studio](https://aistudio.google.com/app/apikey).

## Coding style and conventions

*   **Python Code**: Please follow [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
*   **Linters**: Using a linter like [Flake8](https://flake8.pycqa.org/en/latest/) or [Pylint](https://www.pylint.org/) is encouraged to help maintain code quality and consistency.
    ```bash
    pip install flake8
    flake8 .
    ```
*   **Comments**: Write clear and concise comments to explain complex logic or non-obvious parts of your code. Docstrings should be used for all public modules, classes, and functions.
*   **Main Script**: The core logic of the application is in `ebook_organizer.py`.
*   **Commit Messages**: Write clear and concise commit messages. Start with a verb in the imperative mood (e.g., "Fix bug in X", "Add feature Y").

## How to submit bug reports and feature requests

*   **Use GitHub Issues**: Submit all bug reports and feature requests through [GitHub Issues](https://github.com/jtwiborg/AI_Book_Sort/issues).
*   **Check Existing Issues**: Before creating a new issue, please check if a similar one already exists.
*   **Bug Reports**:
    *   Use a clear and descriptive title.
    *   Provide a detailed description of the bug, including:
        *   Steps to reproduce the bug.
        *   What you expected to happen.
        *   What actually happened.
        *   Your operating system, Python version, and versions of relevant libraries.
        *   Any error messages or stack traces.
*   **Feature Requests**:
    *   Use a clear and descriptive title starting with "Feature Request:".
    *   Provide a clear explanation of the proposed functionality.
    *   Explain why this feature would be beneficial to the project and its users.
    *   If possible, suggest how the feature might be implemented.

## Process for submitting pull requests

1.  **Fork the Repository**: Click the "Fork" button at the top right of the [repository page](https://github.com/jtwiborg/AI_Book_Sort/). This creates your own copy of the project.
2.  **Clone Your Fork**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/jtwiborg/AI_Book_Sort.git
    cd AI_Book_Sort
    ```
3.  **Create a New Branch**: Create a new branch for your changes. Choose a descriptive branch name (e.g., `fix-pdf-parsing-bug`, `add-korean-language-support`).
    ```bash
    git checkout -b your-branch-name
    ```
4.  **Make Your Changes**: Implement your fix or feature.
    *   Ensure your code adheres to the "Coding style and conventions" mentioned above.
    *   Add or update tests if applicable. (Currently, the project does not have a formal testing suite, but if you add new functionality, consider how it could be tested).
    *   Make sure your changes do not break existing functionality.
5.  **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
    ```bash
    git add .
    git commit -m "Brief description of your changes"
    ```
6.  **Push to Your Fork**: Push your changes to your forked repository on GitHub.
    ```bash
    git push origin your-branch-name
    ```
7.  **Open a Pull Request (PR)**:
    *   Go to the original repository on GitHub.
    *   You should see a prompt to create a Pull Request from your new branch. Click it.
    *   If not, go to the "Pull requests" tab and click "New pull request". Choose your fork and branch.
    *   Write a clear and descriptive title for your PR.
    *   In the PR description, explain the changes you've made and why.
    *   Link to any relevant issues (e.g., "Closes #123").
    *   Submit the pull request.
8.  **Review Process**:
    *   Your PR will be reviewed by the maintainers.
    *   Be prepared to answer questions or make further changes based on feedback.
    *   Once approved, your changes will be merged into the main codebase.

Thank you for your contribution!
