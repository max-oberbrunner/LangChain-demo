# Code Analysis Tool

## Project Overview

The Code Analysis Tool is an interactive application designed to analyze code snippets, generate explanations, and provide suggestions for improvement using advanced natural language processing techniques. The tool leverages LangChain and OpenAI's models to understand codebases and facilitate better programming practices.

## Goals

- Understand codebases through vectorization of code snippets.
- Generate clear explanations for functions in the given codebase.
- Provide actionable suggestions for code improvement based on user input queries.
- Enable automated documentation generation in the future.

## Technologies Used

- Python
- LangChain
- OpenAI API (GPT-3.5 / GPT-4)
- Faiss for vector similarity search
- dotenv for environment variable management

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd code_analysis_tool
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the root of the project:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

To run the application, execute the following command:

```bash
python src/main.py
```

This will start the interactive prompt, allowing you to enter queries related to the code snippets in the tool.

## Running Tests

To ensure the functionality of the application, you can run the tests using the following command:

```bash
python -m unittest discover -s tests
```

This will discover and run all the unit tests present in the `tests/` directory.

## Future Development

- Implement automated documentation generation based on function explanations.
- Extend support for additional programming languages.
- Enhance the user interface for better experience and accessibility.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the creators of LangChain and OpenAI for their invaluable tools which make this project possible.