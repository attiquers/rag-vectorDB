-----

# 🧠 Langchain Chatbot

This project is a Streamlit-based chatbot that leverages Langchain to provide answers to your questions based on content extracted from provided web URLs. You can choose to use either Google's Gemini models or OpenAI's models for the AI backend.

-----

🚀 Live Demo

Experience the Langchain Chatbot live at:
https://rag-vectordb-j5dhjhcn4vxfhdnbnnwjn7.streamlit.app/

-----
## ✨ Features

  * **Dynamic Document Loading**: Load content from multiple web URLs to use as a knowledge base.
  * **Flexible AI Models**: Choose between **Gemini** (via `gemini-2.0-flash`) and **OpenAI** (via `gpt-4o-mini`) as your large language model.
  * **Contextual Understanding**: The chatbot uses the loaded document context and chat history to provide relevant answers.
  * **Relevance Validation**: An intelligent validation step ensures the AI's answer is relevant to the provided context.
  * **Interactive Chat Interface**: A user-friendly Streamlit interface for seamless interaction.
  * **API Key Management**: Securely input your Gemini or OpenAI API keys within the app.

-----

## 🚀 Getting Started

Follow these steps to set up and run the Langchain Chatbot.

### Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.9+**
  * **Poetry** (recommended for dependency management) or `pip`

You will also need API keys for either:

  * **Google Gemini API Key**: Obtain one from [Google AI Studio](https://ai.google.dev/).
  * **OpenAI API Key**: Obtain one from [OpenAI](https://platform.openai.com/account/api-keys).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

    (Replace `your-username/your-repo-name.git` with your actual repository information)

2.  **Install dependencies:**

    If using **Poetry**:

    ```bash
    poetry install
    poetry shell
    ```

    If using **pip**:

    ```bash
    pip install -r requirements.txt
    ```

    (You'll need to create a `requirements.txt` file first, see the "Dependencies" section below.)

3.  **Create a `.env` file:**
    Create a file named `.env` in the root directory of your project to store environment variables. While the Streamlit app allows direct API key input, it's good practice to have this for other potential environment variables or if you prefer to load keys directly from here.

    ```dotenv
    # .env
    # GOOGLE_API_KEY="your_gemini_api_key_here"
    # OPENAI_API_KEY="your_openai_api_key_here"
    ```

    *Note: The Streamlit app will prioritize keys entered directly into the UI. You do not strictly need to set these in `.env` if you prefer to paste them in the app.*

### Running the Application

Once the dependencies are installed, you can run the Streamlit application:

```bash
streamlit run your_app_file_name.py
```

(Replace `your_app_file_name.py` with the actual name of your Python script, e.g., `app.py` or `main.py`).

Your browser will automatically open to the Streamlit application.

-----

## 💡 Usage

1.  **Enter API Keys**: On the left sidebar, enter your **Gemini API Key** or **OpenAI API Key** depending on which model you plan to use.
2.  **Choose AI Model**: Select "Gemini" or "OpenAI" using the radio buttons.
3.  **Enter Web URLs**: In the text area, paste the URLs of the web pages you want the chatbot to read from. Enter one URL per line.
4.  **Load Documents**: Click the "🔍 Load Documents and Initialize Chat" button. The application will fetch content from the URLs, split it into chunks, and create a searchable vector store.
5.  **Start Chatting**: Once the documents are loaded, a "💬 Ask a question" input field will appear. Type your question and press "Send ✈️". The chatbot will retrieve relevant information from the loaded documents and respond.
6.  **Review Chat History**: Your conversation history will be displayed below the chat input, with the most recent interactions at the top.

-----

## 🛠️ Project Structure

  * `your_app_file_name.py` (e.g., `app.py` or `main.py`): The main Streamlit application script containing all the logic.
  * `.env`: (Optional) Stores API keys and other environment variables.
  * `requirements.txt`: (If using pip) Lists all the Python dependencies.

-----

## ⚙️ Dependencies

Below are the key libraries used in this project. You can generate a `requirements.txt` file from the following:

```
streamlit
python-dotenv
requests
langchain==0.3.26
langchain-community==0.3.27
langchain-core==0.3.68
langchain-google-genai==2.1.6
langchain-openai
langchain-text-splitters==0.3.8
langsmith==0.4.4
sentence-transformers
faiss-cpu
beautifulsoup4
html2text
tqdm
pandas
```

To create `requirements.txt` from this list, you can manually copy and paste, or if you are using Poetry, you can export it:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

-----
