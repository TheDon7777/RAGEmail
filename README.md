# AI Email Processor for Orders and Inquiries

## Description

This Python script automates the processing of customer emails for a fashion store (or similar business). It uses AI (OpenAI GPT-4o via API) and Retrieval-Augmented Generation (RAG) to classify emails, handle product inquiries, process order requests, check stock availability, and generate appropriate responses. Input data is read from a Google Sheet, and structured results are written to a new Google Sheet.

This project serves as a back-end implementation demonstrating how to leverage LLMs and vector stores for practical business automation tasks.

## Features

*   **Email Classification:** Categorizes incoming emails as "order request" or "product inquiry" using an LLM.
*   **Order Processing:**
    *   Extracts product IDs and quantities from order request emails using an LLM.
    *   Checks product availability against stock data.
    *   Assigns order line status ("created" or "out of stock").
    *   Decrements stock levels for fulfilled items.
*   **Product Inquiry Handling (RAG):**
    *   Uses a FAISS vector store built from product descriptions.
    *   Retrieves relevant product information based on inquiry content.
    *   Generates context-aware responses using an LLM.
*   **Automated Response Generation:**
    *   Generates confirmation emails for successful orders.
    *   Generates emails for partially fulfilled or out-of-stock orders, including **suggestions for similar, in-stock alternative products** using RAG.
    *   Generates informative responses to product inquiries based on catalog data.
*   **Google Sheets Integration:** Reads input data and writes structured output (classification, order status, responses) to separate sheets in a new Google Spreadsheet using the Google Sheets API via a Service Account.
*   **Secure Configuration:** Uses environment variables (optionally loaded from a `.env` file) for API keys and configuration, avoiding hardcoded credentials.

## Technology Stack

*   **Python 3.x**
*   **AI/LLM:** OpenAI API (GPT-4o)
*   **Libraries:**
    *   `openai`: Official OpenAI Python client.
    *   `langchain`, `langchain-community`, `langchain-openai`: For LLM integration, embeddings, and RAG components.
    *   `faiss-cpu`: Vector store for efficient similarity search (RAG).
    *   `gspread`, `gspread-dataframe`: For Google Sheets interaction.
    *   `pandas`: Data manipulation.
    *   `numpy`: Numerical operations.
    *   `python-dotenv`: Loading environment variables from `.env` file.
    *   `google-auth`: Google authentication library (dependency of gspread).
*   **Data Storage:** Google Sheets

## Setup and Installation

1.  **Clone the repository (Optional):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install openai langchain langchain-community langchain-openai faiss-cpu tiktoken google-auth gspread gspread_dataframe httpx pandas numpy python-dotenv
    ```

## Configuration

Configuration is handled via environment variables. You can set these directly in your system or create a `.env` file in the project's root directory.

**1. Create `.env` file (Recommended):**

Create a file named `.env` in the root of the project and add the following variables, replacing the placeholder values:

```dotenv
# .env file

# OpenAI API Key (Required)
# Get from: https://platform.openai.com/account/api-keys
OPENAI_API_KEY="sk-YourActualOpenAIKeyHere"

# Google Service Account Key File Path (Required)
# Path to the JSON key file downloaded from Google Cloud Console
GOOGLE_SERVICE_ACCOUNT_JSON="/path/to/your/service-account-key.json"

# Input Google Sheet ID (Required)
# The ID from the URL of the Google Sheet containing 'products' and 'emails' sheets
GOOGLE_SHEET_ID="YourInputGoogleSheetIDHere"

# Output Google Sheet Name (Optional)
# Defaults to 'AI Email Output' if not set
# OUTPUT_SHEET_NAME="My Custom Output Sheet Name"
```

**2. OpenAI API Key:**

*   Set the `OPENAI_API_KEY` variable with your key obtained from OpenAI.

**3. Google Service Account & Sheets API:**

*   **Create Service Account:**
    *   Go to the Google Cloud Console -> IAM & Admin -> Service Accounts.
    *   Create a new service account (e.g., "ai-email-sheets-processor").
    *   Go to the "Keys" tab for the new service account, click "Add Key" -> "Create new key" -> Select "JSON" and download the key file.
*   **Enable Google Sheets API:**
    *   Go to Google Cloud Console -> APIs & Services -> Library.
    *   Search for "Google Sheets API" and enable it for your project.
*   **Set Environment Variable:** Set `GOOGLE_SERVICE_ACCOUNT_JSON` to the *absolute or relative path* where you saved the downloaded JSON key file.
*   **Share Google Resources:** You **MUST** share the following with the **service account's email address** (found in the JSON key file or Cloud Console):
    *   The **Input Google Sheet** (specified by `GOOGLE_SHEET_ID`): Grant **Viewer** access.
    *   The **Google Drive Folder** where the output sheet will be created: Grant **Editor** access (this allows the service account to create the new sheet file within that folder).

**4. Input Google Sheet Setup:**

*   Set the `GOOGLE_SHEET_ID` variable to the ID of your input sheet.
*   Ensure the input sheet contains two sheets named *exactly* (case-sensitive):
    *   `products`: Must have columns `product ID`, `name`, `category`, `season`, `detailed description`, `stock amount`.
    *   `emails`: Must have columns `email ID`, `subject`, `body`.

    *(Example structure shown in previous conversations)*

## Running the Script

1.  Ensure your virtual environment is activated (if you created one).
2.  Make sure your environment variables are set (either system-wide or via the `.env` file).
3.  Execute the script from the project's root directory:
    ```bash
    python AIEmail.py
    ```
    *(Replace `AIEmail.py` with the actual filename if you saved it differently)*

## Output

The script will:

1.  Print progress information to the console (authentication status, loaded data heads, warnings).
2.  Create a new Google Spreadsheet in your Google Drive (in the folder shared with the Service Account). The name will be based on the `OUTPUT_SHEET_NAME` environment variable or default to "AI Email Output".
3.  Populate the output spreadsheet with four sheets:
    *   `email-classification`: Shows the classification result for each email.
    *   `order-status`: Details each item from processed order requests and its fulfillment status.
    *   `order-response`: Contains the generated email response text for orders.
    *   `inquiry-response`: Contains the generated email response text for inquiries.
4.  Print a shareable link to the newly created output Google Sheet to the console.
