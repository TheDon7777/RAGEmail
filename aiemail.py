################################################################################
# FULL END-TO-END IMPLEMENTATION
################################################################################

# ------------------- STEP 0: ENVIRONMENT SETUP ------------------- #

# Required packages:
# pip install openai langchain langchain-community langchain-openai faiss-cpu tiktoken google-auth gspread gspread_dataframe httpx pandas numpy python-dotenv

# Standard library imports
import os           # For accessing environment variables
import json         # For parsing JSON data returned by the LLM
import sys          # For exiting gracefully if setup fails
from urllib.error import HTTPError # For catching specific web errors when reading Google Sheets

# Third-party imports
import pandas as pd # For data manipulation (DataFrames)
from dotenv import load_dotenv # To load environment variables from a .env file (optional but recommended)
from openai import OpenAI      # The OpenAI client library class

# Vector store libraries
import faiss                                  # Efficient similarity search library
from langchain_openai import OpenAIEmbeddings           # Class for creating text embeddings using OpenAI
from langchain.vectorstores import FAISS                # LangChain wrapper for FAISS vector store

# Google Sheets integration
import gspread                               # Library for interacting with Google Sheets API
from gspread_dataframe import set_with_dataframe # Helper to write pandas DataFrames to gspread sheets
# Removed Colab-specific imports

# ------------------- STEP 1: CONFIGURE API KEYS & READ DATA ------------------- #

# Load environment variables from a .env file if it exists
load_dotenv()

# 1.1 Configure OpenAI API credentials
# Fetch the API key securely from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Check if the key was retrieved successfully
if not OPENAI_API_KEY:
    print("Error: Environment variable OPENAI_API_KEY not found.")
    print("Please set the OPENAI_API_KEY environment variable or create a .env file.")
    sys.exit(1) # Exit if key is missing

# Initialize the OpenAI client (using standard OpenAI endpoint)
# Remove the custom base_url to use the default OpenAI API
client = OpenAI(
    api_key=OPENAI_API_KEY
)

# 1.2 Configure Google Sheets Access (using Service Account)
# Fetch the path to the service account key file from environment variables
# Users need to create a service account, enable Sheets API, share sheets with the service account email,
# and download the JSON key file.
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')

if not SERVICE_ACCOUNT_FILE:
    print("Error: Environment variable GOOGLE_SERVICE_ACCOUNT_JSON not found.")
    print("Please set this variable to the path of your Google Service Account key file.")
    sys.exit(1) # Exit if key file path is missing

try:
    gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
    print("Successfully authenticated with Google Sheets using Service Account.")
except Exception as e:
    print(f"Error authenticating with Google Sheets using {SERVICE_ACCOUNT_FILE}: {e}")
    sys.exit(1)

# 1.3 Read data from source Google Spreadsheet
def read_data_frame(document_id, sheet_name):
    """
    Helper function to read a specific sheet from a Google Spreadsheet as CSV export.
    Includes error handling for missing sheets (HTTP 404).
    """
    export_link = f"https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    try:
        return pd.read_csv(export_link)
    except HTTPError as e:
        if e.code == 404:
            # Raise a more specific error if sheet is not found
            raise ValueError(
                f"Error reading sheet '{sheet_name}' from Google Sheet ID '{document_id}'. "
                f"Received HTTP 404: Not Found.\n\n"
                f"Possible causes:\n"
                f"1. Sheet name '{sheet_name}' is incorrect or misspelled (check case sensitivity!).\n"
                f"2. The Google Sheet (ID: {document_id}) does not have 'Anyone with the link' -> 'Viewer' permissions.\n\n"
                f"Please verify the sheet name and sharing permissions in Google Sheets."
            ) from e
        else:
            # Re-raise other HTTP errors
            raise
    except Exception as e:
        # Catch other potential errors during read_csv (e.g., network issues, parsing errors)
        print(f"An unexpected error occurred while reading sheet '{sheet_name}': {e}")
        raise

# Replace with the target Google Spreadsheet ID (could also be an env var)
# Ensure the Service Account email has VIEW access to this sheet.
document_id = os.getenv('GOOGLE_SHEET_ID', "1s0vyhK-GH_i3l2vmXk4wJKzWaMtUizxiC5yu7BSLg70") # Example fallback
if not document_id:
     print("Error: Environment variable GOOGLE_SHEET_ID not set.")
     sys.exit(1)

print(f"Reading data from Google Sheet ID: {document_id}")

try:
    products_df = read_data_frame(document_id, 'products')
    emails_df = read_data_frame(document_id, 'emails')
except ValueError as e:
    print(e) # Print the helpful error from read_data_frame
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred reading the input sheets: {e}")
    sys.exit(1)

print("Products DF (first 5 rows):")
print(products_df.head())
print("Emails DF (first 5 rows):")
print(emails_df.head())

# ------------------- STEP 2: BUILD A VECTOR STORE FROM PRODUCTS (RAG) ------------------- #

# This step is crucial for the RAG (Retrieval-Augmented Generation) requirement.
# We convert product data into text chunks suitable for semantic search.
# Using a vector store allows us to efficiently find products relevant to customer inquiries
# without including the entire catalog in the LLM prompt, which would exceed token limits.

# 2.1 Create text chunks for each product row
product_texts = []
metadata_list = []

for _, row in products_df.iterrows():
    # Combine relevant fields into a chunk of text
    # Adjust as needed; the key is to embed enough info that queries can match
    txt = (
        f"Product ID: {row['product ID']}\n"
        f"Name: {row['name']}\n"
        f"Category: {row['category']}\n"
        f"Season: {row['season']}\n"
        f"Description: {row['detailed description']}\n"
        f"Stock: {row['stock amount']}"
    )
    product_texts.append(txt)
    metadata_list.append({"product_id": row['product ID']})

# 2.2 Create embeddings + store in FAISS (using standard OpenAI endpoint)

# Initialize embeddings using the standard OpenAI endpoint
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
    # No openai_api_base needed anymore
)

# FAISS is selected as an efficient, in-memory vector store suitable for this scale.
# It allows quick similarity searches needed for RAG.
# We store the 'product_id' in metadata to easily link search results back to specific products.
vectorstore = FAISS.from_texts(product_texts, embedding=embeddings, metadatas=metadata_list)

# ------------------- STEP 3: CLASSIFY EMAILS ------------------- #
# Requirement 1: Classify emails as 'order request' or 'product inquiry'.
# We use GPT-4o for this, as LLMs excel at understanding intent from natural language.
# A simple prompt guides the model to produce one of the two required labels.

def classify_email(subject, body):
    """
    Return 'order request' or 'product inquiry' based on email content.
    Uses the new OpenAI client syntax.
    """
    prompt = f"""
Determine if this email is:
1) "order request"
2) "product inquiry"

Subject: {subject}
Body: {body}

Respond with only "order request" or "product inquiry".
    """
    # Use the client object for the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a classification assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    # Access content using attribute access
    label = response.choices[0].message.content.strip().lower()
    # Basic standardization ensures slight variations in the LLM's response (e.g., casing) are handled.
    if "order" in label:
        return "order request"
    else:
        return "product inquiry"

# 3.1 Apply classification to each email
classification_records = []
for _, row in emails_df.iterrows():
    email_id = row['email ID']
    subj = row['subject']
    body = row['body']
    
    category = classify_email(subj, body)
    classification_records.append({
        "email ID": email_id,
        "category": category
    })

email_classification_df = pd.DataFrame(classification_records, columns=["email ID", "category"])
print("Email Classification:")
display(email_classification_df)

# ------------------- STEP 4: PROCESS ORDER REQUESTS (QUANTITIES, STOCK UPDATES) ------------------- #
# Requirement 2: Process order requests, check stock, update levels, generate responses.

# 4.1 Parse the product IDs & quantities from "order request" emails using GPT-4o
# LLMs are effective at extracting structured data (JSON) from unstructured text (emails).
# We instruct the model to return only 'product_id' and 'quantity'.
def parse_order_request(subject, body):
    """
    Call GPT to parse the 'product_id' and 'quantity' from the email.
    Return a list of dicts, e.g. [ {'product_id': 'SKU001', 'quantity': 2}, ... ]
    Uses the new OpenAI client syntax.
    """
    prompt = f"""
Extract the product requests from the following email in JSON format.
Each item should have "product_id" and "quantity" fields only.

Subject: {subject}
Body: {body}
"""
    # Use the client object for the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    raw_content = response.choices[0].message.content
    
    # Clean the raw content
    cleaned_content = raw_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:] # Remove ```json
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:] # Remove ```
        
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3] # Remove trailing ```
        
    cleaned_content = cleaned_content.strip() # Remove any leading/trailing whitespace after fence removal
    
    try:
        # Attempt parsing the CLEANED content
        extracted_json = json.loads(cleaned_content) 
        if isinstance(extracted_json, list):
            return extracted_json
        else:
            return []
    except Exception as e:
        print(f"Warning: JSON Parsing failed for email subject '{subject}'. Error: {e}. Returning empty list.")
        return []

# 4.2 For each "order request," create order lines with statuses
order_status_records = []

# We maintain an in-memory dictionary (`stock_map`) for efficient stock checking and updates.
# This avoids repeatedly querying the main dataframe within the loop.
stock_map = {row['product ID']: row['stock amount'] for _, row in products_df.iterrows()}

# Filter only "order request" emails based on the previous classification step.
order_emails = email_classification_df[email_classification_df['category'] == 'order request']

for _, row in order_emails.iterrows():
    email_id = row['email ID']
    # find the actual subject/body from emails_df
    orig_email = emails_df[emails_df['email ID'] == email_id].iloc[0]
    
    order_items = parse_order_request(orig_email['subject'], orig_email['body'])
    
    # Process each item identified in the email.
    for item in order_items:
        pid = item.get("product_id", "").strip()
        qty = int(item.get("quantity", 0))
        
        # Check against the current stock level in our map.
        current_stock = stock_map.get(pid, 0)
        if current_stock >= qty and qty > 0:
            status = "created"
            # Crucial step: Decrement stock *immediately* after confirming an item can be fulfilled.
            # This prevents overselling if multiple emails order the same last items.
            stock_map[pid] = current_stock - qty
        else:
            status = "out of stock"
        
        order_status_records.append({
            "email ID": email_id,
            "product ID": pid,
            "quantity": qty,
            "status": status
        })

order_status_df = pd.DataFrame(order_status_records, columns=["email ID", "product ID", "quantity", "status"])
print("Order Status:")
display(order_status_df)

# 4.3 Generate order responses

# Function modified to accept product data, vector store, and current stock map
# to enable suggesting alternatives for out-of-stock items, aligning with GuidelineDoc.md.
def generate_order_response(email_id, products_df, vectorstore, stock_map):
    # Get all lines for this email
    lines = order_status_df[order_status_df['email ID'] == email_id]
    if lines.empty:
        return "No order was placed according to our system. Please let us know how we can help."

    created = []
    out_of_stock = []

    for _, ln in lines.iterrows():
        if ln['status'] == 'created':
            created.append((ln['product ID'], ln['quantity']))
        else:
            # Store product ID, quantity, and retrieve product details for similarity search later
            product_info = products_df[products_df['product ID'] == ln['product ID']].iloc[0]
            out_of_stock.append({
                "pid": ln['product ID'],
                "qty": ln['quantity'],
                "name": product_info['name'],
                "description": product_info['detailed description']
            })

    if not out_of_stock:
        # everything was created
        items_str = "\\n".join([f"- {pid}: {qty}" for (pid, qty) in created])
        return (
            f"Hello! Your order has been successfully created:\\n{items_str}\\n\\n"
            f"Thank you for shopping with us. We will ship your items soon."
        )
    else:
        # some or all items out of stock
        created_str = "\\n".join([f"- {pid}: {qty}" for (pid, qty) in created]) if created else "None"

        # Build the out-of-stock string and generate suggestions
        out_str_parts = []
        suggestion_str_parts = ["\\nWe noticed some items were out of stock. You might like these alternatives instead:"]
        suggestions_added = False

        for item in out_of_stock:
            out_str_parts.append(f"- {item['name']} ({item['pid']}): {item['qty']}")

            # Find similar, in-stock alternatives using RAG
            # This directly addresses the guideline requirement to suggest alternatives.
            # We use the vector store (built in Step 2) to find semantically similar products.
            query = f"{item['name']} {item['description']}"
            try:
                similar_docs = vectorstore.similarity_search(query, k=5)
                alternatives = []
                # Filter for items that are *different* from the OOS item and *in stock* (using the updated stock_map).
                for doc in similar_docs:
                    alt_pid = doc.metadata.get('product_id')
                    # Check if it's a different product and has stock > 0
                    if alt_pid and alt_pid != item['pid'] and stock_map.get(alt_pid, 0) > 0:
                        # Retrieve name for the suggestion
                        alt_name = products_df[products_df['product ID'] == alt_pid].iloc[0]['name']
                        alternatives.append(f"{alt_name} (ID: {alt_pid})")
                        if len(alternatives) >= 2: # Limit to 2 suggestions per item
                            break
                
                if alternatives:
                    suggestions_added = True
                    suggestion_str_parts.append(f"  - Instead of {item['name']} ({item['pid']}): {'; '.join(alternatives)}")

            except Exception as e:
                # Log errors during RAG lookup but don't stop the response generation.
                print(f"Error finding alternatives for {item['pid']}: {e}")
                # Continue without suggestions for this item

        out_str = "\\n".join(out_str_parts)
        suggestions_block = "\\n".join(suggestion_str_parts) if suggestions_added else ""

        return (
            f"Hello! Here is your order update:\\n\\n"
            f"Fulfilled items:\\n{created_str}\\n\\n"
            f"Out of stock:\\n{out_str}\\n\\n"
            f"Unfortunately, we could not fulfill the out-of-stock items. {suggestions_block}\\n\\n"
            f"Please let us know if you'd like us to hold your order until items are restocked or if you'd like to modify it with the suggested alternatives."
        )

order_response_records = []
# Loop through the unique email IDs that had order requests
for email_id in order_emails['email ID'].unique(): # Use unique() to avoid processing same email multiple times if it had multiple order lines
    # Pass the necessary dataframes and vectorstore to the generation function
    resp = generate_order_response(email_id, products_df, vectorstore, stock_map)
    order_response_records.append({
        "email ID": email_id,
        "response": resp
    })

order_response_df = pd.DataFrame(order_response_records, columns=["email ID", "response"])
print("Order Response:")
display(order_response_df)

# ------------------- STEP 5: HANDLE PRODUCT INQUIRIES (RAG) ------------------- #
# Requirement 3: Handle product inquiries using RAG.
# This function leverages the vector store created in Step 2.
def generate_inquiry_response(subject, body):
    """
    1) Use the subject & body as a query for semantic search.
    2) Perform similarity_search against the product vectorstore.
    3) Provide top matching documents (product details) as context to GPT-4o.
    4) Ask GPT-4o to generate a helpful answer based on the query and context.
    This RAG approach ensures responses are grounded in actual product data
    and scales better than putting all product info in the prompt.
    Uses the new OpenAI client syntax.
    """
    user_query = f"{subject}\n{body}"
    # k=5: Retrieve the top 5 most similar product chunks.
    top_docs = vectorstore.similarity_search(user_query, k=5)

    # Combine the content of retrieved documents into a single context string for the LLM.
    context_str = "\\n\\n".join([doc.page_content for doc in top_docs])

    # Construct the prompt for the LLM, clearly separating the customer question and the retrieved context.
    system_msg = "You are a helpful assistant for a fashion store."
    user_msg = f"""
Customer's question:
\"\"\"
{user_query}
\"\"\"

Relevant product info:
\"\"\"
{context_str}
\"\"\"

Please provide a concise, helpful response, referencing details from the context when relevant.
"""
    # Call the LLM with the query and context using the client object.
    # Temperature=0.7 allows for slightly more creative/natural-sounding responses than 0.
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.7
    )
    # Access content using attribute access
    return response.choices[0].message.content.strip()

inquiry_emails = email_classification_df[email_classification_df['category'] == 'product inquiry']
inquiry_response_records = []

for _, row in inquiry_emails.iterrows():
    email_id = row['email ID']
    # find original subject/body
    orig_email = emails_df[emails_df['email ID'] == email_id].iloc[0]
    resp_text = generate_inquiry_response(orig_email['subject'], orig_email['body'])
    inquiry_response_records.append({
        "email ID": email_id,
        "response": resp_text
    })

inquiry_response_df = pd.DataFrame(inquiry_response_records, columns=["email ID", "response"])
print("Inquiry Response:")
display(inquiry_response_df)

# ------------------- STEP 6: WRITE EVERYTHING TO A NEW GOOGLE SPREADSHEET ------------------- #
# Final step: Output all processed dataframes into separate sheets in a new Google Sheet.

# 6.1 Authentication is handled in Step 1.2 now.

# 6.2 Create new output document
# Ensure the Service Account email has EDIT access to the folder where it will create this sheet,
# or grant it explicit create permissions.
OUTPUT_SHEET_NAME = os.getenv('OUTPUT_SHEET_NAME', 'AI Email Output')
try:
    print(f"Creating output Google Sheet: {OUTPUT_SHEET_NAME}...")
    output_document = gc.create(OUTPUT_SHEET_NAME)
    print(f"Output sheet created with ID: {output_document.id}")
    # Optional: Share the sheet with specific users or make it public
    # output_document.share('user@example.com', perm_type='user', role='writer')
    output_document.share('', perm_type='anyone', role='reader') # Share publicly readable
    print(f"Shareable link: {output_document.url}")
except Exception as e:
    print(f"Error creating output Google Sheet: {e}")
    sys.exit(1)

# --- Rename the default sheet instead of deleting it --- #
try:
    if len(output_document.worksheets()) == 1:
        default_sheet = output_document.get_worksheet(0)
        default_sheet.update_title("email-classification")
        default_sheet.update([['email ID', 'category']], 'A1:B1')
        email_classification_sheet = default_sheet
    else:
        email_classification_sheet = output_document.add_worksheet(title="email-classification", rows=100, cols=2)
        email_classification_sheet.update([['email ID', 'category']], 'A1:B1')
except Exception as e:
     print(f"Error setting up 'email-classification' sheet: {e}")
     sys.exit(1)
# ----------------------------------------------------------------- #

# 6.3 Add the *other* required sheets
try:
    order_status_sheet = output_document.add_worksheet(title="order-status", rows=100, cols=4)
    order_status_sheet.update([['email ID', 'product ID', 'quantity', 'status']], 'A1:D1')

    order_response_sheet = output_document.add_worksheet(title="order-response", rows=100, cols=2)
    order_response_sheet.update([['email ID', 'response']], 'A1:B1')

    inquiry_response_sheet = output_document.add_worksheet(title="inquiry-response", rows=100, cols=2)
    inquiry_response_sheet.update([['email ID', 'response']], 'A1:B1')
except Exception as e:
     print(f"Error adding required sheets: {e}")
     sys.exit(1)

# 6.4 Now populate each sheet using gspread_dataframe's set_with_dataframe.
try:
    print("Populating output sheets...")
    set_with_dataframe(email_classification_sheet, email_classification_df, row=2, col=1, include_column_header=False)
    set_with_dataframe(order_status_sheet, order_status_df, row=2, col=1, include_column_header=False)
    set_with_dataframe(order_response_sheet, order_response_df, row=2, col=1, include_column_header=False)
    set_with_dataframe(inquiry_response_sheet, inquiry_response_df, row=2, col=1, include_column_header=False)
except Exception as e:
     print(f"Error populating sheets with data: {e}")
     sys.exit(1)

# 6.5 Sharing is handled after creation in 6.2

print("\nAll tasks completed successfully!")
print(f"Output written to Google Sheet: {output_document.url}")
