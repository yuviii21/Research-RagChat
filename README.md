# Research RagChat: AI-Powered Research Paper Chatbot

## Run the interactive UI

The project now includes a Streamlit interface for uploading a research PDF, indexing it locally for the session, and asking cited questions about its contents.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="your-api-key"
streamlit run app.py
```

Open the local URL printed by Streamlit. You can also paste an API key into the app's sidebar; it is used only in the current browser session. Upload a PDF, choose **Process paper**, then ask questions in the chat. The Evidence desk shows the source passages behind the latest answer.

## Problem Statement:
In the rapidly evolving landscape of academic research, students and researchers often face challenges in efficiently navigating and comprehending extensive research papers. Research RagChat addresses this issue by providing an AI-driven platform that simplifies the process of extracting insights and engaging with research content.

## Context:
Utilizing advanced machine learning, natural language processing, and generative AI, Research RagChat automates the interaction with research papers, enabling users to ask questions and receive tailored responses. This tool enhances understanding, promotes deeper engagement with academic literature, and streamlines the research process.

## Objectives:
Research RagChat has two primary objectives:
Research Understanding and Insight Extraction: The platform offers clear and concise answers, summaries, and explanations based on user inquiries related to uploaded research papers.
Interactive Research Assistance: Users can engage with the chatbot to clarify concepts, explore methodologies, and gain insights into specific sections of research papers.
How It Works:
Document Upload and Processing: Users upload research papers in PDF format, which the application processes to extract textual content.
Text Embedding and Semantic Search: The extracted text is transformed into numerical embeddings, allowing for efficient semantic search and retrieval of relevant information.
Question Answering: User queries are processed and matched against the indexed content, with the OpenAI Large Language Model (LLM) generating comprehensive responses based on the most relevant information.

## Architecture:

![image](https://github.com/user-attachments/assets/15e5661b-4648-40e9-b1fb-a7c9e2d14cf8)

Step 1: Data Collection: Upon document upload, Research RagChat extracts data from the research paper and divides it into manageable chunks for processing.
Step 2: Embeddings Generation: The textual data is converted into numerical embeddings, capturing semantic relationships to facilitate efficient information retrieval.
Step 3: Query Execution: After processing, Research RagChat utilizes the OpenAI LLM to generate responses, incorporating both the context of the research paper and the user's query.

## Product Report:
Research RagChat represents a significant advancement in leveraging AI to enhance the research experience for students and academics. By providing immediate access to insights and facilitating deeper engagement with research content, it empowers users to excel in their academic endeavors and fosters a more interactive learning environment.
