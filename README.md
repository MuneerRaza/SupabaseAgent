# 🧠 Agentic AI System with LangGraph, Supabase & Streamlit

This project implements an intelligent AI agent capable of understanding natural language prompts, interacting with a Supabase database (executing SQL, accessing storage), processing images using Google Gemini Vision, and handling audio files using Groq's Whisper API. The agent is built using Langchain and LangGraph, with a user-friendly web interface powered by Streamlit.

## ✨ Features

* **Natural Language Understanding:** Interprets user prompts to determine intent.
* **Database Interaction:**
    * Connects directly to Supabase PostgreSQL using `psycopg2`.
    * Generates and executes SQL queries (CRUD operations) based on prompts.
    * Handles SQL errors with an LLM-powered correction and retry mechanism.
* **Image Processing:**
    * Retrieves images from Supabase Storage.
    * Uses Google Gemini 1.5 Flash API to extract information from images (specifically receipts, returning JSON).
* **Audio Processing:**
    * Retrieves audio files (URLs) from Supabase tables/storage.
    * Uses Groq's Whisper API for transcription.
    * Uses an LLM (LLaMA 3) for potential translation (e.g., Urdu to English) and summarization/analysis.
* **Agentic Framework:** Uses LangGraph to define a stateful, multi-step execution flow with clear nodes (Rewrite, Route, SQL, Image, Audio, Error Handling).
* **Web Interface:** Provides an intuitive Streamlit frontend for easy interaction, showing simulated agent progress.
* **Configuration:** Uses environment variables for secure handling of API keys and credentials.

## 🛠️ Technology Stack

* **Backend:** Python 3.10+
* **AI Framework:** Langchain, LangGraph
* **LLMs:** Groq API (Meta LLaMA 3 70B), Google Generative AI API (Gemini 1.5 Flash)
* **Audio:** Groq API (Whisper Large v3 via OpenAI SDK interface)
* **Database:** Supabase (PostgreSQL)
