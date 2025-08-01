# Normal Door Opening Intelligence

A RAG (Retrieval-Augmented Generation) powered Q&A system for D&D campaign notes, using Ollama's Deepseek model. This project demonstrates how to create an intelligent chat interface that can answer questions about your D&D campaign by referencing your campaign notes.

## Features

- Vector storage-based document retrieval
- Intelligent context-aware responses using LLM
- Support for Obsidian markdown notes
- Score-threshold filtering for high-quality responses
- Interactive Q&A interface

## Prerequisites

1. Python 3.13 or higher
2. [UV](https://github.com/astral/uv) package manager
3. [Ollama](https://ollama.ai/) installed
4. Deepseek model pulled in Ollama

## Usage
1. Install Ollama and get deepseek running
 ```shell
   ollama run deepseek-r1
```
2. Pull notes project from git
3. The first time we run, we need to set up the vector store
```shell
uv run main.py --notes_dir PATH_TO/content --vectorize
```
4. ```shell uv run main.py```

## Acknowledgments

- Based on the [Normal Door Opening Force D&D campaign](https://github.com/wordlewarriors/normal-door-opening)
- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Ollama](https://ollama.ai/) and Deepseek
