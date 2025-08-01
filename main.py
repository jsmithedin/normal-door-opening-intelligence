import os
import argparse
import sys

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import ObsidianLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from typing import List, Any


class ScoreThresholdRetriever(BaseRetriever):
    retriever: Any = Field(description="Base retriever to filter")
    score_threshold: float = Field(default=0.7, description="Minimum score threshold")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, retriever: Any, score_threshold: float = 0.7) -> None:
        super().__init__(retriever=retriever, score_threshold=score_threshold)

    def _get_relevant_documents(self, query: str) -> List[Any]:
        # First get the documents with scores
        docs_with_scores = self.retriever.get_relevant_documents.invoke(
            query,
            search_kwargs={
                "k": 5,
                "fetch_k": 8,
                "lambda_mult": 0.7,
                "return_score": True,
            },
        )

        # Handle both possible return formats
        if isinstance(docs_with_scores, tuple) and len(docs_with_scores) == 2:
            docs, scores = docs_with_scores
            filtered = [
                (doc, score)
                for doc, score in zip(docs, scores)
                if score >= self.score_threshold
            ]
            return [doc for doc, _ in filtered]
        else:
            # If scores are not returned, just return the documents
            return docs_with_scores

    async def _aget_relevant_documents(self, query: str) -> List[Any]:
        raise NotImplementedError("Async retrieval not implemented")


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="RAG-based Q&A system for D&D campaign notes"
    )
    parser.add_argument(
        "--notes_dir", help="Input file path (required only with --vectorize)"
    )
    parser.add_argument(
        "--filepath",
        help="Alternative input file path (required only with --vectorize)",
    )
    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="Only vectorize the content without starting Q&A session",
    )
    args = parser.parse_args()

    # Only check for notes directory if vectorizing
    if args.vectorize:
        notes_dir = args.notes_dir or args.filepath
        if not notes_dir:
            parser.error("--notes_dir or --filepath is required when using --vectorize")

    return args


def remove_all_files_in_folder(directory: str) -> None:
    os.system(f"rm -rf {directory}/*")


def clean_and_prepare_text(text: str) -> str:
    # Remove unnecessary whitespace
    text = " ".join(text.split())
    # Remove very short segments that might not be meaningful
    if len(text.strip()) < 10:
        return ""
    return text


def setup_vectorstore(notes_dir: str) -> Chroma:
    vectorstore_path = "vectorstore"

    # Create new vectorstore
    print("Creating new vectorstore...")
    loader = ObsidianLoader(path=notes_dir)
    data = loader.load()

    # Clean documents before splitting
    cleaned_data = [doc.model_copy() for doc in data]
    for doc in cleaned_data:
        doc.page_content = clean_and_prepare_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    all_splits = text_splitter.split_documents(cleaned_data)

    # Filter out empty or very short chunks
    all_splits = [split for split in all_splits if len(split.page_content.strip()) > 50]

    # Hard reset to be sure we're clean
    remove_all_files_in_folder(vectorstore_path)

    vectorstore: Chroma = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="deepseek-r1"),
        persist_directory=vectorstore_path,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("Vectorization complete!")
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    vectorstore_path = "vectorstore"
    if not os.path.exists(vectorstore_path) or not os.listdir(vectorstore_path):
        raise ValueError(
            "No existing vectorstore found. Please run with --vectorize first."
        )

    vectorstore: Chroma = Chroma(
        embedding_function=OllamaEmbeddings(model="deepseek-r1"),
        persist_directory=vectorstore_path,
    )
    print("Loaded existing vectorstore!")
    return vectorstore


def main(question: str, vectorstore: Chroma) -> str:
    rag_prompt = PromptTemplate(
        template="""
    You are the lore keeper for a dungeons and dragons campaign.
    Use the following pieces of context to answer the question. If you don't find 
    the answer in the context, just say "I don't know".
    
    When answering, consider these guidelines:
    - Focus on information explicitly stated in the context
    - If multiple contexts provide relevant information, synthesize them
    - Keep answers under 4 sentences
    - If context pieces seem contradictory, mention the inconsistency
    
    Context: {context}
    
    Question: {question}
    
    Answer:
"""
    )
    llm = ChatOllama(model="deepseek-r1", callbacks=[StreamingStdOutCallbackHandler()])

    base_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 8, "lambda_mult": 0.7}
    )
    filtered_retriever = ScoreThresholdRetriever(base_retriever, score_threshold=0.7)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=filtered_retriever,
        chain_type_kwargs={"prompt": rag_prompt},
    )
    return qa_chain.invoke({"query": question})["result"]


if __name__ == "__main__":
    args = get_args()

    if args.vectorize:
        notes_dir = args.notes_dir or args.filepath
        setup_vectorstore(notes_dir)
        print(
            "\nVectorization completed. You can now run the script without arguments to start Q&A session."
        )
        sys.exit(0)

    try:
        vectorstore = load_existing_vectorstore()
    except ValueError as e:
        print(f"Error: {e}")
        print(
            "Hint: Run the script with --vectorize and --notes_dir flags first to create the vectorstore."
        )
        sys.exit(1)

    print("\nVectorstore loaded successfully! You can start asking questions.")
    print("(Type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nEnter your question: ")
        if question.lower() == "quit":
            break
        answer = main(question, vectorstore)
        print("\nAnswer:", answer)
        print("\n" + "-" * 50)
