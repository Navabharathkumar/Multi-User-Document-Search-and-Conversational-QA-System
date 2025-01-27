
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional
import logging

DEFAULT_VECTORSTORE_PATH = "./chroma_langchain_db"


def load_pdf(pdf_path: Path) -> List[str]:
    """Load and process a single PDF file asynchronously."""
    if not pdf_path.exists():
        return []
        
    try:
        loader = PyPDFLoader(str(pdf_path))
        return loader.load()
    except Exception as e:
        return []

def process_pdfs(
    pdf_files: List[Path],
    embeddings,
    vectorstore: Optional[Chroma] = None,
    collection_name: str = "rag-chroma"
) -> Chroma:
    """Process multiple PDFs concurrently and update vector store."""
    try:
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        # User access control
            # User access control
        user_access = {
            "user1@example.com": ["Amazon.pdf", "IBM.pdf", "IBM2.pdf"],
            "user2@example.com": ["JPMC.pdf", "Wells.pdf"],
            "user3@example.com": ["Alphabet.pdf", "JPMC.pdf", "IBM.pdf","IBM2.pdf"],
        }

        # Process PDFs and assign user access metadata
        docs = []
        for pdf in pdf_files:
            pdf_name = pdf.name
            pdf_pages = load_pdf(pdf)
            for user, accessible_pdfs in user_access.items():
                if pdf_name in accessible_pdfs:
                    for page in pdf_pages:
                        if isinstance(page, dict):
                            page["metadata"]["user"] = user
                        else:
                            page.metadata["user"] = user
                        docs.append(page)
        # Split text into chunks
    
        text_splits =  text_splitter.split_documents(docs)
        
      
        print("Processing text splits...")
        # Update or create vector store
        print("Updating vector store...")
        try:
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=text_splits,
                    collection_name=collection_name,
                    embedding=embeddings,
                    persist_directory="./chroma_langchain_db",)
                
                    

            else:
                vectorstore.add_documents(text_splits)
          
            return vectorstore
            
        except Exception as e:
            raise
            
    except Exception as e:
        raise

if __name__ == "__main__":

    
    # Get paths
    print("Processing PDFs...")
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"
    pdf_files = list(data_dir.glob("*.pdf"))
    # Validate data directory
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    # from langchain_google_vertexai import VertexAIEmbeddings

    # embeddings = VertexAIEmbeddings(model="text-embedding-004")
    # Initialize embeddings
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
    process_pdfs(pdf_files, embeddings)


    #testing the IR system

    # vectorstore = Chroma(collection_name='rag-chroma', persist_directory="./chroma_langchain_db", embedding_function = embeddings)
    # # Retrieve content from vector store given a query
    # query = "I want to see jp morgan results for this quarter"
    # results = vectorstore.similarity_search(query, k=5,filter= {"user": "user3@example.com"})  # Retrieve top 5 similar documents
    # print(results)
    # print("###############") 
    
    # # Print the results
    # for result in results:
    #     print(result)
        # print(f"Document: {result['document']}")
        # print(f"Score: {result['score']}")
        # print(f"Metadata: {result['metadata']}")