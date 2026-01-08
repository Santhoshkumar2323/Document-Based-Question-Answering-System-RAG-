Document-Based Question Answering System (RAG)

This project is a local document question-answering system built on standard Retrieval-Augmented Generation (RAG) components.

It indexes documents from disk, retrieves relevant text chunks, and uses an LLM to generate answers grounded in retrieved content.

Data and scope:

Works only on local PDF and TXT files
Documents are read from a fixed folder on disk
All indexing and deletion is explicit, not automatic

What the system does

1.Document ingestion

Reads PDF and TXT files from a directory
Normalizes text conservatively (no aggressive cleaning)

2.Chunking

Splits documents into fixed-size overlapping chunks
Chunk IDs are deterministic and tied to document content

3.Indexing

Stores embeddings in a persistent vector database
Builds a separate keyword (BM25) index
Maintains a registry file as the source of truth

4.Retrieval

Performs hybrid search:
semantic vector search
keyword search
Deduplicates and reranks results using a cross-encoder

5.Answer generation

Passes retrieved evidence to the LLM
Generates answers with explicit use of evidence when available

6.Traceability

Records which evidence was used or ignored
Flags gaps when documents do not support the question
Outputs a confidence level

Deletion and updates:

The filesystem is the source of truth
Deleting a document file does not immediately delete its vectors
Deletions are applied only when the indexing pipeline is rerun

This is intentional and avoids silent or accidental data loss.

Development note:

The code was produced using LLM-assisted development.
The work here is system assembly, configuration, and validation

How to run:
python main.py index
python main.py query "your question"
python main.py chat
