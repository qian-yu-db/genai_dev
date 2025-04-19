import os
from typing import List
from langchain_core.documents import Document


def recursive_file_loader(
    directory_path: str, file_extensions: list = [".md", ".txt", ".yml", ".yaml"]
) -> List[Document]:
    """Recursively load files with specified extensions from a directory and its subdirectories."""
    documents = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create a Document with metadata
                    relative_path = os.path.relpath(file_path, directory_path)
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "relative_path": relative_path,
                            "file_name": os.path.basename(file_path),
                            "file_type": os.path.splitext(file_path)[1],
                        },
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return documents
