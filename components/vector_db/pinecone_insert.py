from typing import List, Optional

from langflow.base.data import BaseFileComponent
from langflow.base.data.utils import parallel_load_data
from langflow.io import BoolInput, IntInput, SecretStrInput, FileInput, NestedDictInput, Output
from langflow.base.data.utils import TEXT_FILE_TYPES
from langflow.schema.message import Message
from langflow.schema import Data, DataFrame
from pydantic import BaseModel, Field


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

class PineconeIndexerComponent(BaseFileComponent):
    """Process and index documents to Pinecone with parallel processing support.
    
    This component handles multiple files concurrently, creates embeddings,
    and indexes them to Pinecone while returning processed document data.
    """
    
    display_name = "Pinecone Inserter"
    description = """This component processes and indexes documents to Pinecone.

        To ensure proper categorization and retrieval, please include the following required metadata fields when uploading a document:
        
        • source: Origin of the document (e.g., website, internal upload)  
        • user_id: Your IVC email address  
        • client: Name of the client the document pertains to  
        • title: Descriptive title of the document  
        • tag: List of relevant tags (e.g., ["cannabis", "compliance"])
        
        Example input:
        {
          "source": "email",
          "user_id": "jane.doe@ivc.media",
          "client": "GreenWell Industries",
          "title": "2024 Market Strategy Overview",
          "tag": ["strategy", "Q2", "marketing"]
        }"""
    icon = "database"
    name = "PineconeIndexer"

    
    inputs = [
        FileInput(
            name="file",
            display_name="File",
            file_types=TEXT_FILE_TYPES,
            required=True,
        ),
        NestedDictInput(
            name="metadata",
            display_name="Metadata",
            tool_mode=True,
        ),
        SecretStrInput(
            name="pinecone_api_key",
            display_name="Pinecone API Key",
            required=True,
            info="Your Pinecone API key"
        ),
        SecretStrInput(
            name="openai_api_key",
            display_name="OpenAI API Key",
            required=True,
            info="Your OpenAI API key for embeddings"
        ),
    ]
    
    outputs = [
        
        Output(
            display_name="DataFrame",
            name="dataframe",
            method="process_files",
            info="A DataFrame built from each Data object's fields plus a 'text' column.",
        ),
    ]


    def process_files(self) -> DataFrame:
        
        # validate Metadata
        class Schema(BaseModel):
            source: str = Field(description = 'where this document came from')
            user_id: str = Field(description = 'ivc email of uploader')
            client: str = Field(description = 'client this document is for')
            title: str = Field(description = 'the title of the document')
            tag: list = Field(description = 'list of tags of the document e.g. ["cannabis"]')
                    
            class Config:
                extra = "allow"
            
            
        valid_meta = Schema(**self.metadata)
        
        file_list = [self.file] if isinstance(self.file, str) else self.file

        """Process multiple files with optional parallel processing.
        
        Args:
            file_list (List[BaseFileComponent.BaseFile]): List of files to process
            
        Returns:
            List[Optional[Data]]: Processed data for each file, or None if processing fails
        """
        if not file_list:
            msg = "No files to process."
            raise ValueError(msg)

        try:
            # Initialize Pinecone and OpenAI
            pc = Pinecone(api_key=self.pinecone_api_key)
            pinecone_index = pc.Index("quickstart")
            
            embed_model = OpenAIEmbedding(
                model="text-embedding-ada-002",
                api_key=self.openai_api_key
            )
            
            # Set up vector store
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Convert file objects to file paths
            file_paths = file_list
            self.log(file_paths)
            print(file_paths)

            # Load documents from file paths
            print("trying to make the documents using llama")
            reader = SimpleDirectoryReader(input_files=file_paths)
            
            def custom_metadata(filename):

                default_metadata = default_file_metadata_func(filename, fs=reader.fs)
                
                # Add custom metadata
                custom_metadata = self.metadata
                
                # Merge default and custom metadata
                merged_metadata = {**default_metadata, **custom_metadata}
                
                return merged_metadata
            
            reader.file_metadata = custom_metadata
            documents = reader.load_data()
            print("we made the documents!!!!!")
           
            
            # Process each document and collect results
            processed_data = []
            print("starting the loop of documents")
            i=0
            for doc in documents:
                i+=1
                print("dealing with document", i)
                try:
                    # Extract file path and metadata
                    print("getting the filepath: ")
                    file_path = doc.metadata.get("file_path", "Unknown")
                    print(file_path)
                    
                    print("________________")
                    print("structuring the data:")
                    data = Data(
                        text=doc.text,
                        data={
                            "metadata": doc.metadata,
                            "file_path": file_path,
                        }
                    )
                    print("made the data!!!!!")
                    print("appending the data")
                    processed_data.append(data)
                    print("appended the data!!!!")
                except Exception as e:
                    print(f"Error processing document: {str(e)}")
                    self.log(f"Error processing document: {str(e)}")
            print("all done with all", i, " documents!!!")
            # Index documents to Pinecone
            print("now lets try to turn this into a dataframe!")
            self.log(f"Indexing {len(processed_data)} documents to Pinecone")
            
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model
            )
            
            self.log("Successfully indexed documents to Pinecone")
            result = DataFrame()
            return result.add_rows(processed_data)
            
        except Exception as e:
            error_msg = f"Error during processing and indexing: {str(e)}"
            self.log(error_msg)
            raise ValueError(error_msg) from e