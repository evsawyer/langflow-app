from typing import Any
from llama_index_cloud_sql_pg import PostgresEngine
from llama_index_cloud_sql_pg import PostgresDocumentStore
import nest_asyncio

nest_asyncio.apply()
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.schema import NodeRelationship

from langflow.custom import Component
from langflow.io import StrInput, SecretStrInput, Output
from langflow.schema import Message



class PineconeDocumentRetriever(Component):
    """Retrieves document text from Pinecone based on file path."""
    
    display_name = "Pinecone Document Retriever"
    description = "Retrieves document text from Pinecone vector store using file path as filter."
    icon = "pinecone"
    documentation = "https://docs.pinecone.io/docs/overview"

    inputs = [
        StrInput(
            name="file_id",
            display_name="File ID",
            required=True,
        ),
        StrInput(
            name="index_name",
            display_name="Index Name",
            info="Name of the Pinecone index",
            required=True,
        ),
        StrInput(
            name="namespace",
            display_name="Namespace",
            info="Namespace in the Pinecone index",
            required=True,
        ),
        SecretStrInput(
            name="PINECONE_API_KEY",
            display_name="Pinecone API Key",
            info="Your Pinecone API key",
            required=True,
        ),
        SecretStrInput(
            name="docstore_password",
            display_name="docstore password",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="text",
            display_name="Retrieved Text",
            description="The text content retrieved from the document",
            method="retrieve_document",
        ),
    ]

    async def retrieve_document(self) -> Message:
        """
        Retrieves document text from Pinecone using file path as filter.
        
        Returns:
            str: The text content of the retrieved document
        
        Raises:
            ValueError: If no document is found or if there's an error connecting to Pinecone
        """
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=self.PINECONE_API_KEY)
            
            # PostgresDocumentStore
            engine = await PostgresEngine.afrom_instance(
                project_id="knowledge-base-458316",
                region="us-central1",
                instance="llamaindex-docstore",
                database="docstore",
                user="docstore_rw",
                password=self.docstore_password,
                ip_type="public",
            )
            
            doc_store = PostgresDocumentStore.create_sync(
                engine=engine,
                table_name="document_store",
                # schema_name=SCHEMA_NAME
            )
            
            # Get the specified index
            try:
                pinecone_index = pc.Index(self.index_name)
            except Exception as e:
                msg = f"Error accessing Pinecone index '{self.index_name}': {str(e)}"
                raise ValueError(msg) from e
            
            try:
                # Create vector store
                vector_store = PineconeVectorStore(
                    pinecone_index=pinecone_index,
                    namespace=self.namespace
                )
            except Exception as e:
                msg = f"Error creating vector store with namespace '{self.namespace}': {str(e)}"
                raise ValueError(msg) from e

            # Create metadata filter
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="file id",
                        value=self.file_id,
                        operator=FilterOperator.EQ
                    )
                ],
            )

            # Retrieve nodes
            nodes = vector_store.get_nodes(filters=filters)
            


            if not nodes:
                msg = f"No document found with file path: {self.file_id}"
                raise ValueError(msg)
                
            relationships = nodes[0].relationships
            source_node_info = relationships.get(NodeRelationship.SOURCE)
            document = doc_store.get_document(doc_id=source_node_info.node_id)

            # Get text from first node
            document_text = document.text
            
            # Log success
            self.log(f"Successfully retrieved document from {self.file_id}")
            self.log(f"document: {document.metadata.get('file path')}")
            
            message = Message(text=document_text)
            return message

        except Exception as e:
            error_msg = f"Error retrieving document: {str(e)}"
            self.log(error_msg)
            raise ValueError(error_msg) from e

    def build(self) -> Any:
        """
        Validates the inputs and returns the component.
        
        Returns:
            PineconeDocumentRetriever: The built component
        """
        return self