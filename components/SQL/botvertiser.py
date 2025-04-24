from langflow.custom import Component
from langflow.io import MessageTextInput, MultilineInput, SecretStrInput, DropdownInput, Output
from langflow.schema import Message
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from google.cloud import bigquery
from google.oauth2 import service_account
from decimal import Decimal
import json

# Custom encoder to handle Decimal values from BigQuery
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class NLToBigQueryComponent(Component):
    display_name = "Botvertiser"
    description = "Converts natural language to SQL and executes it on BigQuery"
    icon = "hand-metal"
    name = "NLToBigQueryComponent"
    
    inputs = [
        MessageTextInput(
            name="question",
            display_name="User Question",
            required=True,
            info="Ask a question like: 'How many clicks did Olas Media get on 2025-04-16?'",
            tool_mode=True
        ),
        DropdownInput(
            name="client",
            display_name="Client",
            options=["Olas Media", "TechFirm", "MarketCorp", "DigitalSolutions"],
            value="Olas Media",
            required=True,
            info="Select the client for this query"
        ),
        MultilineInput(
            name="prompt",
            display_name="LLM Prompt",
            required=True,
            info="Provide the base prompt to instruct the LLM how to format SQL"
        ),
        MultilineInput(
            name="schema",
            display_name="Table Schema",
            required=True,
            info="Paste a plain-text version of the table schema for LLM context"
        ),
        DropdownInput(
            name="model_name",
            display_name="OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
            value="gpt-4o",
            required=True
        ),
        SecretStrInput(
            name="openai_api_key",
            display_name="OpenAI API Key",
            required=True,
            value="OPENAI_API_KEY"
        ),
        SecretStrInput(
            name="service_account_json",
            display_name="BigQuery Service Account JSON",
            required=True,
            value="bq_service_account",
            info="Google Cloud service account JSON stored as a Langflow secret"
        ),
        DropdownInput(
            name="show_sql",
            display_name="Show Generated SQL",
            options=["Yes", "No"],
            value="Yes",
            required=True,
            info="Choose whether to include the generated SQL in the output"
        )
    ]
    
    outputs = [
        Output(display_name="Query Result", name="output", method="build_output"),
    ]
    
    def build_output(self) -> Message:
        try:
            # Step 1: Generate SQL from natural language
            generated_sql = self.generate_sql(
                self.question,
                self.client,
                self.prompt,
                self.schema,
                self.model_name,
                self.openai_api_key
            )
            
            # Step 2: Execute the SQL on BigQuery
            query_result = self.execute_sql(
                generated_sql,
                self.service_account_json
            )
            
            # Step 3: Format the response
            if self.show_sql == "Yes":
                output = f"Generated SQL:\n{generated_sql}\n\nQuery Result:\n{query_result}"
            else:
                output = query_result
                
            return Message(text=output)
        except Exception as e:
            return Message(text=f"Error: {str(e)}")
    
    def generate_sql(self, question, client, prompt, schema, model_name, api_key):
        # Combine prompt with dynamic input including client
        full_prompt = f"{prompt}\n\nTABLE SCHEMA:\n{schema}\n\nCLIENT: {client}\n\nUSER QUESTION: {question}\n\nSQL:"
        
        # Create prompt template and LLM chain
        llm_prompt = PromptTemplate.from_template("{prompt}")
        llm = ChatOpenAI(temperature=0, model_name=model_name, api_key=api_key)
        chain = LLMChain(llm=llm, prompt=llm_prompt)
        
        # Run the chain
        sql = chain.run({"prompt": full_prompt}).strip()
        
        # Clean up the response (remove potential markdown formatting)
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        return sql
    
    def execute_sql(self, query, service_account_json):
        # Get service account JSON from secrets
        service_account_info = json.loads(service_account_json)
        
        # Setup BigQuery client
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        client = bigquery.Client(credentials=credentials, project=service_account_info["project_id"])
        
        # Execute query
        query_job = client.query(query)
        results = query_job.result()
        
        # Format results
        rows = [dict(row.items()) for row in results]
        return json.dumps(rows, indent=2, cls=DecimalEncoder) or "No results found."