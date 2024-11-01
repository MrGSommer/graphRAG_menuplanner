from dotenv import load_dotenv
import os
import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# load neo4j credentials (and openai api key in background).
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

ex_llm=OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={
        "response_format": {"type": "json_object"}, # use json_object formatting for best results
        "temperature": 0 # turning temperature down for more deterministic results
    }
)

#create text embedder
embedder = OpenAIEmbeddings()

#define node labels
basic_node_labels = ["Nutzer", "Zutat", "Ernährungsplan", "Supplement", "Rezept", "RezeptTyp"]

node_labels = basic_node_labels

# define relationship types
rel_types = ["ENTHÄLT", "HAT_PLAN", "VERBUNDEN_MIT", "HAT_TYP"]

prompt_template = '''
You are a medical researcher tasks with extracting information from papers 
and structuring it in a property graph to inform further medical and research Q&A.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 


Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

- Use only the information from the Input text.  Do not add any additional information.  
- If the input text is empty, return empty Json. 
- Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
- An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
- Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. 

Use only fhe following nodes and relationships (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Do not return any additional information other than the JSON in it.

Examples:
{examples}

Input text:

{text}
'''



kg_builder_pdf = SimpleKGPipeline(
    llm=ex_llm,
    driver=driver,
    text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
    embedder=embedder,
    entities=node_labels,
    relations=rel_types,
    prompt_template=prompt_template,
    from_pdf=True
)

"Below, we run the SimpleKGPipeline to construct our knowledge graph from 3 pdf documents and store in Neo4j."

pdf_file_paths = ['truncated-pdfs/biomolecules-11-00928-v2-trunc.pdf', 
             'truncated-pdfs/GAP-between-patients-and-clinicians_2023_Best-Practice-trunc.pdf', 
             'truncated-pdfs/pgpm-13-39-trunc.pdf']

for path in pdf_file_paths:
    print(f"Processing : {path}")
    pdf_result = kg_builder_pdf.run_async(file_path=path)
    print(f"Result: {pdf_result}")


    