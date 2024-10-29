from dotenv import load_dotenv
import os
import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

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
basic_node_labels = ["Nutzer", "Zutat", "Ernährungsplan"]

academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

medical_node_labels = ["Anatomy", "BiologicalProcess", "Cell", "CellularComponent", 
                       "CellType", "Condition", "Disease", "Drug",
                       "EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
                       "MolecularFunction", "Pathway"]

node_labels = basic_node_labels + academic_node_labels + medical_node_labels

# define relationship types
rel_types = ["ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED",
    "BIOMARKER_FOR", "CAUSES", "CITES", "CONTRIBUTES_TO", "DESCRIBES", "EXPRESSES",
    "HAS_REACTION", "HAS_SYMPTOM", "INCLUDES", "INTERACTS_WITH", "PRESCRIBED",
    "PRODUCES", "RECEIVED", "RESULTS_IN", "TREATS", "USED_FOR"]