from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG
import json

from KnowledgeGraphRetrieval import vector_retriever, vc_retriever


"""
At a minimum, you will need to pass the constructor an LLM and a retriever.
You can optionally pass a custom prompt template.
We will do so here just to provide a bit more guidance for the LLM to stick to information from our data source.

Below we create GraphRAG objects for both the vector and vector-cypher retrievers.
"""

llm = LLM(model_name="gpt-4o",  model_params={"temperature": 0.0})

rag_template = RagTemplate(template='''Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned. 

# Question:
{query_text}
 
# Context:
{context}

# Answer:
''', expected_inputs=['query_text', 'context'])

v_rag  = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)
vc_rag = GraphRAG(llm=llm, retriever=vc_retriever, prompt_template=rag_template)


"""
Now we can run GraphRAG and examine the outputs.
"""
q = "How is precision medicine applied to Lupus? provide in list format."
print(f"Vector Response: \n{v_rag.search(q, retriever_config={'top_k':5}).answer}")
print("\n===========================\n")
print(f"Vector + Cypher Response: \n{vc_rag.search(q, retriever_config={'top_k':5}).answer}")

q = "Can you summarize systemic lupus erythematosus (SLE)? including common effects, biomarkers, and treatments? Provide in detailed list format."

v_rag_result = v_rag.search(q, retriever_config={'top_k': 5}, return_context=True)
vc_rag_result = vc_rag.search(q, retriever_config={'top_k': 5}, return_context=True)

print(f"Vector Response: \n{v_rag_result.answer}")
print("\n===========================\n")
print(f"Vector + Cypher Response: \n{vc_rag_result.answer}")
for i in v_rag_result.retriever_result.items: print(json.dumps(eval(i.content), indent=1))
vc_ls = vc_rag_result.retriever_result.items[0].content.split('\\n---\\n')
for i in vc_ls:
    if "biomarker" in i: print(i)
vc_ls = vc_rag_result.retriever_result.items[0].content.split('\\n---\\n')
for i in vc_ls:
    if "treat" in i: print(i)
q = "Can you summarize systemic lupus erythematosus (SLE)? including common effects, biomarkers, treatments, and current challenges faced by Physicians and patients? provide in list format with details for each item."
print(f"Vector Response: \n{v_rag.search(q, retriever_config={'top_k': 5}).answer}")
print("\n===========================\n")
print(f"Vector + Cypher Response: \n{vc_rag.search(q, retriever_config={'top_k': 5}).answer}")