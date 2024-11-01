from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever
import json
from neo4j_graphrag.retrievers import VectorCypherRetriever

from KnowledgeGraphRetrieval import driver, embedder


"""
We will leverage Neo4j's vector search capabilities here.
To do this, we need to begin by creating a vector index on the text chunks from the PDFs, which are stored on Chunk nodes in our knowledge graph.
"""
create_vector_index(driver, name="text_embeddings", label="Chunk",
                    embedding_property="embedding", dimensions=1536, similarity_fn="cosine")


"""
Now that the index is set up, we will start simple with a VectorRetriever.
The VectorRetriever just queries Chunk nodes via vector search, bringing back the text and some metadata.
"""
vector_retriever = VectorRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    return_properties=["text"],
)

"""
Below we visualize the context we get back when submitting a search prompt.
"""
vector_res = vector_retriever.get_search_results(query_text = "How is precision medicine applied to Lupus?", 
                                                 top_k=3)
for i in vector_res.records: print("====\n" + json.dumps(i.data(), indent=4))


"""
Below we will use the VectorCypherRetriever, which allows you to run a graph traversal after finding nodes with vector search.
This uses Cypher, Neo4j's graph query language, to define the logic for traversing the graph.
As a simple starting point, we'll traverse up to 3 hops out from each Chunk, capture the relationships encountered, and include them in the response alongside our text chunks.
"""
vc_retriever = VectorCypherRetriever(
    driver,
    index_name="text_embeddings",
    embedder=embedder,
    retrieval_query="""
//1) Go out 2-3 hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
UNWIND relList AS rel

//2) collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks, 
  collect(DISTINCT rel) AS rels

//3) format and return context
RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
  apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
"""
)


"""
Below we visualize the context we get back when submitting a search prompt.
"""
vc_res = vc_retriever.get_search_results(query_text = "How is precision medicine applied to Lupus?", top_k=3)

# print output
kg_rel_pos = vc_res.records[0]['info'].find('\n\n=== kg_rels ===\n')
print("# Text Chunk Context:")
print(vc_res.records[0]['info'][:kg_rel_pos])
print("# KG Context From Relationships:")
print(vc_res.records[0]['info'][kg_rel_pos:])