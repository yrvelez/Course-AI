from flask import Flask, render_template, request, jsonify
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, GPTVectorStoreIndex, SimpleDirectoryReader
import weaviate

app = Flask(__name__)
documents = SimpleDirectoryReader('materials').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5,
                                     response_mode='tree_summarize')

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
  user_input = request.form['user_input']
  response = query_engine.query(user_input)
  sources_data = [
    {
        'source': f'{dict(dict(response.source_nodes[i])["node"])["metadata"]["file_name"].replace(".pdf", "").replace("_", " ").title()} (pg. {dict(dict(response.source_nodes[i])["node"])["metadata"]["page_label"]})',
        'text': dict(dict(response.source_nodes[i])['node'])['text']
    }
    for i in range(min(5, len(response.source_nodes)))
  ]

  return jsonify({'response': str(response), 'sources': sources_data})


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=8080)
