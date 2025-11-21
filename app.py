import os
import base64
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from flask import Flask, render_template, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

load_dotenv()
api_key = os.getenv("DEEPINFRA_API_KEY")
llm = ChatOpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=api_key,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    temperature=0
)


class AgentState(TypedDict):
    product: str
    context: Optional[str]
    sources: Optional[List[dict]]
    analysis: Optional[str]

def retrieve_node(state: AgentState):
    docs = retriever.invoke(state["product"])
    if docs:
        context = "\n".join([f"{d.page_content} Data: {d.metadata}" for d in docs])
        sources = [{"name": d.page_content, "details": d.metadata} for d in docs]
        return {"context": context, "sources": sources}
    return {"context": None, "sources": None}

def web_search_node(state: AgentState):
    search = DuckDuckGoSearchRun()
    try:
        res = search.run(f"valeurs nutritionnelles {state['product']}")
    except Exception:
        res = "Aucune information trouvée."
    return {"context": res, "sources": [{"name": "Web Search", "details": "DuckDuckGo"}]}

def generate_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un expert nutritionnel. Analyse les données et donne une évaluation concise en français."),
        ("user", "Produit: {product}\nInfo:\n{context}")
    ])
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"product": state["product"], "context": state["context"]})
    return {"analysis": analysis}

def decide_next_step(state: AgentState):
    if state["context"]:
        return "generate"
    return "web_search"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    decide_next_step,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app_flow = workflow.compile()

def find_alternatives(product_name: str):
    docs = retriever.invoke(f"alternative saine {product_name}")
    context = "\n".join([f"{d.page_content}: {d.metadata}" for d in docs]) if docs else "Aucune donnée."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Suggère 3 alternatives plus saines. Format: ### [Nom] \n - Pourquoi c'est mieux."),
        ("user", "Produit: {product_name}\nDispo:\n{context}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"product_name": product_name, "context": context})

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            vision_chain = (
                {"image_data": lambda path: encode_image(path)}
                | ChatPromptTemplate.from_messages([
                    ("user", [
                        {"type": "text", "text": "Nom du produit uniquement."}, 
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}
                    ])
                ])
                | llm | StrOutputParser()
            )
            
            product_name = vision_chain.invoke(filepath).strip()
            result = app_flow.invoke({"product": product_name})
            
            return render_template('result.html', 
                                 product_name=product_name, 
                                 analysis=result["analysis"], 
                                 sources=result["sources"],
                                 image=file.filename)
            
    return render_template('index.html')

@app.route('/alternatives', methods=['POST'])
def alternatives():
    product_name = request.form['product_name']
    image = request.form['image']
    suggestions = find_alternatives(product_name)
    return render_template('result.html', product_name=product_name, alternatives=suggestions, image=image, show_original=False)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    product_name = request.args.get('product', '') or request.form.get('product', '')
    if request.method == 'POST':
        user_msg = request.json.get('message')
        docs = retriever.invoke(product_name)
        context = "\n".join([d.page_content for d in docs])
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "Assistant nutrition. Contexte: {context}"),
                ("user", "{msg}")
            ]) | llm | StrOutputParser()
        )
        return jsonify({"response": chain.invoke({"context": context, "msg": user_msg})})
    return render_template('chat.html', product_name=product_name)

if __name__ == '__main__':
    app.run(debug=True, port=5001)