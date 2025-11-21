import polars as pl
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

cols_selection = [
    "product_name", "brands_tags","nutriments"
]

medium_coverage_nutrition_keys = [
    "energy-kcal",
    "fat",
    "saturated-fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "proteins",
    "salt",
]

def extract_nutrients(nutrient_list):
    nutrients = {}
    
    for item in nutrient_list:
        name = item['name']
        value_100g = item['100g']
        if name == 'energy':
            continue
        nutrients[name] = value_100g
    
    return nutrients

def create_full_rag_text(row):
    return (
        f"Produit : {row['product_name']}. "
        f"Marque : {row['brand_tag']}. "
        f"Énergie : {row['energy-kcal']} kcal. "
        f"Matières grasses totales : {row['fat']} g, dont acides gras saturés : {row['saturated-fat']} g. "
        f"Glucides : {row['carbohydrates']} g, dont sucres : {row['sugars']} g. "
        f"Protéines : {row['proteins']} g. "
        f"Fibres alimentaires : {row['fiber']} g. "
        f"Sel : {row['salt']} g (Sodium : {row['sodium']} g). "
    )

df_france = (
    pl.scan_parquet("food.parquet")
    .filter(
        (pl.col("countries_tags").list.contains("en:france")) &
        (pl.col("product_name").list.len() > 0)&
        (pl.col("brands_tags").is_not_null())&
        (pl.col("nutriments").is_not_null())&
        (pl.col("brands_tags").list.len() > 0)
    )
    .select(cols_selection)
    .collect()
)
df_france = df_france.to_pandas()
df_france["len_product_name"] = df_france.product_name.apply(lambda x: len(x))
df_france = df_france.reset_index(drop=True)
df_france = df_france[df_france["len_product_name"]>1]
df_france = df_france[df_france.len_product_name >0]
df_france = df_france[df_france.product_name.apply(lambda x: "fr" in [el["lang"] for el in x])]
df_france["product_name"] = df_france.product_name.apply(lambda x: [el["text"] for el in x if el["lang"] == "fr"][0])
df_france["brand_tag"] = df_france.brands_tags.apply(lambda x: x[0].replace("-"," ").replace("xx:",""))
df_france = df_france.drop(columns=["brands_tags","len_product_name"])
df_france =df_france.reset_index(drop=True)

dfn = pd.DataFrame(df_france.nutriments.apply(extract_nutrients).to_list())
dfn = dfn[medium_coverage_nutrition_keys]
dfn = pd.DataFrame(df_france.nutriments.apply(extract_nutrients).to_list())
dfn = dfn[medium_coverage_nutrition_keys]
df_france = pd.concat([df_france,dfn],axis=1)
df_france = df_france[["product_name","brand_tag"] + medium_coverage_nutrition_keys]
del dfn

df_france = df_france.drop(columns=["nova-group","fruits-vegetables-nuts-estimate-from-ingredients","nutrition-score-fr"])
df_france.to_parquet("clean_food.parquet",index=False)
df_france['text_for_rag'] = df_france.apply(create_full_rag_text, axis=1)

documents = [
    Document(
        page_content=f"{row['product_name']} {row['brand_tag']}", 
        metadata=row.to_dict()
    ) for _, row in df_france.iterrows()
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents, 
    embeddings, 
    persist_directory="./chroma_db"
)