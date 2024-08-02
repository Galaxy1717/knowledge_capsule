import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from networkx.readwrite import json_graph
import os
import json

with open('data/adaptive_reco_mapping2.json', 'r') as f:
    data_ref = json.load(f)

with open('data/adaptive_reco_mapping_notion2.json', 'r') as f:
    data_notion = json.load(f)
# Sample data
references = list(data_ref.keys())
notions = list(data_notion.keys())


# Streamlit app
st.title("'Adaptive v2' version graphe de connaissance")

# Dropdown to select reference
selected_reference = st.selectbox("Choix de la référence:", references + notions)

concept_keys = ["label", "definition"]
exercise_keys = ["id", "question", "query", "statement", "answers", "distractors", "veracity", "feedback"]

# Display selected reference data
if selected_reference:
    if selected_reference in references:
        data = data_ref
    else:
        data = data_notion
    st.header("Synthèse d'information")
    st.write(data[selected_reference]["synthesis"])

    st.header("Concepts génératifs liés")
    concept_collection = data[selected_reference]["concepts"]
    new_collection = []
    for concept_dict in concept_collection:
        new_dic = {}
        if concept_dict["score"] >= 8:
            new_dic["score"] = concept_dict["score"]
            new_concept = {}
            for key in concept_keys:
                if key in concept_dict["concept"]:
                    new_concept[key] = concept_dict["concept"][key]
            new_dic["concept"] = new_concept
            new_collection.append(new_dic)
    st.write(f"{len(new_collection)} concepts génératifs liés")
    st.write(new_collection)

    st.header("Exercices liés")
    exercise_collection = data[selected_reference]["exercises"]
    new_collection = []
    for exercise_dict in exercise_collection:
        new_dic = {}
        if exercise_dict["score"] >= 7:
            new_dic["score"] = exercise_dict["score"]
            new_exercise = {}
            for key in exercise_keys:
                if key in exercise_dict["exercise"]:
                    new_exercise[key] = exercise_dict["exercise"][key]
            new_dic["exercise"] = new_exercise
            new_collection.append(new_dic)
    st.write(f"{len(new_collection)} exercices liés")
    st.write(new_collection)
