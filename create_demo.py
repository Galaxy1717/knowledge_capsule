import streamlit as st
import streamlit.components.v1 as components
import json

import torch
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class StelliaE:
    """Encodes texts using the Stellia model."""
    def __init__(self, model_name: str = 'ProfessorBob/retrieval-mutli-1024'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 256
    ) -> list[list[float]]:
        self.model.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=1022
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            model_output = self.model(**batch_dict)
            embeddings = model_output[0][:, 0]  # CLS token embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings


with open('data/knowledge_capsule_mapping_references2.json', 'r') as f:
    data_ref = json.load(f)

with open('data/knowledge_capsule_mapping_topics2.json', 'r') as f:
    data_notion = json.load(f)

with open("data/mapping_ref_to_ref.json", "r") as f:
    mapping_ref_to_ref = json.load(f)

embedder = None
selected_reference = None
# Sample data
references = list(data_ref.keys())
notions = list(data_notion.keys())


st.set_page_config(page_title="Demo Selector", layout="wide")

# Create a sidebar selector
st.sidebar.title("Navigation")
demo = st.sidebar.radio("Choix de la démo:", ("Capsules HG 5ème", "Recommendation de capsules"))

# Initialize the session state for 'selected_reference' if it doesn't exist
if 'selected_reference' not in st.session_state:
    st.session_state.selected_reference = references[0]


# Function to change the selected reference
def change_reference(ref):
    if ref in references + notions:
        st.session_state.selected_reference = ref
    else:
        st.error(f"The reference '{ref}' is not available.")


def cross_string(text):
    return ''.join(char + '\u0336' for char in text)


if demo == "Capsules HG 5ème":
    # Streamlit app
    st.title(cross_string("Adaptive v2"))
    st.title("Création de capsule de connaissance")

    # Dropdown to select reference
    selected_reference = st.selectbox("Choix de la référence:", references + notions, key="selected_reference")

    concept_keys = ["label", "definition"]
    exercise_keys = ["id", "question", "query", "statement", "answers", "distractors", "veracity", "feedback"]

    # Display selected reference data
    if selected_reference:
        if selected_reference in references:
            data = data_ref
        else:
            data = data_notion
        st.header("Introduction")
        st.write(data[selected_reference]["introduction"])
        st.header("Synthèse d'information")
        st.write(data[selected_reference]["synthesis"])

        st.header("Concepts important")
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
        st.write(f"{len(new_collection)} concepts gérératifs pertinents")
        st.write(new_collection)

        st.header("Je teste mes connaissances")
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
        st.write(f"{len(new_collection)} exercices pertinents")
        st.write(new_collection)

        st.header("Sources")
        relevant_sources = data[selected_reference]["sources"]
        new_collection = []
        for source_dict in relevant_sources:
            new_dic = {}
            new_dic["name"] = source_dict["name"]
            new_dic["titre"] = source_dict["title"]
            new_dic["top_pages"] = source_dict["top_pages"]
            new_dic["score"] = source_dict["score"]
            new_collection.append(new_dic)
        st.write(f"{len(new_collection)} sources pertinentes")
        st.write(new_collection)

        st.header("Capsules de connaissances similaires")
        if mapping_ref_to_ref:
            similar_refs = mapping_ref_to_ref[selected_reference]
            # add a button for each similar reference
            for ref in similar_refs[:4]:
                st.button(
                    ref,
                    on_click=change_reference,
                    args=(ref,)
                )

        st.header("Visualisation du sous-graphe")
        if st.button("Visualiser le sous-graphe de connaissance"):
            html_path = f"data/graph_html/graph_capsule_{selected_reference}.html"
            HtmlFile = open(html_path, 'r', encoding='utf-8')
            # Load HTML file in HTML component for display on Streamlit page
            components.html(HtmlFile.read(), height=1000, width=1000)

if demo == "Recommendation de capsules":
    if embedder is None:
        embedder = StelliaE()
        introductions = [data_ref[ref]["introduction"] for ref in references] + [data_notion[notion]["introduction"] for notion in notions]
        ref_embedding = embedder.encode(introductions)
        # mapping_ref_to_ref = {}
        # for i, ref in enumerate(references + notions):
        #     sim_scores = [dot(ref_embedding[i], ref_embedding[j])/(norm(ref_embedding[i])*norm(ref_embedding[j]))
        #                   for j in range(len(ref_embedding))]
        #     sorted_idx = sorted(range(len(sim_scores)), key=lambda x: sim_scores[x], reverse=True)
        #     for index in sorted_idx:
        #         if index != i and sim_scores[index] > 0.5:
        #             if ref not in mapping_ref_to_ref:
        #                 mapping_ref_to_ref[ref] = []
                    # mapping_ref_to_ref[ref].append(references[index] if index < len(references) else notions[index - len(references)])
    user_input = st.text_area("Enter your text here:")

    if user_input:
        user_embedding = embedder.encode(user_input)[0]
        # compute cosine similarity with np
        sim_scores = [dot(ref_embedding[i], user_embedding)/(norm(ref_embedding[i])*norm(user_embedding))
                      for i in range(len(ref_embedding))]
        # sort by similarity
        sorted_idx = sorted(range(len(sim_scores)), key=lambda x: sim_scores[x], reverse=True)
        # display top 3
        st.write("Capsules recommandées:")
        for index in sorted_idx[:3]:
            if sim_scores[index] > 0.4:
                if st.button(references[index] if index < len(references) else notions[index - len(references)]):
                    selected_reference = references[index] if index < len(references) else notions[index - len(references)]
                    demo = "Capsules HG 5ème"
