import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from networkx.readwrite import json_graph
import os
import json

# Set header title
st.title('Knowledge graph visualization')

# Define list of selection options and sort alphabetically
options_list = os.listdir("data")
options_list.sort()

# Implement multiselect dropdown menu for option selection (returns a list)
selected_graph = st.selectbox('Select graph to visualize', options_list)
if st.button('Show the graph'):
    with open("data/" + selected_graph, 'r') as f:
        graph_data = json.load(f)
    graph = json_graph.node_link_graph(graph_data)
    for node in graph.nodes:
        graph.nodes[node]["title"] = graph.nodes[node]["definition"]
        graph.nodes[node]["size"] = len(graph.nodes[node]["aliases"]) + 3
        docids = graph.nodes[node]["docids"]
        graph.nodes[node]["group"] = docids[0] if docids else -1000
    # Initiate PyVis network object
    drug_net = Network(width="100%", bgcolor='#222222', font_color='white')

    # Take Networkx graph and translate it to a PyVis graph format
    drug_net.from_nx(graph)
    # Generate network with specific layout settings
    drug_net.repulsion(node_distance=420, central_gravity=0.33,
                       spring_length=110, spring_strength=0.10,
                       damping=0.95)
    drug_net.show_buttons(filter_=['physics'])

    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        path = '/tmp'
        drug_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except Exception:
        path = '/html_files'
        drug_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=1000, width=1000)