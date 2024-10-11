
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from nonconformist.nc import RegressorNc
from nonconformist.cp import IcpRegressor
from nonconformist.nc import AbsErrorErrFunc
import ipywidgets as widgets
from IPython.display import display
from omikb.omikb import kb_toolbox
import sys
import os
sys.path.append(os.path.abspath('../'))
from discomat.cuds.cuds import Cuds
from discomat.cuds.session import Session
from discomat.visualisation.cuds_vis import gvis
from discomat.ontology.namespaces import MIO
import csv
from rdflib import Literal, Graph
import networkx as nx

def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def process_response(response):
    data = response.json()
    values = []
    for binding in data['results']['bindings']:
        o = binding['o']
        if o['type'] == 'literal':
            try:
                value = float(o['value'])
                values.append(value)
            except ValueError:
                continue
    return values

def load_and_process_data(kb_toolbox_instance, d_key, e_key):
    response1 = kb_toolbox_instance.search_keyword(d_key)
    response2 = kb_toolbox_instance.search_keyword(e_key)
    key1_values = process_response(response1)
    key2_values = process_response(response2)

    if not key1_values or not key2_values:
        print("Error: No data available for the given keys.")
        return np.array([]), np.array([])

    d_values = np.array(key1_values, dtype=float).reshape(-1, 1)
    e_atomization = np.array(key2_values, dtype=float)

    return d_values, e_atomization

def train_and_predict(kb_toolbox_instance, new_d_values, save_figure, d_key, e_key):
    d_values, e_atomization = load_and_process_data(kb_toolbox_instance, d_key, e_key)

    if d_values.size == 0 or e_atomization.size == 0:
        print("Error: No data available for the given keys.")
        return new_d_values, None  # Return empty results

    scaler = StandardScaler()
    d_normalized = scaler.fit_transform(d_values)

    if d_normalized.shape[0] < 2:
        print("Error: Not enough data to train the model.")
        return new_d_values, None  # Return empty results

    X_train, X_test, y_train, y_test = train_test_split(d_normalized, e_atomization, test_size=0.2, random_state=42)

    regressor = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', alpha=0.001, max_iter=1000,
                             random_state=42)
    regressor.fit(X_train, y_train)

    nc = RegressorNc(regressor, err_func=AbsErrorErrFunc())
    icp = IcpRegressor(nc)
    icp.fit(X_train, y_train)
    icp.calibrate(X_test, y_test)

    new_d_values = np.array(new_d_values, dtype=float).reshape(-1, 1)
    new_d_normalized = scaler.transform(new_d_values)
    prediction_intervals = icp.predict(new_d_normalized, significance=0.05)

    # Save the predicted intervals for new d values to output.txt
    with open('output.txt', 'w') as f:
        for i, d in enumerate(new_d_values):
            f.write(f"Predicted {e_key} for {d_key} = {d[0]}: {prediction_intervals[i, 0]:.2f} to {prediction_intervals[i, 1]:.2f} eV\n")

    # Return predictions and intervals for RDF generation
    return new_d_values.flatten(), prediction_intervals

def plot_results(X_train, y_train, X_test, y_test, new_d_values, prediction_intervals, scaler, xlabel, ylabel, save_figure):
    plt.figure(figsize=(10, 6))
    plt.scatter(scaler.inverse_transform(X_train), y_train, color='blue', label='Training Data')
    plt.scatter(scaler.inverse_transform(X_test), y_test, color='green', label='Testing Data')
    new_d_values = new_d_values.flatten()
    lower_bounds = prediction_intervals[:, 0]
    upper_bounds = prediction_intervals[:, 1]
    plt.errorbar(new_d_values, (lower_bounds + upper_bounds) / 2, yerr=(upper_bounds - lower_bounds) / 2, fmt='o', color='red', label='Predictions')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_figure:
        plt.savefig("output.png")  # Save the plot if the switch is set\

def visualize_triples(existing_triples, added_triples):
    # Limit the number of triples from the existing knowledge base
    existing_triples_subset = existing_triples[:2000]

    # Create NetworkX graphs from RDF triples
    nx_existing_graph = nx.DiGraph()
    for subj, pred, obj in existing_triples_subset:
        nx_existing_graph.add_node(subj, label=subj)
        nx_existing_graph.add_node(obj, label=obj)
        nx_existing_graph.add_edge(subj, obj, label=pred)

    nx_added_graph = nx.DiGraph()
    for subj, pred, obj in added_triples:
        nx_added_graph.add_node(subj, label=subj)
        nx_added_graph.add_node(obj, label=obj)
        nx_added_graph.add_edge(subj, obj, label=pred)

    # Create a PyVis network
    net = Network(notebook=True, cdn_resources='in_line')

    # Add existing triples to the network with dark blue nodes and edges
    net.from_nx(nx_existing_graph)
    for node in net.nodes:
        node_id = node['id']
        if node_id in [s for s, _, _ in existing_triples_subset] or node_id in [o for _, _, o in
                                                                                existing_triples_subset]:
            node['color'] = 'darkblue'
            node['font'] = {'color': 'black'}

    for edge in net.edges:
        if (edge['from'], edge['to']) in [(s, o) for s, _, o in existing_triples_subset]:
            edge['color'] = 'darkblue'

    # Add added triples to the network with green nodes and edges
    net.from_nx(nx_added_graph)
    for node in net.nodes:
        if node['id'] in [s for s, _, _ in added_triples] or node['id'] in [o for _, _, o in added_triples]:
            node['color'] = 'green'
            node['font'] = {'color': 'black'}

    for edge in net.edges:
        if (edge['from'], edge['to']) in [(s, o) for s, _, o in added_triples]:
            edge['color'] = 'green'

    # Save the visualization to an HTML file
    net.show("visualization.html")


def verification():
    # Initialize kb_toolbox instance
    kb_instance = kb_toolbox()

    # Widgets for user input
    endpoint_url_widget = widgets.Text(description='Database:', placeholder='Enter database endpoint URL')
    new_d_values_widget = widgets.Text(description='Prediction:',
                                       placeholder='Enter prediction values separated by commas')
    d_key_widget = widgets.Text(description='Key 1:', placeholder='Enter key for values')
    e_key_widget = widgets.Text(description='Key 2:', placeholder='Enter key for values')
    save_figure_widget = widgets.Checkbox(value=False, description='Plot', tooltip='Switch to plot figure')

    # Button to trigger processing
    button = widgets.Button(description='Run V&V')

    # Define button click handler
    def on_button_click(b):
        d_key = d_key_widget.value
        e_key = e_key_widget.value
        try:
            new_d_values = [float(x) for x in new_d_values_widget.value.split(',') if x.strip()]
        except ValueError:
            print("Error: Invalid input for new_d_values. Ensure all values are numbers separated by commas.")
            return

        save_figure = save_figure_widget.value

        train_and_predict(kb_instance, new_d_values, save_figure, d_key, e_key)

    button.on_click(on_button_click)

    # Display widgets
    display(endpoint_url_widget, new_d_values_widget, d_key_widget, e_key_widget, save_figure_widget, button)


def upload():
    kb_instance = kb_toolbox()

    # Widgets for user input
    endpoint_url_widget = widgets.Text(description='Database:', placeholder='Enter database endpoint URL')
    dataset_widget = widgets.Text(description='Dataset:', placeholder='Enter dataset .ttl file')
    visualize_widget = widgets.Checkbox(value=False, description='Visualize',
                                        tooltip='Check to visualize triples')

    # Button to trigger processing
    button = widgets.Button(description='Upload')

    # Define button click handler
    def on_button_click(b):
        dataset = dataset_widget.value
        visualize = visualize_widget.value

        try:
            kb_instance.import_ontology(dataset)

            if visualize:
                # Load existing graph with a SPARQL query
                response = kb_instance.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
                existing_graph_data = response.json()
                existing_triples = [(binding['s']['value'], binding['p']['value'], binding['o']['value'])
                                    for binding in existing_graph_data['results']['bindings']]

                # Load and parse the new dataset
                g = Graph()
                g.parse(dataset, format="turtle")
                added_triples = [(str(subj), str(pred), str(obj)) for subj, pred, obj in g]

                # Visualize triples
                visualize_triples(existing_triples, added_triples)

        except ValueError:
            print("Error: Could not upload to knowledge base")
            return

    button.on_click(on_button_click)

    # Display widgets
    display(endpoint_url_widget, dataset_widget, visualize_widget, button)

def store_results_in_rdf(new_d_values, prediction_intervals, d_key, e_key):
    # Create a graph to store results
    g_total = Graph()

    # Add predictions to the graph
    for i, d in enumerate(new_d_values):
        lower_bound, upper_bound = prediction_intervals[i]  # Get bounds from prediction_intervals

        vv_result = Cuds(ontology_type=MIO.dataset, description="VV Results")
        vv_result.add(MIO.hasKey1, Literal(d))  # The key1 value
        vv_result.add(MIO.hasKey2, Literal(e_key)) # The key2 value
        vv_result.add(MIO.lowerBound, Literal(lower_bound))  # Use the lower_bound from prediction_intervals
        vv_result.add(MIO.upperBound, Literal(upper_bound))  # Use the upper_bound from prediction_intervals
        
        # Add to total graph
        g_total += vv_result.graph

    output_file = "output"
    iri = "cuds_iri_f373f80e-4909-45fb-957e-4cc88da788bc"

    # # Serialize to TTL format
    g_total.serialize(output_file+".ttl", format="ttl")

    kb_instance = kb_toolbox()
    # Upload the RDF to the knowledge base
    kb_instance.import_ontology(output_file+".ttl")

    # Visualize the RDF graph and save it as a PNG file
    visualize_rdf(g_total, output_file+".png", iri)


def extract_label(iri):
    """Extract the last part of the IRI after the # or /"""
    return iri.split('#')[-1].split('/')[-1]

def visualize_rdf(graph, output, training_data_iri):
    """ Visualizes the RDF graph with improved layout, labels, shapes, and training data connection."""
    G = nx.DiGraph()  # Create a directed graph

    # Add subject, object nodes, and edges with simplified labels
    for subj, pred, obj in graph:
        subj_label = extract_label(str(subj))
        obj_label = extract_label(str(obj))
        pred_label = extract_label(str(pred))

        G.add_node(subj_label, label=subj_label)  # Add subject node with simplified label
        G.add_node(obj_label, label=obj_label)  # Add object node with simplified label
        G.add_edge(subj_label, obj_label, label=pred_label)  # Add edge with simplified predicate label

    # Add training data IRI as a new node
    training_label = extract_label(training_data_iri)
    G.add_node(training_label, label=training_label)
    
    # Connect the training data IRI to the central node with the predicate 'has_vv_result'
    central_node = list(G.nodes())[0]  # Get the central node (VV results)
    G.add_edge(training_label, central_node, label='has_vv_result')

    # Set up positions for a cleaner layout
    pos = nx.spring_layout(G, seed=42, k=0.7)  # k controls the spacing

    plt.figure(figsize=(12, 8))

    # Define colors for nodes
    iri_color = '#4682B4'  # Blue for IRIs
    literal_color = '#FFD700'  # Gold for literals
    dataset_color = '#FF6347'  # Red for datasets
    edge_color = 'gray'  # Color for edges

    # Separate nodes by type (IRIs, literals, etc.)
    iri_nodes = [n for n in G.nodes if 'iri' in n]  # Simplified rule to detect IRIs
    literal_nodes = [n for n in G.nodes if 'literal' in n]  # Simplified rule for literals
    dataset_nodes = [n for n in G.nodes if 'dataset' in n]  # Simplified rule for datasets

    # Draw nodes with shapes and colors based on type
    nx.draw_networkx_nodes(G, pos, nodelist=iri_nodes, node_color=iri_color, node_shape='o', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=literal_nodes, node_color=literal_color, node_shape='D', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=dataset_nodes, node_color=dataset_color, node_shape='s', node_size=3000)

    # Draw the edges with arrows and labels
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20, edge_color=edge_color)
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', label_pos=0.5)

    # Draw node labels
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')

    # Set title and save the graph as a PNG file
    plt.axis('off')  # Hide the axis
    plt.savefig(output, format="png")
    

def vv_from_yaml(yaml_file_path):
    yaml_content = load_yaml_file(yaml_file_path)
    workflow_step = yaml_content['steps'][0]
    inputs = workflow_step['inputs']

    database_url = inputs['database']
    d_key = inputs['Key1']
    e_key = inputs['Key2']
    prediction_value = inputs['Prediction']

    kb_instance = kb_toolbox()

    new_d_values = [prediction_value]
    save_figure = False  # You can modify based on other YAML inputs

    # Run the training and prediction process
    new_d_values, prediction_intervals = train_and_predict(kb_instance, new_d_values, save_figure, d_key, e_key)

    # Store results in RDF and upload to KB
    if prediction_intervals is not None:  # Ensure that predictions were successful
        store_results_in_rdf(new_d_values, prediction_intervals, d_key, e_key)

