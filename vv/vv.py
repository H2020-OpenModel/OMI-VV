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
from pyvis.network import Network
import networkx as nx
from rdflib import Graph
import json

# Import kb_toolbox
from omikb.omikb import kb_toolbox


def process_response(response):
    # Parse JSON response
    data = response.json()

    # Extract values
    values = []
    for binding in data['results']['bindings']:
        o = binding['o']
        # Check if the object is a literal value and not a URI
        if o['type'] == 'literal':
            try:
                # Attempt to convert the value to a float
                value = float(o['value'])
                values.append(value)
            except ValueError:
                # If conversion fails, skip the value
                continue

    return values


def load_and_process_data(kb_toolbox_instance, d_key, e_key):
    # Execute queries for both keys
    response1 = kb_toolbox_instance.search_keyword(d_key)
    response2 = kb_toolbox_instance.search_keyword(e_key)

    # Extract values using the process_response function
    key1_values = process_response(response1)
    key2_values = process_response(response2)

    # Ensure data arrays are not empty
    if not key1_values or not key2_values:
        print("Error: No data available for the given keys.")
        return np.array([]), np.array([])

    # Convert lists to numpy arrays
    d_values = np.array(key1_values, dtype=float).reshape(-1, 1)
    e_atomization = np.array(key2_values, dtype=float)

    return d_values, e_atomization


def train_and_predict(kb_toolbox_instance, new_d_values, save_figure, d_key, e_key):
    # Load and process data
    d_values, e_atomization = load_and_process_data(kb_toolbox_instance, d_key, e_key)

    # Check if the data is empty
    if d_values.size == 0 or e_atomization.size == 0:
        print("Error: No data available for the given keys.")
        return

    # Normalize d_values using StandardScaler
    scaler = StandardScaler()
    d_normalized = scaler.fit_transform(d_values)

    # Check if there is sufficient data for training
    if d_normalized.shape[0] < 2:
        print("Error: Not enough data to train the model.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(d_normalized, e_atomization, test_size=0.2, random_state=42)

    # Initialize and fit MLPRegressor
    regressor = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', alpha=0.001, max_iter=1000,
                             random_state=42)
    regressor.fit(X_train, y_train)

    # Initialize conformal predictor
    nc = RegressorNc(regressor, err_func=AbsErrorErrFunc())
    icp = IcpRegressor(nc)
    icp.fit(X_train, y_train)
    icp.calibrate(X_test, y_test)

    # Predict with conformal intervals
    new_d_values = np.array(new_d_values, dtype=float).reshape(-1, 1)
    new_d_normalized = scaler.transform(new_d_values)
    prediction_intervals = icp.predict(new_d_normalized, significance=0.05)

    # Save the predicted intervals for new d values to output.txt
    with open('output.txt', 'w') as f:
        for i, d in enumerate(new_d_values):
            f.write(
                f"Predicted {e_key} for {d_key} = {d[0]}: {prediction_intervals[i, 0]:.2f} to {prediction_intervals[i, 1]:.2f} eV\n")

    xlabel = d_key
    ylabel = e_key

    plot_results(X_train, y_train, X_test, y_test, new_d_values, prediction_intervals, scaler, xlabel, ylabel,
                 save_figure)


def plot_results(X_train, y_train, X_test, y_test, new_d_values, prediction_intervals, scaler, xlabel, ylabel,
                 save_figure):
    plt.figure(figsize=(10, 6))

    # Plot training data in blue
    plt.scatter(scaler.inverse_transform(X_train), y_train, color='blue', label='Training Data')

    # Plot testing data in green
    plt.scatter(scaler.inverse_transform(X_test), y_test, color='green', label='Testing Data')

    # Plot predictions for new d values in red with error bars
    new_d_values = new_d_values.flatten()
    lower_bounds = prediction_intervals[:, 0]
    upper_bounds = prediction_intervals[:, 1]
    plt.errorbar(new_d_values, (lower_bounds + upper_bounds) / 2, yerr=(upper_bounds - lower_bounds) / 2, fmt='o',
                 color='red', label='Predictions')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_figure:
        plt.savefig("output.png")  # Save the plot if the switch is set

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
    net.show("triples_visualization.html")


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

