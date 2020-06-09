import sys
import random
import base64
import io
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
import plotly.graph_objs as go
from textwrap import dedent as d
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate


##################################################################################################################################
# Docs on NetworkX: https://networkx.github.io/documentation/stable/index.html
#
# Docs on Dash Plotly: https://dash.plotly.com/
#
# Docs on Plotly: https://plotly.com/python/network-graphs/
##################################################################################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "FOL Network"

# Global variable for storing paths
# If the app will be used by multiple people, then the paths should not be saved in a global var
node_paths=[]
# Its very important to have a seed for graph stability considering the implementation
random.seed(3)

def process_file(content, file_extension):
    """ Process the file containing the graph and return its nodes and edges.
        Only processes .txt and .p files.

    Arguments:
        file {string} -- The file
        file_extension {string} -- Type of file. Either .txt or .p

    Returns:
        list -- Two lists, nodes and edges of the graph
    """
    if file_extension == 'txt':
        data = [el.strip() for el in content.split(".")]
    elif file_extension == 'p':
        data = [el.strip() for el in content.split(":")[1].split("%")[0].split(".")]
    else:
        data=[]

    nodes = []
    edges = []
    for row in data:
        if len(row) > 1:
            # This string manipulation is dependent on the structure of the files
            split = row.split("(", maxsplit=1)[1]
            if (split.find("(")==-1) & (split.find(",")!=-1):
                edges.append(d)
            else:
                nodes.append(d)
    return nodes, edges

def rreplace(string, old, new, occurrence):
    """ Replace the last occurrences of a string.
        Python's build in function only works starting from the beginning and not the end.
    """
    li = string.rsplit(old, occurrence)
    return new.join(li)

def build_graph(nodes, edges):
    """ Build a graph given its nodes and edges.

   Arguments:
       nodes {list} -- The nodes of the graph
       edges {list} -- The edges of the graph

   Returns:
       DiGraph -- A directed graph representing the network
   """
    G = nx.DiGraph()

    # Add SK nodes
    [G.add_node(rreplace(node.split("(", maxsplit=1)[1], ")", "", 1)) for node in nodes]

    # Add other nodes
    for node in nodes:
        n, sk = node.split("(",maxsplit=1)[0], rreplace(node.split("(", maxsplit=1)[1], ")", "", 1)

        # If there are nodes with the name 'node', neato (which is used for visualization) throws
        # an odd error for some reason. Therefore we simply append an empty space if this is the case
        if n == 'node':
            n = " "+n

        # If the node is already in the graph, then add random spaces till we get a unique node
        while n in G.nodes():
            if random.random() > 0.5:
                n = " "+n
            else:
                n = n + " "
        G.add_node(n)

        # Add the parent SK as a label
        G.nodes[n]['sk']=sk

        # Add an edge with the parent sK
        G.add_edge(n, sk)
        G.add_edge(sk, n)


    # Add edges for SK
    for edge in edges:
        first, second = edge.split("(",maxsplit=1)[1].split(",")
        second = second.replace(")", "", 1).strip()

        for node in G.nodes():
            if node == first:
                for node2 in G.nodes():
                    if node2 == second:
                        G.add_edge(node, node2, Label = edge.split("(")[0])
    return G

def get_search_indices(search, G):
    """Get the indices of the searched nodes. There can be three kinds of search.
       The user can search individual nodes(1), nodes with the sK parent(2) and paths(3).
       Paths can be searched only between attributes of sK-nodes.

      #TODO Refactor the Gnodes variable as it is used 3 times
      #TODO Refactor the return line

    Arguments:
        search {string} -- The node/nodes
        G {Graph} -- The graph

    Returns:
        list -- A list of indices of the searched nodes
        search1 -- First type of search, node with sK
        search2 -- Second type of search, only a node
        search3 -- Third type of search, paths
    """

    search1 = True
    search2 = False
    search3 = False

    try:
        search, sk = search.split(",")

        # Check if a path has been searched
        if (search.find('sK') == -1) & (sk.find('sK') == -1):
            search1=False
            search3=True
    except ValueError:
        search1 = False
        search2 = True

    if search1:
        # Flag to check if the node search combination was found
        found=False

        # Get the real value of the original node. When the graph was generated,
        # random spaces were mixed with the node name because many nodes occur
        # more than once and a node name must be unique
        for node in G.nodes:
            if len(G.nodes[node].items()) > 0:
                if (G.nodes[node]['sk'] == sk) & (node.lstrip(" ").rstrip(" ")==search):
                    search=node
                    found=True
                    break

        # The nodes that will be highlighted. This includes by default the searched node
        # and its sK parent node. Optionally, if there is an outgoing edge, the neighboring
        # node is also highlighted
        highlighted=[]
        if found:
            # First we add all the sk nodes and then the attributes
            # Add parent sK node
            highlighted.append(sk)

            # Add the neighboring Sk if an edge is going out
            for edge in G.edges:
                if edge[0] == sk:
                    if edge[1].find("sK") != -1:
                        highlighted.append(edge[1])

            highlighted.append(search)

            # Turn the list to the corresponding node indices
            Gnodes = np.array(G.nodes)
            highlighted=[int(np.where(Gnodes==node)[0]) for node in highlighted]
        return highlighted, search1, search2, search3

    elif search2:
        highlighted = []
        for node in G.nodes:
            if node.lstrip(" ").rstrip(" ") == search:
                highlighted.append(node)

        Gnodes = np.array(G.nodes)
        highlighted = [int(np.where(Gnodes==node)[0]) for node in highlighted]
        return highlighted, search1, search2, search3
    elif search3:
        searchNodes = [search]

        # In order to make it possible for nodes of the same name to be on the same graph, spaces
        # have been  added to already existing nodes in the graph before they are added
        # to this graph. (this is how the implementation was thought from day 1)
        #
        # We will add possible other nodes to our searchNodes array by adding such spaces. They can
        # have spaces before, after or both in the beginning of the word. Admittedly, this solution
        # doesn't sound clean/right
        for i in range(1, 4):
            searchNodes.append(search+" "*i)
            searchNodes.append(i*" "+search)
            searchNodes.append(i*" "+search+" "*i)

        searchNodes.append(sk)
        for i in range(1,4):
            searchNodes.append(sk+" "*i)
            searchNodes.append(i*" "+sk)
            searchNodes.append(i*" "+sk+" "*i)

        # Now filter all the nodes that really are in the graph
        searchNodes = [node for node in searchNodes if G.has_node(node)]

        # Find the index where the second nodes start
        pos = 0
        for i in range(len(searchNodes)):
            if searchNodes[i].strip() != searchNodes[i-1].strip():
                pos = i

        # Finally, get the paths
        highlighted=[]
        for i in range(0, pos):
            for j in range(pos, len(searchNodes)):
        #         print("Examining: ", searchNodes[i], " ", searchNodes[j])
                paths = nx.all_simple_paths(G, searchNodes[i], searchNodes[j])
                for path in paths:
                    highlighted.append(path)

        # Assign the value to the global variable paths for future usage
        global node_paths
        node_paths = highlighted

        # Return the indices for the first path
        Gnodes = np.array(G.nodes)
        if len(highlighted) > 0:
            highlighted = [int(np.where(Gnodes==node)[0]) for node in highlighted[0]]
        return highlighted, search1, search2, search3

def get_clicked_path(n_clicks, paths):
    """Get the path according to the number of times the button was clicked.
       The paths are stored in the global node_paths variable once

    Arguments:
        n_clicks {[type]} -- [description]
        paths {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if n_clicks < len(paths):
        return paths[n_clicks]
    else:
        while n_clicks > len(paths)-1:
            n_clicks = (n_clicks/len(paths)-1) * len(paths)
        return paths[round(n_clicks)]

#TODO Method documentation
def visualize_graph(G, pos, searchValue='', highlighted=[]):

    # # Parse nodes and edges
    # nodes, edges = process_file(content, file_extension)
    # # Build graph
    # G = build_graph(nodes, edges)

    # ########## Visualize the graph

    # # Set node positions
    # pos = nx.nx_pydot.graphviz_layout(G)

    # Nodes information
    node_x = []
    node_y = []
    node_labels = []

    for key, value in pos.items():
        x, y = value[0], value[1]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(key)

    # By default there are only two types of color: sK and the parent
    # if a node has been searched the color types can go up to four
    node_color = np.array([1.0 if node.find('sK')!=-1  else 0 for node in node_labels])

    # String containing error messages
    errorMessage=" "
    search1=False
    search2=False
    search3=False

    # another path is being searched by clicking on the button
    if len(highlighted) > 0:
        # Convert node names to indices
        Gnodes = np.array(G.nodes)
        print("Is there any search value? ", searchValue)
        print("Highlighted: ", highlighted)
        # Get the indices
        # highlighted=[int(np.where(Gnodes==node)[0]) for node in highlighted]
        tmp = []
        for node in highlighted:
            tt1 = np.where(Gnodes==node)[0]
            print(np.where(Gnodes==node))
            print(tt1)
            tt=int(np.where(Gnodes==node)[0])
            tmp.append(tt)
            print(tt)

        highlighted = tmp

        print("Indeces: ",highlighted)
        for i in range(len(highlighted)):
            node_color[highlighted[i]] = 0.5
        search3 = True

    # A node has been searched
    elif len(searchValue)>0:
        # Reset any global paths
        global node_paths
        node_paths=[]

        highlighted, search1, search2, search3 = get_search_indices(searchValue, G)

        if len(highlighted)==0:
            errorMessage="The searched node/path does not exist."
        else:
            if search1:
                # First search type was performed (refer to get_search_indices docs for more info)
                #
                # If length is two, then there is only one sK node to highlight. This means
                # that there was no outgoing edge from the parent sK to any neighboring sk
                if len(highlighted) == 2:
                    node_color[highlighted[0]] = 0.3
                    node_color[highlighted[1]] = 0.5
                else:
                    for i in range(len(highlighted)):
                        if i<len(highlighted)-1:
                            node_color[highlighted[i]] = 0.3
                        else:
                            node_color[highlighted[i]] = 0.5
            elif (search2) | (search3):
                # A node was searched irrespective of sK
                # Or paths were searched
                for i in range(len(highlighted)):
                    node_color[highlighted[i]] = 0.5

    # Attempt to debug the problem with color highlight with some files
    # print("*********** Node COLORS ***********")
    # print(node_color)

    # Edges information for edge trace
    edge_x = []
    edge_y = []
    edge_labels = []

    for edge in G.edges().data():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        # Get the edge label
        label = [val for val in edge[2].values()]

        # Create the middle line coordinates where we shall add the name of the edge
        ax = (x0+x1)/2
        ay = (y0+y1)/2

        # Not all edges have a label
        if len(label) > 0:
            edge_labels.append((label[0], ax, ay))
        else:
            edge_labels.append((None, None, None))

    if (len(highlighted)>0) & (search1==True):
        # create node trace with multiple colors according to the highlighted nodes (including sk)
        node_trace = go.Scatter( x=node_x, y=node_y, text=node_labels, textposition='bottom center',
                        mode='markers+text', hoverinfo='text', name='Nodes',
                        marker=dict( showscale=False,
        #                 symbol='circle',
                        color=node_color,
                        # 0.3 -> red; 0.5->blue
                        colorscale = [[0, 'rgba(41, 128, 185, 0.2)'], [0.3, 'rgba(192, 57, 43, 1)'],
                                    [0.5, 'rgba(41, 128, 185, 1)'],  [1.0, 'rgba(192, 57, 43, 0.2)']],
                        size=15,
                        line=dict(color='rgb(180,255,255)', width=1))
                            )
    elif (len(highlighted)>0) & ((search2==True) | (search3==True)):
        # create node trace with multiple colors according to the highlighted nodes (excluding sk)
        node_trace = go.Scatter( x=node_x, y=node_y, text=node_labels, textposition='bottom center',
                        mode='markers+text', hoverinfo='text', name='Nodes',
                        marker=dict( showscale=False,
        #                 symbol='circle',
                        color=node_color,
                        colorscale = [[0, 'rgba(41, 128, 185, 0.4)'], [0.5, 'rgba(0, 255, 0, 1)'],
                                     [1.0, 'rgba(192, 57, 43, 0.4)']],
                        size=15,
                        line=dict(color='rgb(180,255,255)', width=1))
                           )
    else:
        # create node trace without color highlight
        node_trace = go.Scatter( x=node_x, y=node_y, text=node_labels, textposition='bottom center',
                        mode='markers+text', hoverinfo='text', name='Nodes',
                        marker=dict( showscale=False,
        #                 symbol='circle',
                        color=node_color,
                        colorscale = [[0, 'rgba(41, 128, 185, 1)'], [1, 'rgba(192, 57, 43, 1)']],
                        size=15,
                        line=dict(color='rgb(180,255,255)', width=1))
                            )


    # create edge trace
    edge_trace = go.Scatter( x=edge_x, y=edge_y,
        mode = 'lines', line=dict(width=1, color='rgb(90, 90, 90)'),
        hoverinfo='none'
    )

    # Annotations in order to add labels for the edges
    annotations_list = [
        dict(
            x = None if label[0] == None else label[1],
            y = None if label[0] == None else label[2],
            xref = 'x',
            yref = 'y',
            text = "" if label[0] == None else label[0],
            showarrow=False,
            opacity=0.7,
            ax = label[1],
            ay = label[2]
        )
    for label in edge_labels
    ]

    data = [edge_trace, node_trace]

    # Finally, create layout
    layout = go.Layout(
                width = 1000,
                height = 600,
                showlegend=False,
                plot_bgcolor="rgb(255, 255, 250)",
                hovermode='closest',
                #clickmode='event+select',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=annotations_list,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig, errorMessage

##################################################################################################################################


if __name__ == '__main__':

    # Initialize the graph with no data
    fig = go.Figure(data=None, layout = go.Layout(
                width = 1000,
                height = 600,
                showlegend=False,
                plot_bgcolor="rgb(255, 255, 250)",
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=None,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    # Define app layout
    app.layout = html.Div([
        ### Title
        html.Div([html.H1("First-order Logic Network Graph")],
                className="row",
                style={'textAlign': "center"}),

        ### Define the components, row
        html.Div(
            className="row",
            children=[

            ### Upload component
                html.Div(
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'display': 'inline-block',
                            'width': '25%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                        },
                    ),
                style={'text-align': 'center',}
                ),

            ### Middle graph component
                html.Div(
                    className="eight columns",
                    children=[
                        dcc.Graph(id='fol-graph', figure=fig),
                        # Store the graph and more importantly node positions here between callbacks
                        html.Div(id='graph-pos-intermediary', style={'display':'none'}),
                        html.Div(id='graph-intermediary', style={'display':'none'}),
                        ]
                ),

            ### Left side components
                html.Div(
                    className='two columns',
                    children=[
                        dcc.Markdown(d("""
                        **Search Node **

                        Search for individual nodes, connecting nodes or paths.
                        """)),
                        dcc.Input(id='input', type='text', placeholder='node/paths', value='',
                                debounce=True),
                        html.Div(id="error", style={'margin-top': '100px'}),

                        ### Button for graph paths
                        html.Button('Next Path', id='next-path-btn', n_clicks=0, hidden=True,),
                    ],
                    style={'margin-left': '200px', 'height': '300px'}
                ),
            ]
        )
    ])

    ###### Callback for all components
    @app.callback(
        [dash.dependencies.Output(component_id='fol-graph', component_property='figure'),
        dash.dependencies.Output('graph-intermediary', 'children'),
        dash.dependencies.Output('graph-pos-intermediary', 'children'),
        dash.dependencies.Output('next-path-btn', 'style'),
        dash.dependencies.Output('error', 'children')],

        [dash.dependencies.Input(component_id='upload-data', component_property='contents'),
        dash.dependencies.Input('input', 'value'),
        dash.dependencies.Input('next-path-btn', 'n_clicks'),],

        [dash.dependencies.State('upload-data', 'filename'),
        dash.dependencies.State('graph-intermediary', 'children'),
        dash.dependencies.State('graph-pos-intermediary', 'children'),]
    )
    def process_graph(content, search_value, n_clicks, filepath,  G, pos):
        """ Update/rebuild the graph when the user picks a new file or searches something.
           Stores the graph and its nodes positions in an intermediary value (div in the app).
           This little maneuver greatly improves computational time.

        Arguments:
            content -- [The content of the uploaded file]
            search_value -- [The value searched by the user. Single node, double nodes, paths]
            n_clicks -- [Number of times the button was clicked]
            filepath -- [Contains the file extension. Used to differentiate .txt from .p files]
        """
        ctx = dash.callback_context
        component_name = ctx.triggered[0]['prop_id'].split('.')[0]
        component_value = ctx.triggered[0]['value']

        if (component_value == None) | (component_value == 0):
            raise PreventUpdate
        if component_name == 'upload-data':
            content = content.split(',')[1]
            decoded_content = base64.b64decode(content).decode('utf-8')
            file_extension = filepath.split(".")[1]

            nodes, edges = process_file(decoded_content, file_extension)
            G = build_graph(nodes, edges)

            # Calculate node positions
            pos = nx.nx_pydot.graphviz_layout(G)
            graph, _ = visualize_graph(G, pos)
            return graph, json.dumps(json_graph.node_link_data(G)), json.dumps(pos), {'display': 'none'}, ''

        else:
            G = json_graph.node_link_graph(json.loads(G))
            pos = json.loads(pos)
            global node_paths

            if component_name == 'input':
                graph, error = visualize_graph(G, pos, search_value)
                print("NOde paths length: ", len(node_paths))
                if len(node_paths) > 1:
                    return graph, json.dumps(json_graph.node_link_data(G)), json.dumps(pos), {'display': 'block'}, error
                return graph, json.dumps(json_graph.node_link_data(G)), json.dumps(pos), {'display': 'none'}, error
            elif (component_name == 'next-path-btn') & (component_value > 0):
                if n_clicks > 0:
                    # Display other paths
                    highlighted = get_clicked_path(n_clicks, node_paths)
                    graph, error = visualize_graph(G, pos, '', highlighted)
                    return graph, json.dumps(json_graph.node_link_data(G)), json.dumps(pos), {'display': 'block'}, error

    ###### Start server
    app.run_server(debug=True, use_reloader=False)
    # app.run_server(debug=True,dev_tools_ui=False,dev_tools_props_check=False)