import sys
import random
import numpy as np
import base64
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go
from textwrap import dedent as d


# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "FOL Network"

##################################################################################################################################

#TODO Refactor for new file versions
def process_file(content, file_extension):
    """ Process the text file containing the graph and return its nodes and edges.
        Only processes .txt and .p files.

    Arguments:
        file {string} -- The text file

    Returns:
        list -- A list of nodes and edges of the graph
    """

    # Read the data -according to the file ending-, clean and split into an array
    if file_extension=='txt':
        data= [el.strip() for el in content.split(".")]
    elif file_extension=='p':
        data = [el.strip() for el in content.split(":")[1].split("%")[0].split(".")]
    else:
        data=[]

    nodes = []
    edges = []
    for d in data:
        if len(d) > 1:

            # String manipulation in order to find out if we have a node or edge
            # This is strictly dependend on the structure of the files encountered so far
            # and specifically takes care to include hyperoutput2.txt file which has a
            # different structure from the other files
            split = d.split("(", maxsplit=1)[1]
            if (split.find("(")==-1) & (split.find(",")!=-1):
                edges.append(d)
            else:
                nodes.append(d)

    return nodes, edges


def rreplace(string, old, new, occurrence):
    """ Replace function constructed to replace the last occurrences of a string.
        Python's inbuild function only works starting from the beginning and not the end.
    """
    li = string.rsplit(old, occurrence)
    return new.join(li)


#TODO Refactor for new file versions
def build_graph(nodes, edges):
    """Build a graph given its nodes and its edges.

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

        # Add edge with the corresponding SK
        G.add_edge(n, sk)
        G.add_edge(sk,n)


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


def get_search_indices2(search, G):
    """Get the search indices of the searched node without its sK parent.
    Note: Because of error handling a separate function had to be declared.

    Arguments:
        search {string} -- The node to be searched
        G {Graph} -- The graph where the node will be searched

    Returns:
        list -- A list of indices
    """

    highlighted_indices = []
    for node in G.nodes:
        if node.lstrip(" ").rstrip(" ") ==search:
            highlighted_indices.append(node)

    Gnodes = np.array(G.nodes)
    highlighted_indices = [int(np.where(Gnodes==node)[0]) for node in highlighted_indices]
    return highlighted_indices


def get_search_indices1(search, G):
    """Get the search indices of the searched nodes. This method will return indices for
       the searched node, the parent sK as well as the neighboring sK.

    Arguments:
        search {string} -- The node to be searched
        G {Graph} -- The graph where the node will be searched

    Returns:
        list -- A list of indices of the searched nodes
    """

    # Throws an error if the format is not as specified
    search, sk = search.split(",")

    # Flag to check if the node search combination was found
    found=False

    # Get the real value of the original node
    # When the graph was generated, random spaces were mixed with the node name because
    # many node occur more than once
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
        # Add parent sK node
        highlighted.append(sk)

        # Add the neighboring Sk if an edge is going out
        for edge in G.edges:
            if edge[0] == sk:
                if edge[1].find("sK") != -1:
                    # neighbor_sk=edge[1]
                    highlighted.append(edge[1])

        highlighted.append(search)

        # Turn the list to the corresponding node indices
        # Throws an error if the nodes do not exist (parent sk, searched or both)
        Gnodes = np.array(G.nodes)
        highlighted=[int(np.where(Gnodes==node)[0]) for node in highlighted]

    return highlighted


#TODO Method documentation
def visualize_graph(search1='', search2='', content='', file_extension=''):

    # Parse nodes and edges
    nodes, edges = process_file(content, file_extension)
    # Build graph
    G = build_graph(nodes, edges)

    ########## Visualize the graph

    # Set node positions
    pos = nx.nx_pydot.graphviz_layout(G)

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

    # Array containing nodes that are to be highlighted when searched
    # The rest of the nodes will be blurred out
    highlighted=[]
    # String containing error messages
    errorMessage=" "

    # A node has been searched
    if len(search1)>0:
        # Searched for a node and its sK parent
        try:
            highlighted = get_search_indices1(search1, G)
        except ValueError:
            errorMessage="""
                Given input is not in the correct format.

                Correct format is: node,sKx
                """
        if len(highlighted)==0:
            errorMessage="The searched node does not exist."
        else:
            # If length is two, then there is only one sK node to highlight. This also means
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
    elif len(search2)>0:
        # A node was searched irrespective of sK
        try:
            highlighted = get_search_indices2(search2, G)
        except TypeError:
            errorMessage="The searched node does not exist."
        else:
            if len(highlighted) == 0:
                errorMessage="The searched node does not exist."
            for i in range(len(highlighted)):
                node_color[highlighted[i]] = 0.5

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

    if (len(search1) > 0) & (len(highlighted)>0):
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
    elif (len(search2) > 0) & (len(highlighted)>0):
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

    # Initialize the figure with no data
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
        html.Div([html.H1("FOL Network Graph")],
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
                    children=[dcc.Graph(id='fol-graph', figure=fig)]
                ),
            ### Left side search components
                html.Div(
                    className='two columns',
                    children=[
                        dcc.Markdown(d("""
                        **Search Node with its Instantiation**

                        Input the node and its instantiation.
                        """)),
                        dcc.Input(id='input1', type='text', placeholder='node,sKx', value='',
                                debounce=True),
                        html.Div(id="output1", style={'margin-top': '100px'}),
                    ],
                    style={'margin-left': '200px', 'height': '300px'}
                ),
                html.Div(
                    className='two columns',
                    children=[
                        dcc.Markdown(d("""
                            **Search Node without Instantiation**

                            Input only the node.
                            """)),
                        dcc.Input(id='input2', type='text', placeholder='node', value='',
                                debounce=True),
                        html.Div(id="output2")
                    ],
                    style={'margin-left': '200px', 'margin-top':'80px', 'height': '300px'}
                )
            ]
        )
    ])

    ###### Callback search components and file component
    @app.callback(
        [dash.dependencies.Output(component_id='fol-graph', component_property='figure'),
        dash.dependencies.Output('output1', 'children')],
        [dash.dependencies.Input(component_id='input1', component_property='value'),
        dash.dependencies.Input('input2', 'value'),
        dash.dependencies.Input('upload-data', 'contents'),],
        [dash.dependencies.State('upload-data', 'filename'),],
    )
    def search_update(value1, value2, content, filepath):
        if content is not None:
            # File has been choosen
            content_string = content.split(',')[1]
            decoded_content = base64.b64decode(content_string).decode('utf-8')
            file_extension = filepath.split(".")[1]

            # Only one search works at one time
            if (len(value1)>0) & (len(value2)>0):
                graph, error = visualize_graph(content=decoded_content, file_extension=file_extension)
                error="Can only use one search box at a time."
            else:
                # Rebuild the graph over again for every new search initiated
                graph, error = visualize_graph(value1, value2, decoded_content, file_extension)
                if len(error)>0:
                    print("Error:", error)
            return graph, error
        return fig,""

    ###### Start server
    app.run_server(debug=True, use_reloader=False)