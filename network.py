import random
import base64
import numpy as np
import networkx as nx
import json
import plotly.graph_objs as go
from textwrap import dedent as d
import dash
import dash_core_components as dcc
import dash_html_components as html
from itertools import product
import gensim
import multiprocessing

##################################################################################################################################
# Docs on NetworkX: https://networkx.github.io/documentation/stable/index.html                                                   #
#                                                                                                                                #
# Docs on Dash Plotly: https://dash.plotly.com/                                                                                  #
#                                                                                                                                #
# Docs on Plotly: https://plotly.com/python/network-graphs/                                                                      #
#                 https://plotly.com/python/reference/                                                                           #
#                                                                                                                                #
# Docs on Genism: https://radimrehurek.com/gensim/models/keyedvectors.html                                                       #
#                                                                                                                                #
# ConceptNet Numberbatch: https://github.com/commonsense/conceptnet-numberbatch                                                  #
##################################################################################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "FOL Network"

# If the app will be used by multiple users, for storing paths in a global variables should be reconsidered
global_paths=[]
# Its very important to have a seed for graph stability considering the implementation
random.seed(3)
# NLP model. If the app will be used by multiple users, another solution has to be thought of
NLP_MODEL = None

def load_conceptnet_model(result):
    print('Loading the NLP model...')
    nlp = gensim.models.KeyedVectors.load_word2vec_format('numberbatch-en-17.04b.txt', binary=False, unicode_errors='ignore')
    result.append(nlp)

def process_file(content, file_extension):
    """ Process the file (.txt or .p) containing the graph and return its nodes and edges.
    """
    data, nodes, edges = [], [], []
    if file_extension == 'txt':
        data = [el.strip() for el in content.split(".") if len(el.strip())>0]
    elif file_extension == 'p':
        data = [el.strip() for el in content.split(":")[1].split("%")[0].split(".") if len(el.strip())>0]

    for row in data:
        split = row.split("(", maxsplit=1)[1]
        if (split.find("(")==-1) & (split.find(",")!=-1):
            edges.append(row)
        else:
            nodes.append(row)
    return nodes, edges

def rreplace(string, old, new, occurrence):
    """ Replace the last occurrences of a string starting from the end.
    """
    li = string.rsplit(old, occurrence)
    return new.join(li)

def build_graph(nodes, edges):
    """ Build a graph given its nodes and edges.
    """
    G = nx.Graph()
    [G.add_node(rreplace(node.split("(", maxsplit=1)[1], ")", "", 1)) for node in nodes] # sK nodes

    # other nodes
    for node in nodes:
        n, sk = node.split("(",maxsplit=1)[0], rreplace(node.split("(", maxsplit=1)[1], ")", "", 1)

        # For nodes named 'node', neato (which is used for visualization) throws an (odd) error
        if n == 'node':
            n = " "+n

        # A graph cannot have nodes with the same name
        while n in G.nodes():
            if random.random() > 0.5:
                n = " "+n
            else:
                n = n + " "
        G.add_node(n)
        G.nodes[n]['sk']=sk # Node label = (parent) sK
        G.add_edge(n, sk) # Edge to (parent) sK

    # sK edges
    for edge in edges:
        first, second = edge.split("(",maxsplit=1)[1].split(",")
        second = second.replace(")", "", 1).strip()
        G.add_edge(first,second,Label=edge.split("(")[0])
    return G

def unflatten(l, ind):
    """ Unflatten a list given the starting indices of the different node groups inside that list.
    """
    s = 0
    u_l = []
    for i in range(len(ind)):
        c_l = l[s:ind[i]+1]
        u_l.append(c_l)
        s = ind[i]+1
    return u_l

def get_search_type(search_type):
    search1, search2, search3, search4 = False, False, False, False

    if search_type == 'node,sKx':
        search1=True
    elif search_type == 'node(s)':
        search2=True
    elif search_type == 'node1,node2':
        search3=True
    elif search_type == 'word,n':
        search4=True
    return search1, search2, search3, search4


def get_search_nodes(search, search_type, G):
    highlighted = []
    search1, search2, search3, search4 = get_search_type(search_type)
    searched = search.partition(',')
    searched = [s.strip() for s in searched]

    if search1:
        found=False
        search, sk = searched[0], searched[2]

        # Get the original node with starting/ending whitespace
        for node in G.nodes:
            if len(G.nodes[node].items())>0:
                if (G.nodes[node]['sk'] == sk) & (node.strip()==search):
                    search=node
                    found=True
                    break

        if found:
            highlighted.append(sk)

            # neighboring sK
            for edge in G.edges:
                if edge[0] == sk:
                    if edge[1].find("sK") != -1:
                        highlighted.append(edge[1])
                elif edge[1] == sk:
                    if edge[0].find("sK") != -1:
                        highlighted.append(edge[0])
            highlighted.append(search)

    elif search2:
        search, rest = searched[0], searched[2]
        for node in G.nodes:
            if node.lstrip(" ").rstrip(" ") == search:
                highlighted.append(node)

        # If the user searched more than one node
        if len(rest) > 0:
            rest = rest.split(',')
            for search in rest:
                for node in G.nodes:
                    if node.lstrip(" ").rstrip(" ") == search.strip():
                        highlighted.append(node)

    elif search3:
        global global_paths
        searchNodes = [searched[0]]
        for i in range(1, 4):
            searchNodes.append(searched[0]+" "*i)
            searchNodes.append(i*" "+searched[0])
            searchNodes.append(i*" "+searched[0]+" "*i)

        otherNodes=[el for el in searched[2].split(',')]
        for node in otherNodes:
            searchNodes.append(node)
            for i in range(1, 4):
                searchNodes.append(node+" "*i)
                searchNodes.append(i*" "+node)
                searchNodes.append(i*" "+node+" "*i)
        # Filter all the nodes that actually are in the graph
        searchNodes = [node for node in searchNodes if G.has_node(node)]

        pos=[]
        for i in range(0, len(searchNodes)-1):
            if searchNodes[i].strip() != searchNodes[i+1].strip():
                pos.append(i)
        pos.append(len(searchNodes)-1)

        # One single path exists
        if pos[-1] == pos[len(pos)-2]:
            pos[-1]+=1

        searchNodes = unflatten(searchNodes, pos)
        for items in product(*searchNodes):
            c_paths=[]
            is_path=False
            for i in range(len(items)-1):
                is_path = False
                for path in nx.all_simple_paths(G, items[i], items[i+1]):
                    is_path=True
                    c_paths += path
                if not is_path:
                    break
            if is_path:
                highlighted.append(c_paths)
        global_paths = highlighted
    elif search4:
        #global NLP_MODEL # Can comment this out?
        if searched[2].isdigit(): # problem here
            result = NLP_MODEL.most_similar(searched[0], topn=50)
            result = [res[0] for res in result if G.has_node(res[0])]
            if len(result) > int(searched[2]):
                highlighted = result[:int(searched[2])]
            else:
                highlighted = result

    if len(highlighted) == 0:
        return [], False, False, False, False
    elif (type(highlighted[0]) is list) & (len(highlighted) > 0):
        return highlighted[0], search1, search2, search3, search4
    else:
        return highlighted, search1, search2, search3, search4

def get_clicked_path(n_clicks, paths):
    return paths[n_clicks % len(paths)]

def visualize_graph(G, node_pos, search_value='', search_type='', highlighted=[]):
    # Nodes information
    node_x = []
    node_y = []
    node_labels = []
    for key, value in node_pos.items():
        x, y = value[0], value[1]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(key)

    node_color = np.array([1.0 if node.find('sK')!=-1  else 0 for node in node_labels])
    error_message=""
    search1, search2, search3, search4 = False, False, False, False
    highlighted_names=[] # Names of nodes highlighted, used for highlighting the corresponding edges

    # "Next Path" button search
    if len(highlighted) > 0:
        highlighted_names = highlighted
        Gnodes = np.array(G.nodes)
        highlighted=[int(np.where(Gnodes==node)[0]) for node in highlighted] # Get the indices

        for i in range(len(highlighted)):
            node_color[highlighted[i]] = 0.5
        search3 = True

    # Node/Path has been searched
    elif len(search_value)>0:
        # Reset any global paths
        global global_paths
        global_paths=[]
        highlighted, search1, search2, search3, search4 = get_search_nodes(search_value, search_type, G)

        if len(highlighted)==0:
            error_message="The searched node/path/similarity does not exist or the input format is incorrect."
        else:
            highlighted_names = highlighted
            Gnodes = np.array(G.nodes)
            highlighted = [int(np.where(Gnodes==node)[0]) for node in highlighted]
            if search1: # Node,sK search
                for i in range(len(highlighted)):
                    if i<len(highlighted)-1:
                        node_color[highlighted[i]] = 0.3
                    else:
                        node_color[highlighted[i]] = 0.5
            elif (search2) | (search4): # Node search
                val = 0
                for i in range(len(highlighted)):
                    val+=0.1
                    node_color[highlighted[i]] = val
            else: # Path search
                for i in range(len(highlighted)):
                    node_color[highlighted[i]] = 0.5

    # Colorscale corresponding to colors for nodes
    if (len(highlighted)>0) & (search1):
        colorscale = [[0, 'rgba(41, 128, 185, 0.2)'], [0.3, 'rgba(192, 57, 43, 1)'],
                    [0.5, 'rgba(41, 128, 185, 1)'],  [1.0, 'rgba(192, 57, 43, 0.2)']]
    elif (len(highlighted)>0) & (search3):
        colorscale = [[0, 'rgba(41, 128, 185, 0.4)'], [0.5, 'rgba(0, 255, 0, 1)'],
                    [1.0, 'rgba(192, 57, 43, 0.4)']]
    elif (len(highlighted)>0) & ((search2) | (search4)):
        colorscale = [[0, 'rgba(41, 128, 185, 0.4)'], [0.1, 'rgba(0, 255, 0, 1)'],
                    [0.2, 'rgba(0, 255, 255, 1)'], [0.3, 'rgba(255, 255, 0, 1)'],
                    [0.4, 'rgba(0, 0, 0, 1)'], [0.5, 'rgba(220, 20, 60, 1)'],
                    [1.0, 'rgba(192, 57, 43, 0.4)']]
    else:
        colorscale = [[0, 'rgba(41, 128, 185, 1)'], [1, 'rgba(192, 57, 43, 1)']]

    node_trace = go.Scatter( x=node_x, y=node_y, text=node_labels, textposition='bottom center',
                    mode='markers+text', hoverinfo='text', name='Nodes',
                    marker=dict(
                        showscale=False,
                        color=node_color,
                        colorscale=colorscale,
                        size=15,
                        line=dict(color='rgb(180,255,255)', width=1)
                    )
                )

    edge_trace1 = go.Scatter( x=[], y=[], mode='lines',
        line=dict(width=1), hoverinfo='none',
    )
    edge_trace2 = go.Scatter( x=[], y=[], mode='lines',
        line=dict(width=1), hoverinfo='none',
    )
    edge_labels = []

    for i, edge in enumerate(G.edges().data()):
        x0, y0 = node_pos[edge[0]]
        x1, y1 = node_pos[edge[1]]

        if len(global_paths)>0:
            changed_color=False
            for i in range(0, len(highlighted_names)-1):
                if (((edge[0] == highlighted_names[i]) & (edge[1] == highlighted_names[i+1])) |
                    ((edge[0] == highlighted_names[i+1]) & (edge[1] == highlighted_names[i]))):
                    edge_trace1['x'] += tuple([x0,x1,None])
                    edge_trace1['y'] += tuple([y0,y1,None])
                    changed_color=True
            if changed_color == False:
                edge_trace2['x'] += tuple([x0,x1,None])
                edge_trace2['y'] += tuple([y0,y1,None])
        else:
            edge_trace2['x'] += tuple([x0,x1,None])
            edge_trace2['y'] += tuple([y0,y1,None])
        # Get the edge label
        label = [val for val in edge[2].values()]

        # Create the middle line coordinates for adding edge label
        ax = (x0+x1)/2
        ay = (y0+y1)/2
        if len(label) > 0: # Not all edges have a label
            edge_labels.append((label[0], ax, ay))

    normal_edge_color = 'rgba(100,100,100,0.6)'
    if len(global_paths)>0:
        normal_edge_color = 'rgba(100,100,100,0.1)'

    edge_trace1['marker'] = dict(color='rgb(0,255,0)')
    edge_trace2['marker'] = dict(color=normal_edge_color)

    # Annotations in order to add labels for the edges
    annotations_list = [
        dict(
            x = label[1],
            y = label[2],
            xref = 'x',
            yref = 'y',
            text = label[0],
            showarrow=False,
            opacity=0.7,
            ax = label[1],
            ay = label[2]
        )
        for label in edge_labels
    ]
    edge_trace = [edge_trace1, edge_trace2]

    data = edge_trace +[node_trace]
    layout = go.Layout(
        width = 1250,
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
    return go.Figure(data=data, layout=layout), error_message

##################################################################################################################################


if __name__ == '__main__':
    # Load the model in parallel

    # Initialize the graph with no data
    fig = go.Figure(data=None, layout = go.Layout(
                width = 1250,
                height = 600,
                showlegend=False,
                plot_bgcolor="rgb(255, 255, 250)",
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=None,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    app.layout = html.Div([
        ### Title
        html.Div([html.H1("First-order Logic Network Graph")],
                className="row",
                style={'textAlign': "center"}
        ),
        ### Define the components
        html.Div(
            className="row",
            children=[
            ### Top Upload component
            html.Div(
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'display': 'inline-block',
                        'width': '30%',
                        'height': '80px',
                        'lineHeight': '80px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                    },
                ),
                style={'text-align': 'center', 'margin-bottom':'10px'}
            ),
            ### Top components (below upload)
            html.Div(
                children=[
                    html.Div(
                        children=[
                            dcc.Dropdown(
                                id='search_dropdown',
                                options=[
                                    {'label':'node-sk', 'value': 'node,sKx'},
                                    {'label':'single node', 'value': 'node(s)'},
                                    {'label':'paths', 'value':'node1,node2'},
                                    {'label':'similarity', 'value':'word,n'}
                                ],
                                value='node,sKx',
                                placeholder='Select a search mode',
                                style={'margin-top':'-37px'}
                            ),
                            dcc.Markdown("**Select search mode**")
                        ],
                        style={'width':'20%', 'display':'inline-block'},
                    ),
                    html.Div(
                        children=[
                            dcc.Input(id='input', type='text', placeholder='node/paths', value='',
                                    debounce=True),
                            dcc.Markdown("**Search Node(s)/Paths**")
                        ],
                        style={'width':'40%', 'display':'inline-block'},
                    ),
                    ### Button for graph paths
                    html.Div(
                        html.Button('Next Path', id='next-path-btn', n_clicks=0, hidden=True),
                        style={'width':'20%', 'display':'inline-block'}
                    ),
                    ### Button for enabling model loading
                    html.Div(
                        html.Button('Load NLP model', id='nlp_button', n_clicks=0),
                        style={'width':'20%', 'display':'inline-block'}
                    ),
                    html.Div(id="error", style={'color':'red'}),
                    html.Div(id='nlp_message', style={'color':'blue'}),
                ],
            ),
            ### Middle graph component
                html.Div(
                    children=[
                        dcc.Graph(id='fol-graph', figure=fig),
                        # Store the graph node positions here between callbacks
                        html.Div(id='graph-pos-intermediary', style={'display':'none'}),
                        html.Div(id='graph-intermediary', style={'display':'none'}),
                        ]
                ),
            ]
        )
    ])

    # Possibility: You you return from the model that loads the message "model loaded"
    # This message you return from the callback
    @app.callback(
        [dash.dependencies.Output(component_id='nlp_button', component_property='style'),],
        [dash.dependencies.Input(component_id='nlp_button', component_property='n_clicks'),]
    )
    def load_nlp_model(n_clicks):
        if n_clicks == 1:
            manager = multiprocessing.Manager()
            result = manager.list()
            p = multiprocessing.Process(target=load_conceptnet_model, args=(result,))
            p.start()
            p.join()
            global NLP_MODEL
            NLP_MODEL = result[0]
            print('Model fully integrated')
            return [{'display': 'none'}]
        elif n_clicks > 1: # Get rid of this condition here
            return[{'display':'none'}]
        return [{'display':'block'}]

    ###### Callback for placeholder
    @app.callback(
        [dash.dependencies.Output(component_id='input', component_property='placeholder'),
        dash.dependencies.Output('input', 'value'),
        dash.dependencies.Output('input', 'disabled'),
        dash.dependencies.Output('nlp_message', 'children')],
        [dash.dependencies.Input(component_id='search_dropdown', component_property='value'),]
        )
    def update_mode_search(mode):
        """Update the placeholder of the search box based on the drop-down options & reset the input's value.
        """
        if (mode == 'word,n') & (NLP_MODEL == None):
            return mode, '', True, 'Model not loaded'
        elif (mode=='word,n') & (NLP_MODEL != None):
            return mode, '', False, 'Model loaded'
        return mode, '', False, ''

    ###### Callback for all components
    @app.callback(
        [dash.dependencies.Output(component_id='fol-graph', component_property='figure'),
        dash.dependencies.Output('graph-intermediary', 'children'),
        dash.dependencies.Output('graph-pos-intermediary', 'children'),
        dash.dependencies.Output('next-path-btn', 'style'),
        dash.dependencies.Output('error', 'children')],

        [dash.dependencies.Input(component_id='upload-data', component_property='contents'),
        dash.dependencies.Input('input', 'value'),
        dash.dependencies.Input('next-path-btn', 'n_clicks'),
        dash.dependencies.Input(component_id='search_dropdown', component_property='value')],

        [dash.dependencies.State(component_id='upload-data', component_property='filename'),
        dash.dependencies.State('graph-intermediary', 'children'),
        dash.dependencies.State('graph-pos-intermediary', 'children'),]
    )
    def process_graph(content, search_value, n_clicks, search_type, filepath,  G, pos):
        """ Update/rebuild the graph when the user picks a new file or searches something.
           Stores the graph and its nodes positions in an intermediary div.
           This little maneuver greatly improves run-time.

        Arguments:
            content -- [The content of the uploaded file]
            search_value -- [The value searched by the user: nodes/paths]
            n_clicks -- [Number of times the button was clicked]
            filepath -- [Contains the file extension. Used to differentiate .txt from .p files]
        """
        ctx = dash.callback_context
        component_name = ctx.triggered[0]['prop_id'].split('.')[0]
        component_value = ctx.triggered[0]['value']

        if (component_value == None) | (component_value == 0):
            raise dash.exceptions.PreventUpdate
        if component_name == 'upload-data':
            content = content.split(',')[1]
            decoded_content = base64.b64decode(content).decode('utf-8')
            file_extension = filepath.split(".")[1]

            nodes, edges = process_file(decoded_content, file_extension)
            G = build_graph(nodes, edges)

            pos = nx.nx_pydot.graphviz_layout(G)
            graph, _ = visualize_graph(G, pos)
            return graph, json.dumps(nx.readwrite.json_graph.node_link_data(G)), json.dumps(pos), {'display': 'none'}, ''
        else:
            try:
                G = nx.readwrite.json_graph.node_link_graph(json.loads(G))
                pos = json.loads(pos)
            except TypeError:
                raise dash.exceptions.PreventUpdate
            global global_paths

            if component_name == 'input':
                graph, error = visualize_graph(G, pos, search_value, search_type)
                if len(global_paths) > 1:
                    button_display = {'display': 'block'}
                else:
                    button_display = {'display':'none'}
                return graph, json.dumps(nx.readwrite.json_graph.node_link_data(G)), json.dumps(pos), button_display, error
            elif (component_name == 'next-path-btn'):
                if n_clicks > 0:
                    # Display other paths
                    highlighted = get_clicked_path(n_clicks, global_paths)
                    graph, error = visualize_graph(G, pos, '', '', highlighted)
                    return graph, json.dumps(nx.readwrite.json_graph.node_link_data(G)), json.dumps(pos), {'display': 'block'}, error
            else:
                raise dash.exceptions.PreventUpdate

    app.run_server(debug=True, use_reloader=False,dev_tools_hot_reload=True)
    # app.run_server(debug=True,dev_tools_ui=False,dev_tools_props_check=False)