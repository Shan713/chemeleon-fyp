import io
import threading
import json
from queue import Queue

from ase.io import write

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State

from chemeleon import Chemeleon
from chemeleon.visualize import Visualizer
from app.utils import empty_fig, atoms_to_dict, dict_to_atoms


# Initialize the Dash app with Bootstrap stylesheet
app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME]
)
server = app.server

USE_CLIENT = True
if USE_CLIENT:
    from app.server_client import client
else:
    # Load pre-trained Chemeleon model in local machine
    chemeleon = Chemeleon.load_from_default_checkpoint()

# Define Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Chemeleon", href="#"),
            dbc.Nav(
                [
                    dbc.NavItem(
                        dbc.NavLink(
                            "GitHub Repo", href="https://github.com/your_repo_link"
                        )
                    ),
                    dbc.NavItem(dbc.NavLink("Paper", href="https://link_to_paper")),
                    # Additional links can be added here
                    dbc.NavItem(dbc.NavLink("About", href="#")),
                ],
                navbar=True,
            ),
        ],
        fluid=True,  # Make the navbar container fluid to span 100% width
    ),
    color="dark",
    dark=True,
)

# Define footer
footer = html.Footer(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.P(
                            "© 2024 WMD Group, Imperial College London.",
                            className="text-light mb-0 fs-5",
                        ),
                        md=6,
                        className="text-center text-md-start",
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.A(
                                    "Group GitHub",
                                    href="https://github.com/wmd-group",
                                    className="text-light me-3 fs-5",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                ),
                                html.A(
                                    "Group Website",
                                    href="https://wmd-group.github.io/",
                                    className="text-light fs-5",
                                    target="_blank",
                                    rel="noopener noreferrer",
                                ),
                            ],
                            className="d-flex justify-content-center justify-content-md-end",
                        ),
                        md=6,
                        className="mt-3 mt-md-0",
                    ),
                ],
                className="align-items-center",
            ),
        ],
        fluid=True,
        className="py-4",  # Added vertical padding
    ),
    style={
        "backgroundColor": "#343a40",  # Dark background
    },
    className="mt-auto",
)

# Initialize the structure queue
structure_queue = Queue()

# Define the layout of the app
app.layout = html.Div(
    [
        # Wrapper div for entire content
        html.Div(
            [
                # Navbar
                navbar,
                # Main container with margins on the sides
                dbc.Container(
                    [
                        # Header
                        html.H1(
                            "Welcome to Chemeleon!",
                            className="text-center text-success my-4",
                        ),
                        # Text input section
                        html.H2(
                            "Input your text prompt to generate crystal structures",
                            className="text-left",
                        ),
                        dbc.Form(
                            [
                                dbc.CardGroup(
                                    [
                                        dcc.Textarea(
                                            id="input-text",
                                            placeholder="e.g. A Crystal Structure of LiMnO4 with orthorhombic symmetry",
                                            value="A Crystal Structure of LiMnO4 with orthorhombic symmetry",
                                            style={"width": "100%", "height": "80px"},
                                        ),
                                    ]
                                ),
                                # Divide into two columns below input-text
                                dbc.Row(
                                    [
                                        # Left Column: Logo
                                        dbc.Col(
                                            html.Img(
                                                id="logo-image",
                                                src="/assets/logo_static.jpg",
                                                style={
                                                    "width": "40%",
                                                    "height": "auto",
                                                    "margin-top": "15px",
                                                },
                                            ),
                                            width=6,
                                        ),
                                        # Right Column: Number of Atoms and Generate Button
                                        dbc.Col(
                                            [
                                                dbc.CardGroup(
                                                    [
                                                        dbc.Label(
                                                            "Number of Atoms (Max 20):",
                                                            style={
                                                                "font-size": "1.2rem",
                                                                "margin-right": "10px",
                                                            },
                                                        ),
                                                        dbc.Input(
                                                            id="num-atoms",
                                                            type="number",
                                                            value=6,
                                                            min=1,
                                                            max=20,
                                                            step=1,
                                                            size="md",
                                                            style={
                                                                "width": "80px",
                                                                "display": "inline-block",
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "align-items": "center",
                                                        "justify-content": "flex-end",
                                                    },
                                                    className="mt-3",
                                                ),
                                                dbc.CardGroup(
                                                    [
                                                        dbc.Button(
                                                            "Generate Structures",
                                                            id="generate-button",
                                                            n_clicks=0,
                                                            color="primary",
                                                            className="mt-3",
                                                            disabled=False,  # Ensure button is enabled initially
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "justify-content": "flex-end",
                                                    },
                                                ),
                                            ],
                                            width=6,
                                            className="mt-3",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        # Progress bar
                        dbc.Progress(
                            id="progress-bar",
                            striped=True,
                            animated=True,
                            style={"height": "30px"},
                            color="success",
                            className="my-3",
                            value=0,
                            max=100,
                        ),
                        # Output section with two columns
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        # Left Column: Structure Selection
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        "Structure Selection"
                                                    ),
                                                    dbc.CardBody(
                                                        [
                                                            dbc.RadioItems(
                                                                id="structure-radio",
                                                                options=[],  # Empty options
                                                                value=None,
                                                                className="my-3",
                                                            ),
                                                        ],
                                                        id="structure-selection-cardbody",
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            width=4,
                                        ),
                                        # Right Column: Structure Output
                                        dbc.Col(
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("Visualizer"),
                                                    dbc.CardBody(
                                                        [
                                                            html.H3(
                                                                id="visualization-title",
                                                                className="text-center",
                                                            ),
                                                            html.Center(
                                                                dcc.Graph(
                                                                    id="visualization-graph",
                                                                    figure=empty_fig,
                                                                )
                                                            ),
                                                            dbc.CardGroup(
                                                                [
                                                                    dbc.Button(
                                                                        "Download CIF File",
                                                                        id="download-button",
                                                                        n_clicks=0,
                                                                        color="success",
                                                                        className="mt-3",
                                                                        disabled=True,
                                                                    ),
                                                                ],
                                                                className="text-center",
                                                            ),
                                                        ],
                                                        id="visualization-cardbody",
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            width=8,
                                        ),
                                    ],
                                    className="mt-4",
                                ),
                            ],
                            id="output-sections",
                        ),
                        # Hidden divs to store the structures
                        dcc.Store(id="stored-structures"),
                        # Hidden div to store progress bar value
                        dcc.Store(id="progress-value", data=0),
                        # Download component
                        dcc.Download(id="download-cif"),
                        # Interval component for periodic updates
                        dcc.Interval(
                            id="interval-component",
                            interval=1000,  # Update every 1 second
                            n_intervals=0,
                            disabled=True,  # Initially disabled
                        ),
                    ],
                    fluid=False,  # Keep this False to add margins to left and right
                    className="flex-grow-1",  # Allow this container to grow
                ),
            ],
            className="d-flex flex-column min-vh-100",  # Make this div full height of viewport
        ),
        # Footer
        footer,
    ]
)


def generate_structure_chemeleon(num_atoms, input_text, queue, use_client=USE_CLIENT):
    """
    Chemeleon function to generate random atomic structures based on the input text.
    Simulates a long-running process by yielding intermediate structures.
    """
    n_samples = 5
    timesteps = 1000
    natoms = [num_atoms] * n_samples
    texts = [input_text] * n_samples

    step = 0
    if use_client:
        response = client(
            url="https://8000-01j80snre5xdhq828s1q5brs0m.cloudspaces.litng.ai/predict",
            num_samples=n_samples,
            num_atoms=num_atoms,
            text_input=input_text,
        )
        for line in response.iter_lines():
            output = json.loads(line)["output"]
            atom_dict = json.loads(output)
            atoms_list = [dict_to_atoms(atoms_dict) for atoms_dict in atom_dict]
            data = {
                "structures": [atoms_to_dict(atoms) for atoms in atoms_list],
                "names": [
                    f"Sample {i+1}: {atoms.get_chemical_formula()}"
                    for i, atoms in enumerate(atoms_list)
                ],
                "progress": int(step / timesteps * 100),
            }
            queue.put(data)
            step += 1
    else:
        for atoms_list in chemeleon.sample(natoms, texts, stream=True):
            data = {
                "structures": [atoms_to_dict(atoms) for atoms in atoms_list],
                "names": [
                    f"Sample {i+1}: {atoms.get_chemical_formula()}"
                    for i, atoms in enumerate(atoms_list)
                ],
                "progress": int(step / timesteps * 100),
            }
            queue.put(data)
            step += 1
    queue.put("DONE")


def visualize_structure(atoms):
    """
    Visualizes the given atomic structure using Plotly.
    """
    visualizer = Visualizer([atoms], atomic_size=0.6, resolution=20)
    fig = visualizer.view()
    return fig


# Combined Callback for starting generation and updating outputs
@app.callback(
    [
        Output("progress-bar", "value"),
        Output("progress-bar", "label"),
        Output("progress-value", "data"),
        Output("stored-structures", "data"),
        Output("structure-radio", "options"),
        Output("structure-radio", "value"),
        Output("download-button", "disabled"),
        Output("interval-component", "disabled"),
        Output("generate-button", "disabled"),
    ],
    [
        Input("generate-button", "n_clicks"),
        Input("interval-component", "n_intervals"),
    ],
    [
        State("stored-structures", "data"),
        State("input-text", "value"),
        State("num-atoms", "value"),
        State("structure-radio", "value"),
    ],
)
def update_output(
    n_clicks, n_intervals, stored_data, input_text, num_atoms, current_value
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "generate-button":
        if n_clicks > 0:
            # Clear the queue
            with structure_queue.mutex:
                structure_queue.queue.clear()
            # Start the background thread
            thread = threading.Thread(
                target=generate_structure_chemeleon,
                args=(num_atoms, input_text, structure_queue),
            )
            thread.start()
            # Reset outputs
            progress = 0
            progress_label = f"{progress} %"
            progress_value = progress
            stored_data = None
            structure_options = []
            selected_value = None
            download_disabled = True  # Disable download button
            interval_disabled = False  # Enable interval
            generate_disabled = True  # Disable generate button
            return (
                progress,
                progress_label,
                progress_value,
                stored_data,
                structure_options,
                selected_value,
                download_disabled,
                interval_disabled,
                generate_disabled,
            )
        else:
            raise dash.exceptions.PreventUpdate

    elif triggered_id == "interval-component":
        # Check the queue
        while not structure_queue.empty():
            # get last data in queue
            data = structure_queue.queue[-1]
            if data == "DONE":
                # Process is complete
                progress = 100
                progress_label = f"{progress} %"
                progress_value = progress
                download_disabled = False  # Enable download button
                interval_disabled = True  # Disable interval component
                generate_disabled = False  # Re-enable generate button
                return (
                    progress,
                    progress_label,
                    progress_value,
                    stored_data,
                    dash.no_update,
                    dash.no_update,
                    download_disabled,
                    interval_disabled,
                    generate_disabled,
                )
            else:
                # Update progress and structures
                progress = data["progress"]
                progress_label = f"{progress} %"
                progress_value = progress
                structures = data["structures"]
                names = data["names"]
                # Update stored_data with new structures
                stored_data = structures
                # Update structure selection options
                options = [
                    {"label": name, "value": idx} for idx, name in enumerate(names)
                ]
                # Preserve the selected value if it exists
                if current_value is not None and current_value < len(names):
                    selected_value = current_value
                else:
                    selected_value = 0
                download_disabled = True  # Disable download button until completion
                interval_disabled = False  # Keep interval running
                generate_disabled = True  # Disable generate button
                return (
                    progress,
                    progress_label,
                    progress_value,
                    stored_data,
                    options,
                    selected_value,
                    download_disabled,
                    interval_disabled,
                    generate_disabled,
                )
        # If queue is empty, return no updates
        raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate


# Callback to update the visualization when a structure is selected or when new data arrives or initial load
@app.callback(
    Output("visualization-graph", "figure"),
    [Input("structure-radio", "value"), Input("stored-structures", "data")],
    prevent_initial_call=True,
)
def update_visualization(selected_idx, stored_data):
    if stored_data is not None and len(stored_data) > 0:
        if selected_idx is None:
            selected_idx = 0
        atoms_dict = stored_data[selected_idx]
        atoms = dict_to_atoms(atoms_dict)
        fig = visualize_structure(atoms)
        return fig
    else:
        return empty_fig


# Callback to handle CIF file download
@app.callback(
    Output("download-cif", "data"),
    [Input("download-button", "n_clicks")],
    [State("structure-radio", "value"), State("stored-structures", "data")],
)
def download_cif(n_clicks, selected_idx, stored_data):
    if n_clicks > 0 and stored_data is not None and selected_idx is not None:
        atoms_dict = stored_data[selected_idx]
        atoms = dict_to_atoms(atoms_dict)
        # Write CIF to a string buffer
        buffer = io.BytesIO()
        write(buffer, atoms, format="cif")
        buffer.seek(0)
        cif_content = buffer.read()
        cif_content_str = cif_content.decode("utf-8")
        return dict(content=cif_content_str, filename=f"{str(atoms.symbols)}.cif")
    else:
        raise dash.exceptions.PreventUpdate


# Callback to update the logo image
@app.callback(
    Output("logo-image", "src"),
    [
        Input("interval-component", "disabled"),
        Input("generate-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def update_logo(interval_disabled, n_clicks):
    if not interval_disabled:
        return "/assets/logo.gif"
    return "/assets/logo_static.jpg"


if __name__ == "__main__":
    app.run_server(debug=True)
