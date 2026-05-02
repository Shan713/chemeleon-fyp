import time
import json
import random
import base64
from io import BytesIO
from fire import Fire

import streamlit as st
from ase.atoms import Atoms
from ase.build import bulk
from ase.io import write
from chemeleon import Chemeleon
from chemeleon.visualize import Visualizer

from app.server_client import client
from app.utils import dict_to_atoms

# Constants
TIMESTEPS = 1000
TRAJECTORY_STEPS = 100
DEFAULT_NUM_SAMPLES = 3
DEMO = False

# Set page configuration
st.set_page_config(page_title="Chemeleon", layout="wide")

# Hide Streamlit's default menu and footer for a cleaner look
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def demo_generator_structures(num_atoms, text_input, num_samples):
    """
    Generate crystal structures for demonstration purposes.
    """
    elements = random.choices(["Si", "Ge", "C", "Na", "Cl"], k=num_samples)
    random_elements = random.choices(elements, k=num_atoms)

    for step in range(TIMESTEPS):
        time.sleep(0.001)
        random_atoms = Atoms(
            "Li",
            positions=[[random.random() * 5 for _ in range(3)]],
        )
        atoms_list = [bulk(element, "fcc", a=5.43) for element in random_elements]
        new_atoms_list = []

        for atoms in atoms_list:
            # Adding random atoms to each bulk structure
            combined_atoms = atoms + random_atoms
            new_atoms_list.append(combined_atoms)

        yield new_atoms_list


def generator_structures_chemeleon(
    num_atoms, test_input, num_samples, use_client=False
):
    """
    Generate crystal structures based on the given number of atoms and input text.
    """
    if use_client:
        response = client(
            url="https://8000-01j80snre5xdhq828s1q5brs0m.cloudspaces.litng.ai/predict",
            n_samples=num_samples,
            n_atoms=num_atoms,
            text_input=test_input,
        )

        for line in response.iter_lines():
            output = json.loads(line)["output"]
            atom_dict = json.loads(output)
            atoms_list = [dict_to_atoms(atoms_dict) for atoms_dict in atom_dict]
            yield atoms_list
    else:
        chemeleon = Chemeleon.load_general_text_model()
        for atoms_list in chemeleon.sample(
            text_input=test_input,
            n_atoms=num_atoms,
            n_samples=num_samples,
            stream=True,
        ):
            yield atoms_list


def visualize_structure(atoms):
    """
    Visualize the given atomic structure using Plotly.
    """
    visualizer = Visualizer([atoms], atomic_size=0.6, resolution=20)
    fig = visualizer.view()
    return fig


def visualize_trajectory(atoms_list):
    """
    Visualize the given atomic structure trajectory using Plotly.
    """
    visualizer = Visualizer(atoms_list, atomic_size=0.6, resolution=20)
    fig = visualizer.view_trajectory(duration=1000)
    return fig


# Main application function
def main(use_client=False):
    # Initialize session state
    if "structures" not in st.session_state:
        st.session_state.structures = []
    if "trajectory" not in st.session_state:
        st.session_state.trajectory = []
    if "progress_in_generating" not in st.session_state:
        st.session_state["progress_in_generating"] = False

    # Sidebar for user inputs
    with st.sidebar:
        st.image("app/assets/logo_static.jpg", width=200)
        st.markdown(
            """
            <h1 style='text-align: center; color: #4CAF50;'>Chemeleon</h1>
            <h3 style='text-align: center;'>A text-guided diffusion model for crystal structure generation</h3>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        description = st.text_input(
            "Input your text prompt to generate crystal structures",
            "A Crystal Structure of LiMnO4 with orthorhombic symmetry",
            help="Examples: 'LiMnO4' or 'A Crystal Structure of BaTiO3 with cubic symmetry'",
        )
        num_atoms = st.slider(
            "🔢 Number of Atoms:",
            min_value=1,
            max_value=20,
            value=6,
            help="Select the number of atoms in the unit cell.",
        )
        num_samples = st.number_input(
            "🧪 Number of Samples:",
            min_value=1,
            max_value=5,
            value=DEFAULT_NUM_SAMPLES,
            step=1,
            help="Determine how many structure samples to generate.",
        )

    # Generate Structures when button is clicked
    if st.session_state["progress_in_generating"]:
        # Clear previous structures
        st.session_state.structures = []
        st.session_state.trajectory = []

        # Initialize progress bar in the sidebar
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        # Initialize loading animation
        image_placeholder = st.empty()
        with st.spinner("Generating structures..."):
            with image_placeholder:
                data_url = base64.b64encode(
                    open("app/assets/logo.gif", "rb").read()
                ).decode()
                image_placeholder.markdown(
                    f'<img src="data:image/gif;base64,{data_url}" width=100>',
                    unsafe_allow_html=True,
                )

        # Generate structures
        trajectory = []
        if DEMO:
            generator = demo_generator_structures(num_atoms, description, num_samples)
        else:
            generator = generator_structures_chemeleon(
                num_atoms, description, num_samples, use_client
            )
        for step, atoms_list in enumerate(generator):
            progress_bar.progress((step + 1) / TIMESTEPS)
            if step % TRAJECTORY_STEPS == 0 or step == TIMESTEPS - 1:
                st.session_state.structures = atoms_list
                trajectory.append(atoms_list)

        st.session_state.trajectory = trajectory

        # Remove the progress bar
        progress_placeholder.empty()

        # Remove the loading animation
        image_placeholder.empty()

        # Reset the progress state
        st.session_state["progress_in_generating"] = False

        # Display success message
        st.sidebar.success("✨ Structures generated successfully!")

    with st.sidebar:
        if st.button(
            "Generate Structures 🚀",
            disabled=st.session_state["progress_in_generating"],
        ):
            st.session_state["progress_in_generating"] = True
            st.rerun()

    # Check if structures are generated
    if st.session_state.structures:
        # Tabs for visualization
        tabs = st.tabs(["Structure Visualization", "Trajectory Analysis"])

        # Structure Visualization Tab
        with tabs[0]:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.session_state.selected_sample_index = (
                    st.radio(
                        "Select Sample",
                        options=list(range(1, num_samples + 1)),
                        index=0,
                        help="Choose which sample to visualize.",
                    )
                    - 1
                )  # Adjust for zero-based indexing
                # Download file
                atoms = st.session_state.structures[
                    st.session_state.selected_sample_index
                ]
                buffer = BytesIO()
                write(buffer, atoms, format="cif")
                buffer.seek(0)
                st.download_button(
                    label="Download CIF File",
                    data=buffer,
                    file_name=f"{str(atoms.symbols)}.cif",
                    mime="chemical/cif",
                )
            with col2:
                atoms = st.session_state.structures[
                    st.session_state.selected_sample_index
                ]
                fig = visualize_structure(atoms)
                st.plotly_chart(fig, use_container_width=True)

        # Trajectory Analysis Tab
        with tabs[1]:
            if st.session_state.trajectory:
                trajectory = [
                    traj[st.session_state.selected_sample_index]
                    for traj in st.session_state.trajectory
                ]
                tabs_2 = st.tabs(["Animation", "Step View"])
                # Animation
                with tabs_2[0]:
                    fig = visualize_trajectory(trajectory)
                    st.plotly_chart(fig, use_container_width=True)
                # Slider
                with tabs_2[1]:
                    trajectory_index = st.slider(
                        "Select Trajectory Step",
                        min_value=0,
                        max_value=len(trajectory) - 1,
                        value=0,
                        step=1,
                        help="Navigate through different steps of the structure generation.",
                    )
                    selected_atoms = trajectory[trajectory_index]
                    trajectory_fig = visualize_structure(selected_atoms)
                    st.plotly_chart(trajectory_fig, use_container_width=True)
            else:
                st.info("No trajectory data available.")

    # Footer
    st.markdown(
        """
        <div style="text-align: center; color: grey; margin-top: 50px;">
            <p style="font-size: 14px; margin: 0;">
                Developed by 
                <a href="https://hspark1212.github.io" target="_blank">Hyunsoo Park</a>, 
                as a part of <a href="https://github.com/wmd-group" target="_blank">Materials Design Group</a> 
                at Imperial College London
            </p>
            <p>
                <a href="https://chemrxiv.org/engage/chemrxiv/article-details/6728e27cf9980725cf118177" target="_blank">Research Paper</a> | 
                <a href="https://github.com/hspark1212/chemeleon" target="_blank">Repository</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    Fire(main)  # Usage example: streamlit run app/streamlit_app.py -- --use_client=True
