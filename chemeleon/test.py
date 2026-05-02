import chemeleon
from chemeleon.visualize import Visualizer

# Load default model checkpoint (general text types)
chemeleon = chemeleon.load_general_text_model()

# Set parameters
n_samples = 5
n_atoms = 6
text_inputs = "A crystal structure of LiMnO4 with orthorhombic symmetry"

# Generate crystal structure
atoms_list = chemeleon.sample(text_inputs, n_atoms, n_samples)

# Visualize the generated crystal structure
visualizer = Visualizer(atoms_list)
visualizer.view(index=1)