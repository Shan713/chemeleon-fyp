import json
import litserve as ls
from chemeleon import Chemeleon
from app.utils import atoms_to_dict


class SimpleStreamAPI(ls.LitAPI):
    def setup(self, device, stream=True):
        self.model = Chemeleon.load_general_text_model()
        self.stream = stream

    def decode_request(self, request):
        n_samples = request.get("n_samples", 1)
        n_atoms = request["n_atoms"]
        text_input = request["text_input"]
        return {
            "n_samples": n_samples,
            "n_atoms": n_atoms,
            "text_input": text_input,
        }

    def predict(self, input_request):
        n_samples = input_request["n_samples"]
        n_atoms = input_request["n_atoms"]
        text_input = input_request["text_input"]

        # Sample
        if self.stream:
            yield from self._sample_stream(text_input, n_atoms, n_samples)
        else:
            return self._sample_batch(text_input, n_atoms, n_samples)

    def _sample_stream(self, text_input, n_atoms, n_samples):
        yield from self.model.sample(
            text_input=text_input,
            n_atoms=n_atoms,
            n_samples=n_samples,
            stream=True,
        )

    def _sample_batch(self, text_input, n_atoms, n_samples):
        return self.model.sample(
            text_input=text_input,
            n_atoms=n_atoms,
            n_samples=n_samples,
            stream=False,
        )

    def encode_response(self, output):
        for out in output:
            out_dict = [atoms_to_dict(atoms) for atoms in out]
            yield {"output": json.dumps(out_dict)}


if __name__ == "__main__":
    api = SimpleStreamAPI()
    server = ls.LitServer(api, workers_per_device=4, stream=True)
    server.run(port=8000)
