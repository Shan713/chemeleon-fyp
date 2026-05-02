import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pymatgen.core import Composition

dotenv.load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def process_row(row, llm_chain, prompt_dir):
    prompt_data = row.to_dict()

    chat_val = llm_chain.invoke(prompt_data)

    with open(prompt_dir / f"{row['material_id']}.txt", "w") as f:
        f.write(chat_val["text"])


def main(data_dir: str = "."):

    data_dir = Path(data_dir)
    prompt_dir = data_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True, parents=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")

    df_total = pd.read_csv(data_dir / "mp-40-total.csv")
    print(f"Total crystal text: {len(df_total)}")
    # remove rows if there is prompt already
    already_prompt = prompt_dir.glob("*.txt")
    already_prompt = [p.stem for p in already_prompt]
    print(f"Already prompt: {len(already_prompt)}")
    df_total = df_total[~df_total["material_id"].isin(already_prompt)]
    print(f"Remaining crystal text: {len(df_total)}")

    # get reduced formula
    df_total["reduced_formula"] = df_total["composition"].apply(
        lambda x: Composition(x).reduced_formula
    )

    print("Creating LLMChain")
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

    template = """Provide five concise captions for "{reduced_formula}, {crystal_system}"

Here are some examples for other crystal systems:
1. Orthorhombic crystal structure of ZnMnO4
2. Crystal structure of LiO2 in orthorhombic symmetry
3. Cubic symmetry in SiC crystal structure

Please provide five captions for the crystal structure of {reduced_formula} in {crystal_system} symmetry.
"""

    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Assuming df_crystal_text is your DataFrame
    with ThreadPoolExecutor() as executor:
        # Submit all tasks and create a list of futures
        futures = [
            executor.submit(process_row, row, llm_chain, prompt_dir)
            for i, row in df_total.iterrows()
        ]

        # Wrap as_completed in tqdm to display progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing rows"
        ):
            try:
                result = (
                    future.result()
                )  # In this case, we don't expect a result since the function writes to a file
            except Exception as exc:
                print(f"Task generated an exception: {exc}")


if __name__ == "__main__":
    main()
