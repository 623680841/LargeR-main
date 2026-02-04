import fm
import torch
import requests
from rdkit import Chem
import numpy as np
from scipy import sparse
from transformers import BertTokenizer, BertModel
from larger.llm_tools import run_LLM, run_LLM_json_auto_retry

# 1. Load RNA-FM model
device = 'cuda:2'
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
def process_RNA(sequence):
    batch_labels, batch_strs, batch_tokens = batch_converter([('whatever', sequence)])
    batch_tokens = batch_tokens.to(device)
    model.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
    return results["representations"][12][0].cpu().numpy()


def get_fingerprint(smile):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + smile + "/JSON"
    
    # send request
    response = requests.get(url)
    
    # check response
    if response.status_code == 200:
        data = response.json()
        # print(data)  
    else:
        print("Error retrieving data from PubChem. " + smile)
        return '0'* 230
    fingerprint = None
    for compound in data.get("PC_Compounds", []):
        for prop in compound.get("props", []):
            if prop.get("urn", {}).get("label") == "Fingerprint":
                fingerprint = prop.get("value", {}).get("binary")
                break
        if fingerprint:
            break
    fingerprint = [int(fingerprint[i:i+2], 16) for i in range(0, len(fingerprint), 2)]
    return np.array(fingerprint)

def split_smiles(smiles, kekuleSmiles=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
    except:
        pass
    splitted_smiles = []
    for j, k in enumerate(smiles):
        if len(smiles) == 1:
            return [smiles]
        if j == 0:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            else:
                splitted_smiles.append(k)
        elif j != 0 and j < len(smiles) - 1:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            elif k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
        elif j == len(smiles) - 1:
            if k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
    return np.array(splitted_smiles)

def one_hot_coding(smi, words, kekuleSmiles=True, max_len=1000):
    coord_j = []
    coord_k = []
    spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
    if spt is None:
        return None
    for j,w in enumerate(spt):
        if j >= max_len:
            break
        try:
            k = words.index(w)
        except:
            continue
        coord_j.append(j)
        coord_k.append(k)
    data = np.repeat(1, len(coord_j))
    output = sparse.csr_matrix((data, (coord_j, coord_k)), shape=(max_len, len(words)))
    return output


def ligand2onehot(ligand, words):
    return np.array(one_hot_coding(ligand, words, max_len=len(ligand)).toarray().tolist())

def get_dict(all_smiles, kekuleSmiles=True):
    words = [' ']
    for smi in all_smiles:
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        for w in spt:
            if w in words:
                continue
            else:
                words.append(w)

    return words

from larger.llm_tools import run_LLM_json_auto_retry

def TaskReconginzeAgent(input_text):
    # dialogue_text = "\n".join([f"user: {u}\nAI: {a}" for u, a in conversation_history])
    prompt = (
        "You are an expert in the field of RNA molecular structure and function. Please break down the user's question based on their inquiry："
        f"The user's question is as follows:\n{input_text}\n"
    )
    prompt += (
        "\nPlease provide a preliminary assessment and return the content in JSON format, including the following fields:\n"
        "- clear_enough: The user's description of the question is clear enough to be rated as 1, otherwise it should be rated as 0, with no other possibilities.\n"
        "- using_protein_name: The RNA the user intends to inquire about is referred to by its name or identifier, rated as 1; if presented as a nucleotide sequence, it is rated as 0. Any other situation is categorized as the user's description being unclear.\n"
        "- protein: The RNA name or nucleotide sequence that the user intends to inquire about.\n"
        "- question: The user's question description should refer to the target RNA as 'this RNA', for example, 'What diseases is this RNA associated with?'"
    )
    return run_LLM_json_auto_retry(prompt)

