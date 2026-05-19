import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoConfig, GPT2LMHeadModel
import numpy as np
import re
import numpy as np
import os

# vocab_list
custom_vocab = []
custom_vocab_long = []
special_tokens = ["<pad>","<bos>","<eos>","<sep>"]
custom_vocab.extend(special_tokens)
custom_vocab_long.extend(special_tokens)
ATOMS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
    "Ga", "Ge", "As", "Se", "Br", "Kr", 
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
    "In", "Sn", "Sb", "Te", "I", "Xe", 
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
    "Ho", "Er", "Tm", "Yb", "Lu", 
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", 
    "Po", "At", "Rn", 
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", 
    "Es", "Fm", "Md", "No", "Lr", 
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", 
    "Lv", "Ts", "Og"
]
custom_vocab.extend(ATOMS)
custom_vocab_long.extend(ATOMS)
DIGITS = [str(d) for d in list(range(301))]
custom_vocab.extend(DIGITS)
custom_vocab_long.extend(DIGITS)
KeyWords = [
    #'-', 
    ' ',
    'total_atoms',
    'num_atoms', 'species', '[', ']', '{', '}', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Ele','unk_prop' , ':', '.'
]
custom_vocab.extend(KeyWords)
custom_vocab_long.extend(KeyWords)
coordinate_keywords = []
coordinate_keywords_long = []
for i in range(100):
    coordinate_keywords.append(f'x{i}')
    coordinate_keywords.append(f'y{i}')
    coordinate_keywords.append(f'z{i}')
for i in range(300):
    coordinate_keywords_long.append(f'x{i}')
    coordinate_keywords_long.append(f'y{i}')
    coordinate_keywords_long.append(f'z{i}')
custom_vocab.extend(coordinate_keywords)
#custom_vocab_long.extend(coordinate_keywords_long)
custom_vocab_long.extend(coordinate_keywords_long)
custom_vocab_dict = {word: idx for idx, word in enumerate(custom_vocab)}
custom_vocab_dict_long = {word: idx for idx, word in enumerate(custom_vocab_long)}
id2vocab_dict = {v: k for k, v in custom_vocab_dict.items()}
id2vocab_dict_long = {v: k for k, v in custom_vocab_dict_long.items()}
# tokenizer and decoder
def crystalTokenizer(prompt_batch, seq_batch, vocab_dict):
    input_ids = []
    attention_mask = []
    labels = []
    #transfer word to id
    for p,s in zip(prompt_batch, seq_batch):
        ids = torch.tensor(list(map(lambda token: vocab_dict[token], s)),dtype=torch.int64)
        input_ids.append(ids)
        mask = (ids != 0).to(torch.int64)
        attention_mask.append(mask)
        label = ids.clone()
        label[: len(p)] = -100
        labels.append(label)
    return {'input_ids':input_ids,
            'attention_mask':attention_mask,
            'labels':labels,
    } 
def crystalBatchDecoder(batch, id2vocab_dict):
    decoded_batch=[]
    for sentence in batch:
        decoded_sentence = list(map(lambda tokenid: id2vocab_dict[int(tokenid)] if int(tokenid) !=0 else '', sentence))
        decoded_batch.append(decoded_sentence)
    return decoded_batch

# dataset and dataloader
class CrystalStructureDataset(Dataset):
    def __init__(self, json_file, randomchose=True, dim=2):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.random = randomchose
        self.dim = dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        if self.random and self.dim == 2:
            sampled_object = random.choice(row)
        elif self.random == False and self.dim == 2:
            sampled_object =row[0]
        else:
            sampled_object = row
        #return sampled_object
        def preprocess_object(obj):
            #species = []
            #coordinates = []
            elements = []
            seq_Ele = []
            seq_noEle = []
            crystPrompt = []
            crystPromptShort = []
            #lattice_params = []
            atom_idx = 0
            #numclass = 119
            lattice_params=[obj['a'], obj['b'], obj['c'], obj['alpha'], obj['beta'], obj['gamma']]
            #seq_Ele.extend(['a']+list(str(obj['a']))+['b']+list(str(obj['b']))+['c']+list(str(obj['c']))+['alpha']+list(str(obj['alpha']))+['beta']+list(str(obj['beta']))+['gamma']+list(str(obj['gamma'])))
            #seq_noEle.extend(['a']+list(str(obj['a']))+['b']+list(str(obj['b']))+['c']+list(str(obj['c']))+['alpha']+list(str(obj['alpha']))+['beta']+list(str(obj['beta']))+['gamma']+list(str(obj['gamma'])))
            while f'x{atom_idx}' in obj:
                #coordinates.append([obj[f'x{atom_idx}'], obj[f'y{atom_idx}'], obj[f'z{atom_idx}']])
                seq_Ele.extend([f'x{atom_idx}']+list(str(obj[f'x{atom_idx}']))+[f'y{atom_idx}']+list(str(obj[f'y{atom_idx}']))+[f'z{atom_idx}']+list(str(obj[f'z{atom_idx}'])))
                seq_noEle.extend([f'x{atom_idx}']+list(str(obj[f'x{atom_idx}']))+[f'y{atom_idx}']+list(str(obj[f'y{atom_idx}']))+[f'z{atom_idx}']+list(str(obj[f'z{atom_idx}'])))
                #element = element_to_atomic_number[obj[f'Ele{atom_idx}']]
                element = obj[f'Ele{atom_idx}']
                seq_Ele.extend([element])
                elements.append(element)
                '''if not element in species:
                    species.extend(element)''' 
                atom_idx += 1
            species, num_atoms = np.unique(elements, return_counts=True)
            species = species.tolist()
            total_atoms = sum(num_atoms.tolist())
            num_atoms = list(map(str,num_atoms))
            seq_Ele.extend(['a']+list(str(obj['a']))+['b']+list(str(obj['b']))+['c']+list(str(obj['c']))+['alpha']+list(str(obj['alpha']))+['beta']+list(str(obj['beta']))+['gamma']+list(str(obj['gamma'])))
            seq_noEle.extend(['a']+list(str(obj['a']))+['b']+list(str(obj['b']))+['c']+list(str(obj['c']))+['alpha']+list(str(obj['alpha']))+['beta']+list(str(obj['beta']))+['gamma']+list(str(obj['gamma'])))
            seq_Ele.append('<eos>')
            seq_noEle.append('<eos>')
            crystPrompt.extend(['species']+species+['total_atoms']+[str(total_atoms)]+['num_atoms']+ num_atoms+['<sep>'])
            crystPromptShort.extend(['species']+species+['total_atoms']+[str(total_atoms)])
            #crystPrompt.extend(species)
            return {'num_atom': num_atoms,
                    'species': species,
                    'prompt': crystPrompt,
                    'prompt_short': crystPromptShort,
                    'cryst_seq_noele': seq_noEle,
                    'cryst_seq_ele': seq_Ele,
                    'crystal_json': obj
                    }
        return preprocess_object(sampled_object)
def collate_fn(batch_samples):
    batch_seq = []
    batch_seq_short = []
    batch_prompt = []
    batch_prompt_short = []
    max_len = 0
    max_len_short = 0
    for sample in batch_samples:
        batch_seq.append(sample['prompt']+sample['cryst_seq_ele'])
        batch_seq_short.append(sample['prompt_short']+sample['cryst_seq_ele'])
        batch_prompt.append(sample['prompt'])
        batch_prompt_short.append(sample['prompt_short'])
        max_len = max(max_len, len(sample['prompt']+sample['cryst_seq_ele']))
        max_len_short = max(max_len_short, len(sample['prompt_short']+sample['cryst_seq_ele']))
    batch_seq = list(map(lambda seq: seq + ["<pad>"] * (max_len - len(seq)), batch_seq))
    batch_seq_short = list(map(lambda seq: seq + ["<pad>"] * (max_len_short - len(seq)), batch_seq_short))
    tokenized_data=crystalTokenizer(batch_prompt, batch_seq, custom_vocab_dict)
    tokenized_data_short = crystalTokenizer(batch_prompt_short, batch_seq_short, custom_vocab_dict)
    input_ids = torch.stack(tokenized_data['input_ids'],0)
    attention_mask = torch.stack(tokenized_data['attention_mask'],0)
    labels = torch.stack(tokenized_data['labels'],0)
    input_ids_short = torch.stack(tokenized_data_short['input_ids'],0)
    attention_mask_short = torch.stack(tokenized_data_short['attention_mask'],0)
    prompt_lenths=list(map(lambda prompt: len(prompt),batch_prompt))
    prompt_lenths_short=list(map(lambda prompt_short: len(prompt_short),batch_prompt_short))
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'prompt_lenths': prompt_lenths,
        'input_ids_short':input_ids_short,
        'attention_mask_short':attention_mask_short, 
        'prompt_lenths_short': prompt_lenths_short
    }

def check_for_nan(arr):
    if np.any(np.isnan(arr)):
        raise ValueError('nan found in array')
#
def format_crystal_structure(text_list):
    sampled_strutures = []
    for idx,text in enumerate(text_list):
        try:
            eos_index = text.index('<eos>')
            if eos_index != -1:
                text = text[:eos_index+1]

            a_value = ''.join(text[text.index('a')+1:text.index('b')])
            b_value = ''.join(text[text.index('b')+1:text.index('c')])
            c_value = ''.join(text[text.index('c')+1:text.index('alpha')])
            alpha_value = ''.join(text[text.index('alpha')+1:text.index('beta')])
            beta_value = ''.join(text[text.index('beta')+1:text.index('gamma')])
            gamma_value = ''.join(text[text.index('gamma')+1:text.index('<eos>')])

            lattice_constants = {
                'a': float(a_value),
                'b': float(b_value),
                'c': float(c_value),
                'alpha': float(alpha_value),
                'beta': float(beta_value),
                'gamma': float(gamma_value)
            }
            coords = []
            elements = []
            for i in text[text.index('<sep>'):]:
                try:
                    if custom_vocab_dict[i] >= 4 and custom_vocab_dict[i] <= 121:
                        elements.append(i)
                except:
                    pass
            total_atoms = len(elements)
            species, num_atoms = np.unique(elements, return_counts=True)
            ats=[]
            for i in range(total_atoms):
                #try:
                    x = float(''.join(text[text.index(f'x{i}')+1:text.index(f'y{i}')]))
                    y = float(''.join(text[text.index(f'y{i}')+1:text.index(f'z{i}')]))
                    if i == total_atoms-1:
                        z = z = float(''.join(text[text.index(f'z{i}')+1:text.index('a')-1]))
                    else:
                        z = float(''.join(text[text.index(f'z{i}')+1:text.index(f'x{i+1}')-1]))
                #except:
                    #print(f'error happend in reading struture coordinates (atom {i}): {text}')
                    #pass
                    coords.append([x, y, z])
                    ats.append([x,y,z,elements[i]])
            ats=np.array(ats)
            ordered_ats = []
            #print(ats)
            for element in species:
                indices = np.where(ats[:,3] == element)[0]
                ordered_ats.extend(ats[indices].tolist())

            ordered_ats = np.array(ordered_ats)
            coords_matrix = np.array(coords)
            elements_matrix = np.array(elements)
            sampled_struture = {
            'num_atoms': num_atoms,
            'species': species,
            'lattice_constants': lattice_constants,
            'coords_matrix': coords_matrix,
            'elements_matrix': elements_matrix,
            'ats':ordered_ats
        }
            sampled_strutures.append(sampled_struture)  
        except:
            sampled_struture = {
            'num_atoms': None,
            'species': None,
            'lattice_constants': None,
            'coords_matrix': None,
            'elements_matrix': None,
            'ats':None
        }
            sampled_strutures.append(sampled_struture) 
            #print(f'error happend in reading struture {idx}: {text}')
            pass
    return sampled_strutures

def format_crystal_structure_L(text_list):
    sampled_strutures = []
    for idx,text in enumerate(text_list):
        try:
            eos_index = text.find('<eos>')
            if eos_index != -1:
                text = text[:eos_index]

            #text = re.sub(r'<.*?>', '', text)
            #print(text)
            text = re.sub(r'\b(alpha|beta|gamma)\b', lambda x: x.group(0).upper(), text)
            num_atoms_search = re.search(r'num_atoms\s+(\d+)', text)
            species_search = re.search(r'species\s+([A-Za-z]+)', text)
            
            if not num_atoms_search or not species_search:
                raise ValueError("num_atoms or species information missing")

            num_atoms = int(num_atoms_search.group(1))
            species = species_search.group(1)
            lattice_start_index = max(text.find('<sep>'),text.find('<bos>'))+6
            #print(text[lattice_start_index:text.find('x0')])
            if lattice_start_index == -1:
                raise ValueError("Lattice constants section not found")

            lattice_section = text[lattice_start_index:text.find('x0')]
            coordinate_section = text[text.find('x0'):]

            a_value = lattice_section[1:lattice_section.find('b')].replace(' ', '')
            b_value = lattice_section[lattice_section.find('b')+1 : lattice_section.find('c')].replace(' ', '')
            c_value = lattice_section[lattice_section.find('c')+1:lattice_section.find('ALPHA')].replace(' ', '')
            alpha_value = lattice_section[lattice_section.find('ALPHA')+6:lattice_section.find('BETA')].replace(' ', '')
            beta_value = lattice_section[lattice_section.find('BETA')+5:lattice_section.find('GAMMA')].replace(' ', '')
            gamma_value = lattice_section[lattice_section.find('GAMMA')+6:].replace(' ', '')

            lattice_constants = {
                'a': float(a_value),
                'b': float(b_value),
                'c': float(c_value),
                'ALPHA': float(alpha_value),
                'BETA': float(beta_value),
                'GAMMA': float(gamma_value)
            }
            coords = []
            elements = []
            
            for i in range(num_atoms):
                x = float(coordinate_section[coordinate_section.find(f'x{i}')+len(str(i))+1:coordinate_section.find(f'y{i}')].replace(' ', ''))
                y = float(coordinate_section[coordinate_section.find(f'y{i}')+len(str(i))+1:coordinate_section.find(f'z{i}')].replace(' ', ''))
                z = float(coordinate_section[coordinate_section.find(f'z{i}')+len(str(i))+1:coordinate_section.find(f'x{i+1}')-2].replace(' ', ''))
                
                coords.append([x, y, z])

                element = coordinate_section[coordinate_section.find(f'x{i+1}')-2:coordinate_section.find(f'x{i+1}')].replace(' ', '')
                elements.append(element)

            coords_matrix = np.array(coords)

            elements_matrix = np.array(elements)
            sampled_struture = {
            'num_atoms': num_atoms,
            'species': species,
            'lattice_constants': lattice_constants,
            'coords_matrix': coords_matrix,
            'elements_matrix': elements_matrix
        }
            sampled_strutures.append(sampled_struture)  
        except:
            #print(f'error happend in reading struture {idx}: {text}')
            pass
    return sampled_strutures

def crystText2Poscar(datas):
    poscars=[]
    err_stru_ids=[]
    for idx,data in enumerate(datas):
        try:
            num_atoms = data['num_atoms']
            species = data['species']
            lattice_constants = data['lattice_constants']
            #coords_matrix = data['coords_matrix']
            #elements_matrix = data['elements_matrix']
            ats = data['ats']
            coord_matrix = ats[:,:3].tolist()
            element_matrix = ats[:,3]
            scaling_factor = 1.0

            a, b, c = lattice_constants['a'], lattice_constants['b'], lattice_constants['c']
            alpha, beta, gamma = lattice_constants['alpha'], lattice_constants['beta'], lattice_constants['gamma']
            
            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)
            #print(1 - (np.cos(alpha_rad) ** 2 + np.cos(beta_rad) ** 2 + np.cos(gamma_rad) ** 2 - 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)) / (np.sin(gamma_rad) ** 2))
            a1 = [a, 0.0, 0.0]
            a2 = [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0.0]
            a3 = [c * np.cos(beta_rad), 
                c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad), 
                c * (np.sqrt(1 - (np.cos(alpha_rad) ** 2 + np.cos(beta_rad) ** 2 + np.cos(gamma_rad) ** 2) + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))]
            #print(a3)
            #check_for_nan(a1)
            #check_for_nan(a2)
            check_for_nan(a3)
            name = 'structure_dafault_name'
            poscar_data = [name]
            poscar_data.append(f"{scaling_factor:.16f}")
            poscar_data.append("   ".join([f"{x:.16f}" for x in a1]))
            poscar_data.append("   ".join([f"{x:.16f}" for x in a2]))
            poscar_data.append("   ".join([f"{x:.16f}" for x in a3]))
            
            poscar_data.append("   ".join(species))
            poscar_data.append("   ".join(list(map(str,num_atoms))))

            #poscar_data.append("Selective dynamics")
            
            poscar_data.append("Direct")
            
            for coord, element in zip(coord_matrix, element_matrix):
                poscar_data.append(f"{float(coord[0]):.16f} {float(coord[1]):.16f} {float(coord[2]):.16f}  {element.strip()}")
            poscar_data = "\n".join(poscar_data)
            poscars.append(poscar_data)
        except:
            #print(f'error happend in structure {idx}')
            err_stru_ids.append(idx)
            pass
    #print(err_stru_ids)
    return poscars, err_stru_ids

def batch_prompt_padding(input_ids, attention_mask, prompt_lengths, maxlen):
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    range_tensor = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(input_ids.device)

    prompt_masks = (range_tensor < prompt_lengths.unsqueeze(-1)).int()

    prompt_input_ids = input_ids * prompt_masks
    prompt_attention_masks = attention_mask * prompt_masks

    padding_sizes = maxlen - prompt_lengths

    def left_pad(tensor, padding_sizes):
        padded_tensors = [F.pad(tensor[i][:(tensor[i] == 0).nonzero()[0][0]], (padding_sizes[i].item(), 0), "constant", 0) for i in range(batch_size)]
        return torch.stack(padded_tensors, dim=0)

    padded_prompts_input_ids = left_pad(prompt_input_ids, padding_sizes)
    padded_prompts_attention_masks = left_pad(prompt_attention_masks, padding_sizes)

    return {'input_ids':padded_prompts_input_ids, 'attention_mask':padded_prompts_attention_masks}


def write_poscar(poscar_list,path,start_id=0,namelist=None):
    if os.path.exists(path) == False:
        os.mkdir(path)
    if namelist != None and len(namelist) == len(poscar_list):
        for poscar,name in zip(poscar_list,namelist):
            with open(os.path.join(path,name),'w') as f:
                f.write(poscar)
    elif namelist != None and len(namelist) != len(poscar_list):
        print('Error, the lenth of poscar_list should be as same as name_list')
    elif namelist == None:
        for id, poscar in enumerate(poscar_list):
            with open(os.path.join(path,f'POSCAR_{id+start_id}'),'w') as f:
                f.write(poscar)
    return 0

def crystText2Atoms(decoded_texts):
    from ase import Atoms
    atoms_list = []
    for text in decoded_texts:
        try:
            eos_index = text.index('<eos>')
            if eos_index != -1:
                text = text[:eos_index+1]

            a = float(''.join(text[text.index('a')+1:text.index('b')]))
            b = float(''.join(text[text.index('b')+1:text.index('c')]))
            c = float(''.join(text[text.index('c')+1:text.index('alpha')]))
            alpha = float(''.join(text[text.index('alpha')+1:text.index('beta')]))
            beta = float(''.join(text[text.index('beta')+1:text.index('gamma')]))
            gamma = float(''.join(text[text.index('gamma')+1:text.index('<eos>')]))

            elements = []
            for i in text[text.index('<sep>'):]:
                try:
                    if custom_vocab_dict[i] >= 4 and custom_vocab_dict[i] <= 121:
                        elements.append(i)
                except:
                    pass
            
            total_atoms = len(elements)
            coords = []
            for i in range(total_atoms):
                x = float(''.join(text[text.index(f'x{i}')+1:text.index(f'y{i}')]))
                y = float(''.join(text[text.index(f'y{i}')+1:text.index(f'z{i}')]))
                if i == total_atoms-1:
                    z_end_tag = 'a'
                    z = float(''.join(text[text.index(f'z{i}')+1:text.index(z_end_tag)-1]))
                else:
                    z_end_tag = f'x{i+1}'
                    z = float(''.join(text[text.index(f'z{i}')+1:text.index(z_end_tag)-1]))
                coords.append([x, y, z])
            
            atoms = Atoms(symbols=elements,
                          scaled_positions=coords,
                          cell=[a, b, c, alpha, beta, gamma],
                          pbc=True)
            check_for_nan(atoms.get_cell())
            atoms_list.append(atoms)
        except Exception as e:
            # print(f"Failed to parse structure: {e}")
            atoms_list.append(None)
    return atoms_list

def extract_traj_to_poscar(traj_path, output_dir):
    from ase.io import read, write
    from ase.io.trajectory import Trajectory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    traj = Trajectory(traj_path)
    for i, atoms in enumerate(traj):
        write(os.path.join(output_dir, f'POSCAR_{i}'), atoms, format='vasp')
    print(f"Extracted {len(traj)} structures from {traj_path} to {output_dir}")