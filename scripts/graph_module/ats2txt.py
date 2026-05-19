import numpy as np
import json
from ase.geometry import permute_axes

def write_txt(fd, ats, perm, direct=True, wrap=True):
    symbols = ats.get_chemical_symbols()

    if direct:
        pos = ats.get_scaled_positions(wrap=wrap)
    else:
        pos = ats.get_positions(wrap=wrap)
    intercept = -pos[perm[0]]
    for i in range(len(perm)):
        coords = (pos[perm[i]] + intercept) % 1
        fd.write('x y z {} {} {} {} '.format(round(coords[0], 2), round(coords[1], 2), \
        round(coords[2], 2), symbols[perm[i]]))

    cell = ats.cell.cellpar()
    fd.write('L a b c alpha beta gamma {} {} {} {} {} {} L\n'.format(round(cell[0], 1), round(cell[1], 1), \
    round(cell[2], 1), round(cell[3], 1), round(cell[4], 1), round(cell[5], 1)))

def one_json(ats, perm, direct=True, wrap=True):
    symbols = ats.get_chemical_symbols()
    data = {}

    pos = ats.get_scaled_positions(wrap=wrap)
    cell = np.array(ats.get_cell())
    intercept = -pos[perm[0]]
    for i in range(len(perm)):
        coords = (pos[perm[i]] + intercept) % 1
        if not direct:
            coords = np.dot(coords, cell)
        #data["Pos{}".format(i)] = {"x": round(coords[0], 2), "y": round(coords[1], 2), "z": round(coords[2], 2)}
        data["Ele{}".format(i)] = str(symbols[perm[i]])
        data["x{}".format(i)] = round(coords[0], 2)
        data["y{}".format(i)] = round(coords[1], 2)
        data["z{}".format(i)] = round(coords[2], 2)

    cell = ats.cell.cellpar()
    #data["Cellpar"] = {"a": round(cell[0], 1), "b": round(cell[1], 1), "c": round(cell[2], 1), \
    #"alpha": round(cell[3], 1), "beta": round(cell[4], 1), "gamma": round(cell[5], 1)}
    data["a"] = round(cell[0], 1)
    data["b"] = round(cell[1], 1)
    data["c"] = round(cell[2], 1)
    data["alpha"] = round(cell[3], 1)
    data["beta"] = round(cell[4], 1)
    data["gamma"] = round(cell[5], 1)
    data["Endpoint"] = "L"

    return data

def write_json(ats, perms, direct=True, wrap=True, sym=None):
    struct_entry = dict()
    property_list = ["dft_band_gap", "energy_above_hull"]
    if "dft_band_gap" in ats.info:
        for prop in property_list:
            struct_entry[prop] = ats.info[prop]
    else:
        for prop in property_list:
            struct_entry[prop] = None
    struct_entry["spacegroup"] = sym

    total_data = []
    for perm in perms:
        data = one_json(ats, perm, direct, wrap)
        total_data.append(data)
        ats2 = permute_axes(ats, [1,0,2])
        data2 = one_json(ats2, perm, direct, wrap)
        total_data.append(data2)
        ats3 = permute_axes(ats, [2,1,0])
        data3 = one_json(ats3, perm, direct, wrap)
        total_data.append(data3)
    struct_entry["variants"] = total_data

    return struct_entry

if __name__ == "__main__":
    from ase.io.trajectory import Trajectory
    from ase.io import write
    
    ats = Trajectory('train.traj')[0]
    ats2 = permute_axes(ats, [1,0,2])
    ats3 = permute_axes(ats, [2,1,0])
    write('POSCAR1.vasp', ats, format='vasp')
    write('POSCAR2.vasp', ats2, format='vasp')
    write('POSCAR3.vasp', ats3, format='vasp')
    
