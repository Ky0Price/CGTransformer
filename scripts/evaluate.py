import itertools
import numpy as np
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import smact
from smact.screening import pauling_test
from ase.neighborlist import neighbor_list
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from constants import CompScalerMeans, CompScalerStds
from data_utils import StandardScaler
from matbench_genmetrics.core.metrics import GenMetrics
from rich.progress import track
import os


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

CompScaler = StandardScaler(means=np.array(CompScalerMeans), stds=np.array(CompScalerStds), replace_nan_token=0.)

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

def smact_validity(atoms, use_pauling_test=True, include_alloys=True):
    """
    Check whether the chemical composition of an ASE Atoms object is chemically valid
    according to SMACT-based oxidation-state screening.

    Input:
        atoms (ase.Atoms): Structure to be checked.
        use_pauling_test (bool): Whether to apply the Pauling electronegativity test.
        include_alloys (bool): Whether to automatically accept all-metal systems.

    Output:
        None. The result is written to atoms.info["comp_valid"].
    """
    # Get chemical symbols from the ASE Atoms object
    atom_types = atoms.get_chemical_symbols()

    # Count each unique element and enforce a deterministic element order
    elem_counter = Counter(atom_types)
    ordered_elems = tuple(sorted(elem_counter.keys()))

    # Build the reduced integer stoichiometry in the same order as ordered_elems
    counts = np.array([elem_counter[elem] for elem in ordered_elems], dtype=int)
    gcd_value = np.gcd.reduce(counts)
    counts = counts // gcd_value
    count = tuple(counts.tolist())

    # Single-element systems are always accepted
    if len(ordered_elems) == 1:
        atoms.info["comp_valid"] = True
        return

    # All-metal systems can be directly accepted if alloy mode is enabled
    if include_alloys:
        is_metal_list = [elem in smact.metals for elem in ordered_elems]
        if all(is_metal_list):
            atoms.info["comp_valid"] = True
            return

    # Get SMACT element objects in the exact same order as ordered_elems
    space = smact.element_dictionary(ordered_elems)
    smact_elems = [space[elem] for elem in ordered_elems]

    # Extract Pauling electronegativities and oxidation-state candidates
    electronegs = [elem.pauling_eneg for elem in smact_elems]
    ox_combos = [elem.oxidation_states for elem in smact_elems]

    # Guard against combinatorial explosion, following the published logic
    oxn = 1
    for ox_states in ox_combos:
        oxn *= len(ox_states)
    if oxn > 1e7:
        atoms.info["comp_valid"] = False
        return

    # Use the maximum reduced stoichiometric coefficient as the neutral-ratio threshold
    threshold = int(np.max(count))
    stoichs = [(c,) for c in count]

    # Enumerate oxidation-state combinations and return immediately once a valid one is found
    for ox_states in itertools.product(*ox_combos):
        # Check charge neutrality
        cn_e, cn_r = smact.neutral_ratios(
            ox_states,
            stoichs=stoichs,
            threshold=threshold,
        )

        if not cn_e:
            continue

        # Optionally apply the Pauling electronegativity test
        if use_pauling_test:
            try:
                electroneg_ok = pauling_test(ox_states, electronegs)
            except TypeError:
                # If electronegativity data are missing, follow the common fallback behavior
                electroneg_ok = True
        else:
            electroneg_ok = True

        if electroneg_ok:
            atoms.info["comp_valid"] = True
            return

    # No valid oxidation-state assignment was found
    atoms.info["comp_valid"] = False

def get_fingerprints(atoms):
    """
    Calculate the fingerprint features of the given atomic structure.

    Args:
        atoms (ase.Atoms): The atomic structure to calculate fingerprint features for.

    Returns:
        None: The results are stored in the `atoms.info` dictionary with keys "comp_fp" and "struct_fp".
    """
    # Get the list of chemical symbols in the atomic structure
    comp_ase = atoms.get_chemical_symbols()
    # Count the occurrences of each element
    elem_counter = Counter(comp_ase)
    # Create a Composition object
    comp = Composition(elem_counter)
    # Calculate and store the composition fingerprint features
    atoms.info["comp_fp"] = CompFP.featurize(comp)
    try:
        # Convert the ASE atomic structure to a pymatgen structure
        structure = AseAtomsAdaptor.get_structure(atoms)
        # Calculate and store the site fingerprint features
        site_fps = [CrystalNNFP.featurize(structure, i) for i in range(len(structure))]
    except Exception:
        # If the fingerprint features cannot be constructed, mark the atomic structure as invalid
        atoms.info["valid"] = False
        atoms.info["comp_fp"] = None
        atoms.info["struct_fp"] = None
        return
    # Calculate and store the average site fingerprint feature
    atoms.info["struct_fp"] = np.array(site_fps).mean(axis=0)

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps

def get_validity_one(atoms, cutoff=0.5):
    try:
        pos = atoms.get_scaled_positions()
        cell = np.array(atoms.get_cell())

        if np.all(np.isnan(pos)==False) and np.all(np.isnan(cell)==False) and atoms.get_volume() > 0.1:
            atoms.info["constructed"] = True
        else:
            atoms.info["constructed"] = False
    except Exception:
        atoms.info["constructed"] = False

    if atoms.info["constructed"]:
        try:
            i_indices = neighbor_list("i", atoms, cutoff=cutoff, max_nbins=100.0)
            atoms.info["struct_valid"] = (len(i_indices) == 0)
        except Exception:
            atoms.info["struct_valid"] = False
    else:
        atoms.info["struct_valid"] = False

    atoms.info["valid"] = atoms.info["comp_valid"] and atoms.info["struct_valid"]

def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()

def get_match_rms_one(
    struct_pred,
    struct_true,
    matcher,
    is_pred_valid=True,
    require_same_formula=True,
):
    """
    Compute RMSD for one predicted structure against one target structure.

    Args:
        struct_pred: Predicted ASE Atoms or pymatgen Structure.
        struct_true: Target ASE Atoms or pymatgen Structure.
        matcher: A pymatgen StructureMatcher instance.
        is_pred_valid (bool): Whether the predicted structure is considered valid.
        require_same_formula (bool): Whether to require identical chemical formulas.

    Returns:
        float or None:
            - float if matched successfully
            - None if invalid or unmatched
    """
    if not is_pred_valid:
        return None

    try:
        if require_same_formula:
            if struct_pred.get_chemical_formula() != struct_true.get_chemical_formula():
                return None

        pred_pm = AseAtomsAdaptor.get_structure(struct_pred)
        true_pm = AseAtomsAdaptor.get_structure(struct_true)

        rms_dist = matcher.get_rms_dist(pred_pm, true_pm)
        return None if rms_dist is None else rms_dist[0]
    except Exception:
        return None
    
def get_match_rms(
    gen_struct_groups,
    true_structs,
    matcher,
    validity_fn=None,
    require_same_formula=True,
    return_best_matches=False,
    traj_match_pred_path=None,
    traj_match_true_path=None,
):
    """
    Compute match rate and mean RMSD in a generic grouped format.

    This function can optionally save the best matched prediction-target pairs
    into two aligned trajectory files for later analysis and visualization.
    """
    if len(gen_struct_groups) != len(true_structs):
        raise ValueError("gen_struct_groups and true_structs must have the same length.")

    if (traj_match_pred_path is None) != (traj_match_true_path is None):
        raise ValueError(
            "traj_match_pred_path and traj_match_true_path must be both provided or both omitted."
        )

    need_store_best_pairs = return_best_matches or (
        traj_match_pred_path is not None and traj_match_true_path is not None
    )

    rms_dists = []
    best_matches = []
    best_true_structs = []

    for candidates, true_struct in zip(gen_struct_groups, true_structs):
        best_rms = None
        best_pred = None

        for pred_struct in candidates:
            is_pred_valid = validity_fn(pred_struct) if validity_fn is not None else True

            rms_dist = get_match_rms_one(
                struct_pred=pred_struct,
                struct_true=true_struct,
                matcher=matcher,
                is_pred_valid=is_pred_valid,
                require_same_formula=require_same_formula,
            )

            if rms_dist is not None:
                if best_rms is None or rms_dist < best_rms:
                    best_rms = rms_dist
                    best_pred = pred_struct

        rms_dists.append(best_rms)

        if need_store_best_pairs:
            best_matches.append(best_pred)
            best_true_structs.append(true_struct if best_pred is not None else None)

    # Save aligned best-match pairs for later analysis.
    if traj_match_pred_path is not None and traj_match_true_path is not None:
        with Trajectory(traj_match_pred_path, "w") as pred_traj, Trajectory(traj_match_true_path, "w") as true_traj:
            for best_pred, true_struct in zip(best_matches, best_true_structs):
                if best_pred is not None and true_struct is not None:
                    pred_traj.write(best_pred)
                    true_traj.write(true_struct)

    n_match = sum(item is not None for item in rms_dists)
    denominator = len(true_structs)
    match_rate = n_match / denominator if denominator > 0 else 0.0

    matched_rms = [item for item in rms_dists if item is not None]
    mean_rms_dist = float(np.mean(matched_rms)) if len(matched_rms) > 0 else None

    result = {
        "n_match": n_match,
        "denominator": denominator,
        "match_rate": match_rate,
        "mean_rms_dist": mean_rms_dist,
        "rms_dists": np.array(rms_dists, dtype=object),
    }

    if return_best_matches:
        result["best_matches"] = best_matches
        result["best_true_structs"] = best_true_structs

    return result

class GenEval:

    def __init__(self, traj_pred, traj_true, traj_train, traj_test, traj_match_path, Eval):
        self.traj_pred = traj_pred
        self.traj_true = traj_true
        self.traj_train = traj_train
        self.traj_test = traj_test
        self.traj_match_path = traj_match_path
        self.Eval = Eval
        self.n_samples = Eval.get("n_samples", None)

        valid_ats = [atoms for atoms in traj_pred if atoms.info["valid"]]
        if self.n_samples is None:
            self.n_samples = int(0.8 * len(valid_ats))
        #print(type(self.n_samples))
        if len(valid_ats) >= self.n_samples:
            sampled_indices = np.random.choice(len(valid_ats), self.n_samples, replace=False)
            self.valid_samples = [valid_ats[i] for i in sampled_indices]
        else:
            raise Exception(f'not enough valid crystals in the predicted set: {len(valid_ats)}/{self.n_samples}')

    def compute_cov(self, struc_cutoff, comp_cutoff, num_gen_crystals=None):    
        struc_fps = [atoms.info["struct_fp"] for atoms in self.traj_pred]
        comp_fps = [atoms.info["comp_fp"] for atoms in self.traj_pred]
        gt_struc_fps = [atoms.info["struct_fp"] for atoms in self.traj_true]
        gt_comp_fps = [atoms.info["comp_fp"] for atoms in self.traj_true]

        assert len(struc_fps) == len(comp_fps)
        assert len(gt_struc_fps) == len(gt_comp_fps)

        if num_gen_crystals is None:
            num_gen_crystals = len(struc_fps)

        struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)
        gt_struc_fps, gt_comp_fps = filter_fps(gt_struc_fps, gt_comp_fps)

        comp_fps = CompScaler.transform(comp_fps)
        gt_comp_fps = CompScaler.transform(gt_comp_fps)

        struc_fps = np.array(struc_fps)
        gt_struc_fps = np.array(gt_struc_fps)
        comp_fps = np.array(comp_fps)
        gt_comp_fps = np.array(gt_comp_fps)

        struc_pdist = cdist(struc_fps, gt_struc_fps)
        comp_pdist = cdist(comp_fps, gt_comp_fps)

        struc_recall_dist = struc_pdist.min(axis=0)
        struc_precision_dist = struc_pdist.min(axis=1)
        comp_recall_dist = comp_pdist.min(axis=0)
        comp_precision_dist = comp_pdist.min(axis=1)

        cov_recall = np.mean(np.logical_and(struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff))
        cov_precision = np.sum(np.logical_and(struc_precision_dist <= struc_cutoff, comp_precision_dist <= comp_cutoff)) / num_gen_crystals

        metrics_dict = {
            'cov_recall': cov_recall,
            'cov_precision': cov_precision,
            'amsd_recall': np.mean(struc_recall_dist),
            'amsd_precision': np.mean(struc_precision_dist),
            'amcd_recall': np.mean(comp_recall_dist),
            'amcd_precision': np.mean(comp_precision_dist),
        }

        combined_dist_dict = {
            'struc_recall_dist': struc_recall_dist.tolist(),
            'struc_precision_dist': struc_precision_dist.tolist(),
            'comp_recall_dist': comp_recall_dist.tolist(),
            'comp_precision_dist': comp_precision_dist.tolist(),
        }

        return metrics_dict, combined_dist_dict
    
    def get_validity(self):
        comp_valid = np.array([atoms.info["comp_valid"] for atoms in self.traj_pred]).mean()
        struct_valid = np.array([atoms.info["struct_valid"] for atoms in self.traj_pred]).mean()
        valid = np.array([atoms.info["valid"] for atoms in self.traj_pred]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [atoms.info["comp_fp"] for atoms in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([atoms.info["struct_fp"] for atoms in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [AseAtomsAdaptor.get_structure(atoms).density for atoms in self.valid_samples]
        gt_densities = [AseAtomsAdaptor.get_structure(atoms).density for atoms in self.traj_true]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(atoms.get_chemical_symbols())) for atoms in self.valid_samples]
        gt_nelems = [len(set(atoms.get_chemical_symbols())) for atoms in self.traj_true]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_coverage(self):
        cutoff_dict = self.Eval["cov_cutoffs"]
        (cov_metrics_dict, combined_dist_dict) = self.compute_cov(
            struc_cutoff=float(cutoff_dict['struc']),
            comp_cutoff=float(cutoff_dict['comp']))
        return cov_metrics_dict
    
    def get_novelty_unique(self):
        train_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in self.traj_train]
        test_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in self.traj_test]
        gen_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in self.traj_pred]
        gen_metrics = GenMetrics(train_structures=train_structures, test_structures=test_structures, gen_structures=gen_structures)

        return {'novelty': gen_metrics.novelty, 'uniqueness': gen_metrics.uniqueness}

    def get_metrics(self):
        metrics = {}
        if self.Eval["calc_validity"]:
            metrics.update(self.get_validity())
        if self.Eval["calc_comp_div"]:
            try:
                metrics.update(self.get_comp_diversity())
            except:
                metrics.update({'comp_div': None})
        if self.Eval["calc_struct_div"]:
            metrics.update(self.get_struct_diversity())
        if self.Eval["calc_wdist_density"]:
            metrics.update(self.get_density_wdist())
        if self.Eval["calc_wdist_num_elems"]:
            metrics.update(self.get_num_elem_wdist())
        if self.Eval["calc_novelty_unique"]:
            metrics.update(self.get_novelty_unique())
        if self.Eval["calc_coverage"]:
            metrics.update(self.get_coverage())
        return metrics