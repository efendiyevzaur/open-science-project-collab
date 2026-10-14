#!/usr/bin/env python3
"""Generate a fictional epilepsy-surgery cohort for teaching purposes.

The output is entirely synthetic and must not be used to estimate clinical
effects or guide patient care. A small number of data-quality problems are
introduced deliberately so the raw dataset can support cleaning exercises.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 20261013
DEFAULT_N = 300


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def yes_no(rng: np.random.Generator, probability: float) -> str:
    return "Yes" if rng.random() < probability else "No"


def localization_result(
    rng: np.random.Generator,
    performed: bool,
    concordance_probability: float,
    nonlocalizing_probability: float,
) -> str:
    if not performed:
        return "Not performed"
    draw = rng.random()
    if draw < concordance_probability:
        return "Concordant"
    if draw < concordance_probability + nonlocalizing_probability:
        return "Non-localizing"
    return "Discordant"


def draw_age(rng: np.random.Generator) -> int:
    """Draw age at surgery from pediatric, young-adult, and older-adult groups."""
    group = rng.choice(["pediatric", "young_adult", "older_adult"], p=[0.22, 0.48, 0.30])
    if group == "pediatric":
        return int(rng.integers(2, 18))
    if group == "young_adult":
        return int(rng.integers(18, 46))
    return int(rng.integers(46, 76))


def draw_localization(rng: np.random.Generator, pediatric: bool) -> str:
    if pediatric:
        probabilities = [0.42, 0.25, 0.13, 0.20]
    else:
        probabilities = [0.63, 0.20, 0.10, 0.07]
    return str(
        rng.choice(
            [
                "Temporal",
                "Frontal",
                "Parietal/occipital",
                "Hemispheric/multilobar",
            ],
            p=probabilities,
        )
    )


def draw_pathology(
    rng: np.random.Generator, localization: str, pediatric: bool
) -> str:
    if localization == "Temporal" and not pediatric:
        values = [
            "Hippocampal sclerosis",
            "Focal cortical dysplasia",
            "Low-grade tumor",
            "Vascular malformation",
            "Gliosis/other",
            "No definitive pathology",
        ]
        probabilities = [0.44, 0.20, 0.12, 0.06, 0.10, 0.08]
    elif localization == "Hemispheric/multilobar":
        values = [
            "Focal cortical dysplasia",
            "Low-grade tumor",
            "Vascular malformation",
            "Gliosis/other",
            "No definitive pathology",
        ]
        probabilities = [0.49, 0.10, 0.08, 0.23, 0.10]
    else:
        values = [
            "Focal cortical dysplasia",
            "Low-grade tumor",
            "Vascular malformation",
            "Gliosis/other",
            "No definitive pathology",
        ]
        probabilities = [0.45, 0.20, 0.10, 0.15, 0.10]
    return str(rng.choice(values, p=probabilities))


def draw_surgery(
    rng: np.random.Generator,
    localization: str,
    pathology: str,
    pediatric: bool,
) -> str:
    discrete_lesion = pathology in {
        "Focal cortical dysplasia",
        "Low-grade tumor",
        "Vascular malformation",
    }
    if localization == "Temporal":
        if pathology == "Hippocampal sclerosis":
            return str(
                rng.choice(
                    ["Temporal resection", "Laser ablation"], p=[0.80, 0.20]
                )
            )
        if discrete_lesion:
            return str(
                rng.choice(
                    ["Temporal resection", "Lesionectomy", "Laser ablation"],
                    p=[0.45, 0.40, 0.15],
                )
            )
        return str(
            rng.choice(["Temporal resection", "Laser ablation"], p=[0.85, 0.15])
        )
    if localization == "Hemispheric/multilobar" and pediatric:
        return str(
            rng.choice(
                ["Hemispherotomy", "Extratemporal resection"], p=[0.66, 0.34]
            )
        )
    if discrete_lesion:
        return str(
            rng.choice(
                ["Lesionectomy", "Extratemporal resection", "Laser ablation"],
                p=[0.55, 0.35, 0.10],
            )
        )
    return str(
        rng.choice(
            ["Extratemporal resection", "Lesionectomy", "Laser ablation"],
            p=[0.70, 0.20, 0.10],
        )
    )


def generate_clean_cohort(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for index in range(1, n + 1):
        age_at_surgery = draw_age(rng)
        pediatric = age_at_surgery < 18
        maximum_onset = max(age_at_surgery, 1)
        onset_fraction = rng.beta(1.7, 2.1)
        age_at_onset = int(round(onset_fraction * maximum_onset))
        age_at_onset = min(age_at_onset, age_at_surgery)
        epilepsy_duration = age_at_surgery - age_at_onset

        sex = str(rng.choice(["Female", "Male"], p=[0.51, 0.49]))
        localization = draw_localization(rng, pediatric)
        laterality = str(
            rng.choice(
                ["Left", "Right", "Bilateral"],
                p=[0.46, 0.46, 0.08]
                if localization != "Hemispheric/multilobar"
                else [0.38, 0.38, 0.24],
            )
        )
        pathology = draw_pathology(rng, localization, pediatric)

        # Monthly frequency is right-skewed and tends to be higher in pediatric
        # and hemispheric/multilobar cases.
        log_mean = 1.25
        log_mean += 0.35 if pediatric else 0.0
        log_mean += 0.45 if localization == "Hemispheric/multilobar" else 0.0
        monthly_frequency = round(float(np.clip(rng.lognormal(log_mean, 0.75), 0.2, 90)), 1)

        lesion_pathologies = {
            "Hippocampal sclerosis",
            "Focal cortical dysplasia",
            "Low-grade tumor",
            "Vascular malformation",
        }
        lesion_probability = 0.86 if pathology in lesion_pathologies else 0.28
        mri_performed = rng.random() < 0.99
        mri_lesion = mri_performed and rng.random() < lesion_probability
        mri_concordance_probability = 0.82 if mri_lesion else 0.32
        mri_result = localization_result(
            rng,
            mri_performed,
            mri_concordance_probability,
            0.11 if mri_lesion else 0.45,
        )

        scalp_eeg_performed = rng.random() < 0.995
        scalp_concordance_probability = 0.66
        scalp_concordance_probability -= 0.18 if laterality == "Bilateral" else 0.0
        scalp_concordance_probability -= (
            0.10 if localization == "Hemispheric/multilobar" else 0.0
        )
        scalp_result = localization_result(
            rng, scalp_eeg_performed, scalp_concordance_probability, 0.22
        )

        noninvasive_uncertainty = (
            (not mri_lesion)
            or mri_result != "Concordant"
            or scalp_result != "Concordant"
        )
        pet_probability = 0.38 + (0.36 if noninvasive_uncertainty else 0.0)
        pet_performed = rng.random() < min(pet_probability, 0.90)
        pet_result = localization_result(
            rng,
            pet_performed,
            0.62 + (0.08 if localization == "Temporal" else 0.0),
            0.22,
        )

        meg_probability = 0.19
        meg_probability += 0.24 if pediatric else 0.0
        meg_probability += 0.26 if noninvasive_uncertainty else 0.0
        meg_performed = rng.random() < min(meg_probability, 0.82)
        meg_result = localization_result(
            rng,
            meg_performed,
            0.60 - (0.10 if localization == "Hemispheric/multilobar" else 0.0),
            0.25,
        )

        preliminary_results = [mri_result, scalp_result, pet_result, meg_result]
        concordant_preliminary = preliminary_results.count("Concordant")
        discordant_preliminary = preliminary_results.count("Discordant")
        ieeg_probability = 0.08
        ieeg_probability += 0.30 if not mri_lesion else 0.0
        ieeg_probability += 0.20 if discordant_preliminary > 0 else 0.0
        ieeg_probability += 0.18 if concordant_preliminary < 2 else 0.0
        ieeg_performed = rng.random() < min(ieeg_probability, 0.78)
        ieeg_result = localization_result(
            rng, ieeg_performed, 0.77, 0.13
        )

        all_results = preliminary_results + [ieeg_result]
        concordance_count = all_results.count("Concordant")
        performed_count = sum(result != "Not performed" for result in all_results)

        surgery_type = draw_surgery(rng, localization, pathology, pediatric)

        # Engel outcome depends probabilistically on features already generated.
        # Coefficients are simulation assumptions, not estimated clinical effects.
        favorable_score = -0.55
        favorable_score += 0.38 * concordance_count
        favorable_score += 0.32 if mri_lesion else -0.12
        favorable_score += 0.22 if pathology != "No definitive pathology" else -0.18
        favorable_score -= 0.018 * min(epilepsy_duration, 45)
        favorable_score -= 0.45 if laterality == "Bilateral" else 0.0
        favorable_score -= (
            0.30 if localization == "Hemispheric/multilobar" else 0.0
        )
        favorable_score += 0.22 if ieeg_result == "Concordant" else 0.0
        p_engel_i = float(np.clip(logistic(favorable_score), 0.18, 0.88))

        if rng.random() < p_engel_i:
            engel_outcome = "I"
        else:
            poor_score = (
                0.28 * discordant_preliminary
                + (0.45 if laterality == "Bilateral" else 0.0)
                + 0.012 * min(epilepsy_duration, 45)
            )
            p_iv_given_not_i = float(np.clip(logistic(poor_score - 1.55), 0.10, 0.48))
            if rng.random() < p_iv_given_not_i:
                engel_outcome = "IV"
            else:
                p_ii_given_middle = float(np.clip(0.58 + 0.05 * concordance_count, 0.50, 0.78))
                engel_outcome = "II" if rng.random() < p_ii_given_middle else "III"

        records.append(
            {
                "patient_id": f"SYN-{index:04d}",
                "sex": sex,
                "age_at_surgery_years": age_at_surgery,
                "age_group": "Pediatric" if pediatric else "Adult",
                "age_at_epilepsy_onset_years": age_at_onset,
                "epilepsy_duration_years": epilepsy_duration,
                "monthly_seizure_frequency": monthly_frequency,
                "epilepsy_localization": localization,
                "laterality": laterality,
                "pathology": pathology,
                "mri_performed": "Yes" if mri_performed else "No",
                "mri_lesion_detected": (
                    "Yes" if mri_lesion else "No" if mri_performed else "Not performed"
                ),
                "mri_localization_result": mri_result,
                "scalp_eeg_performed": "Yes" if scalp_eeg_performed else "No",
                "scalp_eeg_localization_result": scalp_result,
                "pet_performed": "Yes" if pet_performed else "No",
                "pet_localization_result": pet_result,
                "meg_performed": "Yes" if meg_performed else "No",
                "meg_localization_result": meg_result,
                "intracranial_eeg_performed": "Yes" if ieeg_performed else "No",
                "intracranial_eeg_localization_result": ieeg_result,
                "modalities_performed_count": performed_count,
                "multimodal_concordance_count": concordance_count,
                "surgery_type": surgery_type,
                "engel_outcome_1yr": engel_outcome,
            }
        )

    return pd.DataFrame.from_records(records)


def add_deliberate_data_quality_issues(
    cohort: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Introduce a small, reproducible set of issues into the raw dataset."""
    rng = np.random.default_rng(seed + 991)
    raw = cohort.copy()
    available = list(raw.index)
    rng.shuffle(available)

    cursor = 0

    def take(count: int) -> list[int]:
        nonlocal cursor
        selected = available[cursor : cursor + count]
        cursor += count
        return selected

    # Unexplained missing values, distinct from explicit "Not performed" values.
    missing_frequency = take(3)
    raw.loc[missing_frequency, "monthly_seizure_frequency"] = np.nan

    missing_pathology = take(2)
    raw.loc[missing_pathology, "pathology"] = np.nan

    performed_result_rows = raw.index[
        (raw["pet_performed"] == "Yes") & raw["pet_localization_result"].notna()
    ].tolist()
    missing_pet_result = list(rng.choice(performed_result_rows, size=2, replace=False))
    raw.loc[missing_pet_result, "pet_localization_result"] = np.nan

    # Inconsistent capitalization in an otherwise categorical field.
    inconsistent_case = take(3)
    raw.loc[inconsistent_case, "epilepsy_localization"] = (
        raw.loc[inconsistent_case, "epilepsy_localization"].str.lower()
    )

    # Internal inconsistencies and an extreme value for validation practice.
    onset_after_surgery = take(1)[0]
    raw.loc[onset_after_surgery, "age_at_epilepsy_onset_years"] = (
        raw.loc[onset_after_surgery, "age_at_surgery_years"] + 3
    )

    negative_duration = take(1)[0]
    raw.loc[negative_duration, "epilepsy_duration_years"] = -2

    extreme_frequency = take(1)[0]
    raw.loc[extreme_frequency, "monthly_seizure_frequency"] = 999.0

    # Append two exact duplicate records, including their patient identifiers.
    duplicate_source_rows = take(2)
    raw = pd.concat([raw, raw.loc[duplicate_source_rows]], ignore_index=True)

    issue_counts = {
        "unexplained_missing_values": 7,
        "inconsistent_category_case_rows": 3,
        "onset_after_surgery_rows": 1,
        "negative_duration_rows": 1,
        "extreme_frequency_rows": 1,
        "duplicate_rows_appended": 2,
    }
    return raw, issue_counts


def build_data_dictionary() -> pd.DataFrame:
    rows = [
        ("patient_id", "identifier", "", "Synthetic patient identifier; duplicates indicate duplicated records."),
        ("sex", "categorical", "Female; Male", "Simulated sex category."),
        ("age_at_surgery_years", "integer", "years", "Age at epilepsy surgery."),
        ("age_group", "categorical", "Pediatric; Adult", "Derived from age at surgery using <18 years as pediatric."),
        ("age_at_epilepsy_onset_years", "integer", "years", "Age at first epilepsy onset."),
        ("epilepsy_duration_years", "integer", "years", "Age at surgery minus age at onset before deliberate raw-data issues are added."),
        ("monthly_seizure_frequency", "numeric", "seizures/month", "Approximate preoperative monthly seizure frequency."),
        ("epilepsy_localization", "categorical", "Temporal; Frontal; Parietal/occipital; Hemispheric/multilobar", "Primary presurgical localization category."),
        ("laterality", "categorical", "Left; Right; Bilateral", "Primary lateralization category."),
        ("pathology", "categorical", "Hippocampal sclerosis; Focal cortical dysplasia; Low-grade tumor; Vascular malformation; Gliosis/other; No definitive pathology", "Simulated postoperative pathology category."),
        ("mri_performed", "categorical", "Yes; No", "Whether MRI was performed."),
        ("mri_lesion_detected", "categorical", "Yes; No; Not performed", "Whether a potentially relevant lesion was detected."),
        ("mri_localization_result", "categorical", "Concordant; Discordant; Non-localizing; Not performed", "MRI localization relative to the eventual surgical target."),
        ("scalp_eeg_performed", "categorical", "Yes; No", "Whether scalp EEG was performed."),
        ("scalp_eeg_localization_result", "categorical", "Concordant; Discordant; Non-localizing; Not performed", "Scalp EEG localization relative to the eventual surgical target."),
        ("pet_performed", "categorical", "Yes; No", "Whether PET was performed; more likely when noninvasive results are uncertain."),
        ("pet_localization_result", "categorical", "Concordant; Discordant; Non-localizing; Not performed", "PET localization relative to the eventual surgical target."),
        ("meg_performed", "categorical", "Yes; No", "Whether MEG was performed; more likely in pediatric or uncertain cases."),
        ("meg_localization_result", "categorical", "Concordant; Discordant; Non-localizing; Not performed", "MEG localization relative to the eventual surgical target."),
        ("intracranial_eeg_performed", "categorical", "Yes; No", "Whether intracranial EEG was performed; more likely in uncertain or discordant cases."),
        ("intracranial_eeg_localization_result", "categorical", "Concordant; Discordant; Non-localizing; Not performed", "Intracranial EEG localization relative to the eventual surgical target."),
        ("modalities_performed_count", "integer", "count", "Number of MRI, scalp EEG, PET, MEG, and intracranial EEG investigations performed."),
        ("multimodal_concordance_count", "integer", "count", "Number of performed modalities with a concordant localization result before deliberate missingness is added."),
        ("surgery_type", "categorical", "Temporal resection; Extratemporal resection; Lesionectomy; Laser ablation; Hemispherotomy", "Simulated surgical procedure."),
        ("engel_outcome_1yr", "ordinal", "I; II; III; IV", "One-year Engel class: I free of disabling seizures; II rare disabling seizures; III worthwhile improvement; IV no worthwhile improvement."),
    ]
    return pd.DataFrame(rows, columns=["variablename", "type", "unit_or_labels", "description"])


def write_outputs(output_dir: Path, n: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_cohort = generate_clean_cohort(n=n, seed=seed)
    raw_cohort, issue_counts = add_deliberate_data_quality_issues(clean_cohort, seed=seed)
    dictionary = build_data_dictionary()

    data_path = output_dir / "simulated_epilepsy_surgery_raw.csv"
    dictionary_path = output_dir / "epilepsy_surgery_data_dictionary.csv"
    raw_cohort.to_csv(data_path, index=False, na_rep="")
    dictionary.to_csv(dictionary_path, index=False)

    print(f"Wrote {len(raw_cohort)} rows to {data_path}")
    print(f"Wrote {len(dictionary)} variable definitions to {dictionary_path}")
    print(f"Random seed: {seed}")
    print(f"Base patients: {n}")
    print(f"Deliberate issue counts: {issue_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of unique patients")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated CSV files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.n < 50:
        raise ValueError("Use at least 50 patients so the simulated categories remain useful.")
    write_outputs(arguments.output_dir, arguments.n, arguments.seed)
