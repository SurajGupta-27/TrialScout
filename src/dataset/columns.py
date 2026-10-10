"""Every column in the dataset: its type, where it comes from, and whether it is known at trial start.

This list is the single source for the data dictionary (data/DATA_DICTIONARY.md),
so the documentation can't drift away from the code.

"known_at_start" values:
    yes      - decided when the trial is designed/registered.
    caution  - planned at start, but often edited later, and the registry only keeps
               the latest version (e.g. sites added or removed while the trial runs).
    no       - only known during or after the trial. LEAKAGE: never use as a feature.
    label    - the target (or the raw status it comes from).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    """Description of one dataset column."""

    name: str
    dtype: str
    source: str  # path in the API record
    meaning: str
    known_at_start: str


P = "protocolSection"
COLUMNS: list[Column] = [
    # --- identity and label
    Column("nct_id", "string", f"{P}.identificationModule.nctId", "ClinicalTrials.gov ID", "yes"),
    Column("brief_title", "string", f"{P}.identificationModule.briefTitle", "Short title", "yes"),
    Column("overall_status", "string", f"{P}.statusModule.overallStatus",
           "Final status: COMPLETED or TERMINATED", "label"),
    Column("label", "int", "derived from overall_status", "Target: 0 = COMPLETED, 1 = TERMINATED", "label"),

    # --- dates
    Column("start_date", "date", f"{P}.statusModule.startDateStruct.date",
           "Trial start (month-only dates are set to the 1st)", "yes"),
    Column("start_year", "int", "derived from start_date", "Year the trial started (used for the time split)", "yes"),
    Column("start_date_type", "string", f"{P}.statusModule.startDateStruct.type",
           "ACTUAL or ESTIMATED; empty in many older records", "caution"),
    Column("first_submit_date", "date", f"{P}.statusModule.studyFirstSubmitDate",
           "Date first submitted to the registry", "yes"),
    Column("primary_completion_date", "date", f"{P}.statusModule.primaryCompletionDateStruct.date",
           "When the primary outcome data collection ended", "no"),
    Column("completion_date", "date", f"{P}.statusModule.completionDateStruct.date",
           "When the whole trial ended", "no"),
    Column("last_update_date", "date", f"{P}.statusModule.lastUpdatePostDateStruct.date",
           "Last time the record was updated", "no"),
    Column("why_stopped", "string", f"{P}.statusModule.whyStopped",
           "Free-text reason a trial stopped (almost only filled for TERMINATED)", "no"),
    Column("has_results", "bool", "hasResults", "Results were posted to the registry", "no"),

    # --- design
    Column("phases", "list[string]", f"{P}.designModule.phases", "Phase codes, e.g. [PHASE2, PHASE3]", "yes"),
    Column("phase", "string", "derived from phases", "Phases joined with '/', e.g. PHASE2/PHASE3; NA = not applicable",
           "yes"),
    Column("primary_purpose", "string", f"{P}.designModule.designInfo.primaryPurpose",
           "TREATMENT, PREVENTION, BASIC_SCIENCE, ...", "yes"),
    Column("allocation", "string", f"{P}.designModule.designInfo.allocation", "RANDOMIZED, NON_RANDOMIZED or NA",
           "yes"),
    Column("intervention_model", "string", f"{P}.designModule.designInfo.interventionModel",
           "PARALLEL, SINGLE_GROUP, CROSSOVER, ...", "yes"),
    Column("masking", "string", f"{P}.designModule.designInfo.maskingInfo.masking",
           "NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE", "yes"),
    # Reclassified caution -> no in Phase 8: ~98% of finished trials hold the ACTUAL count.
    Column("enrollment_count", "int", f"{P}.designModule.enrollmentInfo.count",
           "Number of participants. Planned at start, but replaced by the ACTUAL number at the end "
           "(true for ~98% of rows), so it is leakage", "no"),
    Column("enrollment_type", "string", f"{P}.designModule.enrollmentInfo.type",
           "ACTUAL or ESTIMATED: tells whether enrollment_count was updated after the trial", "no"),

    # --- arms and interventions
    Column("n_arms", "int", f"{P}.armsInterventionsModule.armGroups", "Number of arms (groups)", "yes"),
    Column("n_interventions", "int", f"{P}.armsInterventionsModule.interventions", "Number of interventions",
           "yes"),
    Column("intervention_types", "list[string]", f"{P}.armsInterventionsModule.interventions[].type",
           "Distinct intervention types: DRUG, DEVICE, BEHAVIORAL, ...", "yes"),
    Column("intervention_mesh_terms", "list[string]", "derivedSection.interventionBrowseModule.meshes[].term",
           "MeSH terms for the interventions (added by the NLM)", "yes"),

    # --- sponsor and oversight
    Column("lead_sponsor_name", "string", f"{P}.sponsorCollaboratorsModule.leadSponsor.name", "Lead sponsor",
           "yes"),
    Column("lead_sponsor_class", "string", f"{P}.sponsorCollaboratorsModule.leadSponsor.class",
           "INDUSTRY, NIH, OTHER_GOV, OTHER, ...", "yes"),
    Column("n_collaborators", "int", f"{P}.sponsorCollaboratorsModule.collaborators", "Number of collaborators",
           "caution"),
    Column("responsible_party_type", "string", f"{P}.sponsorCollaboratorsModule.responsibleParty.type",
           "SPONSOR, PRINCIPAL_INVESTIGATOR or SPONSOR_INVESTIGATOR", "yes"),
    Column("has_dmc", "bool", f"{P}.oversightModule.oversightHasDmc", "Has a data monitoring committee", "yes"),
    Column("is_fda_regulated_drug", "bool", f"{P}.oversightModule.isFdaRegulatedDrug",
           "Studies an FDA-regulated drug (newer field, often empty for older records)", "yes"),
    Column("is_fda_regulated_device", "bool", f"{P}.oversightModule.isFdaRegulatedDevice",
           "Studies an FDA-regulated device (newer field, often empty for older records)", "yes"),

    # --- eligibility
    Column("healthy_volunteers", "bool", f"{P}.eligibilityModule.healthyVolunteers", "Accepts healthy volunteers",
           "yes"),
    Column("sex", "string", f"{P}.eligibilityModule.sex", "ALL, FEMALE or MALE", "yes"),
    Column("min_age_years", "float", f"{P}.eligibilityModule.minimumAge", "Minimum age, converted to years",
           "yes"),
    Column("max_age_years", "float", f"{P}.eligibilityModule.maximumAge",
           "Maximum age, converted to years (empty = no upper limit)", "yes"),
    Column("std_ages", "list[string]", f"{P}.eligibilityModule.stdAges", "CHILD, ADULT, OLDER_ADULT", "yes"),
    Column("eligibility_criteria_chars", "int", f"{P}.eligibilityModule.eligibilityCriteria",
           "Length of the eligibility criteria text (a rough complexity measure)", "caution"),

    # --- conditions and outcomes
    Column("conditions", "list[string]", f"{P}.conditionsModule.conditions", "Conditions as written by the sponsor",
           "yes"),
    Column("n_conditions", "int", f"{P}.conditionsModule.conditions", "Number of conditions", "yes"),
    Column("n_keywords", "int", f"{P}.conditionsModule.keywords", "Number of keywords", "yes"),
    Column("condition_mesh_terms", "list[string]", "derivedSection.conditionBrowseModule.meshes[].term",
           "MeSH terms for the conditions (added by the NLM)", "yes"),
    Column("condition_mesh_ancestors", "list[string]", "derivedSection.conditionBrowseModule.ancestors[].term",
           "Broader MeSH terms, useful for grouping into disease areas", "yes"),
    Column("n_primary_outcomes", "int", f"{P}.outcomesModule.primaryOutcomes", "Number of primary outcomes",
           "caution"),
    Column("n_secondary_outcomes", "int", f"{P}.outcomesModule.secondaryOutcomes", "Number of secondary outcomes",
           "caution"),

    # --- locations
    Column("n_locations", "int", f"{P}.contactsLocationsModule.locations", "Number of sites (latest record)",
           "caution"),
    Column("n_countries", "int", f"{P}.contactsLocationsModule.locations[].country", "Number of distinct countries",
           "caution"),
    Column("countries", "list[string]", f"{P}.contactsLocationsModule.locations[].country",
           "Distinct site countries, sorted", "caution"),
]

COLUMN_NAMES = [c.name for c in COLUMNS]
LEAKAGE_COLUMNS = [c.name for c in COLUMNS if c.known_at_start == "no"]
