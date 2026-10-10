# Data dictionary – `data/processed/trials.parquet`

Generated from `src/dataset/columns.py` by `python -m src.dataset.build`. One row per trial.

**Known at start** says whether a column could be used to predict the outcome *when the trial begins*:
- **yes**: decided when the trial is designed or registered.
- **caution**: planned at start but often edited later. The registry keeps only the latest version, so the value may reflect what happened (e.g. sites added or removed during the trial).
- **no**: only known during or after the trial. **Leakage: never use as a feature.** This includes `enrollment_count`, which holds the *actual* final number for ~98% of finished trials.
- **label**: the target.

Note: every value comes from the *current* registry record, not a copy from the start date. Which columns become features, and why the others don't, is in `data/LEAKAGE_AUDIT.md` (Phase 8).

| Column | Type | Known at start | Meaning | API source |
|---|---|---|---|---|
| `nct_id` | string | yes | ClinicalTrials.gov ID | `protocolSection.identificationModule.nctId` |
| `brief_title` | string | yes | Short title | `protocolSection.identificationModule.briefTitle` |
| `overall_status` | string | label | Final status: COMPLETED or TERMINATED | `protocolSection.statusModule.overallStatus` |
| `label` | int | label | Target: 0 = COMPLETED, 1 = TERMINATED | `derived from overall_status` |
| `start_date` | date | yes | Trial start (month-only dates are set to the 1st) | `protocolSection.statusModule.startDateStruct.date` |
| `start_year` | int | yes | Year the trial started (used for the time split) | `derived from start_date` |
| `start_date_type` | string | caution | ACTUAL or ESTIMATED; empty in many older records | `protocolSection.statusModule.startDateStruct.type` |
| `first_submit_date` | date | yes | Date first submitted to the registry | `protocolSection.statusModule.studyFirstSubmitDate` |
| `primary_completion_date` | date | no | When the primary outcome data collection ended | `protocolSection.statusModule.primaryCompletionDateStruct.date` |
| `completion_date` | date | no | When the whole trial ended | `protocolSection.statusModule.completionDateStruct.date` |
| `last_update_date` | date | no | Last time the record was updated | `protocolSection.statusModule.lastUpdatePostDateStruct.date` |
| `why_stopped` | string | no | Free-text reason a trial stopped (almost only filled for TERMINATED) | `protocolSection.statusModule.whyStopped` |
| `has_results` | bool | no | Results were posted to the registry | `hasResults` |
| `phases` | list[string] | yes | Phase codes, e.g. [PHASE2, PHASE3] | `protocolSection.designModule.phases` |
| `phase` | string | yes | Phases joined with '/', e.g. PHASE2/PHASE3; NA = not applicable | `derived from phases` |
| `primary_purpose` | string | yes | TREATMENT, PREVENTION, BASIC_SCIENCE, ... | `protocolSection.designModule.designInfo.primaryPurpose` |
| `allocation` | string | yes | RANDOMIZED, NON_RANDOMIZED or NA | `protocolSection.designModule.designInfo.allocation` |
| `intervention_model` | string | yes | PARALLEL, SINGLE_GROUP, CROSSOVER, ... | `protocolSection.designModule.designInfo.interventionModel` |
| `masking` | string | yes | NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE | `protocolSection.designModule.designInfo.maskingInfo.masking` |
| `enrollment_count` | int | no | Number of participants. Planned at start, but replaced by the ACTUAL number at the end (true for ~98% of rows), so it is leakage | `protocolSection.designModule.enrollmentInfo.count` |
| `enrollment_type` | string | no | ACTUAL or ESTIMATED: tells whether enrollment_count was updated after the trial | `protocolSection.designModule.enrollmentInfo.type` |
| `n_arms` | int | yes | Number of arms (groups) | `protocolSection.armsInterventionsModule.armGroups` |
| `n_interventions` | int | yes | Number of interventions | `protocolSection.armsInterventionsModule.interventions` |
| `intervention_types` | list[string] | yes | Distinct intervention types: DRUG, DEVICE, BEHAVIORAL, ... | `protocolSection.armsInterventionsModule.interventions[].type` |
| `intervention_mesh_terms` | list[string] | yes | MeSH terms for the interventions (added by the NLM) | `derivedSection.interventionBrowseModule.meshes[].term` |
| `lead_sponsor_name` | string | yes | Lead sponsor | `protocolSection.sponsorCollaboratorsModule.leadSponsor.name` |
| `lead_sponsor_class` | string | yes | INDUSTRY, NIH, OTHER_GOV, OTHER, ... | `protocolSection.sponsorCollaboratorsModule.leadSponsor.class` |
| `n_collaborators` | int | caution | Number of collaborators | `protocolSection.sponsorCollaboratorsModule.collaborators` |
| `responsible_party_type` | string | yes | SPONSOR, PRINCIPAL_INVESTIGATOR or SPONSOR_INVESTIGATOR | `protocolSection.sponsorCollaboratorsModule.responsibleParty.type` |
| `has_dmc` | bool | yes | Has a data monitoring committee | `protocolSection.oversightModule.oversightHasDmc` |
| `is_fda_regulated_drug` | bool | yes | Studies an FDA-regulated drug (newer field, often empty for older records) | `protocolSection.oversightModule.isFdaRegulatedDrug` |
| `is_fda_regulated_device` | bool | yes | Studies an FDA-regulated device (newer field, often empty for older records) | `protocolSection.oversightModule.isFdaRegulatedDevice` |
| `healthy_volunteers` | bool | yes | Accepts healthy volunteers | `protocolSection.eligibilityModule.healthyVolunteers` |
| `sex` | string | yes | ALL, FEMALE or MALE | `protocolSection.eligibilityModule.sex` |
| `min_age_years` | float | yes | Minimum age, converted to years | `protocolSection.eligibilityModule.minimumAge` |
| `max_age_years` | float | yes | Maximum age, converted to years (empty = no upper limit) | `protocolSection.eligibilityModule.maximumAge` |
| `std_ages` | list[string] | yes | CHILD, ADULT, OLDER_ADULT | `protocolSection.eligibilityModule.stdAges` |
| `eligibility_criteria_chars` | int | caution | Length of the eligibility criteria text (a rough complexity measure) | `protocolSection.eligibilityModule.eligibilityCriteria` |
| `conditions` | list[string] | yes | Conditions as written by the sponsor | `protocolSection.conditionsModule.conditions` |
| `n_conditions` | int | yes | Number of conditions | `protocolSection.conditionsModule.conditions` |
| `n_keywords` | int | yes | Number of keywords | `protocolSection.conditionsModule.keywords` |
| `condition_mesh_terms` | list[string] | yes | MeSH terms for the conditions (added by the NLM) | `derivedSection.conditionBrowseModule.meshes[].term` |
| `condition_mesh_ancestors` | list[string] | yes | Broader MeSH terms, useful for grouping into disease areas | `derivedSection.conditionBrowseModule.ancestors[].term` |
| `n_primary_outcomes` | int | caution | Number of primary outcomes | `protocolSection.outcomesModule.primaryOutcomes` |
| `n_secondary_outcomes` | int | caution | Number of secondary outcomes | `protocolSection.outcomesModule.secondaryOutcomes` |
| `n_locations` | int | caution | Number of sites (latest record) | `protocolSection.contactsLocationsModule.locations` |
| `n_countries` | int | caution | Number of distinct countries | `protocolSection.contactsLocationsModule.locations[].country` |
| `countries` | list[string] | caution | Distinct site countries, sorted | `protocolSection.contactsLocationsModule.locations[].country` |
