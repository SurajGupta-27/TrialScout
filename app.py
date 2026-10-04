"""TrialScout web UI. Run locally with:  streamlit run app.py"""

import streamlit as st

from src.config import TOP_N_FOR_SUMMARY
from src.pipeline import PipelineResult, run_pipeline
from src.trials_api import format_phase, format_status
from src.validation import find_nct_ids

EXAMPLE_QUESTIONS = [
    "Phase 3 diabetes trials recruiting in India",
    "Completed Phase 2 asthma trials in the US",
    "Upcoming Alzheimer's trials in Japan",
    "Recruiting breast cancer trials in Germany",
]


def use_example(question: str) -> None:
    """Button callback: put the example in the search box and run it."""
    st.session_state.question = question
    st.session_state.run_now = True


def get_result(question: str) -> PipelineResult:
    """Run the pipeline, reusing earlier answers in this session to save API quota.

    Only successful results are kept, so a temporary error is retried next time.
    """
    cache = st.session_state.setdefault("results_cache", {})
    if question not in cache:
        result = run_pipeline(question)
        if result.error:
            return result
        cache[question] = result
    return cache[question]


def show_filters(result: PipelineResult) -> None:
    """Show the filters the AI extracted, so users can see how it read the question."""
    st.subheader("1. Filters extracted from your question")
    labels = {"condition": "Condition", "phase": "Phase", "status": "Status", "country": "Country"}
    for column, (key, label) in zip(st.columns(4), labels.items()):
        value = result.filters.get(key)
        if value and key == "phase":
            value = format_phase([value])
        elif value and key == "status":
            value = format_status(value)
        column.metric(label, value or "Any")
    for warning in result.filter_warnings:
        st.warning(warning)


def show_trials_table(result: PipelineResult) -> None:
    """Show the fetched trials as a table with links to ClinicalTrials.gov."""
    st.subheader(f"2. Trials found: {result.total_count} (showing {len(result.trials)})")
    rows = [
        {
            "NCT ID": t["url"],
            "Title": t["title"],
            "Phase": format_phase(t["phase"]),
            "Status": format_status(t["status"]),
            "Sponsor": t["sponsor"],
            "Countries": ", ".join(t["countries"]) or "Not listed",
        }
        for t in result.trials
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            # Show the NCT ID as the link text; clicking opens the trial page.
            "NCT ID": st.column_config.LinkColumn(display_text=r"study/(NCT\d+)"),
        },
    )


def show_summary(result: PipelineResult) -> None:
    """Show the validated AI summary and anything the validator removed or flagged."""
    st.subheader(f"3. AI summary of the top {min(len(result.trials), TOP_N_FOR_SUMMARY)} trials")
    validation = result.validation
    st.markdown(validation.summary)
    if validation.missing_ids:
        st.info(f"The summary skipped {', '.join(validation.missing_ids)}. They are in the table above.")

    if validation.hallucination_count == 0:
        verified = len(find_nct_ids(validation.summary))
        st.success(f"Validation passed: all {verified} NCT IDs and their phase, status and sponsor match the fetched data.")
        return

    st.warning(f"Validation caught {validation.hallucination_count} problem(s) in the AI summary.")
    for line in validation.removed_lines:
        st.markdown(f"- **Removed** (NCT ID not in the search results): ~~{line.lstrip('- ')}~~")
    for issue in validation.issues:
        st.markdown(f"- **Flagged:** {issue}")


def main() -> None:
    """Draw the page."""
    st.set_page_config(page_title="TrialScout", page_icon="🔬", layout="wide")
    st.title("🔬 TrialScout")
    st.caption(
        "Ask about clinical trials in plain English. Results come live from ClinicalTrials.gov; "
        "the AI summary is checked against that data before you see it."
    )

    with st.form("search"):
        st.text_input(
            "Your question",
            key="question",
            placeholder="e.g. Phase 3 diabetes trials recruiting in India",
        )
        submitted = st.form_submit_button("Search", type="primary")

    st.write("Or try an example:")
    for column, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        column.button(example, on_click=use_example, args=(example,), width="stretch")

    run_now = st.session_state.pop("run_now", False)
    question = st.session_state.get("question", "").strip()
    if not (submitted or run_now):
        return
    if not question:
        st.warning("Please type a question first.")
        return

    with st.spinner("Reading your question, searching ClinicalTrials.gov, and checking the summary..."):
        result = get_result(question)

    if result.error:
        st.error(f"Something went wrong: {result.error}")
        return
    show_filters(result)
    if result.message:
        st.info(result.message)
        return
    show_trials_table(result)
    show_summary(result)

    timings = " · ".join(f"{step.replace('_', ' ')}: {secs:.1f}s" for step, secs in result.timings.items())
    st.caption(f"Timings: {timings}")
    st.caption("TrialScout is a search aid, not medical advice. Always confirm details on ClinicalTrials.gov.")


if __name__ == "__main__":
    main()
