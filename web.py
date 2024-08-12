import streamlit as st
import pandas as pd
from ConcreteMixOptimiser.utils.constants import DATA_DIR
from ConcreteMixOptimiser.optimiser import RatioAllocation
from ConcreteMixOptimiser.utils.logger import get_logger
import re
import streamlit.components.v1 as components

logger = get_logger(__name__)


def load_csv(file):
    if file is not None:
        return pd.read_csv(file)
    return None


def is_valid_email(email):
    # Basic email validation regex
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


class FileProcessor:
    def __init__(self, files):
        self.files = files

    def process_files(self):
        results = []
        for file in self.files:
            content = file.read().decode("utf-8")  # Assuming text files
            results.append(content)
        return results


def main():
    # update the webpage title
    st.set_page_config(page_title="Mix Concrete", page_icon=":construction:")

    st.markdown("""
    <style>
    .uploaded-file {
        color: green;
        font-weight: bold;
    }
    .not-uploaded-file {
        color: red;
        font-weight: bold;
    }
    .material-card {
        background-color: #f9f9f9;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.info("Welcome to Concrete Mix Optimiser developed by UWA")

    st.title("Concrete Mix Optimiser")

    # Ask user to input a name
    uid_name = st.text_input("Enter your email address to get started:")
    if not uid_name:
        st.info("For demo purpose, you can type in the following email address: cmouwa@demo.com")
        st.warning("Please provide a name.")
        return

    if not is_valid_email(uid_name):
        st.warning("Please provide a valid email.")
        return

    raw_csv_dir = DATA_DIR / uid_name
    raw_csv_dir.mkdir(exist_ok=True, parents=True)

    # File uploader
    file_names = [
        "class_g_cement", "crumb_rubber", "crumb_rubber_powder",
        "quartz_powder", "quartz_sand", "silica_fume", "sand"
    ]

    files = {}
    files_status = {name: (raw_csv_dir / f"{name}.csv").exists() for name in file_names}

    with st.expander("Upload Files", expanded=not all(files_status.values())):
        st.markdown("### Upload the required material files:")
        st.markdown(
            "Upload the CSV files for each material used in the concrete mix. If a file has been uploaded, it will be indicated below.")
        # Create rows with 2 columns each
        for i in range(0, len(file_names), 2):
            col1, col2 = st.columns(2)

            with col1:
                name = file_names[i]
                file_path = raw_csv_dir / f"{name}.csv"
                status_class = "uploaded-file" if files_status[name] else "not-uploaded-file"

                st.markdown(f'<div class="{status_class}">{name.replace("_", " ").title()} CSV</div>',
                            unsafe_allow_html=True)

                file = st.file_uploader(f"Upload {name}.csv", type="csv", key=f"upload_{i}")
                files[name] = file

            if i + 1 < len(file_names):
                with col2:
                    name = file_names[i + 1]
                    file_path = raw_csv_dir / f"{name}.csv"
                    status_class = "uploaded-file" if files_status[name] else "not-uploaded-file"

                    st.markdown(f'<div class="{status_class}">{name.replace("_", " ").title()} CSV</div>',
                                unsafe_allow_html=True)

                    file = st.file_uploader(f"Upload {name}.csv", type="csv", key=f"upload_{i + 1}")
                    files[name] = file

        if st.button("Process Uploaded Files"):
            st.header("File Previews")
            for name, file in files.items():
                file_path = raw_csv_dir / f"{name}.csv"
                if file is not None:
                    df = load_csv(file)
                    st.subheader(f"{name}.csv")
                    st.write(df.head())
                    # save the file to the raw_data directory
                    df.to_csv(file_path, index=False)
                    st.success(f"{name}.csv processed and saved successfully.")
                    files_status[name] = True
                elif files_status[name]:
                    st.info(f"{name}.csv already exists")
                else:
                    st.warning(f"{name}.csv not uploaded.")

            # Check if all files are ready after processing
            if not all(files_status.values()):
                st.warning("Some files are still missing. Please upload them.")

    if not all(files_status.values()):
        return
    # Initialize the RatioAllocation
    ratio_allocation = RatioAllocation(raw_csv_dir, show_plotly=False)

    with st.expander("Data Analysis", expanded=True):
        analyse_html = ratio_allocation.plot()
        components.html(analyse_html, height=400 * 7)
    st.success("All files are ready for processing!")
    st.divider()

    # Material Ratio Allocation
    st.header("Material Ratio Allocation")
    # Selection option for calculation type
    calculation_type = st.selectbox(
        "Select Calculation Type:",
        ("Calculate based on Given Ratio", "Optimise based on Given Ratio Range", "Search within Given Ratio Range")
    )

    st.info("Select the ratio range for each material involved in the mix.")

    selected_materials = {}
    with st.expander("Material Ratio Selection", expanded=True):
        for name in file_names:
            col1, col2 = st.columns([1, 2])
            with col1:
                use_material = st.checkbox(f"Use {name.replace('_', ' ').title()}", key=f"check_{name}")
            if use_material:
                if calculation_type == 'Calculate based on Given Ratio':
                    # only one value for slider
                    with col2:
                        ratio_value = st.number_input(
                            f"Select ratio for {name.replace('_', ' ').title()}",
                            min_value=0.0000,
                            max_value=1.0000,
                            value=0.5000,
                            step=0.0001,  # Step size, controls the precision
                            format="%.4f",  # Ensures 4 decimal places are displayed
                            key=f"input_{name}"
                        )
                        selected_materials[name] = (ratio_value, ratio_value)
                else:
                    with col2:
                        ratio_range = st.slider(
                            f"Select ratio range for {name.replace('_', ' ').title()}",
                            min_value=0.0000,
                            max_value=1.0000,
                            value=(0.0000, 1.0000),
                            step=0.0001,  # Step size, controls the precision
                            key=f"slider_{name}"
                        )
                        selected_materials[name] = ratio_range

        if calculation_type == "Search within Given Ratio Range":
            total_iterations = st.number_input(
                "Total Iterations",
                min_value=1,
                max_value=1000,
                value=100,
                step=1,
                key="total_iterations"
            )

    if st.button("Process"):
        # Extracting selected component names and their bounds
        component_names = list(selected_materials.keys())
        given_bound = [list(ratio) for ratio in selected_materials.values()]
        logger.info(f"Component Names: {component_names}")
        logger.info(f"Given Bound: {given_bound}")

        if calculation_type == "Calculate based on Given Ratio":
            st.write("Calculating given ratio...")
            given_ratio = {name: ratio[0] for name, ratio in selected_materials.items()}
            component_names = list(given_ratio.keys())
            given_ratio = list(given_ratio.values())

            # sum the value, if it is not within 1 range, then alert the user
            if not (1.02 >= sum(given_ratio) >= 0.98):
                st.error("The sum of the given ratio must be equal to 1. Current sum: {}".format(sum(given_ratio)))
                return

            # Add your calculation logic here
            mse, html, _ = ratio_allocation.process(
                component_names=component_names,
                given_ratio=given_ratio
            )
            st.write("Calculation complete.")

            st.write(f"For given ratio, Mean Squared Error (MSE): {mse}")
            components.html(html, height=800)

        elif calculation_type == "Optimise based on Given Ratio Range":
            st.write("Running optimizer calculation...")
            # Add your optimization logic here
            # Example: mse = ratio_allocation.optimize(...)
            mse, html, optimised_ratio = ratio_allocation.process(
                component_names=component_names,
                given_bound=given_bound
            )
            st.write("Optimization complete.")
            optimised_ratio_html_text = "<ul>"
            for name, ratio in zip(component_names, optimised_ratio):
                optimised_ratio_html_text += f"<li>{name.replace('_', ' ').title()}: {ratio:.4f}</li>"
            optimised_ratio_html_text += "</ul>"
            optimised_ratio_html_text += "<p>Mean Squared Error (MSE): {:.4f}</p>".format(mse)
            st.write("Optimised Ratio:")
            st.write(optimised_ratio_html_text, unsafe_allow_html=True)
            components.html(html, height=800)

        elif calculation_type == "Search within Given Ratio Range":
            st.write("Searching within the given range...")
            if "class_g_cement" not in component_names:
                st.error("The Class G Cement must be included in the search.")
                return
            progress_bar = st.progress(0)
            best_one = None
            best_html = None
            best_optimised_ratio = None
            best_mse = 0
            for concrete in range(1, total_iterations):
                concrete_left_b = 0.38 + 0.001 * concrete
                if concrete_left_b > 0.999:
                    break
                given_bound[-1][1] = concrete_left_b
                mse, html, optimised_ratio = ratio_allocation.process(
                    component_names=component_names,
                    given_bound=given_bound,
                )
                progress_bar.progress(concrete / total_iterations)
                if (best_one is None) or (mse < best_mse):
                    best_one = concrete_left_b
                    best_mse = mse
                    best_html = html
                    best_optimised_ratio = optimised_ratio
            st.write("Search complete.")
            optimised_ratio_html_text = "<ul>"
            for name, ratio in zip(component_names, best_optimised_ratio):
                optimised_ratio_html_text += f"<li>{name.replace('_', ' ').title()}: {ratio:.4f}</li>"
            optimised_ratio_html_text += "</ul>"
            optimised_ratio_html_text += "<p>Mean Squared Error (MSE): {:.4f}</p>".format(best_mse)
            st.write("Optimised Ratio:")
            st.write(optimised_ratio_html_text, unsafe_allow_html=True)
            components.html(best_html, height=800)


if __name__ == "__main__":
    main()
