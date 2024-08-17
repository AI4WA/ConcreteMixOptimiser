import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ConcreteMixOptimiser.optimiser import RatioAllocation
from ConcreteMixOptimiser.utils.constants import DATA_DIR
from ConcreteMixOptimiser.utils.logger import get_logger

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

    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    st.info("Welcome to Concrete Mix Optimiser developed by UWA Civil Engineer Team")
    st.markdown(
        "Github Repository: "
        "<a href='https://github.com/AI4WA/ConcreteMixOptimiser' target='__blank'>Concrete Mix Optimiser</a>",
        unsafe_allow_html=True)

    st.title("Concrete Mix Optimiser")

    with st.expander("How to use it", expanded=False):
        # https://youtu.be/kx85c3kUAyw, add this video to the page
        st.video("https://youtu.be/kx85c3kUAyw")

    # Ask user to input a name
    uid_name = st.text_input("Enter your email address to get started:")
    if not uid_name:
        st.info(
            "For demo purpose, you can type in the following email address: cmouwa@demo.com"
        )
        st.warning("Please provide a name.")
        return

    if not is_valid_email(uid_name):
        st.warning("Please provide a valid email.")
        return

    raw_csv_dir = DATA_DIR / uid_name
    raw_csv_dir.mkdir(exist_ok=True, parents=True)

    # Customizable file names
    default_file_names = [
        "material_1",
        "material_2",
        "material_3",
        "material_4",
    ]

    # Allow user to customize file names
    with st.expander("Customize Mixture Materials", expanded=False):
        custom_file_names = st.text_area(
            "Enter material names (one per line), name it as example shows. \n You can include two and more materials you want to use, doesn't necessary to be how many we put here as an example:",
            "\n".join(default_file_names),
            height=300,
        )
        file_names = [
            name.strip() for name in custom_file_names.split("\n") if name.strip()
        ]

    # File uploader
    files = {}
    files_status = {name: (raw_csv_dir / f"{name}.csv").exists() for name in file_names}

    with st.expander("Upload Files", expanded=not all(files_status.values())):
        st.markdown(
            "Upload the CSV files for each material used in the concrete mix. \n"
            "If a file has been uploaded, it will be indicated below."
        )
        # give an example of the file format, what contains in the file
        # size, finer
        # 0.523, 0.08
        # 0.594, 0.17
        # 0.675, 0.26
        # 0.767, 0.32
        # 0.872, 0.34
        st.markdown(
            """
        **File Format Example:**

        ```
        size, finer
        0.523, 0.08
        0.594, 0.17
        0.675, 0.26
        0.767, 0.32
        0.872, 0.34
        ```
        """
        )

        # Create rows with 2 columns each
        for i in range(0, len(file_names), 2):
            col1, col2 = st.columns(2)

            with col1:
                name = file_names[i]
                file_path = raw_csv_dir / f"{name}.csv"
                status_class = (
                    "uploaded-file" if files_status[name] else "not-uploaded-file"
                )

                st.markdown(
                    f'<div class="{status_class}">{name.replace("_", " ").title()} CSV</div>',
                    unsafe_allow_html=True,
                )

                file = st.file_uploader(
                    f"Upload {name}.csv", type="csv", key=f"upload_{i}"
                )
                files[name] = file

            if i + 1 < len(file_names):
                with col2:
                    name = file_names[i + 1]
                    file_path = raw_csv_dir / f"{name}.csv"
                    status_class = (
                        "uploaded-file" if files_status[name] else "not-uploaded-file"
                    )

                    st.markdown(
                        f'<div class="{status_class}">{name.replace("_", " ").title()} CSV</div>',
                        unsafe_allow_html=True,
                    )

                    file = st.file_uploader(
                        f"Upload {name}.csv", type="csv", key=f"upload_{i + 1}"
                    )
                    files[name] = file

        if st.button("Process Uploaded Files"):
            st.header("File Previews")
            for name, file in files.items():
                logger.info(f"Processing {name}.csv")
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
        components.html(analyse_html, height=450 * len(file_names))
    st.success("All files are ready for processing!")
    st.divider()

    # Material Ratio Allocation
    st.header("Material Ratio Allocation")
    # Selection option for calculation type
    calculation_type = st.selectbox(
        "Select Calculation Type:",
        (
            "Calculate based on Given Ratio",
            "Optimise based on Given Ratio Range",
            "Search within Given Ratio Range",
        ),
    )

    st.info("Select the ratio range for each material involved in the mix.")

    selected_materials = {}
    with st.expander("Material Ratio Selection", expanded=True):
        for name in file_names:
            col1, col2 = st.columns([1, 2])
            with col1:
                use_material = st.checkbox(
                    f"Use {name.replace('_', ' ').title()}", key=f"check_{name}"
                )
            if use_material:
                if calculation_type == "Calculate based on Given Ratio":
                    # only one value for slider
                    with col2:
                        ratio_value = st.number_input(
                            f"Select ratio for {name.replace('_', ' ').title()}",
                            min_value=0.0000,
                            max_value=1.0000,
                            value=0.5000,
                            step=0.0001,  # Step size, controls the precision
                            format="%.4f",  # Ensures 4 decimal places are displayed
                            key=f"input_{name}",
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
                            key=f"slider_{name}",
                        )
                        selected_materials[name] = ratio_range

        if calculation_type == "Search within Given Ratio Range":
            total_iterations = st.number_input(
                "Total Iterations",
                min_value=1,
                max_value=1000,
                value=100,
                step=1,
                key="total_iterations",
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
                st.error(
                    "The sum of the given ratio must be equal to 1. Current sum: {}".format(
                        sum(given_ratio)
                    )
                )
                return

            # Add your calculation logic here
            mse, html, _ = ratio_allocation.process(
                component_names=component_names, given_ratio=given_ratio
            )
            st.write("Calculation complete.")
            given_ratio_html_text = "<ul>"
            for name, ratio in zip(component_names, given_ratio):
                given_ratio_html_text += (
                    f"<li>{name.replace('_', ' ').title()}: {ratio:.4f}</li>"
                )
            given_ratio_html_text += "</ul>"
            given_ratio_html_text += (
                "<p>Mean Squared Error (MSE): {:.4f}</p>".format(mse)
            )
            st.write("Given Ratio:")
            st.write(given_ratio_html_text, unsafe_allow_html=True)
            components.html(html, height=800)

        elif calculation_type == "Optimise based on Given Ratio Range":
            st.write("Running optimizer calculation...")
            # Add your optimization logic here
            # Example: mse = ratio_allocation.optimize(...)
            mse, html, optimised_ratio = ratio_allocation.process(
                component_names=component_names, given_bound=given_bound
            )
            st.write("Optimization complete.")
            optimised_ratio_html_text = "<ul>"
            for name, ratio in zip(component_names, optimised_ratio):
                optimised_ratio_html_text += (
                    f"<li>{name.replace('_', ' ').title()}: {ratio:.4f}</li>"
                )
            optimised_ratio_html_text += "</ul>"
            optimised_ratio_html_text += (
                "<p>Mean Squared Error (MSE): {:.4f}</p>".format(mse)
            )
            st.write("Optimised Ratio:")
            st.write(optimised_ratio_html_text, unsafe_allow_html=True)
            components.html(html, height=800)

        elif calculation_type == "Search within Given Ratio Range":
            st.write("Searching based on the first material within the given range...")
            progress_bar = st.progress(0)
            best_one = None
            best_html = None
            best_optimised_ratio = None
            best_mse = 0
            for concrete in range(1, total_iterations):
                concrete_left_b_start = given_bound[0][0]
                concrete_left_b = concrete_left_b_start + 0.001 * concrete
                if concrete_left_b > 0.999:
                    break
                given_bound[0][0] = concrete_left_b
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
                optimised_ratio_html_text += (
                    f"<li>{name.replace('_', ' ').title()}: {ratio:.4f}</li>"
                )
            optimised_ratio_html_text += "</ul>"
            optimised_ratio_html_text += (
                "<p>Mean Squared Error (MSE): {:.4f}</p>".format(best_mse)
            )
            st.write("Optimised Ratio:")
            st.write(optimised_ratio_html_text, unsafe_allow_html=True)
            components.html(best_html, height=800)

    # add a footer: developed by UWA, contact email
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
        }
        </style>
        <div class="footer">
            <p>Developed by UWA</p>
            <p>Contact: 
                Software Related: <a href="mailto:pascal.sun@research.uwa.edu.au">pascal.sun@research.uwa.edu.au</a> or 
                Research Related: <a href="mailto:xin.lyu@research.uwa.edu.au">xin.lyu@research.uwa.edu.au</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
