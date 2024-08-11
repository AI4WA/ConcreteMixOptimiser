import streamlit as st
import pandas as pd
from ConcreteMixOptimiser.utils.constants import DATA_DIR
import re


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

    st.info("Welcome to Concrete Mix Optimiser developed by UWA")

    st.title("Concrete Mix Optimiser")

    # Ask user to input a name
    uid_name = st.text_input("Enter your email address to get started:")
    if not uid_name:
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

    # check whether all the files are ready
    all_files_ready = all((raw_csv_dir / f"{name}.csv").exists() for name in file_names)

    files = {}
    with st.expander("Upload Files", expanded=(not all_files_ready)):
        # Create rows with 2 columns each
        for i in range(0, len(file_names), 2):
            col1, col2 = st.columns(2)

            with col1:
                name = file_names[i]
                file_path = raw_csv_dir / f"{name}.csv"

                if file_path.exists():
                    st.info(f"{name}.csv exists")

                file = st.file_uploader(f"Upload {name}.csv", type="csv", key=f"upload_{i}")
                files[name] = file

            if i + 1 < len(file_names):
                with col2:
                    name = file_names[i + 1]
                    file_path = raw_csv_dir / f"{name}.csv"

                    if file_path.exists():
                        st.info(f"{name}.csv exists")

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
                elif file_path.exists():
                    st.info(f"{name}.csv already exists")
                else:
                    st.warning(f"{name}.csv not uploaded.")

            # Here you would typically call your RatioAllocation process
            # For example:
            # ratio_allocation = RatioAllocation(raw_data_dir=raw_csv_dir)
            # ratio_allocation.process()

            st.success("Files processed successfully!")


if __name__ == "__main__":
    main()
