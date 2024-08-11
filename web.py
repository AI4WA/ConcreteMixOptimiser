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
    st.title("Concrete Mix Optimiser")

    # Ask user to input a name
    uid_name = st.text_input("Give your session a unique name, can be your email:")
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

    files = {}
    for name in file_names:
        file_path = raw_csv_dir / f"{name}.csv"

        if file_path.exists():
            st.info(f"{name}.csv already exists. You can re-upload if needed.")

        file = st.file_uploader(f"Upload {name}.csv", type="csv")
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
                st.info(f"{name}.csv already exists. Displaying existing file:")
                existing_df = pd.read_csv(file_path)
                st.write(existing_df.head())
            else:
                st.warning(f"{name}.csv not uploaded.")

        # Here you would typically call your RatioAllocation process
        # For example:
        # ratio_allocation = RatioAllocation(raw_data_dir=raw_csv_dir)
        # ratio_allocation.process()

        st.success("Files processed successfully!")


if __name__ == "__main__":
    main()
