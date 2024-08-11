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
    class_g_cement_file = st.file_uploader("Upload class_g_cement.csv", type="csv")
    crumb_rubber_file = st.file_uploader("Upload crumb_rubber.csv", type="csv")
    crumb_rubber_powder_file = st.file_uploader("Upload crumb_rubber_powder.csv", type="csv")
    quartz_powder_file = st.file_uploader("Upload quartz_powder.csv", type="csv")
    quartz_sand_file = st.file_uploader("Upload quartz_sand.csv", type="csv")
    silica_fume_file = st.file_uploader("Upload silica_fume.csv", type="csv")
    sand_file = st.file_uploader("Upload sand.csv", type="csv")

    # Load and display previews
    files = {
        "class_g_cement": class_g_cement_file,
        "crumb_rubber": crumb_rubber_file,
        "crumb_rubber_powder": crumb_rubber_powder_file,
        "quartz_powder": quartz_powder_file,
        "quartz_sand": quartz_sand_file,
        "silica_fume": silica_fume_file,
        "sand": sand_file
    }

    if st.button("Process Uploaded Files"):
        st.header("File Previews")
        for name, file in files.items():
            df = load_csv(file)
            if df is not None:
                st.subheader(f"{name}.csv")
                st.write(df.head())
                # save the file to the raw_data directory
                df.to_csv(raw_csv_dir / f"{name}.csv", index=False)
            else:
                st.warning(f"{name}.csv not uploaded.")

        # Here you would typically call your RatioAllocation process
        # For example:
        # ratio_allocation = RatioAllocation(raw_data_dir=".")
        # ratio_allocation.process()

        st.success("Files processed successfully!")


if __name__ == "__main__":
    main()
