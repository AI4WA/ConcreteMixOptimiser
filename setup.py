from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt", "r") as req:
        content = req.read()
        requirements = content.split("\n")
    return requirements


setup(
    name="ConcreteMixOptimiser",
    author="AI4WA",
    author_email="admin@ai4wa.com",
    description="ConcreteMixOptimiser",
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Adjust the location where setuptools looks for packages
    include_package_data=True,  # To include other types of files specified in MANIFEST.in or found in your packages
    install_requires=read_requirements(),
    python_requires=">=3.8",  # Specify your Python version compatibility
    classifiers=[
        # Classifiers help users find your project by categorizing it
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # license: GNU LESSER GENERAL PUBLIC LICENSE
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        # topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
