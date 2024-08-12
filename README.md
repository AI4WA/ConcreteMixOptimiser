# Concrete Mix Optimiser

## How to use it?

Link: [Concrete Mix Optimiser](https://cmouwa.streamlit.app/)

Open the link, and put your email address to get started.
Upload the csv files for different materials.
Then after processed, you will be able to do three different mode of calculations:

- Given ratio of selected materials, calculate the strength of the concrete.
- Given ratio range of selected materials, find out the best ratio for the highest strength.
- Search different ratio ranges of Concrete Mix, find out the best ratio for the highest strength.

### Demo

For demo purpose, you can type in the following email address: `cmouwa@demo.com`

And then you will be able to explore the three different modes of calculations.

### Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/kx85c3kUAyw" frameborder="0" allowfullscreen></iframe>

## How to run it locally?

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the app using `streamlit run web.py`.

Core code for optimiser is in `ConcreteMixOptimiser/optimiser.py` file, you can adpat it to your own use case.

