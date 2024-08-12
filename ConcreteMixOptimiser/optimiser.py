from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from pathlib import Path

from ConcreteMixOptimiser.utils.constants import DATA_DIR, REPORT_DIR
from ConcreteMixOptimiser.utils.logger import get_logger


class RatioAllocation:
    def __init__(self, data_dir: Path = None, show_plotly: bool = True):
        self.logger = get_logger()
        self.show_plotly = show_plotly
        if data_dir is not None:
            self.data_dir = data_dir
            self.raw_data_dir = self.data_dir
        else:
            self.data_dir = DATA_DIR
            self.raw_data_dir = self.data_dir / "raw"
        self.report_dir = REPORT_DIR
        self.class_g_cement_df = pd.read_csv(self.raw_data_dir / "class_g_cement.csv")
        self.crumb_rubber_df = pd.read_csv(self.raw_data_dir / "crumb_rubber.csv")
        self.crumb_rubber_powder_df = pd.read_csv(
            self.raw_data_dir / "crumb_rubber_powder.csv"
        )
        self.quartz_powder_df = pd.read_csv(self.raw_data_dir / "quartz_powder.csv")
        self.quartz_sand_df = pd.read_csv(self.raw_data_dir / "quartz_sand.csv")
        self.silica_fume_df = pd.read_csv(self.raw_data_dir / "silica_fume.csv")
        self.sand_df = pd.read_csv(self.raw_data_dir / "sand.csv")

        # add category column to each df
        self.class_g_cement_df["category"] = "class_g_cement"
        self.crumb_rubber_df["category"] = "crumb_rubber"
        self.crumb_rubber_powder_df["category"] = "crumb_rubber_powder"
        self.quartz_powder_df["category"] = "quartz_powder"
        self.quartz_sand_df["category"] = "quartz_sand"
        self.silica_fume_df["category"] = "silica_fume"
        self.sand_df["category"] = "sand"

        # get the type of the dfs as float
        self.class_g_cement_df["size"] = self.class_g_cement_df["size"].astype(float)
        self.crumb_rubber_df["size"] = self.crumb_rubber_df["size"].astype(float)
        self.crumb_rubber_powder_df["size"] = self.crumb_rubber_powder_df[
            "size"
        ].astype(float)
        self.quartz_powder_df["size"] = self.quartz_powder_df["size"].astype(float)
        self.quartz_sand_df["size"] = self.quartz_sand_df["size"].astype(float)
        self.silica_fume_df["size"] = self.silica_fume_df["size"].astype(float)
        self.sand_df["size"] = self.sand_df["size"].astype(float)
        # and the finer
        self.class_g_cement_df["finer"] = self.class_g_cement_df["finer"].astype(float)
        self.crumb_rubber_df["finer"] = self.crumb_rubber_df["finer"].astype(float)
        self.crumb_rubber_powder_df["finer"] = self.crumb_rubber_powder_df[
            "finer"
        ].astype(float)
        self.quartz_powder_df["finer"] = self.quartz_powder_df["finer"].astype(float)
        self.quartz_sand_df["finer"] = self.quartz_sand_df["finer"].astype(float)
        self.sand_df["finer"] = self.sand_df["finer"].astype(float)

        self.logger.info("RatioAllocation initialized")
        self.logger.info(f"Data directory: {self.data_dir}")

        # log the shape of the dataframes
        self.logger.info(f"class_g_cement_df shape: {self.class_g_cement_df.shape}")
        self.logger.info(f"crumb_rubber_df shape: {self.crumb_rubber_df.shape}")
        self.logger.info(
            f"crumb_rubber_powder_df shape: {self.crumb_rubber_powder_df.shape}"
        )
        self.logger.info(f"quartz_powder_df shape: {self.quartz_powder_df.shape}")
        self.logger.info(f"quartz_sand_df shape: {self.quartz_sand_df.shape}")
        self.logger.info(f"silica_fume_df shape: {self.silica_fume_df.shape}")
        self.logger.info(f"sand_df shape: {self.sand_df.shape}")

    def process(
            self, component_names: List[str] = None, given_bound=None, given_ratio=None
    ):
        self.logger.info("RatioAllocation process started")
        self.plot()
        return self.calculate_best_ratio(component_names, given_bound, given_ratio)

    def calculate_best_ratio(
            self, component_names: List[str], given_bound=None, given_ratio=None
    ):
        """
        calculate the best ratio for the given components
        Parameters
        ----------
        component_names
        given_bound
        given_ratio

        Returns
        -------

        """
        # first grab the combined data frame
        # create a new dataframe with the size, finer and type
        df = pd.DataFrame(columns=["size", "finer", "type"])
        for component_name in component_names:
            df = pd.concat(
                [
                    df,
                    self.__getattribute__(f"{component_name}_df")[
                        ["size", "finer"]
                    ].assign(type=component_name),
                ]
            )
        # sort the df via the size value asc
        # get the range of the size
        # plot the target blue
        df = df.sort_values(by="size", ascending=True)
        self.logger.info(f"df shape: {df.shape}")
        size_min = df[df["finer"] != 0]["size"].min()
        # size max is the largest size without finer == 0
        size_max = df[df["finer"] != 0]["size"].max()

        q = 0.23  # constant
        # calculate the target blue
        size_range = df["size"].unique()
        # get the size range that is between size_min and size_max
        size_range = [i for i in size_range if (size_min <= i <= size_max)]
        size_range_readable = [f"{val:.2f}" for val in size_range]
        self.logger.info(f"size range: {size_range_readable}")
        # sort size range asc
        size_range.sort()
        x, y_pred, html = self.plot_target_blue(
            size_ranges=size_range, d_min=size_min, d_max=size_max, q=q
        )
        x_readable = [f"{val:.2f}" for val in x]
        self.logger.info(f"target x value range: {x_readable}")
        self.logger.info(f"target y value range: {y_pred}")

        # the problem is transformed to find the best ratio of the given components, which can
        # minimize the sum of the square of the difference between the target blue and the actual blue
        # we first construct the df, fill missing size's finer value with 0
        # then we can construct len(conponents) vectors, each vector is the finer value of the given component
        # then we can use scipy.optimize.minimize to find the best ratio

        # for each component, create a new df with the size, size = x
        # then merge it with the original df, fill the missing finer with 0
        # then sort the df via the size value asc

        # create a new df with the size, size = x
        df = pd.DataFrame(columns=["size"])
        df["size"] = x
        for component_name in component_names:
            # merge it with the original df, fill the missing finer with 0
            df = pd.merge(
                df,
                self.__getattribute__(f"{component_name}_df")[["size", "finer"]],
                on="size",
                how="left",
            )
            df = df.fillna(0)
            # rename the finer column to the component name
            df = df.rename(columns={"finer": component_name})
        # sort the df via the size value asc
        df = df.sort_values(by="size", ascending=True)
        # get the vectors
        vectors = []
        for component_name in component_names:
            vectors.append(df[component_name].values)
        self.logger.debug(f"vectors: {vectors}")
        # first check all vector size is the same
        assert len(set([len(vector) for vector in vectors])) == 1
        for vector in vectors:
            self.logger.debug(f"vector: {vector}")
            self.logger.debug(f"vector size: {len(vector)}")
            # sum the vector
            self.logger.info(f"vector sum: {np.sum(vector)}")
            # then make sure the sum is roughly around 100
            assert np.sum(vector) > 99.9
            assert np.sum(vector) < 100.1

        # multiply each element in the y_pred by 100, so it can be used as the target blue
        for i in range(len(y_pred)):
            y_pred[i] = y_pred[i] * 100

        if given_ratio is not None:
            # then this is a calculation for the given ratio problem, do not need to do the optimization
            # just calculate the y from the given ratio
            self.logger.info(f"Given ratio sum: {np.sum(given_ratio)}")
            y_actual, mse = self.calculate_y_from_given_ratio(vectors, y_pred, given_ratio)
            filename = "_".join([str(val) for val in given_ratio])

        else:
            optimal_results = self.find_best_ratio(
                vectors, y=y_pred, given_bound=given_bound
            )
            given_ratio = optimal_results[0]
            y_actual = optimal_results[-2]
            filename = "_".join([str(val) for val in optimal_results[0]])
            mse = optimal_results[-1]

        for i in range(len(y_actual)):
            y_actual[i] = y_actual[i] / 100

        _, _, html = self.plot_target_blue(
            size_ranges=size_range,
            d_min=size_min,
            d_max=size_max,
            q=q,
            y_actual=y_actual,
            filename=filename,
        )
        return mse, html, given_ratio

    def calculate_y_from_given_ratio(self, vectors, y, given_ratio):
        n = len(vectors)  # Get the number of vectors
        assert len(given_ratio) == n

        # Convert given_ratio and vectors to a common data type if needed
        given_ratio = np.array(given_ratio, dtype=float)
        vectors = np.array(vectors, dtype=float)

        # Calculate the predicted y values
        y_pred = np.dot(given_ratio, vectors)
        cum_sum = np.cumsum(y_pred)
        self.logger.info(f"Last element of cum_sum: {cum_sum[-1]}")

        # Calculate the mse
        mse = ((cum_sum - y) ** 2).mean()

        # Log the mse
        self.logger.info(f"mse: {mse}")

        return cum_sum, mse

    def find_best_ratio(self, vectors, y, given_bound=None):
        """
        y scale is 0-100
        Parameters
        ----------
        vectors
        y
        given_bound

        Returns
        -------

        """

        n = len(vectors)  # Get the number of vectors
        if given_bound is not None:
            assert len(given_bound) == n

        # Define the objective function
        def objective(params):
            *coefficients, _lambda = params
            y_pred = np.dot(coefficients, vectors)

            cum_sum = np.cumsum(y_pred)
            mse = ((cum_sum - y) ** 2).mean()
            return mse

        # Initial guess for coefficients and lambda
        initial_guess = [1.0 / n] * n + [0.0]
        # get it to ndarray
        initial_guess = np.array(initial_guess, dtype=float)

        if given_bound is None:
            # Define bounds for coefficients and lambda
            bounds = [(0, 1)] * n + [(-1, 1)]
        else:
            bounds = given_bound + [(-1, 1)]

        # add constraint that the sum of coefficients should be 1
        def constraint(params):
            *coefficients, _lambda = params
            return np.sum(coefficients) - 1

        # def constraint_cement_fume(params):
        #     # If the size of coefficients is 4 or more, enforce the constraint
        #     if len(params) >= 4:
        #         first_coefficient, _, _, fourth_coefficient, *_ = params
        #         # The constraint: 0.4 <= first_coefficient + fourth_coefficient <= 0.5
        #         return np.logical_xor(0.4 <= first_coefficient + fourth_coefficient,
        #                               first_coefficient + fourth_coefficient <= 0.5).astype(int)
        #
        #     return 0  # No constraint if there are fewer than 4 coefficients

        # Solve the optimization problem
        result = minimize(
            objective,
            x0=initial_guess,
            bounds=bounds,
            constraints={"type": "eq", "fun": constraint},
        )

        # Extract the optimal values of coefficients
        optimal_coefficients = result.x[:-1]
        optimal_lambda = result.x[-1]
        self.logger.info(f"Sum of coefficients: {np.sum(optimal_coefficients)}")
        # get optimal_coefficients format to 2 decimal and as string
        optimal_coefficients_print = [f"{val:.3f}" for val in optimal_coefficients]
        self.logger.info(f"Optimal coefficients: {optimal_coefficients_print}")

        self.logger.info(f"Optimal lambda: {optimal_lambda}")

        # Calculate the predicted y values
        y_pred = np.dot(optimal_coefficients, vectors)
        cum_sum = np.cumsum(y_pred)
        self.logger.info(f"Last element of cum_sum: {cum_sum[-1]}")

        # Calculate MSE
        mse = ((cum_sum - y) ** 2).mean()
        self.logger.info(f"y_pred: {cum_sum}")
        self.logger.info(f"mse: {mse}")

        return optimal_coefficients, optimal_lambda, cum_sum, mse

    def plot(self):
        html = ""
        html += self.plot_cum_distribution(self.class_g_cement_df, "class_g_cement")
        html += self.plot_cum_distribution(self.crumb_rubber_df, "crumb_rubber")
        html += self.plot_cum_distribution(self.crumb_rubber_powder_df, "crumb_rubber_powder")
        html += self.plot_cum_distribution(self.quartz_powder_df, "quartz_powder")
        html += self.plot_cum_distribution(self.quartz_sand_df, "quartz_sand")
        html += self.plot_cum_distribution(self.silica_fume_df, "silica_fume")
        html += self.plot_cum_distribution(self.sand_df, "sand")
        return html

    def collect_all_size_params(self):
        """
        merge all the df together and collect all the size params
        Returns
        -------

        """
        df = pd.concat(
            [
                self.class_g_cement_df,
                self.crumb_rubber_df,
                self.crumb_rubber_powder_df,
                self.quartz_powder_df,
                self.quartz_sand_df,
                self.silica_fume_df,
            ]
        )
        df = df[["size", "finer"]]
        df = df.sort_values(by="size", ascending=True)
        self.logger.info(f"df shape: {df.shape}")
        self.logger.info(f"df head: {df.head()}")
        for _, row in df.iterrows():
            self.logger.info(f"size: {row['size']}, finer: {row['finer']}")

    def plot_target_blue(  # noqa
            self, size_ranges, d_min=0.2, d_max=600, q=0.23, y_actual=None, filename=None
    ):
        """
        Target blue cum plot for the class_g_cement
        formula: \(y = \frac{{x^q - d_{min}^q}}{{d_{max}^q - d_{min}^q}}\)  # noqa
        formula: \(y = \frac{{x^{0.23} - 0.2^{0.23}}}{{600^{0.23} - 0.2^{0.23}}}\) # noqa
        Returns
        -------

        """
        # set x is 0-100, step is 0.1, plot the x,y with loglog plot
        y = [(i ** q - d_min ** q) / (d_max ** q - d_min ** q) for i in size_ranges]
        # stop it when y > 1, so only plot the range of x that y < 1
        y = [i for i in y if i <= 1]
        # cat the size range to the y
        size_ranges = size_ranges[: len(y)]
        # plot the y with x

        fig = go.Figure()

        # line plot
        fig.add_trace(go.Scatter(x=size_ranges, y=y, mode="lines", name="Curve"))

        # Add the LaTeX-formatted formula annotation
        formula_text = (
                "Formula: $y = \\frac{{x^{{"
                + str(q)
                + f"}} - {d_min}^{{"
                + str(q)
                + "}}}}{{"
                + str(d_max)
                + "^{{"
                + str(q)
                + "}} - "
                + str(d_min)
                + "^{{"
                + str(q)
                + "}}}}$"
        )

        # also add the annotation what's
        formula_annotation = go.layout.Annotation(
            x=0.9,  # Adjust the x-coordinate to move the formula to the right
            y=1.05,  # Adjust the y-coordinate to move the formula higher
            xref="paper",
            yref="paper",
            text=formula_text,
            showarrow=False,
            font=dict(size=20),
        )
        # if y_actual is not None:
        if y_actual is not None:
            # add scatter plot for the actual y
            fig.add_trace(
                go.Scatter(
                    x=size_ranges,
                    y=y_actual,
                    # mode="markers",
                    mode="lines",
                    name="Actual",
                    marker=dict(color="red"),
                )
            )

        # set y axis start from 0
        fig.update_yaxes(range=[0, 1.2])
        fig.add_annotation(formula_annotation)

        fig.update_layout(title="Target blue cumulative sum", xaxis_type="log")

        if self.show_plotly:
            fig.show()

        # Save the figure
        if filename is not None:
            fig.write_image(str(self.report_dir / f"target_blue_{filename}.png"))
        else:
            fig.write_image(str(self.report_dir / "target_blue.png"))
        html = fig.to_html(include_mathjax="cdn", full_html=False)
        return size_ranges, y, html

    def plot_cum_distribution(self, df: pd.DataFrame, name: str):
        report_dir = self.report_dir / "blue_individually"
        report_dir.mkdir(parents=True, exist_ok=True)
        # sort the df via the size value asc
        # and then use plotly to plot a cumulative sum of finer aggregate
        # use plotly to do the plot

        df = df.sort_values(by="size", ascending=True)
        df["cumsum"] = df["finer"].cumsum()
        # show the scatter points and line plot together
        fig = px.scatter(df, x="size", y="cumsum", title=f"{name} cumulative sum")
        fig.add_trace(px.line(df, x="size", y="cumsum").data[0])
        # set x axis to log
        fig.update_xaxes(type="log")

        if self.show_plotly:
            fig.show()
        # write to report dir
        fig.write_image(str(report_dir / f"{name}.png"))
        html = fig.to_html(include_mathjax="cdn", full_html=False)
        return html


if __name__ == "__main__":
    ratio_allocation = RatioAllocation()
    mse, html, given_ratio = ratio_allocation.process(
        component_names=[
            "class_g_cement",
            "quartz_sand",
            "quartz_powder",
            "silica_fume",
            "sand"
        ],
        given_bound=[[0.389, 1],
                     [0, 1],
                     [0.03, 0.089],
                     [0.08, 0.16],
                     [0, 1],
                     # [0, 1]
                     ],
        # given_ratio=[
        #     0.3828,  # class_g_cement
        #     0,  # quartz_sand
        #     0.05,  # quartz_powder
        #     0.1158,  # silica_fume
        #     0.45,  # sand
        # ],
    )
    #
    # best_one = None
    # best_mse = 0
    # for shui_ni in range(1, 1000):
    #     shui_ni_left_b = 0.38 + 0.001 * shui_ni
    #     if shui_ni_left_b > 0.999:
    #         break
    #     mse = ratio_allocation.process(
    #         component_names=[
    #             "class_g_cement",
    #             "quartz_sand",
    #             "quartz_powder",
    #             "silica_fume",
    #             "sand"
    #         ],
    #         given_bound=[[shui_ni_left_b, 1],
    #                      [0, 1],
    #                      [0.03, 0.089],
    #                      [0.08, 0.16],
    #                      [0, 1],
    #                      # [0, 1]
    #                      ],
    #         # given_ratio=[
    #         #     0.3828,  # class_g_cement
    #         #     0,  # quartz_sand
    #         #     0.0889,  # quartz_powder
    #         #     0.1158,  # silica_fume
    #         #     0.4504,  # sand
    #         # ],
    #     )
    #     if (best_one is None) or (mse < best_mse):
    #         best_one = shui_ni_left_b
    #         best_mse = mse
    #
    # print(f"Best one: {best_one}, best mse: {best_mse}")

"""
Dataset1:   0.31,  # class_g_cement
            0.43,  # quartz_sand
            0.08,  # quartz_powder
            0.12, # silica_fume
            MSE: 29

Dataset2:
            0.379,  # class_g_cement
            0.404,  # quartz_sand
            0.1,  # quartz_powder
            0.117,  # silica_fume
            MSE: 14.35

"""

"""
1. Remove the two rubber, And a sand component
2. process the sand component data
3. constraint is also changed
    - 

"""
