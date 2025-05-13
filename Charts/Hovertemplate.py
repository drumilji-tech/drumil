"""Helper Functions and Constants for Hovertemplate Creation."""

from Utils.UiConstants import (
    METRIC_UNITS,
)


FAULT_HOVER_ORDER = [
    "Turbine",
    "Lost Revenue",
    "Lost Energy",
    "Downtime",
    "Count",
    "FaultCode",
    "Description",
]
HOVERTEMPLATE_WITH_PITAG = "label: %{label}<br>%{customdata[0]}<br>value: %{value}"
TREEMAP_HOVERLABEL = dict(font=dict(size=16))


def append_units_and_round(var, metric):
    """Prepare a value for consumption in Plotly's hovertemplate.

    Args:
        var (type): The name of the variable to be displayed in the
            hovertemplate.
        metric (str): This param determines both the units and the
            rounding that this value gets. If the value of `metric`
            is not recognized, no units or rounding will be applied
    Returns:
        hovertemplate (str): A parameter for Plotly's hovertemplate.
            To learn more about hovertemplates, see the official docs:
            https://plotly.com/python/hover-text-and-formatting/
    """
    if metric == "Lost Energy":
        return f"%{{{var}:.0f}} {METRIC_UNITS[metric]}"
    elif metric == "Lost Revenue":
        return f"{METRIC_UNITS[metric]} (%{{{var}:,.0f}})"
    elif metric == "Downtime":
        return f"%{{{var}:.0f}} {METRIC_UNITS[metric]}"
    elif metric == "Count":
        return f"%{{{var}:.0f}} {METRIC_UNITS[metric]}"
    elif metric in ["FaultCode", "Relative Deviation"]:
        return f"%{{{var}:.0f}}"
    else:
        return f"%{{{var}}}"


def join_hover_pair(label, value):
    """Join a label and its associated value for the Plotly hover."""
    return ": ".join(
        [
            label,
            value,
        ]
    )


def join_hover_lines(all_lines):
    """Manually join a bunch of lines to create a hovertemplate."""
    return "<br>".join(all_lines + ["<extra></extra>"])


def gen_hovertemplate(hover_data, x=None, y=None):
    """Generate a consistent hovertemplate for the Fault Charts.

    Plotly Express plotting functions are convenient wrappers to
    the lower-level plotting library; they conveniently abstract
    the details.

    When attempting to attach a custom hovertemplate to a chart
    created by such a wrapper, there are a few nuances we have
    to be mindful of to prevent the hovertemplate from breaking.
    This function exists in service of dealing with these nuances.

    In creating the hovertemplate, this function:
        - order the variables in a set fashion
        - add units to each variable
        - display a desired rounded number for each variable

    Example of Usage:
    ```
    hover_data = [
        "Turbine",
        "FaultCode",
        "Description",
        "Count",
        "Downtime",
        "Lost Revenue",
        "Lost Energy",
    ]
    x = "Count"
    fig = px.scatter(
        data_frame=combined_df,
        x=x,
        y=metric,
        title=title,
        hover_data=hover_data,
    )

    hovertemplate = gen_hovertemplate(
        hover_data=hover_data,
        x="Count",
        y=metric,
    )
    fig.update_traces(hovertemplate=hovertemplate)
    ```

    Args:
        hover_data (list): A list of hover data information that
            was initially provided to the Plotly Express function.
            variable to be displayed in the hovertemplate. Note that
            the hovertemplate order is set from `FAULT_HOVER_ORDER`,
            not `hover_data`.
        x (str, optional): If used, the `x` paramter of the Plotly
            Express function.
        y (str, optional): If used, the `y` paramter of the Plotly
            Express function.

    Returns:
        hovertemplate (str): A Plotly-ready hovertemplate.
    """

    def _decide_variable_name(metric, x=None, y=None):
        """Figure out the `X` that goes in 'Label: %{`X`:.0f}'.

        Normally when you want to see a variable in a hovertemplate
        that is deemed custom data, you need to use the string
        `customdata[i]` where `i` is the index of the customdata
        parameter of the figure.

        The trick is that if the variable was _already_ declared as a
        param when creating the Plotly Express figure initially, you
        have to use the param name itself otherwise the hovertemplate
        breaks.

        Example:

        If you run the following snippet, we see what the native
        hovertemplate looks like:

        ```
        data = {"apple": [1,2,3], "banana": [1,2,3], "grape": [1,2,3]}
        hover_data = ["apple", "banana", "grape"]

        fig = px.scatter(
            data_frame=data,
            x="apple",
            y="banana",
            hover_data=hover_data,
        )
        print(fig["data"][0].hovertemplate)
        ### 'apple=%{x}<br>banana=%{y}<br>grape=%{customdata[0]}<extra></extra>'
        ```

        But if you run the same code and remove the `y="banana",` line, we see
        the hovertemplate becomes

        ```
        ### 'apple=%{x}<br>index=%{y}<br>banana=%{customdata[0]}<br>grape=%{customdata[1]}<extra></extra>'
        ```

        As we see, the index of `customdata[index]` shifts depending on the
        number of params that are declared in `px.scatter`.
        """
        if metric == x:
            return "x"
        elif metric == y:
            return "y"
        else:
            mod_idx = hover_data.index(metric)
            if x:
                mod_idx -= hover_data.index(x) <= hover_data.index(metric)
            if y:
                mod_idx -= hover_data.index(y) <= hover_data.index(metric)
            return f"customdata[{mod_idx}]"

    all_lines = []
    for metric in FAULT_HOVER_ORDER:
        if metric not in hover_data:
            continue
        label = metric
        var = _decide_variable_name(metric, x, y)
        united_var = append_units_and_round(var=var, metric=metric)

        line = join_hover_pair(label=label, value=united_var)
        all_lines.append(line)
    hovertemplate = join_hover_lines(all_lines=all_lines)
    return hovertemplate
