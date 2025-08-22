import numpy as np
import pandas as pd


def get_group_id(metadata, columns=["technology", "donor", "day"]):
    """Assign group IDs for each row of metadata based on the unique combinations of the values of the specified columns.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        The metadata to assign group IDs to.
    columns : tuple, optional
        The columns to use to assign group IDs. Default is ("technology", "donor", "day").

    Returns
    -------
    group_ids : np.ndarray
        List of integer group IDs for each row of metadata.
    """

    assert all(column in metadata.columns for column in columns), f"Some columns of {columns} are not in the metadata"

    group_ids = np.full(len(metadata), fill_value=-1)
    unique_combinations = metadata[columns].value_counts().index.tolist()

    group_id = 0

    for comb in unique_combinations:
        selector = pd.Series(True, index=metadata.index)
        for i, value in enumerate(comb):
            selector &= (metadata[columns[i]] == value)

        group_ids[selector.values] = group_id
        group_id += 1

    assert (group_ids != -1).all()

    return group_ids
