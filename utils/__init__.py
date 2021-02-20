import numpy as np

from sklearn.model_selection import train_test_split

def split_dataframe(df, train_ratio=0.2, seed=42):
    grouping_field = 'doc_id'
    # perform groupby and put groups into dict
    dict_of_docs = dict(tuple(df.groupby(grouping_field)))
    # convert group indexes to numpy list
    list_of_indexes = np.array(list(dict_of_docs.keys()))
    # perform the split on indexes
    train_ids, val_ids = train_test_split(list_of_indexes, train_size = train_ratio, random_state=42)
    # take the df records according to splits
    df_train = df[df[grouping_field].isin(train_ids)]
    df_val = df[df[grouping_field].isin(val_ids)]
    return df_train, df_val