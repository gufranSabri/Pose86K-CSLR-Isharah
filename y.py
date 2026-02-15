import pandas as pd

test_df = pd.read_csv("annotations_v2/isharah2000/SI/test_orig.csv", delimiter="|")[["id"]]
test_df.to_csv("annotations_v2/isharah2000/SI/test.csv", index=False, sep="|")


test_df = pd.read_csv("annotations_v2/isharah2000/US/test_orig.csv", delimiter="|")[["id"]]
test_df.to_csv("annotations_v2/isharah2000/US/test.csv", index=False, sep="|")