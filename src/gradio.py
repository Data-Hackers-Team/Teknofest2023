import gradio as gr
import pandas as pd
from io import StringIO


def auth(username, password):
    if username == "Data_Hackers" and password == "2GDC9C3CN7W4D3WS":
        return True
    else:
        return False


def predict(df):
    # TODO:
    df["offansive"] = 1
    df["target"] = "asdf"

    # ***************************
    # WRITE YOUR INFERENCE STEPS BELOW
    #
    # HERE
    #
    # *********** END ***********
    return df


def get_file(file):
    output_file = "output_Data_Hackers.csv"

    with open(file.name) as f:
        df = pd.read_csv(StringIO(f.read()))

    df = predict(df)
    df.to_csv(output_file, index=False, sep="|")
    return (output_file)


# Launch the interface with user password
iface = gr.Interface(get_file, "file", "file")
