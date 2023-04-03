from src.gradio import iface, auth

if __name__ == "__main__":
    iface.launch(share=True, auth=auth)
