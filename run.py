from src.train import Train


if __name__ == "__main__":
    obj = Train(data_path="teknofest_train_final.csv")
    obj.execute()
