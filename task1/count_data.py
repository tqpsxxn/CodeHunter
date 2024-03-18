import pandas as pd


def countPd(path):
    data = pd.read_pickle(path)
    print(path, ":", len(data))


# countPd('processed_train_ast.pkl')
# countPd('processed_test_ast.pkl')
# countPd('processed_valid_ast.pkl')
countPd('data/train.pkl')
countPd('data/test.pkl')
countPd('data/valid.pkl')