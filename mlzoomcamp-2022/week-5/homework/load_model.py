import pickle

def get_dv_model(dv_path, model_path):
    #load model:
    with open(dv_path, 'rb') as f_in:
        dv = pickle.load(f_in)
    f_in.close()

    #load vectorizer:
    with open(model_path, 'rb') as f_in:
        model = pickle.load(f_in)
    f_in.close()

    return dv, model

def predict_single(data, model_version=1):
    print(f'Using model version {model_version}')
    dv, model = get_dv_model('./dv.bin', f'./model{model_version}.bin')

    X = dv.transform(data)
    predict_score = model.predict_proba(X)[:, 1]
    return round(predict_score[0], 3)


if __name__ == '__main__':
    dv_path = './dv.bin'
    model_path = './model1.bin'
    predict_single(data)