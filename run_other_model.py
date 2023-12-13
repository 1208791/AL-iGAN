from model import run_other_model

if __name__ == '__main__':
    model_list = [
        'rfr',
        'svr',
        'xgb'
    ]
    str_name_list = [
        'rand',
        'un',
        'qbc'
    ]
    for model_name in model_list:
        for str_name in str_name_list:
            for number in range(10):
                run_other_model(model_name=model_name, number=number, strname=str_name).AL()






