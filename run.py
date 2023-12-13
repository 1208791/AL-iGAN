from model import IGAN

if __name__ == '__main__':
    str_name_list = [
        'un',
        'qbc',
        'rand'
    ]
    stand_list = [
        0,
        # 1,
        # 2,
        # 3,
        # 4
    ]
    for str__ in str_name_list:
        for stan_ in stand_list:
            for i in range(10):
                if i == 0:
                    IGAN(i, strname=str__).train_pre_model(k=5, batch_size=32, epochs=200, lr_G=0.005,test_size=0.1, shuffle=False, save_model=True)
                    IGAN(i, strname=str__).train_model(k=5, batch_size=88, epochs=300, pretrain=True,lr_G=0.0001, lr_D=0.0005, lr_T=0.001)
                else:
                    IGAN(i, strname=str__).itrain_model_d2(k=5, batch_size=32, rtrain_size=0.8, epochs=100,lr_G=0.0001, lr_D=0.0005, lr_T=0.001, lamb=1,save_model=True, C=10)