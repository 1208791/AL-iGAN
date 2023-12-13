
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
random.seed(1)
import time
np.random.seed(1)
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from sklearn import ensemble
import xgboost as xgb
from joblib import dump, load
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from math import sqrt

class run_other_model():
    def __init__(self, model_name, number, strname, stand=None, fre=5):
        self.fre = fre
        self.model_name = model_name
        self.number = number
        self.strname = strname
        self.stand = stand
        self.stand_data_src = './基准数据/'
        self.datasrc = './%s/%s/%s/data/' % (self.model_name, self.strname, self.number)
        self.src = './%s/%s/%s/'%(self.model_name, self.strname, self.number)
        self.modelsrc = './%s/%s/%s/model/'%(self.model_name, self.strname, self.number)
        if not os.path.exists(self.modelsrc):
            os.makedirs(self.modelsrc)

        self.new_modelsrc = './%s/%s/%s/model/'%(self.model_name, self.strname, self.number+1)
        if not os.path.exists(self.new_modelsrc):
            os.makedirs(self.new_modelsrc)

        self.new_src = './%s/%s/%s/data/'%(self.model_name, self.strname, self.number+1)
        if not os.path.exists(self.new_src):
            os.makedirs(self.new_src)

        self.result_src = './%s/%s/%s/result/'%(self.model_name, self.strname, self.number)
        if not os.path.exists(self.result_src):
            os.makedirs(self.result_src)

        self.new_result_src = './%s/%s/%s/result/'%(self.model_name, self.strname, self.number+1)
        if not os.path.exists(self.new_result_src):
            os.makedirs(self.new_result_src)
        if self.model_name == 'xgb':
            self.model = MultiOutputRegressor(xgb.XGBRegressor())
        elif self.model_name == 'rfr':
            self.model = ensemble.RandomForestRegressor(bootstrap=True, criterion='squared_error', max_depth=9,
                            max_features='auto', max_leaf_nodes=None,
                            min_impurity_decrease=0.0,  min_samples_leaf=1, min_samples_split=70,
                            min_weight_fraction_leaf=0.0, n_estimators=11, n_jobs=-1,
                            oob_score=False, random_state=None, verbose=0, warm_start=False)
        elif self.model_name == 'svr':
            self.model = MultiOutputRegressor(svm.SVR())

    def AL(self,C=10):
        for i in range(self.fre):
            if self.number == 0:
                self.train_data = np.load(self.stand_data_src + 'train_data.npy')
                self.train_label = np.load(self.stand_data_src+ 'train_label.npy')
                self.test_data = np.load(self.stand_data_src + 'test_data.npy')
                self.test_label = np.load(self.stand_data_src + 'test_label.npy')
                self.geoData = np.load(self.stand_data_src + 'geoData.npy')
                self.pool_data = np.load(self.stand_data_src + 'pool_data.npy')
                self.pool_label = np.load(self.stand_data_src + 'pool_label.npy')
                self.pool_data_unselect = np.load(self.stand_data_src + 'pool_data_unselect.npy', allow_pickle=True)
                self.pool_label_unselect = np.load(self.stand_data_src + 'pool_label_unselect.npy', allow_pickle=True)
            else:
                self.train_data = np.load(self.datasrc + 'train_data_%s.npy'%i)
                self.train_label = np.load(self.datasrc + 'train_label_%s.npy'%i)
                self.test_data = np.load(self.datasrc + 'test_data_%s.npy'%i)
                self.test_label = np.load(self.datasrc + 'test_label_%s.npy'%i)
                self.geoData = np.load(self.datasrc + 'geoData_%s.npy'%i)
                self.pool_data = np.load(self.datasrc + 'pool_data_%s.npy'%i)
                self.pool_label = np.load(self.datasrc + 'pool_label_%s.npy'%i)
                self.pool_data_unselect = np.load(self.datasrc + 'pool_data_unselect_%s.npy'%i, allow_pickle=True)
                self.pool_label_unselect = np.load(self.datasrc + 'pool_label_unselect_%s.npy'%i, allow_pickle=True)
            pooldata_un = self.pool_data_unselect
            time0 = time.time()
            temp_model = self.model
            temp_model.fit(self.train_data, self.train_label)
            time1 = time.time()
            y_pred = temp_model.predict(self.test_data)
            dump(temp_model,self.modelsrc + '/current_model_%s.pkl'%i)
            time2 = time.time()
            np.save(self.result_src + './train_time_%s.npy'%i, time1-time0)
            np.save(self.result_src + './test_time_%s.npy'%i, time2-time1)
            result_mse = mean_squared_error(y_pred, self.test_label)
            np.save(self.result_src + './mse_%s.npy'%i, result_mse)
            value_temp = []
            number_temp = []
            time3 = time.time()
            if self.strname == 'un':
                for j in range(len(pooldata_un)):
                    x = pooldata_un[j]
                    pred = temp_model.predict(x)
                    pred[pred < 0] = 0
                    temp = 0
                    number_final = np.zeros([len(pred), 1])
                    for tt in pred:
                        t = tt
                        h = 0
                        t__ = [np.exp(t_) for t_ in t]
                        t__ = t__/np.sum(t__)
                        for iii in t__:
                            h += -iii*np.log(iii)
                        number_final[temp] = h
                        temp += 1
                    value, number = np.max(number_final), np.argmax(number_final)
                    value_temp.append(value)
                    number_temp.append(number)
                value_temp = np.array(value_temp)
                number_temp = np.array(number_temp)
            elif self.strname == 'qbc':
                temp_value_matrix = []
                for j in range(len(pooldata_un)):
                    x__ = pooldata_un[j]
                    temp_value_matrix.append(np.zeros([len(x__),11]))
                for k in range(C):
                    x_train, _, y_train, __ = train_test_split(self.train_data, self.train_label, test_size=0.2, random_state=k)
                    qbc_temp_model = self.model
                    qbc_temp_model.fit(x_train, y_train)
                    dump(qbc_temp_model, self.modelsrc + './qbc_model%s_%s.pkl'%(i,k))
                    for j in range(len(pooldata_un)):
                        x = pooldata_un[j]
                        pred = qbc_temp_model.predict(x)
                        pred[pred < 0] = 0
                        temp = 0
                        for iiii in pred:
                            t__ = np.argmax(iiii)
                            temp_value_matrix[j][temp, t__] +=1
                            temp += 1
                for j in range(len(pooldata_un)):
                    x = pooldata_un[j]
                    temp_number_ = np.zeros([len(x),1])
                    j_matrix = temp_value_matrix[j]
                    hh = 0
                    for t in j_matrix:
                        h = 0
                        t__ = t/C
                        for iiiii in t__:
                            if iiiii == 0:
                                h=h
                            else:
                                h += -iiiii*np.log(iiiii)
                        temp_number_[hh] = h
                        hh += 1
                    value, number = np.max(temp_number_), np.argmax(temp_number_)
                    value_temp.append(value)
                    number_temp.append(number)
                value_temp = np.array(value_temp)
                number_temp = np.array(number_temp)
            else:
                for j in range(len(pooldata_un)):
                    x = pooldata_un[j]
                    number_final = np.random.randint(0, len(x))
                    value, number = np.max(number_final), np.argmax(number_final)
                    value_temp.append(value)
                    number_temp.append(number)
                value_temp = np.array(value_temp)
                number_temp = np.array(number_temp)
            time4 = time.time()
            np.save(self.result_src + './active_time_%s.npy'%i, time4-time3)
            np.save(self.result_src + './target_pool_index_%s.npy'%i, np.argmax(value_temp))

            index = np.argmax(value_temp)
            if self.number == 0:
                old_train_data = np.load(self.stand_data_src + './train_data.npy')
                old_train_label = np.load(self.stand_data_src + './train_label.npy')
                old_test_data = np.load(self.stand_data_src + './test_data.npy')
                old_test_label = np.load(self.stand_data_src + './test_label.npy')
                old_geoData = np.load(self.stand_data_src + './geoData.npy')
                old_pool_data = np.load(self.stand_data_src + './pool_data.npy')
                old_pool_label = np.load(self.stand_data_src + './pool_label.npy')
                old_pool_data_unselect = np.load(self.stand_data_src + './pool_data_unselect.npy', allow_pickle=True)
                old_pool_label_unselect = np.load(self.stand_data_src + './pool_label_unselect.npy', allow_pickle=True)
            else:
                old_train_data = np.load(self.datasrc + './train_data_%s.npy'%i)
                old_train_label = np.load(self.datasrc + './train_label_%s.npy'%i)
                old_test_data = np.load(self.datasrc + './test_data_%s.npy'%i)
                old_test_label = np.load(self.datasrc + './test_label_%s.npy'%i)
                old_geoData = np.load(self.datasrc + './geoData_%s.npy'%i)
                old_pool_data = np.load(self.datasrc + './pool_data_%s.npy'%i)
                old_pool_label = np.load(self.datasrc + './pool_label_%s.npy'%i)
                old_pool_data_unselect = np.load(self.datasrc + './pool_data_unselect_%s.npy'%i, allow_pickle=True)
                old_pool_label_unselect = np.load(self.datasrc + './pool_label_unselect_%s.npy'%i, allow_pickle=True)

            temp_train_data, _, temp_train_label, __ = train_test_split(old_train_data, old_train_label, test_size=0.2, random_state=i)
            new_train_data = np.vstack([temp_train_data, old_pool_data_unselect[index]])
            new_train_label = np.vstack([temp_train_label, old_pool_label_unselect[index]])
            new_test_data = old_test_data
            new_test_label = old_test_label
            new_pool_data_unselect = np.delete(old_pool_data_unselect, index)
            new_pool_label_unselect = np.delete(old_pool_label_unselect, index)
            new_pool_data = np.vstack(new_pool_data_unselect)
            new_pool_label = np.vstack(new_pool_label_unselect)
            new_geodata = old_geoData

            np.save(self.new_src + './train_data_%s.npy'%i, new_train_data)
            np.save(self.new_src + './train_label_%s.npy'%i, new_train_label)
            np.save(self.new_src + './test_data_%s.npy'%i, new_test_data)
            np.save(self.new_src + './test_label_%s.npy'%i, new_test_label)
            np.save(self.new_src + './pool_data_%s.npy'%i, new_pool_data)
            np.save(self.new_src + './pool_label_%s.npy'%i, new_pool_label)
            np.save(self.new_src + './pool_data_unselect_%s.npy'%i, new_pool_data_unselect)
            np.save(self.new_src + './pool_label_unselect_%s.npy'%i, new_pool_label_unselect)
            np.save(self.new_src + './geoData_%s.npy'%i, new_geodata)


class Multiheadselfattention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_unit = None, num_heads = 8):
        super(Multiheadselfattention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_unit = num_unit
        if self.num_unit is None:
            self.num_unit = self.dim_in
        self.num_heads = num_heads
        self.linear_q = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_k),
            nn.ReLU())
        self.linear_k = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_k),
            nn.ReLU())
        self.linear_v = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_v),
            nn.ReLU())
        self._norm_fact = 1 / sqrt(self.dim_k //self.num_heads)
    def Position_Enbedding(self, x, position_size):
        batch_size, seq_len = x.shape[0], x.shape[1]
        position_j = 1. /torch.pow(10000., 2 * torch.arange(position_size/2) /position_size)
        position_j = position_j.unsqueeze(0)
        position_i = torch.arange(seq_len)
        position_i = position_i.unsqueeze(1)
        position_ij = torch.matmul(position_i.type(torch.float32), position_j.type(torch.float32))
        position_ij = torch.cat([torch.cos(position_ij), torch.sin(position_ij)], dim=1)
        position_embedding = position_ij.unsqueeze(0) + torch.zeros((batch_size, seq_len, position_size))
        return position_embedding

    def forward(self, x):
        batch, n, dim_in = x.shape
        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist = nn.Softmax(dim=-1)(dist)
        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)
        att = att + self.Position_Enbedding(x, self.num_heads).to(device)
        return att
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.atten = Multiheadselfattention(69, 8, 8, 8)
        self.dense0 = nn.Sequential(
            self.atten,
            nn.AvgPool1d(kernel_size=8,stride=3),
            nn.Flatten()
        )
        self.dense1 = nn.Sequential(
            nn.Linear(10,20),
            nn.LeakyReLU()
        )
        self.dense2 = nn.Sequential(
            nn.Linear(20,7),
            nn.LeakyReLU()
        )
        self.outputs = nn.Sequential(
            nn.Linear(7,11),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs):
        out = self.dense0(inputs)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.outputs(out)
        return out
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(11+69,7),
            nn.LeakyReLU()
        )
        self.dense2 = nn.Sequential(
            nn.Linear(7,20),
            nn.LeakyReLU()
        )
        self.dense3 = nn.Sequential(
            nn.Linear(20,1),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        out = self.dense1(inputs)
        out = self.dense2(out)
        out = self.dense3(out)
        return out
def generator_loss(d_fake):
    return -torch.mean(d_fake)
def teacher_loss(gen_data, real_data):
    return torch.mean(torch.square(gen_data-real_data))
def discriminator_loss(d_fake, d_real):
    return torch.mean(d_fake) - torch.mean(d_real)
def acc(y1, y2):
    return torch.mean(torch.eq(torch.argmax(y1, dim=-1), torch.argmax(y2, dim=-1)).float())

class IGAN():
    def __init__(self, number, strname=None, stand = None):
        self.number = number
        self.strname = strname
        self.stand = stand
        self.par_src = './corr/'
        self.datasrc = self.par_src + './GAN2/%s/%s/data/'%(self.strname, self.number)
        if number == 0:
            self.train_data = np.load(self.datasrc + 'train_data.npy')
            self.train_label = np.load(self.datasrc + 'train_label.npy')
            self.test_data = np.load(self.datasrc + 'test_data.npy')
            self.test_label = np.load(self.datasrc + 'test_label.npy')
            self.geoData = np.load(self.datasrc + 'geoData.npy')
            self.pool_data = np.load(self.datasrc + 'pool_data.npy')
            self.pool_label = np.load(self.datasrc + 'pool_label.npy')
            self.pool_data_unselect = np.load(self.datasrc + 'pool_data_unselect.npy', allow_pickle=True)
            self.pool_label_unselect = np.load(self.datasrc + 'pool_label_unselect.npy', allow_pickle=True)
        self.src = self.par_src + './GAN2/%s/%s/'%(self.strname, self.number)
        self.modelsrc = self.par_src +'./GAN2/%s/%s/model/'%(self.strname, self.number)
        if not os.path.exists(self.modelsrc):
            os.makedirs(self.modelsrc)
        self.new_modelsrc = self.par_src +'./GAN2/%s/%s/model/'%(self.strname, self.number+1)
        if not os.path.exists(self.new_modelsrc):
            os.makedirs(self.new_modelsrc)
        self.new_src = self.par_src +'./GAN2/%s/%s/data/'%(self.strname, self.number+1)
        self.new_src = self.par_src +'./GAN2/%s/%s/data/'%(self.strname, self.number+1)
        if not os.path.exists(self.new_src):
            os.makedirs(self.new_src)
        self.new_result_src = self.par_src +'./GAN2/%s/%s/result/'%(self.strname, self.number+1)
        if not os.path.exists(self.new_result_src):
            os.makedirs(self.new_result_src)
        self.G = Generator()
        self.D = Discriminator()
    def AL(self, model ,pool_data_unselect, train_data, train_label, strname, C, k_):
        pooldata_un = pool_data_unselect
        value_temp = []
        number_temp = []
        time1 = time.time()
        if strname == 'qbc':
            for k1 in range(C):
                model = self.train_base_model(np.float32(train_data), train_label, random_s=k_, fu=k1)
        for j in range(len(pooldata_un)):
            x = pooldata_un[j]
            x = torch.tensor(x).to(device)
            if strname == 'un':
                with torch.no_grad():
                    pred = model(x.to(torch.float32))
                    temp = 0
                    number_final = np.zeros([len(pred), 1])
                    for tt in pred:
                        t = tt.cpu()
                        h = 0
                        t__ = [np.exp(t_) for t_ in t]
                        t__ = t__/np.sum(t__)
                        for i in t__:
                            h += -i*np.log(i)
                        number_final[temp] = h
                        temp += 1
            elif strname == 'qbc':
                number_final = np.zeros([len(train_data), 1])
                number_middle = np.zeros([len(train_data), 11])
                for k in range(C):
                    temp = 0
                    number_ = np.zeros([len(train_data),11])
                    model = torch.load(self.modelsrc + 'train_G_model_base_%s_%s.pkl'%(k, k_)).to(device)
                    with torch.no_grad():
                        pred = model(x.to(torch.float32))
                    for i in pred:
                        i = i.cpu()
                        t_ = np.argmax(i)
                        number_[temp, t_] +=1
                        temp += 1
                    number_middle += number_
                hh = 0
                for t in number_middle:
                    h = 0
                    t__  =t/C
                    for i in t__:
                        if i == 0:
                            h = h
                        else:
                            h += -i*np.log(i)
                    number_final[hh] = h
                    hh += 1
            else:
                number_final = np.random.randint(0, len(x))
            value, number = np.max(number_final), np.argmax(number_final)
            value_temp.append(value)
            number_temp.append(number)
        value_temp = np.array(value_temp)
        number_temp = np.array(number_temp)
        time2 = time.time()
        np.save(self.src + 'active_time_%s.npy'%k_, time2-time1)
        np.save(self.src + 'target_pool_index_%s.npy'%k_, np.argmax(value_temp))
        return time2-time1, np.argmax(value_temp), number_temp[np.argmax(value_temp)]
    def AL_old_data(self, model, train_data, train_label,str_name, C, k_, rtrain_size = 1, stand = 0):
        train_label = torch.tensor(train_label)
        if stand == 0:
            temp = np.zeros([train_data.shape[0],1])
            with torch.no_grad():
                pred = model(torch.tensor(train_data).to(torch.float32).to(device)).cpu()
            for i in range(pred.shape[0]):
                temp[i] = acc(pred[i],train_label[i])
            temp_index = np.argsort(temp)

            new_data = train_data[sorted(temp_index[-np.int32(train_data.shape[0]*rtrain_size):])]
            new_label = train_label[sorted(temp_index[-np.int32(train_data.shape[0]*rtrain_size):])].reshape(-1,11)
        else:
            temp_index = np.random.choice(train_data.shape[0], np.int32(train_data.shape[0]*rtrain_size),replace=False)
            new_data = train_data[temp_index]
            new_label = train_label[temp_index]
        return new_data, new_label
    def train_base_model(self, x_train, y_train, random_s, fu ,batch_size = 32,
                         epochs = 200,
                         test_size = 0.2, save_model = True, pretrain = True,lr_G = 0.001, lr_D = 0.005, lr_T = 0.001):
        if pretrain:
            G = torch.load(self.par_src + './GAN2/%s/0/model/'%self.strname+'pre_model_%s.pkl'%random_s)
        else:
            G = self.G
        D = self.D
        G = G.to(device)
        D = D.to(device)
        optimizer_G = torch.optim.Adam(G.parameters(),lr = lr_G)
        optimizer_D = torch.optim.Adam(D.parameters(),lr = lr_D)
        optimizer_T = torch.optim.Adam(G.parameters(),lr = lr_T)
        train_data, _, train_label, _=train_test_split(x_train, y_train, test_size=test_size, random_state=fu)
        tensor_train_data = torch.tensor(self.train_data).to(device)
        tensor_train_label = torch.tensor(self.train_label).to(device)
        tensor_test_data = torch.tensor(self.test_data).to(device)
        tensor_test_label = torch.tensor(self.test_label).to(device)
        tensor_geo_data = torch.tensor(self.geoData)[torch.randint(0, 88, [batch_size])].to(torch.float32).to(device)
        dataset = TensorDataset(tensor_train_data, tensor_train_label)
        data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, drop_last=True)
        G_loss_list = []
        D_loss_list = []
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        for step in range(epochs):
            G_loss_sum = 0
            D_loss_sum = 0
            T_loss_sum = 0
            train_acc_sum = 0
            for num, batch_dataset in enumerate(data_loader):
                for parm in D.parameters():
                    parm.data.clamp_(-0.01,0.01)
                G_fake = G(batch_dataset[0].to(torch.float32)).detach()
                G_pair = torch.cat([batch_dataset[0], G_fake],dim=1).to(torch.float32).detach()
                real_pair = torch.cat([batch_dataset[0], tensor_geo_data],dim=1).to(torch.float32)
                D_fake = D(G_pair)
                D_real = D(real_pair)

                D_loss = discriminator_loss(D_fake, D_real)
                with torch.no_grad():
                    D_loss_sum += D_loss.item()
                optimizer_D.zero_grad()
                D_loss.backward(retain_graph=True)
                optimizer_D.step()

                D_fake = D(G_pair)

                G_loss = generator_loss(D_fake)
                with torch.no_grad():
                    G_loss_sum += G_loss.item()
                optimizer_G.zero_grad()
                G_loss.backward(retain_graph=True)
                optimizer_G.step()

                G_fake = G(batch_dataset[0].to(torch.float32))
                T_loss = teacher_loss(G_fake, batch_dataset[1])
                with torch.no_grad():
                    train_acc = acc(G_fake, batch_dataset[1])
                    train_acc_sum += train_acc.item()
                    T_loss_sum += T_loss.item()
                optimizer_T.zero_grad()
                T_loss.backward()
                optimizer_T.step()

            with torch.no_grad():
                pred = G(tensor_test_data.to(torch.float32))
                test_loss = teacher_loss(pred, tensor_test_label).item()
                test_acc = acc(pred, tensor_test_label).item()
            G_loss_list.append(G_loss_sum/(num+1))
            D_loss_list.append(D_loss_sum/(num+1))
            train_loss_list.append(T_loss_sum/(num+1))
            train_acc_list.append(train_acc_sum/(num+1))
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        if save_model:
            torch.save(G, self.modelsrc + 'train_G_model_base_%s_%s.pkl'%(fu, random_s))
            torch.save(D, self.modelsrc + 'train_D_model_base_%s_%s.pkl'%(fu, random_s))
        return G
    def train_pre_model(self, k=5,batch_size = 32,epochs = 400, lr_G = 0.005, test_size = 0.1, shuffle = False, save_model = True):
        for kk in range(k):
            pre_G = self.G
            pre_G = pre_G.to(device)
            optimizer_G = torch.optim.Adam(pre_G.parameters(),lr = lr_G)
            if self.number != 0:
                self.train_data = np.load(self.datasrc + 'train_data_%s.npy'%kk)
                self.train_label = np.load(self.datasrc + 'train_label_%s.npy'%kk)
                self.test_data = np.load(self.datasrc + 'test_data_%s.npy'%kk)
                self.test_label = np.load(self.datasrc + 'test_label_%s.npy'%kk)
                self.geoData = np.load(self.datasrc + 'geoData_%s.npy'%kk)
                self.pool_data = np.load(self.datasrc + 'pool_data_%s.npy'%kk)
                self.pool_label = np.load(self.datasrc + 'pool_label_%s.npy'%kk)
                self.pool_data_unselect = np.load(self.datasrc + 'pool_data_unselect_%s.npy'%kk, allow_pickle=True)
                self.pool_label_unselect = np.load(self.datasrc + 'pool_label_unselect_%s.npy'%kk, allow_pickle=True)
            train_data, test_data, train_label, test_label = train_test_split(self.train_data, self.train_label, test_size=test_size, random_state=kk)
            tensor_data = torch.tensor(train_data).to(device)
            tensor_label = torch.tensor(train_label).to(device)
            tensor_test_data = torch.tensor(test_data).to(device)
            tensor_test_label = torch.tensor(test_label).to(device)
            dataset = TensorDataset(tensor_data, tensor_label)
            data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle, drop_last=True)
            for step in range(epochs):
                G_loss_sum = 0
                G_acc_sum = 0
                for num, batch_dataset in enumerate(data_loader):
                    G_pred = pre_G(batch_dataset[0].to(torch.float32))
                    G_loss = teacher_loss(G_pred, batch_dataset[1])
                    with torch.no_grad():
                        G_loss_sum += G_loss.item()
                        G_acc = acc(G_pred, batch_dataset[1])
                        G_acc_sum += G_acc.item()
                    optimizer_G.zero_grad()
                    G_loss.backward()
                    optimizer_G.step()
                with torch.no_grad():
                    pred = pre_G(tensor_test_data.to(torch.float32))
                    test_loss = teacher_loss(pred, tensor_test_label).item()
                    test_acc = acc(pred, tensor_test_label).item()
            if save_model:
                torch.save(pre_G, self.modelsrc + 'pre_model_%s.pkl'%kk)
    def train_model(self, k = 5, batch_size = 88, epochs = 300, pretrain = True,lr_G = 0.001, lr_D = 0.005, lr_T = 0.001, save_model = True):
        for kk in range(k):
            if pretrain:
                G = torch.load(self.modelsrc + 'pre_model_%s.pkl'%kk)
            else:
                G = self.G
            D = self.D
            G = G.to(device)
            D = D.to(device)
            optimizer_G = torch.optim.Adam(G.parameters(),lr = lr_G)
            optimizer_D = torch.optim.Adam(D.parameters(),lr = lr_D)
            optimizer_T = torch.optim.Adam(G.parameters(),lr = lr_T)
            if self.number != 0:
                self.train_data = np.load(self.datasrc + 'train_data_%s.npy'%kk)
                self.train_label = np.load(self.datasrc + 'train_label_%s.npy'%kk)
                self.test_data = np.load(self.datasrc + 'test_data_%s.npy'%kk)
                self.test_label = np.load(self.datasrc + 'test_label_%s.npy'%kk)
                self.geoData = np.load(self.datasrc + 'geoData_%s.npy'%kk)
                self.pool_data = np.load(self.datasrc + 'pool_data_%s.npy'%kk)
                self.pool_label = np.load(self.datasrc + 'pool_label_%s.npy'%kk)
                self.pool_data_unselect = np.load(self.datasrc + 'pool_data_unselect_%s.npy'%kk, allow_pickle=True)
                self.pool_label_unselect = np.load(self.datasrc + 'pool_label_unselect_%s.npy'%kk, allow_pickle=True)
            tensor_train_data = torch.tensor(self.train_data).to(device)
            tensor_train_label = torch.tensor(self.train_label).to(device)
            tensor_test_data = torch.tensor(self.test_data).to(device)
            tensor_test_label = torch.tensor(self.test_label).to(device)
            tensor_geo_data = torch.tensor(self.geoData).to(device)
            dataset = TensorDataset(tensor_train_data, tensor_train_label)
            data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, drop_last=True)
            G_loss_list = []
            D_loss_list = []
            train_loss_list = []
            train_acc_list = []
            test_loss_list = []
            test_acc_list = []
            time11 = time.time()
            for step in range(epochs):
                G_loss_sum = 0
                D_loss_sum = 0
                T_loss_sum = 0
                train_acc_sum = 0
                for num, batch_dataset in enumerate(data_loader):
                    for parm in D.parameters():
                        parm.data.clamp_(-0.01,0.01)
                    G_fake = G(batch_dataset[0].to(torch.float32)).detach()
                    G_pair = torch.cat([batch_dataset[0], G_fake],dim=1).to(torch.float32).detach()
                    real_pair = torch.cat([batch_dataset[0], tensor_geo_data],dim=1).to(torch.float32)
                    D_fake = D(G_pair)
                    D_real = D(real_pair)

                    D_loss = discriminator_loss(D_fake, D_real)
                    with torch.no_grad():
                        D_loss_sum += D_loss.item()
                    optimizer_D.zero_grad()
                    D_loss.backward(retain_graph=True)
                    optimizer_D.step()

                    D_fake = D(G_pair)

                    G_loss = generator_loss(D_fake)
                    with torch.no_grad():
                        G_loss_sum += G_loss.item()
                    optimizer_G.zero_grad()
                    G_loss.backward(retain_graph=True)
                    optimizer_G.step()

                    G_fake = G(batch_dataset[0].to(torch.float32))
                    T_loss = teacher_loss(G_fake, batch_dataset[1])
                    with torch.no_grad():
                        train_acc = acc(G_fake, batch_dataset[1])
                        train_acc_sum += train_acc.item()
                        T_loss_sum += T_loss.item()
                    optimizer_T.zero_grad()
                    T_loss.backward()
                    optimizer_T.step()
                time33 = time.time()
                with torch.no_grad():
                    pred = G(tensor_test_data.to(torch.float32))
                    test_loss = teacher_loss(pred, tensor_test_label).item()
                    test_acc = acc(pred, tensor_test_label).item()
                time44 = time.time()
                G_loss_list.append(G_loss_sum/(num+1))
                D_loss_list.append(D_loss_sum/(num+1))
                train_loss_list.append(T_loss_sum/(num+1))
                train_acc_list.append(train_acc_sum/(num+1))
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            time22 = time.time()
            if save_model:
                torch.save(G, self.modelsrc + 'train_G_model_%s.pkl'%kk)
                torch.save(D, self.modelsrc + 'train_D_model_%s.pkl'%kk)
            result = [G_loss_list, D_loss_list, train_loss_list, train_acc_list, test_loss_list, test_acc_list]
            np.save(self.src + 'train_result_%s.npy'%kk, result)
            np.save(self.src + 'train_time_%s.npy'%kk, time22-time11-(time44-time33))
            np.save(self.src + 'test_time_%s.npy'%kk, time44-time33)
    def train_all_data(self, k = 5, batch_size = 88, epochs = 300, pretrain = True,
                    lr_G = 0.001, lr_D = 0.005, lr_T = 0.001, save_model = True, C = 10):
        for kk in range(k):
            if pretrain:
                G = torch.load(self.modelsrc + 'pre_model_%s.pkl'%kk)
            else:
                G = self.G
            D = self.D
            G = G.to(device)
            D = D.to(device)
            optimizer_G = torch.optim.Adam(G.parameters(),lr = lr_G)
            optimizer_D = torch.optim.Adam(D.parameters(),lr = lr_D)
            optimizer_T = torch.optim.Adam(G.parameters(),lr = lr_T)
            if self.number != 0:
                self.train_data = np.load(self.datasrc + 'train_data_%s.npy'%kk)
                self.train_label = np.load(self.datasrc + 'train_label_%s.npy'%kk)
                self.test_data = np.load(self.datasrc + 'test_data_%s.npy'%kk)
                self.test_label = np.load(self.datasrc + 'test_label_%s.npy'%kk)
                self.geoData = np.load(self.datasrc + 'geoData_%s.npy'%kk)
                self.pool_data = np.load(self.datasrc + 'pool_data_%s.npy'%kk)
                self.pool_label = np.load(self.datasrc + 'pool_label_%s.npy'%kk)
                self.pool_data_unselect = np.load(self.datasrc + 'pool_data_unselect_%s.npy'%kk, allow_pickle=True)
                self.pool_label_unselect = np.load(self.datasrc + 'pool_label_unselect_%s.npy'%kk, allow_pickle=True)
            time_11 = time.time()
            actime, index, _ = self.AL(G, pool_data_unselect = self.pool_data_unselect,train_data=self.train_data, train_label=self.train_label, strname=self.strname, C=C, k_=kk)
            time_22 = time.time()
            tensor_train_data = torch.cat((torch.tensor(self.train_data), torch.tensor(self.pool_data_unselect[index].astype(float)).to(torch.float32)), dim = 0).to(device)
            tensor_train_label = torch.cat((torch.tensor(self.train_label), torch.tensor(self.pool_label_unselect[index].astype(float)).to(torch.float32)), dim = 0).to(device)
            self.save_new_data(index = index, k=kk)

    def itrain_model_d2(self, k=5, batch_size = 88, rtrain_size = 0.5,epochs = 300, lr_G = 0.0001, lr_D = 0.0005, lr_T = 0.001, lamb = 1,
                        save_model = True, C = 10):
        for kk in range(k):
            if self.number ==0:
                self.train_data = np.load(self.datasrc + 'train_data.npy')
                self.train_label = np.load(self.datasrc + 'train_label.npy')
                self.test_data = np.load(self.datasrc + 'test_data.npy')
                self.test_label = np.load(self.datasrc + 'test_label.npy')
                self.geoData = np.load(self.datasrc + 'geoData.npy')
                self.pool_data = np.load(self.datasrc + 'pool_data.npy')
                self.pool_label = np.load(self.datasrc + 'pool_label.npy')
                self.pool_data_unselect = np.load(self.datasrc + 'pool_data_unselect.npy', allow_pickle=True)
                self.pool_label_unselect = np.load(self.datasrc + 'pool_label_unselect.npy', allow_pickle=True)
            else:
                self.train_data = np.load(self.datasrc + 'train_data_%s.npy'%kk)
                self.train_label = np.load(self.datasrc + 'train_label_%s.npy'%kk)
                self.test_data = np.load(self.datasrc + 'test_data_%s.npy'%kk)
                self.test_label = np.load(self.datasrc + 'test_label_%s.npy'%kk)
                self.geoData = np.load(self.datasrc + 'geoData_%s.npy'%kk)
                self.pool_data = np.load(self.datasrc + 'pool_data_%s.npy'%kk)
                self.pool_label = np.load(self.datasrc + 'pool_label_%s.npy'%kk)
                self.pool_data_unselect = np.load(self.datasrc + 'pool_data_unselect_%s.npy'%kk, allow_pickle=True)
                self.pool_label_unselect = np.load(self.datasrc + 'pool_label_unselect_%s.npy'%kk, allow_pickle=True)
            if os.path.exists('./GAN2/%s/0/model/'%self.strname + 'train_G_model_%s.pkl'%kk):
                G = torch.load('./GAN2/%s/0/model/'%self.strname + 'train_G_model_%s.pkl'%kk)
                D = torch.load('./GAN2/%s/0/model/'%self.strname + 'train_D_model_%s.pkl'%kk)
            else:
                print('Please train model G and D!')
            D.dense3[0] = nn.Linear(20,2,bias=True)

            G = G.to(device)
            D = D.to(device)
            optimizer_G = torch.optim.Adam(G.parameters(),lr = lr_G)
            optimizer_D = torch.optim.Adam(D.parameters(),lr = lr_D)
            optimizer_T = torch.optim.Adam(G.parameters(),lr = lr_T)
            new_data, new_label = self.AL_old_data(G, train_data=self.train_data, train_label=self.train_label, str_name=self.strname, C=C,k_ = kk, rtrain_size = rtrain_size, stand = self.stand)

            tensor_train_data = torch.tensor(new_data).to(device)
            tensor_train_label = torch.tensor(new_label).to(device)
            tensor_test_data = torch.tensor(self.test_data).to(device)
            tensor_test_label = torch.tensor(self.test_label).to(device)
            tensor_geo_data = torch.tensor(self.geoData)[torch.randint(0, 88, [batch_size])].to(torch.float32).to(device)
            actime, index, _ = self.AL(G, pool_data_unselect = self.pool_data_unselect,train_data=tensor_train_data, train_label=tensor_train_label, strname=self.strname, C=C, k_=kk)
            self.save_new_data(index = index, k=kk)
            tensor_pool_data = torch.tensor(self.pool_data_unselect[index].astype(float)).to(torch.float32).to(device)
            tensor_pool_label = torch.tensor(self.pool_label_unselect[index].astype(float)).to(torch.float32).to(device)
            dataset = TensorDataset(torch.cat([tensor_train_data, tensor_pool_data], dim=0).to(torch.float32),
                                    torch.cat(
                                        [torch.cat(
                                            [tensor_train_label,
                                            torch.zeros(tensor_train_label.shape[0],1).to(torch.float32).to(device)],dim=1
                                        ),
                                        torch.cat(
                                            [tensor_pool_label,
                                            torch.ones(tensor_pool_label.shape[0],1).to(torch.float32).to(device)],dim=1
                                        )],dim=0
                                    ).to(torch.float32))
            data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, drop_last=True)
            G_loss_list = []
            D_loss_list = []
            train_loss_list = []
            train_acc_list = []
            test_loss_list = []
            test_acc_list = []
            for step in range(epochs):
                G_loss_sum = 0
                D_loss_sum = 0
                T_loss_sum = 0
                train_acc_sum = 0
                for num, batch_dataset in enumerate(data_loader):
                    for parm in D.parameters():
                        parm.data.clamp_(-0.01,0.01)
                    G_fake = G(batch_dataset[0].to(torch.float32)).detach()
                    G_pair = torch.cat([batch_dataset[0], G_fake],dim=1).to(torch.float32).detach()
                    real_pair = torch.cat([batch_dataset[0], tensor_geo_data],dim=1).to(torch.float32)
                    D_fake = D(G_pair)[:, 0]
                    D_real = D(real_pair)[:, 0]
                    D_fake_ = D(G_pair)[:, 1]
                    D_real_ = D(real_pair)[:, 1]
                    D_loss = discriminator_loss(D_fake, D_real) + lamb/2*(teacher_loss(D_fake_, batch_dataset[1][:,-1]) + teacher_loss(D_real_, batch_dataset[1][:,-1]))
                    with torch.no_grad():
                        D_loss_sum += D_loss.item()
                    optimizer_D.zero_grad()
                    D_loss.backward(retain_graph=True)
                    optimizer_D.step()

                    D_fake = D(G_pair)[:, 0]
                    D_fake_ = D(G_pair)[:, 1]
                    G_loss = generator_loss(D_fake) + lamb/2*(teacher_loss(D_fake_, batch_dataset[1][:,-1]))
                    with torch.no_grad():
                        G_loss_sum += G_loss.item()
                    optimizer_G.zero_grad()
                    G_loss.backward(retain_graph=True)
                    optimizer_G.step()

                    G_fake = G(batch_dataset[0].to(torch.float32))
                    T_loss = teacher_loss(G_fake, batch_dataset[1][:,:-1])
                    with torch.no_grad():
                        train_acc = acc(G_fake, batch_dataset[1][:,:-1])
                        train_acc_sum += train_acc.item()
                        T_loss_sum += T_loss.item()
                    optimizer_T.zero_grad()
                    T_loss.backward()
                    optimizer_T.step()

                with torch.no_grad():
                    pred = G(tensor_test_data.to(torch.float32))
                    test_loss = teacher_loss(pred, tensor_test_label).item()
                    test_acc = acc(pred, tensor_test_label).item()
                G_loss_list.append(G_loss_sum/(num+1))
                D_loss_list.append(D_loss_sum/(num+1))
                train_loss_list.append(T_loss_sum/(num+1))
                train_acc_list.append(train_acc_sum/(num+1))
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
            if save_model:
                torch.save(G, self.modelsrc + 'd2_G_model_%s.pkl'%kk)
                torch.save(D, self.modelsrc + 'd2_D_model_%s.pkl'%kk)
            result = [G_loss_list, D_loss_list, train_loss_list, train_acc_list, test_loss_list, test_acc_list]
            np.save(self.src + 'itrain_d2_result_%s.npy'%kk, result)

    def save_new_data(self, index, k):
        if self.number == 0:
            old_train_data = np.load(self.datasrc + './train_data.npy')
            old_train_label = np.load(self.datasrc + './train_label.npy')
            old_test_data = np.load(self.datasrc + './test_data.npy')
            old_test_label = np.load(self.datasrc + './test_label.npy')
            old_geoData = np.load(self.datasrc + './geoData.npy')
            old_pool_data = np.load(self.datasrc + './pool_data.npy')
            old_pool_label = np.load(self.datasrc + './pool_label.npy')
            old_pool_data_unselect = np.load(self.datasrc + './pool_data_unselect.npy', allow_pickle=True)
            old_pool_label_unselect = np.load(self.datasrc + './pool_label_unselect.npy', allow_pickle=True)
        else:
            old_train_data = np.load(self.datasrc + './train_data_%s.npy'%k)
            old_train_label = np.load(self.datasrc + './train_label_%s.npy'%k)
            old_test_data = np.load(self.datasrc + './test_data_%s.npy'%k)
            old_test_label = np.load(self.datasrc + './test_label_%s.npy'%k)
            old_geoData = np.load(self.datasrc + './geoData_%s.npy'%k)
            old_pool_data = np.load(self.datasrc + './pool_data_%s.npy'%k)
            old_pool_label = np.load(self.datasrc + './pool_label_%s.npy'%k)
            old_pool_data_unselect = np.load(self.datasrc + './pool_data_unselect_%s.npy'%k, allow_pickle=True)
            old_pool_label_unselect = np.load(self.datasrc + './pool_label_unselect_%s.npy'%k, allow_pickle=True)

        tr_data, _, tr_label, __ = train_test_split(old_train_data, old_train_label, test_size=0.2, random_state=k)
        new_train_data = np.vstack([tr_data, old_pool_data_unselect[index]])
        new_train_label = np.vstack([tr_label, old_pool_label_unselect[index]])
        new_test_data = old_test_data
        new_test_label = old_test_label
        new_pool_data_unselect = np.delete(old_pool_data_unselect, index)
        new_pool_label_unselect = np.delete(old_pool_label_unselect, index)
        new_pool_data = np.vstack(new_pool_data_unselect)
        new_pool_label = np.vstack(new_pool_label_unselect)
        new_geodata = old_geoData

        np.save(self.new_src + './train_data_%s.npy'%k, new_train_data)
        np.save(self.new_src + './train_label_%s.npy'%k, new_train_label)
        np.save(self.new_src + './test_data_%s.npy'%k, new_test_data)
        np.save(self.new_src + './test_label_%s.npy'%k, new_test_label)
        np.save(self.new_src + './pool_data_%s.npy'%k, new_pool_data)
        np.save(self.new_src + './pool_label_%s.npy'%k, new_pool_label)
        np.save(self.new_src + './pool_data_unselect_%s.npy'%k, new_pool_data_unselect)
        np.save(self.new_src + './pool_label_unselect_%s.npy'%k, new_pool_label_unselect)
        np.save(self.new_src + './geoData_%s.npy'%k, new_geodata)