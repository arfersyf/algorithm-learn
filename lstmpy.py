import csv
import numpy as np
csv_file = csv.reader(open('/Users/arfer/Downloads/TIMEDATA2.csv','r'))
x_train = [row[0] for row in csv_file]
csv_file = csv.reader(open('/Users/arfer/Downloads/TIMEDATA2.csv','r'))
y_train = [row[1] for row in csv_file]
print(x_train)
print(y_train)


# 输出单元激活函数
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

hidden_dim = 100
data_dim = 1
learning_rate=0.005
whi = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim),(hidden_dim, hidden_dim))
wxi = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, data_dim))
bi = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, 1))
whf = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim),(hidden_dim, hidden_dim))
wxf = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, data_dim))
bf = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, 1))
who = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim),(hidden_dim, hidden_dim))
wxo = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, data_dim))
bo = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, 1))
wha = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim),(hidden_dim, hidden_dim))
wxa = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, data_dim))
ba = np.random.uniform(-np.sqrt(1.0/data_dim), np.sqrt(1.0/data_dim),(hidden_dim, 1))
wy = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim),(data_dim, hidden_dim))
by = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim),(data_dim, 1))


dwhi = np.zeros(whi.shape)
dwxi = np.zeros(wxi.shape)
dbi = np.zeros(bi.shape)
dwhf = np.zeros(whf.shape)
dwxf = np.zeros(wxf.shape)
dbf = np.zeros(bf.shape)
dwho = np.zeros(who.shape)
dwxo = np.zeros(wxo.shape)
dbo = np.zeros(bo.shape)
dwha = np.zeros(wha.shape)
dwxa = np.zeros(wxa.shape)
dba = np.zeros(ba.shape)
dwy = np.zeros(wy.shape)
dby = np.zeros(by.shape)

delta_ct = np.zeros((hidden_dim, 1))

T = 1

iss = np.array([np.zeros((hidden_dim, 1))] * (T + 1))  # input gate
fss = np.array([np.zeros((hidden_dim, 1))] * (T + 1))  # forget gate
oss = np.array([np.zeros((hidden_dim, 1))] * (T + 1))  # output gate
ass = np.array([np.zeros((hidden_dim, 1))] * (T + 1))  # current inputstate
hss = np.array([np.zeros((hidden_dim, 1))] * (T + 1))  # hidden state
css = np.array([np.zeros((hidden_dim, 1))] * (T + 1))  # cell state
ys = np.array([np.zeros((data_dim, 1))] * T)

stats = {'iss': iss, 'fss': fss, 'oss': oss,'ass': ass, 'hss': hss, 'css': css,'ys': ys}

x_train = x_train[:10000]
y_train = y_train[:10000]
for i in range(len(y_train)):

    losses = []
    num_examples = 0


    ht_pre = np.array(stats['hss'][-1]).reshape(-1, 1)

    # input gate
    stats['iss'][0] = sigmoid(whi.dot(ht_pre)+ wxi.dot(float(x_train[i])).reshape(-1,1) + bi)
    # forget gate
    stats['fss'][0] = sigmoid(whf.dot(ht_pre)+ wxf.dot(float(x_train[i])).reshape(-1,1) + bf)
    # output gate
    stats['oss'][0] = sigmoid(who.dot(ht_pre)+ wxo.dot(float(x_train[i])).reshape(-1,1) + bo)
    # current inputstate
    stats['ass'][0] = tanh(wha.dot(ht_pre)+ wxa.dot(float(x_train[i])).reshape(-1,1) + ba)
    # cell state, ct = ft * ct_pre + it * at
    stats['css'][0] = stats['fss'][0] * stats['css'][-1] + stats['iss'][0] * stats['ass'][0]
    # hidden state, ht = ot * tanh(ct)
    stats['hss'][0] = stats['oss'][0] * tanh(stats['css'][0])
    # output value, yt = softmax(self.wy.dot(ht) + self.by)
    stats['ys'][0] = softmax(wy.dot(stats['hss'][0]) + by)

    # 目标函数对输出 y 的偏导数
    delta_o = stats['ys']
    delta_o[0] -= float(y_train[i])

    # 输出层wy, by的偏导数，由于所有时刻的输出共享输出权值矩阵，故所有时刻累加
    dwy += delta_o[0].dot(stats['hss'][0].reshape(1, -1))
    dby += delta_o[0]

    # 目标函数对隐藏状态的偏导数
    delta_ht = wy.T.dot(delta_o[0])

    # 各个门及状态单元的偏导数
    delta_ot = delta_ht * tanh(stats['css'][0])
    delta_ct += delta_ht * stats['oss'][0] * (1 - tanh(stats['css'][0]) ** 2)
    delta_it = delta_ct * stats['ass'][0]
    delta_ft = delta_ct * stats['css'][-1]
    delta_at = delta_ct * stats['iss'][0]

    delta_at_net = delta_at * (1 - stats['ass'][0] ** 2)
    delta_it_net = delta_it * stats['iss'][0] * (1 - stats['iss'][0])
    delta_ft_net = delta_ft * stats['fss'][0] * (1 - stats['fss'][0])
    delta_ot_net = delta_ot * stats['oss'][0] * (1 - stats['oss'][0])

    dwhf += delta_ft_net * stats['hss'][-1]
    dwxf += delta_ft_net * float(x_train[i])
    dbf += delta_ft_net

    dwhi += delta_it_net * stats['hss'][-1]
    dwxi += delta_it_net * float(x_train[i])
    dbi += delta_it_net

    dwha += delta_at_net * stats['hss'][-1]
    dwxa += delta_at_net * float(x_train[i])
    dba += delta_at_net

    dwho += delta_ot_net * stats['hss'][-1]
    dwxo += delta_ot_net * float(x_train[i])
    dbo += delta_ot_net

    whf -= learning_rate * dwhf
    wxf -= learning_rate * dwxf
    bf -= learning_rate * dbf

    whi -= learning_rate * dwhi
    wxi -= learning_rate * dwxi
    bi -= learning_rate * dbi

    wha -= learning_rate * dwha
    wxa -= learning_rate * dwxa
    ba -= learning_rate * dba

    who -= learning_rate * dwho
    wxo -= learning_rate * dwxo
    bo -= learning_rate * dbo

    stats['iss'][-1] = stats['iss'][0]
    stats['fss'][-1] = stats['fss'][0]
    stats['oss'][-1] = stats['oss'][0]
    stats['ass'][-1] = stats['ass'][0]
    stats['css'][-1] = stats['css'][0]
    stats['hss'][-1] = stats['hss'][0]

    if (i % 1000 == 0):
        print("Error:" + str(delta_o[0]))
        print("last:" + str(x_train[i]))
        print("Pred:" + str(stats['ys'][0]))
        print("True:" + str(y_train[i]))
        print(str(x_train[i]) + " -> " + str(y_train[i]))
        print("------------")