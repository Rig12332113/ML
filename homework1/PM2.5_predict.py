import numpy as np
import csv
import math
import pandas as pd
from argparse import Namespace

def valid(x, y):
  # TODO: Try to filter out extreme values.
    return y < 100
  #  ex: If PM2.5 > 100, then we don't use the data to train (return False), otherwise return True,
  
# Create your dataset
def parse2train(data, feats):
    x = []
    y = []

  # Use data #0~#7 to predict #8 => Total data length should be decresased by 8.
    total_length = data.shape[0] - 8

    for i in range(total_length):
        x_tmp = data[i:i+8, feats] # Use data #0~#7 to predict #8, data #1~#8 to predict #9, etc.
        y_tmp = data[i+8, -1] # last column of (i+8)th row: PM2.5

    # Filter out extreme values to train.
        if valid(x_tmp, y_tmp):
            x.append(x_tmp.reshape(-1,))
            y.append(y_tmp)
    
  # x.shape: (n, 15, 8)
  # y.shape: (n, 1)
    x = np.array(x)
    y = np.array(y)

    return x,y

#TODO: Implement 2-nd polynomial regression version for the report.
def minibatch(x, y, config):
    '''
    # Randomize the data in minibatch
    index = np.arange(x.shape[1])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    '''

    # Initialization
    batch_size = config.batch_size
    lr = config.lr
    epoch = config.epoch

    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    # Linear regression: only contains two parameters (w, b).
    w1= np.full(x[0].shape, 0.008).reshape(-1, 1)
    w2= np.full(x[0].shape, 0.001).reshape(-1, 1)
    bias = 0.1
    m_t1 = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t1 = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t2 = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t2 = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    # Training loop
    for num in range(epoch):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            #print(x_batch.shape)
            #print(y_batch.shape)
            # Prediction of linear regression
            pred = np.dot(np.square(x_batch), w2) + np.dot(x_batch, w1) + bias
            
            # loss
            loss = y_batch - pred

            # Compute gradient
            ## Edit: remove 2 * lam * np.sum(w)  (2022.10.11)
            # https://math.stackexchange.com/questions/1962877/compute-the-gradient-of-mean-square-error
            
            g_t1 = np.dot(x_batch.transpose(),loss) * (-2)
            g_t2 = np.dot(np.square(x_batch).transpose(),loss) * (-2)
            g_t_b = loss.sum(axis=0) * (-2)
            m_t1 = beta_1*m_t1 + (1-beta_1)*g_t1
            m_t2 = beta_1*m_t2 + (1-beta_1)*g_t2
            v_t1 = beta_2*v_t1 + (1-beta_2)*np.multiply(g_t1, g_t1)
            v_t2 = beta_2*v_t2 + (1-beta_2)*np.multiply(g_t2, g_t2)
            m_cap1 = m_t1/(1-(beta_1**t))
            m_cap2 = m_t2/(1-(beta_1**t))
            v_cap1 = v_t1/(1-(beta_2**t))
            v_cap2 = v_t2/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b)
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            
            # Update weight & bias
            w1 -= ((lr*m_cap1)/(np.sqrt(v_cap1)+epsilon)).reshape(-1, 1)
            w2 -= ((lr*m_cap2)/(np.sqrt(v_cap2)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)

    return w2, w1, bias

def parse2test(data, feats):
    x = []
    for i in range(90):
        x_tmp = data[8*i: 8*i+8, feats]
        x.append(x_tmp.reshape(-1,))

    # x.shape: (n, 15, 8)
    x = np.array(x)
    return x

def main():
    ## Edit: use np.random.seed(seed) (2022.10.12)
    seed = 9487
    np.random.seed(seed)

    # Training
    data = pd.read_csv("./train.csv")
    data = data.values
    train_data = np.array(np.float64(data))
    # TODO: Tune the config to boost your performance.
    train_config = Namespace(
    batch_size = 8,
    lr = 1e-4,
    epoch = 5,
    )
    feats = [2, 3, 6, 14]
    #feats = [2]
    #train_data = np.transpose(np.array(np.float64(data)))
    train_x, train_y = parse2train(train_data, feats)
    #print(train_x.shape)
    #print(train_y)
    w2, w1, bias = minibatch(train_x, train_y, train_config)
    print("w2: ")
    print(w2)
    print("w1: ")
    print(w1)
    print("bias: ")
    print(bias)

    # Testing
    data = pd.read_csv('./test.csv')
    data = data.values
    #test_data = np.transpose(np.array(np.float64(data)))
    test_data = np.array(np.float64(data))
    test_x = parse2test(test_data, feats)

    # Write output
    with open('my_sol.csv', 'w', newline='') as csvf:
    # 建立 CSV 檔寫入器
        writer = csv.writer(csvf)
        writer.writerow(['Id','Predicted'])
        #print(test_x.shape)
        for i in range(int(test_x.shape[0])):
        # Prediction of linear regression
            prediction = float(np.dot(np.square(test_x[i]), w2) + np.dot(test_x[i], w1) + bias)
            writer.writerow([i, prediction])

if __name__ == '__main__':
    main()
