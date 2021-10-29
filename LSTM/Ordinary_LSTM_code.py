import numpy as np  # numpy库
import matplotlib.pyplot as plt 
import scipy.io as sio
from sklearn import preprocessing # 归一化
from sklearn.model_selection import train_test_split#划分训练集
import sklearn.metrics as sm #画混淆矩阵
import pandas as pd 
import keras
import joblib

#调整显卡资源分配
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:20:49 2019
@author: wkyy
"""
from math import sqrt, ceil
from matplotlib import pyplot
from pandas import read_excel
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,r2_score,mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load data set
def _load_dataset(dataset, index_col_name, col_to_predict, cols_to_drop):
    
    """
    file_path: the csv file path
    header_row_index: the header row（行） index in the csv file
    index_col_name: the index column（列) (can be None if no index is there)
    col_to_predict: the column name/index to predict
    cols_to_drop: the column names/indices to drop (single label or list-like)
    """
 
    # set index col，设置索引列，参数输入列的名字列表
    if index_col_name:
        dataset.set_index(index_col_name, inplace=True)
    
    # drop nonused colums，删除不需要的列，参数输入列的名字列表
    '''if cols_to_drop:
        if type(cols_to_drop[0]) == int:
            dataset.drop(index=cols_to_drop, axis=0, inplace=True)
        else:
            dataset.drop(columns=cols_to_drop, axis=1, inplace=True)'''
    if cols_to_drop:
        dataset.drop(cols_to_drop, axis =1, inplace = True)
    
    #print('\nprint data set again\n',dataset)
    # get rows and column names
    col_names = dataset.columns.values.tolist() #获取列名
    values = dataset.values #获取数据值，将数据转换为np.array格式
    #print(col_names, '\n values\n', values)
    
    # move the column to predict to be the first col: 把预测列调至第一列
    col_to_predict_index = col_to_predict if type(col_to_predict) == int else col_names.index(col_to_predict)
    output_col_name = col_names[col_to_predict_index]  #用于输出的列名称
    
    #下面的这个if语句，实现把预测列的索引调至第一列
    if col_to_predict_index > 0:
        col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[col_to_predict_index+1:]
        
    #下面这个np.concatenae实现，将预测列的数值调至第一列    
    values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)), values[:,:col_to_predict_index], values[:,col_to_predict_index+1:]), axis=1)
    #print(col_names, '\n values2\n', values)
    # ensure all data is float
    values = values.astype("float32")
    #print(col_names, '\n values3\n', values)
    return col_names, values,values.shape[1], output_col_name


# scale dataset
#def _scale_dataset(values, scale_range = (0,1)):
def _scale_dataset(values, scale_range):
    """
    归一化，可以缩短模型计算时间
    """
    # normalize features
    scaler = MinMaxScaler(feature_range=scale_range or (0, 1))    #实例话 
    scaler_model = scaler.fit(values)   #训练模型
    scaled = scaler.fit_transform(values)  #将特征值进行归一化处理。 
    return (scaler, scaled, scaler_model)

# convert series to supervised learning (ex: var1(t)_row1 = var1(t-1)_row2)，列表打印出来一看就明白了
#def _series_to_supervised(values, n_in=3, n_out=1, dropnan=True, col_names, verbose=True):
def _series_to_supervised(values, n_in, n_out, dropnan, col_names, verbose):
    """
    values: dataset scaled values
    n_in: number of time lags (intervals) to use in each neuron, 与多少个之前的time_step相关,和后面的n_intervals是一样
    n_out: number of time-steps in future to predict，预测未来多少个time_step
    dropnan: whether to drop rows with NaN values after conversion to supervised learning
    col_names: name of columns for dataset
    verbose: whether to output some debug data
    """
 
    n_vars = 1 if type(values) is list else values.shape[1] #返回数据的列数
    if col_names is None: col_names = ["var%d" % (j+1) for j in range(n_vars)] #如果输入的数据中没有指定列名，这里会有代码给定
    df = DataFrame(values)
    cols, names = list(), list()
 
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):#表示从n_in开始，到0为止，步长为-1
        cols.append(df.shift(i))
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))         #这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]
 
    # put it all together
    agg = concat(cols, axis=1)    #将cols中的每一行元素一字排开，连接起来，vala t-n_in, valb t-n_in ... valta t, valb t... vala t+n_out-1, valb t+n_out-1
    agg.columns = names
 
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)#将数据格式转换为监督学习的格式，时间序列转换为python中的监督学习
 
    if verbose:
        print("\nsupervised data shape:", agg.shape)
    return agg

# split into train and test sets
#def _split_data_to_train_test_sets(values, n_intervals=3, n_features, train_percentage=0.67, verbose=True):
def _split_data_to_train_test_sets(values, n_intervals, n_features, train_percentage, val_percentage,verbose):
    """
    values: dataset supervised values
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    train_percentage: percentage of train data related to the dataset series size; (1-train_percentage) will be for test data
    val_percentage:percentage of val data related to the dataset series size; (1-train_percentage-val_percentage) will be for test data
    verbose: whether to output some debug data
    """
 
    n_train_intervals = ceil(values.shape[0] * train_percentage) #ceil(x)->得到最接近的一个不小于x的整数，如ceil(2.001)=3
    n_val_intervals = ceil(values.shape[0] * val_percentage) 
    n_all=n_train_intervals+n_val_intervals
    
    train = values[-n_train_intervals:, :]
    val=values[-n_all:-n_train_intervals, :]
    test = values[:-n_all, :]
    #test = values[150227:190691, :]

 
    # split into input and outputs
    n_obs = n_intervals * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]  #train_Y直接赋值倒数第六列，刚好是t + n_out_timestep-1时刻的0号要预测列
                                                                #train_X此时的shape为[train.shape[0], timesteps * features]
    val_X, val_y = val[:, :n_obs], val[:, -n_features]                                                            
    #print('before reshape\ntrain_X shape:', train_X.shape)
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
 
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_intervals, n_features))
    val_X = val_X.reshape((val_X.shape[0], n_intervals, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_intervals, n_features))
 
    if verbose:
        print("")
        print("train_X shape:", train_X.shape)
        print("train_y shape:", train_y.shape)
        print("val_X shape:", val_X.shape)
        print("val_y shape:", val_y.shape)
        print("test_X shape:", test_X.shape)
        print("test_y shape:", test_y.shape)
 
    return (train_X, train_y,val_X,val_y, test_X, test_y)

# create the nn model
#def _create_model(train_X, train_y, test_X, test_y, n_neurons=20, n_batch=50, n_epochs=60, is_stateful=False, has_memory_stack=False, loss_function='mse', optimizer_function='adam', draw_loss_plot=True, output_col_name, verbose=True):
def _create_model(train_X, train_y, test_X, test_y, n_neurons, n_batch, n_epochs, is_stateful, has_memory_stack, loss_function, optimizer_function, draw_loss_plot, output_col_name, verbose):
    """
    n_neurons: LSTM神经元的个数
    n_batch: 指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
    n_epochs: 训练的次数
    is_stateful: 模型是否有内存状态
    has_memory_stack: 模型是否有内存堆栈
    loss_function: 模型损失函数评估器
    optimizer_function: 损失优化函数
    draw_loss_plot: 是否绘制损失历史图
    output_col_name: 要预测的输出/目标列的名称
    verbose: 是否输出部分调试数据
    """
 
    # design network
    model = Sequential()
    
    if is_stateful:
        # calculate new compatible batch size
        for i in range(n_batch, 0, -1):
            if train_X.shape[0] % i == 0 and test_X.shape[0] % i == 0:
                if verbose and i != n_batch:
                    print ("\n*In stateful network, batch size should be dividable by training and test sets; had to decrease it to %d." % i)
                n_batch = i
                break
 
        model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True, return_sequences=has_memory_stack))
        if has_memory_stack:
            model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True))
    else:
        model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1))
 
    model.compile(loss=loss_function, optimizer=optimizer_function)
 
    if verbose:
        print("--------开始训练----------")
 
    # fit network
    losses = []
    val_losses = []
    if is_stateful:
        for i in range(n_epochs):
            history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch, 
                                validation_data=(test_X, test_y), verbose=0, shuffle=False)#verbose=0，在控制台没有任何输出 
            if verbose:
                print("Epoch %d/%d" % (i + 1, n_epochs))
                print("loss: %f - val_loss: %f" % (history.history["loss"][0], history.history["val_loss"][0]))
 
            losses.append(history.history["loss"][0])
            val_losses.append(history.history["val_loss"][0])
 
            model.reset_states()
    else:
        history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch, 
                            validation_data=(test_X, test_y), verbose=2 if verbose else 0, shuffle=False)
    
    
    if draw_loss_plot:
        pyplot.figure()
        pyplot.plot(history.history["loss"] if not is_stateful else losses, label="Train Loss (%s)" % output_col_name)
        pyplot.plot(history.history["val_loss"] if not is_stateful else val_losses, label="Val Loss (%s)" % output_col_name)
        pyplot.legend()
        pyplot.show()
    
    print(history.history)
    
    return (model, n_batch)

# make a prediction
#def _make_prediction(model, train_X, train_y, test_X, test_y, compatible_n_batch, n_intervals=3, n_features, scaler=(0,1), draw_prediction_fit_plot=True, output_col_name, verbose=True):
def _make_prediction(model, test_X, test_y,
                     compatible_n_batch, n_intervals, n_features, 
                     scaler, draw_prediction_fit_plot, output_col_name, 
                     verbose):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    compatible_n_batch: modified (compatible) nn batch size
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    scaler: the scaler object used to invert transformation to real scale
    draw_prediction_fit_plot: whether to draw the the predicted vs actual fit plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
    """
 
    if verbose:
        print("")
 
    yhat = model.predict(test_X, batch_size=compatible_n_batch, verbose = 1 if verbose else 0)
    test_X = test_X.reshape((test_X.shape[0], n_intervals*n_features)) #从(31189, 10, 2)，变为(31189, 10)
    #print(yhat)
    
    #反归一化
    # invert scaling for 预测值
    inv_yhat = np.concatenate((yhat, test_X[:, (1-n_features):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0] #提取归一化处理后的预测值y
 
    # invert scaling for 真实值
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, (1-n_features):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
 
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat)) #mean_squared_error()计算MSE
    mape = mean_absolute_percentage_error(inv_y,inv_yhat) #计算MAPE
    mse = mean_squared_error(inv_y,inv_yhat)
    r2 = r2_score(inv_y,inv_yhat)
 
    # calculate average error percentage
    avg = np.average(inv_y)
    error_percentage = rmse / avg
 
    if verbose:
        print("回归模型的评价指标")
        print("Test RMSE: %.4f" % rmse)
        print("Test MAPE: %.4f" % mape)
        print("Test MSE : %.4f" % mse)
        print("Test R2  : %.4f" % r2)
#         print("Test Average Value for %s: %.3f" % (output_col_name, avg))
        print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))
        
 # 红色为预测速度
 # 蓝色为真实速度
    if draw_prediction_fit_plot:
        pyplot.figure()
        pyplot.plot(inv_y, "g--", label="Actual (%s)" % output_col_name)
        pyplot.plot(inv_yhat,"r--", label="Predicted (%s)" % output_col_name)
        pyplot.legend()
        pyplot.ylim(min(inv_y)-2, max(inv_y)+2)
        pyplot.show()
 
    return (inv_y, inv_yhat, rmse, error_percentage)


def Examples_training_evaluation(num_people,freq,columns_to_predict,using_columns_name
                                 ,train_size, val_size,input_timestep,output_timestep):
    
    """显卡内存自增长"""
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    
    import numpy as np
    # random seeds must be set before importing keras & tensorflow
    my_seed = 512
    np.random.seed(my_seed)
    import random 
    random.seed(my_seed)
    import tensorflow as tf
    tf.random.set_seed(my_seed)

    """模型训练主函数"""      
    #输入数据，并将数据转换为np.array格式
    file_path= r''.format(d=freq)
    header_row_index = 0 #行索引号
    index_col_name = None #列名称索引

    col_to_predict = columns_to_predict #预测值（标签）
    cols_to_drop = None
    dataset = read_excel(file_path, header=header_row_index, index_col=False)
    if num_people != "all":
        dataset = dataset[dataset["Number of people"].isin([num_people])]

    dataset = dataset.loc[:,using_columns_name]  #用于训练的特征矩阵
    col_names, values,n_features, output_col_name = _load_dataset(dataset, 
                                                                  index_col_name, col_to_predict, cols_to_drop)

    #将特征进行归一化处理
    scaler, values, scaler_model = _scale_dataset(values, None)
    print('\nvalue shape:', values.shape)

    #数据准备
    n_in_timestep = input_timestep     #数据切片时间步
    n_out_timestep = output_timestep     #数据预测时间步
    verbose = 1             #是否选择显示训练过程 0-不显示 1-显示
    dropnan = True
    
    #按照时间序列以及选择的补偿，进行数据的划分。
    agg1 = _series_to_supervised(values, 
                                 n_in_timestep, 
                                 n_out_timestep, 
                                 dropnan, 
                                 col_names, #数据的列名
                                 verbose)
    #训练集和测试集划分
    train_percentage = train_size
    val_percentage = val_size
    train_X, train_Y, val_X, val_Y, test_X, test_Y =_split_data_to_train_test_sets(agg1.values,   #数据集
                                                                                   n_in_timestep, #时间补偿
                                                                                   n_features, #特征维度
                                                                                   train_percentage, #训练集百分比
                                                                                   val_percentage,   #验证集百分比
                                                                                   verbose)
    n=len(train_Y)

    #模型创建
    n_neurons=100    #神经元尺寸
    n_batch=50     #batch size
    n_epochs=50     #循环次数
    is_stateful=False
    has_memory_stack=False
    loss_function='mse'
    optimizer_function='adam'
    draw_loss_plot=True
    model, compatible_n_batch = _create_model(train_X,train_Y, 
                                              val_X, val_Y, 
                                              n_neurons, n_batch, 
                                              n_epochs, is_stateful, 
                                              has_memory_stack, loss_function, 
                                              optimizer_function, draw_loss_plot,
                                              output_col_name, verbose)
    #%%------------------------------------------------------
#     #是否保存模型
#     model.save('speed_prediction_final_test_105_5s3.model')
#     import joblib
#     joblib.dump(scaler_model, "scaler_final_test_105_5s3.model")

#     import scipy.io as scio
#     data_save = 'data_final_test_105_5s.mat'

#     scio.savemat(data_save, 
#                  {'test_X':test_X,
#                   'test_Y':test_Y,
#                   'n_in_timestep':n_in_timestep,
#                   'n_out_timestep':n_out_timestep,
#                   'compatible_n_batch':compatible_n_batch,
#                   'output_col_name':output_col_name,
#                   'n_features':n_features})

    #%%-------------------------------------------------------

    #测试集预测，并绘图
    draw_prediction_fit_plot = True
    actual_target_test, predicted_target_test,\
    error_value_test, error_percentage_test\
    = _make_prediction(model,  
                       test_X, test_Y, compatible_n_batch, 
                       n_in_timestep, n_features, scaler, 
                       draw_prediction_fit_plot, 
                       output_col_name, verbose)
    
if __name__ == "__main__":
    num_people = 1
    freq = 20
    columns_to_predict = "v_resample"
    using_columns_name= ["v_resample"] 
    train_size = 0.7
    val_size = 0.15
    input_timestep = 10
    output_timestep = 1
    Examples_training_evaluation(num_people,freq,columns_to_predict,using_columns_name
                                 ,train_size, val_size,input_timestep,output_timestep)
