import pandas as pd

train = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\ml\\train_lgb20180729.txt', header=None, names=['ml'])
test = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\ml\\result_lgb20180729.txt', header=None, names=['ml'])

train_rnn = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\models\\RNN_dot\\1532881984train_rnn.txt', header=None,
                          names=['rnn'])
test_rnn = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\models\\RNN_dot\\1532617414result_rnn.txt', header=None,
                         names=['rnn'])

train_rnn_bn = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\models\\RNN_dot\\1532881905train_rnn_bn.txt', header=None,
                             names=['rnn_bn'])
test_rnn_bn = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\models\\RNN_dot\\1532660128result_rnn_bn.txt', header=None,
                            names=['rnn_bn'])

train_mvrnn_cos = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\mv_rnn\\1532881832train_mvrnn_cos.txt',
                                header=None,
                                names=['mvrnn_cos'])
test_mvrnn_cos = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\mv_rnn\\1532832093result_mvrnn_cos.txt',
                               header=None,
                               names=['mvrnn_cos'])

train_mvrnn_bilinear = pd.read_table(
    'I:\\temp\\CNNimpl_static\\CIKM\\models\\mv_rnn\\1532881741train_mvrnn_bilinear.txt',
    header=None,
    names=['mvrnn_bilinear'])
test_mvrnn_bilinear = pd.read_table(
    'I:\\temp\\CNNimpl_static\\CIKM\\models\\mv_rnn\\1532874447result_mvrnn_bilinear.txt',
    header=None,
    names=['mvrnn_bilinear'])

train_mvrnn_image = pd.read_table(
    'I:\\temp\\CNNimpl_static\\CIKM\\models\\mv_rnn\\1532911703train_mvrnn_images.txt',
    header=None,
    names=['mvrnn_bilinear'])
test_mvrnn_image = pd.read_table(
    'I:\\temp\\CNNimpl_static\\CIKM\\models\\mv_rnn\\1532911613result_mvrnn_image.txt',
    header=None,
    names=['mvrnn_bilinear'])

train_han = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\HAN\\1532879785train_HAN.txt', header=None,
                          names=['han'])
test_han = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\HAN\\1532879574result_HAN.txt', header=None,
                         names=['han'])

train_decom = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\decomposable\\1532879975train_decom.txt',
                            header=None,
                            names=['decom'])
test_decom = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\decomposable\\1532832307result_decom.txt',
                           header=None,
                           names=['decom'])

train_image = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\cnn_image\\1532880041train_cnn_image.txt',
                            header=None,
                            names=['image'])
test_image = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\cnn_image\\1532796433result_cnn_image.txt',
                           header=None,
                           names=['image'])

train_cnn1d = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\cnn_1d\\1532881338train_cnn1d.txt', header=None,
                            names=['cnn1d'])
test_cnn1d = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\cnn_1d\\1532790146result_cnn1d.txt', header=None,
                           names=['cnn1d'])
y_train = pd.read_table('I:\\temp\\CNNimpl_static\\CIKM\\models\\cnn_1d\\1532881338y_train_cnn1d.txt', header=None,
                        names=['y_train'])

train = pd.concat([train, train_rnn, train_rnn_bn, train_mvrnn_cos, train_mvrnn_bilinear, train_mvrnn_image,
                   train_han, train_decom, train_image, train_cnn1d], axis=1)

test = pd.concat([test, test_rnn, test_rnn_bn, test_mvrnn_cos, test_mvrnn_bilinear, test_mvrnn_image,
                  test_han, test_decom, test_image, test_cnn1d], axis=1)

import numpy as np

# params = {}
# params["objective"] = 'binary'
# params['metric'] = "binary_logloss"
# params["learning_rate"] = 0.01
# params["subsample"] = 0.8
# params["feature_fraction"] = 1.0
# params["max_depth"] = 8
# params["num_leaves"] = 256
# params["lambda_l1"] = 0.01
# params["lambda_l2"] = 0.01
# params["num_iterations"] = 1000
#
# print("start training")
#
# from sklearn.model_selection import train_test_split

y_train = np.squeeze(y_train.values, axis=1)
# print(y_train.shape)
#
# x_train, x_dev, y_train, y_dev = train_test_split(train.values, y_train, test_size=0.1, random_state=10)
# import lightgbm as lgb
#
# train_input = lgb.Dataset(x_train, y_train)
#
# val_input = lgb.Dataset(x_dev, y_dev)
#
# lgb_model = lgb.train(params=params, train_set=train_input, valid_sets=val_input)
#
# print("predict")
# pred = lgb_model.predict(data=test.values)

# np.savetxt("result_ensemble.txt", pred)
# np.savetxt("train_lgb20180729.txt", train_pred)
# print("number of result: %d" % len(pred))
# print("number of result: %d" % len(train_pred))

from sklearn.linear_model import LinearRegression
import time
import numpy as np

lr_model = LinearRegression()
lr_model.fit(train.values, y_train)

result = lr_model.predict(test.values)
timestamp = str(int(time.time()))
file_name = timestamp + "result_ensemble_lr.txt"
np.savetxt(file_name, result)
print("number of result: %d" % len(result))
# print("train shape:")
# print(train.shape)
#
# print("test shape:")
# print(test.shape)
#
# print("y_train shape:")
# print(y_train.shape)
