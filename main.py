import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    ## placeholder
    # os.system("python3 /home/deamoon_uw_nn/uw-nn-adam/nn_pull_param.py")
    os.system("python3 /home/deamoon_uw_nn/uw-nn-adam/nn_train.py")
    os.system("python3 /home/deamoon_uw_nn/uw-nn-adam/nn_optimizer.py")
    os.system("python3 /home/deamoon_uw_nn/uw-nn-adam/nn_predict.py")
