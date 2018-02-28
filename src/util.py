

# import cPickle # for python2
import pickle  # for python3
import librosa
import torch
from torch.autograd import Variable


def save_data(filename, data):
    """Save variable into a pickle file

    Parameters
    ----------
    filename: str
        Path to file

    data: list or dict
        Data to be saved.

    Returns
    -------
    nothing

    """
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump(data, open(filename, 'w'))


def load_data(filename):
    """Load data from pickle file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    data: list or dict
        Loaded file.

    """

    return pickle.load(open(filename, "rb"), encoding='latin1')


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def dic_ESC10():
    return {0: 'Fire crackling', 1: 'Dog bark', 2: 'Rain', 3: 'Sea waves', 4: 'Baby cry', 5: 'Clock tick',
            6: 'Person sneeze', 7: 'Helicopter', 8: 'Chainsaw', 9: 'Rooster'}


def dic_ESC50():
    return {0: 'Hand saw', 1: 'Dog', 2: 'Rooster', 3: 'Pig', 4: 'Cow', 5: 'Frog', 6: 'Cat', 7: 'Hen',
            8: 'Insects (flying)', 9: 'Sheep', 10: 'Crow', 11: 'Rain', 12: 'Sea waves',
            13: 'Crackling fire', 14: 'Crickets', 15: 'Chirping birds', 16: 'Water drops',
            17: 'Wind', 18: 'Pouring water', 19: 'Toilet flush', 20: 'Thunderstorm',
            21: 'Crying baby', 22: 'Sneezing', 23: 'Clapping', 24: 'Breathing', 25: 'Coughing',
            26: 'Footsteps', 27: 'Laughing', 28: 'Brushing teeth', 29: 'Snoring',
            30: 'Drinking - sipping', 31: 'Door knock', 32: 'Mouse click', 33: 'Keyboard typing',
            34: 'Door - wood creaks', 35: 'Can opening', 36: 'Washing machine',
            37: 'Vacuum cleaner', 38: 'Clock alarm', 39: 'Clock tick', 40: 'Glass breaking',
            41: 'Helicopter', 42: 'Chainsaw', 43: 'Siren', 44: 'Car horn', 45: 'Engine',
            46: 'Train', 47: 'Church bells', 48: 'Airplane', 49: 'Fireworks'}


def num_to_id_ESC50(num):
    dic = {'510':0, '101':1, '102':2, '103':3, '104':4, '105':5, '106':6, '107':7, '108':8,
           '109':9, '110':10, '201':11, '202':12, '203':13, '204':14, '205':15, '206':16,
           '207':17, '208':18, '209':19, '210':20, '301':21, '302':22, '303':23, '304':24,
           '305':25, '306':26, '307':27, '308':28, '309':29, '310':30, '401':31, '402':32,
           '403':33, '404':34, '405':35, '406':36, '407':37, '408':38, '409':39, '410':40,
           '501':41, '502':42, '503':43, '504':44, '505':45, '506':46, '507':47, '508':48,
           '509':49}
    return dic[num]


def num_to_id_ESC10(num):
    dic = {'010':0, '001':1, '002':2, '003':3, '004':4, '005':5, '006':6, '007':7, '008':8, '009':9}
    return dic[num]


def id_to_lb(id, dataSet='ESC-50'):
    if dataSet == 'ESC-10':
        dic = dic_ESC10()
    elif dataSet == 'ESC-50':
        dic = dic_ESC50()
    else:
        raise ValueError
    return dic[id]


def lb_to_id(lb, dataSet='ESC-50'):
    if dataSet == 'ESC-10':
        dic = dic_ESC10()
    elif dataSet == 'ESC-50':
        dic = dic_ESC50()
    else:
        raise ValueError
    re_dic = {v: k for k, v in dic.items()}
    return re_dic[lb]
