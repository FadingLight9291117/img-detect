import logging
import os
import psutil
import winreg
from pathlib import Path
import re

import cv2
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
import pandas as pd

from exceptions import RoiException, ImgException
from utils import save2json, yaml2dict, json2dict, save2txt


def scaling_search(imgL, imgS, roi, sub_method, isGray, mae_thresh, enable_adapt, K, enable_scaling, scaling_rates):
    res = main_search(imgL, imgS, roi, sub_method, isGray,
                      mae_thresh, enable_adapt, K)
    if not enable_scaling:
        return res
    for imgS_s in gen_scaled_img(imgS, scaling_rates):
        res_ = main_search(imgL, imgS_s, roi, sub_method, isGray,
                           mae_thresh, enable_adapt, K)
        if res_['mae'] < res['mae']:
            res = res_
        if res['mae'] < mae_thresh:
            break
    return res


def main_search(imgL, imgS, roi, sub_method, isGray, mae_thresh, enable_adapt, K):
    if imgL is None or imgS is None:
        raise ImgException('大图或者小图为None')
    if imgL.size < imgS.size:
        raise ImgException('大图比小图小')

    x1, y1, x2, y2 = roi
    h_l, w_l = imgL.shape[:2]
    h_s, w_s = imgS.shape[:2]

    if roi is None:
        raise RoiException('roi为None')
    if len(roi) != 4:
        raise RoiException('roi元素个数不为4')
    if x1 >= x2 or y1 >= y2:
        raise RoiException('roi两坐标不合法，或左上角坐标大于右下角坐标，或roi面积为0')
    if x1 < 0 or y1 < 0 or x2 > w_l or y2 > h_l:
        raise RoiException('roi超出大图范围')
    if x2 - x1 < w_s or y2 - y1 < h_s:
        raise RoiException('roi比小图小')

    x1 = bound(x1, 0, w_l)
    y1 = bound(y1, 0, h_l)
    x2 = bound(x2, 0, w_l)
    y2 = bound(y2, 0, h_l)

    imgL_n = imgL[y1:y2, x1:x2]
    rate = adaptive_rate(imgS, enable_adapt, K)
    imgL_n = trans_img(imgL_n, rate, isGray)
    imgS_n = trans_img(imgS, rate, isGray)

    coeff = cv2.matchTemplate(imgL_n, imgS_n, sub_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(coeff)
    if sub_method == cv2.TM_SQDIFF or sub_method == cv2.TM_SQDIFF_NORMED:
        box_x1, box_y1 = min_loc
    else:
        box_x1, box_y1 = max_loc

    box_x1 = inv_trans(box_x1, rate)
    box_y1 = inv_trans(box_y1, rate)
    box_x1 += x1
    box_y1 += y1
    box_x2 = box_x1 + w_s
    box_y2 = box_y1 + h_s
    box_x1 = bound(box_x1, 0, w_l)
    box_x2 = bound(box_x2, 0, w_l)
    box_y1 = bound(box_y1, 0, h_l)
    box_y2 = bound(box_y2, 0, h_l)

    conf, mae = compute_conf_mae(
        imgL[box_y1: box_y2, box_x1: box_x2], imgS, mae_thresh)

    box = [box_x1, box_y1, box_x2, box_y2]

    res = {
        'box': box,
        'conf': float(f'{conf:.2f}'),
        'mae': float(f'{mae:.3f}'),
    }

    return res


def adaptive_rate(imgS, b_rate, K=50):
    rate = 1
    if b_rate == True:
        h, w = imgS.shape[:2]
        rate = np.min([h / K, w / K])
        rate = np.max([rate, 1])

    return rate


def trans_img(img, rate, isGray):
    if isGray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img /= 255

    if rate != 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (int(w // rate), int(h // rate)), cv2.INTER_LINEAR)
    return img


def inv_trans(a, rate):
    a = int(a * rate)
    return a


def compute_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


def compute_conf_mae(img1, img2, mae_thresh):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 != h2 or w1 != w2:
        conf = 0
        mae = 1
    else:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        img1 /= 255
        img2 /= 255
        mae = compute_mae(img1, img2)
        conf = 1 - mae / mae_thresh / 3
        # conf = 1 if mae < mae_thresh else 0
        conf = bound(conf, 0, 1)
    conf = float(conf)
    mae = float(mae)
    return conf, mae


def bound(x, a, b):
    x = a if x < a else x
    x = b if x > b else x
    return x


def box_dist(res_box, gt_box):
    dist = np.sqrt(
        np.power(res_box[0] - gt_box[0], 2) + np.power(res_box[1] - gt_box[1], 2))
    return dist


def gen_scaled_img(img, scaling_rates=None):
    if scaling_rates is None:
        yield img
    ranges = list(reversed(np.arange(*scaling_rates)))
    for rate in ranges:
        h, w = img.shape[:2]
        img_s = cv2.resize(img, (int(w * rate), int(h * rate)))
        yield img_s


def detect(test_type='P', img_N=100):
    import logging
    from utils import Timer

    config = yaml2dict('config/config.yaml')

    isGray = config['IS_GRAY']
    sub_method = config['SUB_METHOD']
    enable_adapt = config['ENABLE_ADAPT']
    K = config['K']
    mae_thresh = config['DIFF_THRESH']
    imgP_dir = Path(config['imgP_dir'])
    imgN_dir = Path(config['imgN_dir'])
    imgL_name = config['imgL_name']
    imgS_name = config['imgS_name']

    # 小图缩放相关
    enable_scaling = config['enable_scaling']
    if enable_scaling:
        scaling_rates = config['scaling_rates']
    else:
        scaling_rates = None

    if test_type == 'P':
        img_dir = imgP_dir
    else:
        img_dir = imgN_dir
    if not img_dir.exists():
        raise FileNotFoundError(f'{img_dir} 文件夹不存在。')

    # 预处理，获取图片路径
    imgL_paths_tmp = list(img_dir.glob(imgL_name))
    imgS_paths_tmp = list(img_dir.glob(imgS_name))
    # label_path = imgP_dir / 'label.json'
    imgL_paths = []
    imgS_paths = []
    for path in imgL_paths_tmp:
        split = imgL_name.split('*')[-1]
        filename = path.name
        suffix = filename.replace(split, '')
        imgS_filename = suffix + imgS_name.split('*')[-1]
        imgS_filenames = [i.name for i in imgS_paths_tmp]
        if imgS_filename in imgS_filenames:
            imgL_paths.append(path)
            imgS_paths.append(img_dir / imgS_filename)
        else:
            logging.warning(f'大图 {path} 不存在对应的小图 {img_dir / imgS_filename}')

    # 路径排序
    # imgL_paths.sort(key=lambda path: int(Path(path).stem))
    # imgS_paths.sort(key=lambda path: int(Path(path).stem[:-1]))

    timer = Timer()
    # boxes = json2dict(label_path)['boxes']

    if img_N != -1:
        imgL_paths = imgL_paths[:img_N]
        imgS_paths = imgS_paths[:img_N]

    all_result = []

    lp = enumerate(zip(imgL_paths, imgS_paths))
    lp = tqdm(lp, total=len(imgL_paths))
    lp.set_description(test_type)
    for i, (imgL_path, imgS_path) in lp:
        res = {}
        res['type'] = test_type
        res['id'] = i
        res['imgL_path'] = imgL_path.absolute().__str__()
        res['imgS_path'] = imgS_path.absolute().__str__()
        with timer:
            res_ = {}
            # 读图
            imgL = read_img(imgL_path.as_posix())
            imgS = read_img(imgS_path.as_posix())

            # 图片异常问题
            if imgL is None or imgS is None:
                img_path = imgL_path if imgL is None else imgS_path
                logging.warning(f'{img_path.absolute()} 图片读入出错')
                continue
            h_s, w_s = imgS.shape[:2]
            h_l, w_l = imgL.shape[:2]
            if h_s > h_l or w_s > w_l:
                logging.warning(
                    f'小图 {imgL_path.absolute()} 比大图 {imgS_path.absolute()} 大')
                continue

            res_ = scaling_search(
                imgL, imgS, [0, 0, w_l, h_l], sub_method=sub_method, isGray=isGray,
                mae_thresh=mae_thresh, enable_adapt=enable_adapt, K=K,
                enable_scaling=enable_scaling, scaling_rates=scaling_rates)

            # res['dist'] = int(box_dist(res['box'], boxes[i]))

        for k, v in res_.items():
            res[k] = v
        res['time'] = float(f'{timer.this_time() * 1000:.0f}')
        all_result.append(res)
        logging.debug(res)

    logging.debug(f'run time {timer.total_time():.2f}s')

    return all_result


def read_img(path):
    try:
        im = cv2.imread(path)
    except Exception as e:
        im = None
    finally:
        return im


def compute_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return float(f'{precision:.4f}'), float(f'{recall:.4f}')


def get_cpu_speed():
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                         r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    speed, type = winreg.QueryValueEx(key, "~MHz")
    speed = round(float(speed)/1024, 1)
    return speed


def get_memory():
    return float(f'{psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f}')


def main():
    config = yaml2dict('config/config.yaml')
    thresh = config['threshold']
    result_path = config['result_path']
    detail_path = config['DETAIL_PATH']
    Path(result_path).parent.mkdir(exist_ok=True)
    Path(detail_path).parent.mkdir(exist_ok=True)
    P_N = config['P_N']
    N_N = config['N_N']
    P_res = detect(test_type='P', img_N=P_N)
    N_res = detect(test_type='N', img_N=N_N)
    save2txt(P_res + N_res, detail_path)

    TP = sum([item['conf'] > thresh for item in P_res])
    FN = len(P_res) - TP
    TN = sum([item['conf'] < thresh for item in N_res])
    FP = len(N_res) - TN

    P_N = len(P_res)
    N_N = len(N_res)
    precision, recall = compute_metrics(TP, FP, TN, FN)
    average_time = np.mean([item['time'] for item in P_res + N_res])
    average_time = float(f'{average_time:.0f}')

    # 保存结果
    columns = ["cpu主频(GHz)", "内存(GB)", "实际正样本数", "实际负样本数", "测试正样本数", "测试负样本数",
               "精确度%", "召回率%", "平均时间(ms)"]
    data = [get_cpu_speed(), get_memory(), P_N, N_N, len(P_res), len(
        N_res), precision * 100, recall * 100, average_time]
    data = np.array([data])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(result_path)
    print(df)


def exe():
    # logging.basicConfig(level=logging.DEBUG)
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        os.system('pause')


if __name__ == '__main__':
    main()
