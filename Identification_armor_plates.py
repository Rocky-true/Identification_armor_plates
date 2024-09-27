import numpy as np
import argparse
import cv2

class Armor:
    def __init__(self, left_pole, right_pole):
        self.left_pole = left_pole
        self.right_pole = right_pole
        self.score = None

    def __repr__(self):
        return f"Armor(left_pole={self.left_pole}, right_pole={self.right_pole},  score={self.score})"
    def vertices(self):
        x1 = self.left_pole[0][0]
        x2 = self.right_pole[0][0]
        y1 = self.left_pole[0][1] + self.left_pole[1][1]
        y2 = self.right_pole[0][1] + self.left_pole[1][1]
        x3 = self.left_pole[0][0]
        x4 = self.right_pole[0][0]
        y3 = self.left_pole[0][1] - self.left_pole[1][1]
        y4 = self.right_pole[0][1] - self.left_pole[1][1]
        # print(self.left_pole[1][0], "左边光柱宽度")
        # print(self.left_pole[1][1],"左边光柱高度")
        # print(self.right_pole[1][0], "右边光柱宽度")
        # print(self.right_pole[1][1], "右边光柱高度")
        # print(x2-x1, "装甲板宽度")
        # print(y2, "装甲板高度")

        return [(x1, y1), (x2, y2), (x4, y4), (x3, y3)]

def separate_colors(roi_img, enemy_color):
    # 将3通道图像转换成HSV色彩空间
    hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # 定义颜色的HSV范围
    if enemy_color == 'RED':
        lower_red1 = np.array([0, 20, 80])
        upper_red1 = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        lower_red2 = np.array([156, 20, 80])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        # 将两个掩膜进行逻辑或操作
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower_blue = np.array([70, 50, 50])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # 将掩码应用到原图像上，只保留特定颜色的区域
    gray_img = cv2.bitwise_and(roi_img, roi_img, mask=mask)


    # 将结果转换为灰度图
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    return gray_img
# 阈值化与膨胀
def threshold_and_dilate(gray_img, brightness_threshold):

    _, bin_bright_img = cv2.threshold(gray_img, brightness_threshold, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_bright_img = cv2.dilate(bin_bright_img, element)
    bin_bright_img = cv2.erode(bin_bright_img, element)




    return bin_bright_img

# 灯条轮廓提取
def find_contours(bin_bright_img):
    contours, _ = cv2.findContours(bin_bright_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def adjust_rec(rec):
    center, size, angle = rec
    if angle < -45:
        angle += 90
        size = (size[1], size[0])  # 交换宽高
    elif angle > 45:
        angle -= 90
        size = (size[1], size[0])  # 交换宽高
    return (center, size, angle)

# 灯柱筛选
def filter_contours(contours, params):
    light_infos = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params['light_min_area']:
            continue
        light_rec = cv2.fitEllipse(contour)  # 获取拟合椭圆的参数
        if light_rec[1][0] == 0 or light_rec[1][1] == 0:
            # 如果椭圆的宽度或高度为0，则跳过
            continue
        if light_rec[1][1] / light_rec[1][0] < params['light_min_ratio']:
            continue

        light_infos.append(light_rec)

    return light_infos

# 灯珠匹配
def match_armor(light_infos, params):
    armors = []
    light_infos.sort( key=lambda x: x[0][0])

    for i in range(len(light_infos) - 1):
        for j in range(i + 1, len(light_infos)):
            left_light, right_light = light_infos[i], light_infos[j]
            if is_valid_match(left_light, right_light, params):
                armor = create_armor(left_light, right_light)
                armors.append(armor)
                # 匹配成功后，跳过当前灯柱，避免重复匹配
                break
    return armors

def create_armor(pole1, pole2):
    # 创建装甲板对象
    return Armor(pole1, pole2)
def is_valid_match(pole1, pole2, params):
    # 计算角度差异
    angle_diff = abs(pole1[2] - pole2[2])
    # print(angle_diff)
    # 计算长度比率
    len_ratio = abs(pole1[1][1] - pole2[1][1]) / (pole1[1][1] + pole2[1][1]) / 2
    # print(len_ratio,"长度比率")
    # 计算y方向差异比率
    y_diff_ratio = abs(pole1[0][1] - pole2[0][1]) / (pole1[1][1] + pole2[1][1]) / 2
    # print(y_diff_ratio, "y差异比率")
    # 计算x方向差异比率
    x_diff_ratio = abs(pole1[0][0] - pole2[0][0]) / (pole1[1][1] + pole2[1][1]) / 2
    # print(x_diff_ratio, "x差异比率")
    # 检查是否满足所有条件
    if (y_diff_ratio > params['light_max_y_diff_ratio'] or
        x_diff_ratio < params['light_min_x_diff_ratio'] or
        len_ratio > params['light_armor_max_ratio'] or
        len_ratio < params['light_armor_min_ratio'] or
        angle_diff > params['light_max_angle_diff']):
        return False
    return True



# 定义cv_show
def cv_show(name, img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def load_video():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--video", required=True, help="path to input video")
    args = vars(ap.parse_args())
    cap = cv2.VideoCapture(args["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def main( enemy_color, params):
    # 载入图像
    cap, fps = load_video()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
    out = cv2.VideoWriter('output_1_1.mp4', fourcc, fps, (1600, 1200))  # 创建 VideoWriter 对象
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 假设ROI区域是整个图像
        roi_img = frame

        # 1. 颜色提取
        gray_img = separate_colors(roi_img, enemy_color)

        # 2. 阈值化与膨胀
        bin_bright_img = threshold_and_dilate(gray_img, params['brightness_threshold'])
        bin_bright_img = cv2.bilateralFilter(bin_bright_img, 9, 75, 75)  # 双边滤波
        # 3. 灯条轮廓提取
        contours = find_contours(bin_bright_img)
        # cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)

        # 4 灯柱筛选
        light_recs = filter_contours(contours, params)
        for light_rec in light_recs:
            box = cv2.boxPoints(light_rec)
            box = np.intp(box)  # 将坐标转换为整数
            cv2.polylines(frame, [box], True, (0, 255, 0), 1)  # 使用绿色线条绘制矩形

        # 5 打印装甲板
        armors = match_armor(light_recs, params)
        # print(armors)
        for armor in armors:
            # 遍历装甲板列表。
            points = np.array([[vertex[0], vertex[1]] for vertex in armor.vertices()], dtype=np.int32)
            # 提取每个装甲板的顶点，并将其转换为 NumPy 数组，用于绘制多边形。
            x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
            # 计算长宽比
            aspect_ratio = h / float(w)
            if aspect_ratio >= 0.8:
                # area = cv2.contourArea(points)
                # print(area, "面积")
                cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1,
                              lineType=8, shift=0)
            # 使用 cv2.polylines 函数在图像上绘制多边形。参数 isClosed=True 表示多边形是封闭的，
            # cv_show('roi_img', roi_img)

            out.write(frame)  # 写入处理后的帧到输出视频
    out.release()  # 释放 VideoWriter 对象



if __name__ == '__main__':
    enemy_color = 'RED'
    params = {
        'brightness_threshold': 100,
        'light_min_area': 100,
        'light_min_ratio': 0.6,
        'light_max_angle_diff': 5.0,
        'light_max_y_diff_ratio': 0.2,
        'light_min_x_diff_ratio': 0.25,
        'light_armor_max_ratio': 0.1,
        'light_armor_min_ratio': 0,

    }
    main(enemy_color, params)