import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description="Camera Intrinsic Calibration")
parser.add_argument('-input', '--INPUT_TYPE', default='camera', type=str, help='Input Source: camera/video/image')
parser.add_argument('-type', '--CAMERA_TYPE', default='fisheye', type=str, help='Camera Type: fisheye/normal')
parser.add_argument('-id', '--CAMERA_ID', default=1, type=int, help='Camera ID')
parser.add_argument('-path', '--INPUT_PATH',
                    default='E:\工作\潍柴第一台车内参畸变参数以及相应的单目录像\潍柴第一台车内参畸变参数以及相应的单目录像\WeiChai_FrontRightCamera/',
                    type=str, help='Input Video/Image Path')
parser.add_argument('-video', '--VIDEO_FILE', default='WeiChai_FrontRightCamera.mp4', type=str,
                    help='Input Video File Name (eg.: video.mp4)')
parser.add_argument('-image', '--IMAGE_FILE', default='img_raw', type=str,
                    help='Input Image File Name Prefix (eg.: img_raw)')
parser.add_argument('-mode', '--SELECT_MODE', default='auto', type=str, help='Image Select Mode: auto/manual')
parser.add_argument('-fw', '--FRAME_WIDTH', default=1280, type=int, help='Camera Frame Width')
parser.add_argument('-fh', '--FRAME_HEIGHT', default=1024, type=int, help='Camera Frame Height')
parser.add_argument('-bw', '--BORAD_WIDTH', default=7, type=int, help='Chess Board Width (corners number)')
parser.add_argument('-bh', '--BORAD_HEIGHT', default=6, type=int, help='Chess Board Height (corners number)')
parser.add_argument('-size', '--SQUARE_SIZE', default=10, type=int, help='Chess Board Square Size (mm)')
parser.add_argument('-num', '--CALIB_NUMBER', default=5, type=int, help='Least Required Calibration Frame Number')
parser.add_argument('-delay', '--FRAME_DELAY', default=12, type=int, help='Capture Image Time Interval (frame number)')
parser.add_argument('-subpix', '--SUBPIX_REGION', default=5, type=int, help='Corners Subpix Optimization Region')
parser.add_argument('-fps', '--CAMERA_FPS', default=20, type=int, help='Camera Frame per Second(FPS)')
parser.add_argument('-fs', '--FOCAL_SCALE', default=0.5, type=float, help='Camera Undistort Focal Scale')
parser.add_argument('-ss', '--SIZE_SCALE', default=1, type=float, help='Camera Undistort Size Scale')
parser.add_argument('-store', '--STORE_FLAG', default=False, type=bool, help='Store Captured Images (Ture/False)')
parser.add_argument('-store_path', '--STORE_PATH', default='./data/', type=str, help='Path to Store Captured Images')
parser.add_argument('-crop', '--CROP_FLAG', default=False, type=bool,
                    help='Crop Input Video/Image to (fw,fh) (Ture/False)')
parser.add_argument('-resize', '--RESIZE_FLAG', default=False, type=bool,
                    help='Resize Input Video/Image to (fw,fh) (Ture/False)')
args = parser.parse_args()  # 这段代码用于解析用户在命令行中输入的参数，并将它们保存在 args 变量中，以便在后续代码中使用。


class CalibData:  # CalibData 类定义了一个用于存储相机标定数据的数据结构
    def __init__(self):  # 提供了初始化的方法 __init__() 来为数据成员设置初始值。
        self.type = None  # 成员变量 type，它用于存储相机标定数据的类型。
        self.camera_mat = None  # 存储相机的内参矩阵
        self.dist_coeff = None  # 存储相机的畸变系数。
        self.rvecs = None  # 用于存储相机标定的旋转向量。
        self.tvecs = None  # 用于存储相机标定的平移向量。
        self.map1 = None  # 用于存储相机标定的畸变校正映射参数之一
        self.map2 = None  # 用于存储相机标定的畸变校正映射参数之一。
        self.reproj_err = None  # 用于存储相机标定的重投影误差。
        self.ok = False  # 用于表示相机标定是否成功的标志。初始化为 False，如果相机标定成功，该标志可能会在后续操作中更新为 True。
        self.distance = None
        self.board_centers = None
        self.sorted_frames = None
        self.fw = 1280
        self.fh = 720

class Fisheye:  # 定义了一个名为 Fisheye 的类，用于执行鱼眼相机的标定和畸变校正
    def __init__(self):
        self.data = CalibData()  # 相机标定数据
        self.inited = False  # 布尔标志，表示相机标定是否已完成初始化
        self.BOARD = np.array([[(j * args.SQUARE_SIZE, i * args.SQUARE_SIZE, 0.)]
                               for i in range(args.BORAD_HEIGHT)
                               for j in range(args.BORAD_WIDTH)],
                              dtype=np.float32)  # 创建了一个包含标定板格点坐标的 NumPy 数组，用于标定时的角点计算。

    def update(self, corners, frame_size):  # 更新相机标定数据。corners：在图像中检测到的标定板角点坐标的列表，frame_size 是帧的尺寸。
        board = [self.BOARD] * len(corners)  # self.BOARD标定板格点的坐标 board就是检测到多少个角点就复制几份
        if not self.inited:
            self._update_init(board, corners, frame_size)
            self.inited = True
        else:
            self._update_refine(board, corners, frame_size)
        self._calc_reproj_err(corners)
        self._get_undistort_maps()  # 计算两个映射参数：map1 和 map2，x 和 y 方向上的映射

    # 调用了 _update_init 或 _update_refine 方法进行相机标定的初始化或精化，并计算重投影误差并计算两个映射参数。
    def _update_init(self, board, corners, frame_size):
        data = self.data
        data.type = "FISHEYE"
        data.camera_mat = np.eye(3, 3)  # 赋值内参
        data.dist_coeff = np.zeros((4, 1))  # 径向畸变
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.fisheye.calibrate(
            board, corners, frame_size, data.camera_mat, data.dist_coeff,
            flags=cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            # CALIB_FIX_SKEW 表示固定鱼眼相机的切向畸变参数（skew），CALIB_RECOMPUTE_EXTRINSIC 表示在标定过程中重新计算相机的外参（旋转向量和平移向量）
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6))  # 鱼眼相机的初始化标定，
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)

    # criteria是标定过程的终止条件。cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT 表示迭代次数和误差的终止条件，30 表示最大迭代次数为 30，1e-6 表示迭代的误差阈值为 1e-6。
    # ok检查内参矩阵和畸变系数矩阵是否合理
    # 用于进行相机标定的初始化。在这个方法中，它调用 cv2.fisheye.calibrate() 函数来进行相机标定
    def _update_refine(self, board, corners, frame_size):
        data = self.data
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.fisheye.calibrate(
            board, corners, frame_size, data.camera_mat, data.dist_coeff,
            flags=cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 1e-6))
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)
        # 相机标定的精化。在这个方法中，调用 cv2.fisheye.calibrate() 函数来进行相机标定的精化，并将标定的结果保存到data 中。

    def _calc_reproj_err(self, corners):
        if not self.inited: return
        data = self.data
        data.reproj_err = []
        for i in range(len(corners)):
            corners_reproj, _ = cv2.fisheye.projectPoints(self.BOARD, data.rvecs[i], data.tvecs[i], data.camera_mat,
                                                          data.dist_coeff)  # 重投影坐标
            err = cv2.norm(corners_reproj, corners[i], cv2.NORM_L2) / len(corners_reproj)  # 重投影点的平均误差
            data.reproj_err.append(err)

    # 这是一个私有方法用于计算重投影误差。在这个方法中，它使用相机标定的结果来计算标定板上角点的重投影误差，并将结果保存在data 的 reproj_err 属性中。
    def _get_camera_mat_dst(self, camera_mat):
        camera_mat_dst = camera_mat.copy()
        camera_mat_dst[0][0] *= args.FOCAL_SCALE  # 焦距fx乘缩放因子，用于调整相机的焦距
        camera_mat_dst[1][1] *= args.FOCAL_SCALE  # 焦距fy
        camera_mat_dst[0][2] = args.FRAME_WIDTH / 2 * args.SIZE_SCALE  # 主点cx设置为图像的水平中心点的位置，乘以缩放因子
        camera_mat_dst[1][2] = args.FRAME_HEIGHT / 2 * args.SIZE_SCALE
        return camera_mat_dst

    # 根据输入的相机内参矩阵 camera_mat、给定的缩放和焦距比例，校正后的相机内参矩阵 camera_mat_dst。
    def _get_undistort_maps(self):
        data = self.data
        camera_mat_dst = self._get_camera_mat_dst(data.camera_mat)
        data.map1, data.map2 = cv2.fisheye.initUndistortRectifyMap(
            data.camera_mat, data.dist_coeff, np.eye(3, 3), camera_mat_dst,
            (int(args.FRAME_WIDTH * args.SIZE_SCALE), int(args.FRAME_HEIGHT * args.SIZE_SCALE)), cv2.CV_16SC2)
    # 计算畸变校正的映射参数 map1 和 map2，存在data里。


class Normal:
    def __init__(self):
        self.data = CalibData()
        self.inited = False
        self.BOARD = np.array([[(j * args.SQUARE_SIZE, i * args.SQUARE_SIZE, 0.)]
                               for i in range(args.BORAD_HEIGHT)
                               for j in range(args.BORAD_WIDTH)], dtype=np.float32)

    def update(self, corners, frame_size):
        board = [self.BOARD] * len(corners)
        if not self.inited:
            self._update_init(board, corners, frame_size)
            self.inited = True
        else:
            self._update_refine(board, corners, frame_size)
        self._calc_reproj_err(corners)
        self._get_undistort_maps()

    def _update_init(self, board, corners, frame_size):
        data = self.data
        data.type = "NORMAL"
        data.camera_mat = np.eye(3, 3)
        data.dist_coeff = np.zeros((5, 1))
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.calibrateCamera(
            board, corners, frame_size, data.camera_mat, data.dist_coeff,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6))
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)

    def _update_refine(self, board, corners, frame_size):
        data = self.data
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.calibrateCamera(
            board, corners, frame_size, data.camera_mat, data.dist_coeff,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 1e-6))
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)

    def _calc_reproj_err(self, corners):
        if not self.inited: return
        data = self.data
        data.reproj_err = []
        for i in range(len(corners)):
            corners_reproj, _ = cv2.projectPoints(self.BOARD, data.rvecs[i], data.tvecs[i], data.camera_mat,
                                                  data.dist_coeff)
            err = cv2.norm(corners_reproj, corners[i], cv2.NORM_L2) / len(corners_reproj)
            data.reproj_err.append(err)

    def _get_camera_mat_dst(self, camera_mat):
        camera_mat_dst = camera_mat.copy()
        camera_mat_dst[0][0] *= args.FOCAL_SCALE
        camera_mat_dst[1][1] *= args.FOCAL_SCALE
        camera_mat_dst[0][2] = args.FRAME_WIDTH / 2 * args.SIZE_SCALE
        camera_mat_dst[1][2] = args.FRAME_HEIGHT / 2 * args.SIZE_SCALE
        return camera_mat_dst

    def _get_undistort_maps(self):
        data = self.data
        camera_mat_dst = self._get_camera_mat_dst(data.camera_mat)
        data.map1, data.map2 = cv2.initUndistortRectifyMap(
            data.camera_mat, data.dist_coeff, np.eye(3, 3), camera_mat_dst,
            (int(args.FRAME_WIDTH * args.SIZE_SCALE), int(args.FRAME_HEIGHT * args.SIZE_SCALE)), cv2.CV_16SC2)
    # normal模式和鱼眼就只有径向畸变的格式不同，和调用的cv库的不同


class InCalibrator:
    def __init__(self, camera):
        self.image_center = None
        self.board_centers = None
        if camera == 'fisheye':
            self.camera = Fisheye()
        elif camera == 'normal':
            self.camera = Normal()
        else:
            raise Exception("camera should be fisheye/normal")
        self.corners = []
    @staticmethod
    def get_args():
        return args

    def get_corners(self, img):
        ok, corners = cv2.findChessboardCorners(img, (args.BORAD_WIDTH, args.BORAD_HEIGHT),
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if ok:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 亚像素级别的角点精确化 cv2.cornerSubPix 需要使用灰度图像。
            corners = cv2.cornerSubPix(gray, corners, (args.SUBPIX_REGION, args.SUBPIX_REGION), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))  # 得到精度提升的角点
        return ok, corners
        # 在输入图像中检测标定板的角点。

    def draw_corners(self, img):
        ok, corners = self.get_corners(img)
        cv2.drawChessboardCorners(img, (args.BORAD_WIDTH, args.BORAD_HEIGHT), corners, ok)
        return img
        # 在输入图像上绘制检测到的角点

    def undistort(self, img):
        data = self.camera.data
        return cv2.remap(img, data.map1, data.map2, cv2.INTER_LINEAR)
        # 对输入图像进行畸变校正

    def calibrate(self, img):
        if len(self.corners) >= args.CALIB_NUMBER:
            self.camera.update(self.corners, img.shape[1::-1])
        return self.camera.data
        # 进行相机的标定

    def get_board_center(self, frames):
        board_centers = []
        for frame in frames:
            ok, corners = self.get_corners(frame)
            if not ok:
                board_centers.append([-1,-1])
                continue
            center = np.mean(corners, axis=0)
            board_centers.append(center)
        return board_centers

    def get_distance_to_center(self, frame,board_center):
        fw = self.fw
        fh = self.fh
        # 假设图片的中心点坐标为(image_center_x, image_center_y)
        image_center_x, image_center_y = fw / 2, fh / 2
        self.image_center = np.array([image_center_x, image_center_y])
        # self.board_centers = self.get_board_center(frames)
        distance = np.linalg.norm(board_center - self.image_center)
        return distance

    def sort_frames_by_distance(self, frames, board_centers):
        distances = [self.get_distance_to_center(frames[idx],board_centers[idx]) for idx in range(len(frames))]
        sorted_indices = np.argsort(distances)
        sorted_frames = [frames[i] for i in sorted_indices]
        return sorted_frames


    def __call__(self, raw_frame):
         ok, corners = self.get_corners(raw_frame)
         result = self.camera.data
         if ok:
             self.corners.append(corners)
             result = self.calibrate(raw_frame)
         return result

def centerCrop(img, width, height):
    if img.shape[1] < width or img.shape[0] < height:
        raise Exception("CROP size should be smaller than original size")
    img = img[round((img.shape[0] - height) / 2): round((img.shape[0] - height) / 2) + height,
          round((img.shape[1] - width) / 2): round((img.shape[1] - width) / 2) + width]
    return img
    # 中心剪裁


def get_images(PATH, NAME):
    filePath = [os.path.join(PATH, x) for x in os.listdir(PATH)
                if any(x.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
                ]
    filenames = [filename for filename in filePath if NAME in filename]
    if len(filenames) == 0:
        raise Exception("from {} read images failed".format(PATH))
    return filenames
    # 从指定目录 PATH 获取满足名称条件 NAME 的图像文件的路径列表。
    # 它将在目录中搜索具有 .png、.jpg 或 .jpeg 扩展名（不区分大小写）的文件，
    # 并筛选出文件名中包含特定名称 NAME 的图像文件路径。
    # 如果没有找到满足条件的图像文件，将会抛出异常。这样的方法可以方便地获取指定目录下的特定图像文件，并用于后续的图像处理和分析任务。

class CalibMode():
    def __init__(self, calibrator, input_type, mode):
        self.calibrator = calibrator
        self.input_type = input_type
        self.mode = mode
        self.image_center =None
        self.board_centers = []
        self.sorted_frames = []
        self.distance = None
        self.fw =1280
        self.fh =780
    def imgPreprocess(self, img):
        if args.CROP_FLAG:
            img = centerCrop(img, args.FRAME_WIDTH, args.FRAME_HEIGHT)
        elif args.RESIZE_FLAG:
            img = cv2.resize(img, (args.FRAME_WIDTH, args.FRAME_HEIGHT))
        return img
        # 预处理

    def setCamera(self, cap):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, args.CAMERA_FPS)
        return cap
        # 设置相机的参数：视频编码方式、帧的宽度和高度、帧率

    def runCalib(self, raw_frames, display_raw=True, display_undist=True):
        # frames = []
        calibrator = self.calibrator
        # raw_frame = self.imgPreprocess(raw_frame)
        # raw_frame = calibrator.draw_corners(raw_frame)
        # frames.append(raw_frame)
        self.board_centers = calibrator.get_board_center(raw_frames)
        # result = calibrator(frames)
        sorted_frames = calibrator.sort_frames_by_distance(raw_frames, calibrator.image_center,self.board_centers)
        # 显示排序后的视频帧
        import tqdm
        for frame in tqdm.tqdm(sorted_frames):
            cv2.imshow("Sorted Frames", frame)
            result = calibrator(frame)
            #标定
        # if display_raw:
        #     cv2.namedWindow("raw_frame", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        #     cv2.imshow("raw_frame", raw_frame)  # 显示标定前的图像
        # if len(calibrator.corners) > args.CALIB_NUMBER and display_undist:
        #     undist_frame = calibrator.undistort(raw_frame)
        #     cv2.namedWindow("undist_frame", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        #     cv2.imshow("undist_frame", undist_frame)  # 显示矫正后的图像
        #cv2.waitKey(1)
        return result

    # runCalib 方法通过调用相机标定对象 calibrator 对图像进行相机标定。标定过程中，它会在窗口中显示标定前的图像，
    # 并在满足一定条件后（例如检测到足够多的角点），显示标定后的图像。最终，方法返回相机标定的结果，供调用者使用。
    def imageAutoMode(self):
        filenames = get_images(args.INPUT_PATH, args.IMAGE_FILE)
        for filename in filenames:
            print(filename)
            raw_frame = cv2.imread(filename)  # 将图像文件解码为一个 NumPy 数组，存储在 raw_frame 中。
            result = self.runCalib(raw_frame)
            key = cv2.waitKey(1)
            if key == 27: break  # 按下 ESC 键（键码为 27），则退出循环，中止图像处理。
        cv2.destroyAllWindows()
        return result

    # 自动标定图片的模式
    def imageManualMode(self):
        filenames = get_images(args.INPUT_PATH, args.IMAGE_FILE)
        for filename in filenames:
            print(filename)
            raw_frame = cv2.imread(filename)
            raw_frame = self.imgPreprocess(raw_frame)
            img = raw_frame.copy()
            img = self.calibrator.draw_corners(img)  # img变为带有角点的图像。
            display = "raw_frame: press SPACE to SELECT, other key to SKIP, press ESC to QUIT"
            cv2.namedWindow(display, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(display, img)  ##显示带有角点的图像。
            key = cv2.waitKey(0)
            if key == 32:
                result = self.runCalib(raw_frame, display_raw=False)  # 不显示标定前的图像
            if key == 27: break
        cv2.destroyAllWindows()
        return result

    # 手动模式
    def videoAutoMode(self):
        cap = cv2.VideoCapture(args.INPUT_PATH + args.VIDEO_FILE)
        if not cap.isOpened():
            raise Exception("from {} read video failed".format(args.INPUT_PATH + args.VIDEO_FILE))
        frame_id = 0  # id用于计数当前处理的帧数
        # 下面那行持续处理视频流
        frames = []
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break
            raw_frame = self.imgPreprocess(raw_frame)
            frames.append(raw_frame)
            if frame_id % args.FRAME_DELAY == 0:
                if args.STORE_FLAG:
                    cv2.imwrite(args.STORE_PATH + 'img_raw{}.jpg'.format(len(self.calibrator.corners)), raw_frame)
                # result = self.runCalib(raw_frame)
                # print(len(self.calibrator.corners))
            frame_id += 1
            key = cv2.waitKey(1)
            if key == 27: break
        #end
        result = self.runCalib(frames)
        cap.release()
        cv2.destroyAllWindows()
        return result

    # 自动视频标定
    def videoManualMode(self):
        cap = cv2.VideoCapture(args.INPUT_PATH + args.VIDEO_FILE)
        if not cap.isOpened():
            raise Exception("from {} read video failed".format(args.INPUT_PATH + args.VIDEO_FILE))
        while True:
            key = cv2.waitKey(1)
            ok, raw_frame = cap.read()
            raw_frame = self.imgPreprocess(raw_frame)
            display = "raw_frame: press SPACE to capture image"
            cv2.namedWindow(display, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(display, raw_frame)  # 显示预处理后的原始图像，并等待用户按键。
            if key == 32:
                if args.STORE_FLAG:
                    cv2.imwrite(args.STORE_PATH + 'img_raw{}.jpg'.format(len(self.calibrator.corners)), raw_frame)
                result = self.runCalib(raw_frame)
                print(len(self.calibrator.corners))
            if key == 27: break
        cap.release()
        cv2.destroyAllWindows()
        return result

    # 手动
    def cameraAutoMode(self):
        cap = cv2.VideoCapture(args.CAMERA_ID)  # 传入相机的ID args.CAMERA_ID。这个对象用于读取实时视频流。
        if not cap.isOpened():
            raise Exception("from {} read video failed".format(args.CAMERA_ID))
        cap = self.setCamera(cap)  # 设置相机的分辨率、帧率等参数。
        frame_id = 0
        start_flag = False  # 初始化表示尚未开始标定。
        while True:
            key = cv2.waitKey(1)
            ok, raw_frame = cap.read()
            raw_frame = self.imgPreprocess(raw_frame)
            if key == 32: start_flag = True
            if key == 27: break
            if not start_flag:
                cv2.putText(raw_frame, 'press SPACE to start!', (args.FRAME_WIDTH // 4, args.FRAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)  # 在图像帧 raw_frame 上添加文字提示，提示用户按 SPACE 键来开始标定。
                cv2.imshow("raw_frame", raw_frame)  # 在显示窗口中展示带有文字提示的预处理后的图像帧 raw_frame。
                continue
            if frame_id % args.FRAME_DELAY == 0:  # 如果是帧延迟的倍数，标定储存
                if args.STORE_FLAG:
                    cv2.imwrite(args.STORE_PATH + 'img_raw{}.jpg'.format(len(self.calibrator.corners)), raw_frame)
                result = self.runCalib(raw_frame)
                print(len(self.calibrator.corners))
            frame_id += 1
        cap.release()
        cv2.destroyAllWindows()
        return result

    def cameraManualMode(self):
        cap = cv2.VideoCapture(args.CAMERA_ID)
        if not cap.isOpened():
            raise Exception("from {} read video failed".format(args.CAMERA_ID))
        cap = self.setCamera(cap)
        while True:
            key = cv2.waitKey(1)
            ok, raw_frame = cap.read()
            raw_frame = self.imgPreprocess(raw_frame)
            display = "raw_frame: press SPACE to capture image"
            cv2.namedWindow(display, flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(display, raw_frame)
            if key == 32:
                if args.STORE_FLAG:
                    cv2.imwrite(args.STORE_PATH + 'img_raw{}.jpg'.format(len(self.calibrator.corners)), raw_frame)
                result = self.runCalib(raw_frame)
                print(len(self.calibrator.corners))
            if key == 27: break
        cap.release()
        cv2.destroyAllWindows()
        return result

    def __call__(self):
        input_type = self.input_type
        mode = self.mode
        if input_type == 'image' and mode == 'auto':
            result = self.imageAutoMode()
        if input_type == 'image' and mode == 'manual':
            result = self.imageManualMode()
        if input_type == 'video' and mode == 'auto':
            result = self.videoAutoMode()
        if input_type == 'video' and mode == 'manual':
            result = self.videoManualMode()
        if input_type == 'camera' and mode == 'auto':
            result = self.cameraAutoMode()
        if input_type == 'camera' and mode == 'manual':
            result = self.cameraManualMode()
        return result


def main():
    calibrator = InCalibrator(args.CAMERA_TYPE)
    calib = CalibMode(calibrator, args.INPUT_TYPE, args.SELECT_MODE)
    result = calib()

    if len(calibrator.corners) == 0:
        raise Exception("Calibration failed. Chessboard not found, check the parameters")
    if len(calibrator.corners) < args.CALIB_NUMBER:
        raise Exception("Warning: Calibration images are not enough. At least {} valid images are needed.".format(
            args.CALIB_NUMBER))

    print("Calibration Complete")
    print("Camera Matrix is : {}".format(result.camera_mat.tolist()))
    print("Distortion Coefficient is : {}".format(result.dist_coeff.tolist()))
    print("Reprojection Error is : {}".format(np.mean(result.reproj_err)))
    np.save('camera_{}_K.npy'.format(args.CAMERA_ID), result.camera_mat.tolist())
    np.save('camera_{}_D.npy'.format(args.CAMERA_ID), result.dist_coeff.tolist())


if __name__ == '__main__':
    main()
