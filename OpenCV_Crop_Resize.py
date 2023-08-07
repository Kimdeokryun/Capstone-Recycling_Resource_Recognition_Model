import os
from functools import partial
from urllib.parse import quote
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import EfficientNet_optimize as model_file


def loop_dir(o_dir, c_dir, dataframe,
             export_name):  # dir_name : label_dir or vallabel_dir    export_name : "trainlabel.xlsx"
    # 라벨링 데이터로 부터 json파일들 파싱하기.
    path_list = []
    for path1 in os.listdir(o_dir):  # [T원천]가구류_밥상_밥상 등의 폴더
        dir1 = os.path.join(o_dir, path1)
        save_dir2 = os.path.join(c_dir, path1)
        dirs = [dir1, save_dir2]
        path_list.append(dirs)

    print(len(path_list))

    func = partial(many_working, dataframe)

    with Pool(cpu_count() // 2) as p:
        results = p.starmap(func, path_list)

        # results is a list of tuples. We can separate it into two lists:
        dataframes, filecounts, boundingcounts = zip(*results)
        # now you can aggregate the dataframes and counts as needed
        total_count1 = sum(filecounts)
        total_count2 = sum(boundingcounts)

        combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.to_excel(export_name)

    print("파일의 개수: ", total_count1)
    print("바운딩 박스가 2개 이상인 이미지의 개수: ", total_count2)


def many_working(image_information, dir1, save_dir1):  # 폴더 이름과 이미지 이름 데이터 프레임화,
    # 열 목록 생성
    columns = ["filename", "filepath", "class", "details"]

    # 데이터 프레임 초기화
    image_df = pd.DataFrame(columns=columns)
    count1 = 0
    count2 = 0

    print(dir1)

    for path2 in os.listdir(dir1):  # 11_X001_C012_1215 등의 폴더
        dir2 = dir1 + "/" + path2
        save_dir2 = save_dir1 + "/" + path2

        for file in os.listdir(dir2):  # 사진 파일들
            filepath = dir2 + "/" + file
            savefilepath = save_dir2 + "/" + file

            temp_df = image_information[image_information['filename'] == file]

            box_num = len(temp_df)

            x1_list = temp_df['x1'].tolist()
            y1_list = temp_df['y1'].tolist()
            x2_list = temp_df['x2'].tolist()
            y2_list = temp_df['y2'].tolist()
            class_list = temp_df['class'].tolist()
            details_list = temp_df['details'].tolist()

            for idx in range(box_num):
                # BoundingBox 정의 (x, y, w, h)
                x1 = int(x1_list[idx])
                y1 = int(y1_list[idx])
                x2 = int(x2_list[idx])
                y2 = int(y2_list[idx])

                cla = str(class_list[idx])
                detail = str(details_list[idx])

                if idx > 0:
                    savefilepath = save_dir2 + "/" + "(" + str(idx) + ")" + file
                    count2 += 1

                bbox = np.array([x1, y1, x2 - x1, y2 - y1])  # 예시 값
                image_processing(filepath, savefilepath, bbox)

                image_df = pd.concat([image_df, pd.DataFrame(
                    {"filename": [file], "filepath": [savefilepath], "class": [cla], "details": [cla + "-" + detail]})],
                                     ignore_index=True)
                count1 += 1

    return image_df, count1, count2


def image_processing(image_path, image_save_path, bbox):
    # 이미지 로드
    # img = cv2.imread(path1)
    with open(image_path, 'rb') as f:
        imagebytes = bytearray(f.read())
        numpyarray = np.asarray(imagebytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    # 이미지의 높이와 너비
    img_height, img_width = img.shape[:2]

    # BoundingBox의 중심 좌표 계산
    center_x = bbox[0] + bbox[2] // 2
    center_y = bbox[1] + bbox[3] // 2

    # BoundingBox의 가장 긴 변 길이 계산
    max_len = max(bbox[2], bbox[3])

    # 새로운 BoundingBox 계산
    new_bbox = np.array([center_x - max_len // 2, center_y - max_len // 2, max_len, max_len])

    # 새로운 BoundingBox가 이미지의 경계를 넘어가는 경우, 이미지의 크기에 맞게 조정
    new_bbox[0] = max(0, new_bbox[0])  # x 시작점
    new_bbox[1] = max(0, new_bbox[1])  # y 시작점
    new_bbox[2] = min(img_width - new_bbox[0], new_bbox[2])  # 너비
    new_bbox[3] = min(img_height - new_bbox[1], new_bbox[3])  # 높이

    # 새로운 정사각형의 한 변의 길이를 계산
    new_square_len = min(new_bbox[2], new_bbox[3])

    # 이미지 자르기
    crop = img[new_bbox[1]:new_bbox[1] + new_square_len, new_bbox[0]:new_bbox[0] + new_square_len]

    # 이미지를 224x224 사이즈로 리사이즈
    # resized_crop = cv2.resize(crop, (299, 299))

    """
    # 새로운 바운딩 박스의 중심 좌표 계산
new_center_x = new_bbox[0] + new_bbox[2] // 2
new_center_y = new_bbox[1] + new_bbox[3] // 2

# 이미지 크기에 대한 비율 계산
resize_ratio_width = 224 / new_square_len
resize_ratio_height = 224 / new_square_len

# 새로운 바운딩 박스의 좌표 계산
new_x = (new_center_x - new_square_len // 2) * resize_ratio_width
new_y = (new_center_y - new_square_len // 2) * resize_ratio_height
new_width = new_bbox[2] * resize_ratio_width
new_height = new_bbox[3] * resize_ratio_height

# 새로운 바운딩 박스의 정보
new_bbox = [new_x, new_y, new_width, new_height]
    
    """

    # 결과 보기
    # cv2_imshow(resized_crop)

    # 이미지 저장
    # cv2.imwrite(path2, resized_crop)
    is_success, im_buf_arr = cv2.imencode(".jpg", crop)
    if is_success:
        with open(image_save_path, 'wb') as f_out:
            f_out.write(im_buf_arr)


if __name__ == "__main__":
    """
    # 훈련 데이터셋
    # 이미지 불러오고 전처리 이미지 저장하기
    image_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 이미지/Training/"
    saveimage_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 전처리 이미지/Training/"

    # 엑셀 파일을 읽어들여 데이터프레임으로 변환
    dataframe_path = "D:/생활폐기물이미지데이터셋/train_label.xlsx"
    df = pd.read_excel(dataframe_path)

    export_path = "D:/생활폐기물이미지데이터셋/train_image_label.xlsx"
    
    # 검증 데이터셋

    # 이미지 불러오고 전처리 이미지 저장하기
    image_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 이미지/Validation/"
    saveimage_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 전처리 이미지/Validation/"

    # 엑셀 파일을 읽어들여 데이터프레임으로 변환
    dataframe_path = "D:/생활폐기물이미지데이터셋/val_label.xlsx"
    df = pd.read_excel(dataframe_path)

    export_path = "D:/생활폐기물이미지데이터셋/val_image_label.xlsx"
    """

    # ====================================================================================================================
    # 둘의 변수 이름이 같으므로 순차적으로 실행해야 함   (이 디렉토리는 이미지 데이터중 필요한 이미지 데이터만을 사용하려고 처리함.)

    # 훈련 데이터셋
    # 이미지 불러오고 전처리 이미지 저장하기
    image_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 이미지/Meaning_dataset/Training_use/"
    saveimage_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 전처리 이미지/Training/"

    # 엑셀 파일을 읽어들여 데이터프레임으로 변환
    dataframe_path = "D:/생활폐기물이미지데이터셋/train_label.xlsx"
    df = pd.read_excel(dataframe_path)

    export_path = "D:/생활폐기물이미지데이터셋/trainuse_image_label.xlsx"

    loop_dir(image_dir, saveimage_dir, df, export_path)

    # 검증 데이터셋
    # 이미지 불러오고 전처리 이미지 저장하기
    image_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 이미지/Meaning_dataset/Validaton_use/"
    saveimage_dir = "D:/생활폐기물이미지데이터셋/생활 폐기물 전처리 이미지/Validation/"

    # 엑셀 파일을 읽어들여 데이터프레임으로 변환
    dataframe_path = "D:/생활폐기물이미지데이터셋/val_label.xlsx"
    df = pd.read_excel(dataframe_path)

    export_path = "D:/생활폐기물이미지데이터셋/valuse_image_label.xlsx"

    loop_dir(image_dir, saveimage_dir, df, export_path)