import pandas as pd
import json
import os
from multiprocessing import Pool, cpu_count


def loop_dir(dir_name, export_name):  # dir_name : label_dir or vallabel_dir    export_name : "trainlabel.xlsx"
    # 라벨링 데이터로 부터 json파일들 파싱하기.
    global combined_df, total_count

    path_list = []

    for path1 in os.listdir(dir_name):  # 가구류, 고철류, 나무, 도기류....... 출력
        class_name = path1
        class_path = os.path.join(dir_name, class_name)
        path_list.append(class_path)

    with Pool(cpu_count() // 2) as p:
        results = p.map(analze_json, path_list)

        # results is a list of tuples. We can separate it into two lists:
        dataframes, counts = zip(*results)
        # now you can aggregate the dataframes and counts as needed
        total_count = sum(counts)
        combined_df = pd.concat(dataframes, ignore_index=True)

    print(combined_df)
    print(total_count)
    # 데이터 프레임을 엑셀 파일로 추출
    combined_df.to_excel(export_name)


def analze_json(class_path):
    # 열 목록 생성
    columns = ["filename", "form", "resolution", "class", "details", "x1", "y1", "x2", "y2"]

    # 데이터 프레임 초기화
    df = pd.DataFrame(columns=columns)

    # pass 한 사진들 개수 알아내기
    passnum = 0
    print(class_path)

    for path2 in os.listdir(class_path):  # 밥상, 서랍장, 소파, 수납장....... 출력
        detail_name = path2
        detail_path = os.path.join(class_path, detail_name)
        for path3 in os.listdir(detail_path):  # 같은 물체의 다방면 이미지를 모아둔 폴더이름들
            folder_name = os.path.join(detail_path, path3)
            for file in os.listdir(folder_name):
                file_name = os.path.join(folder_name, file)
                with open(file_name, encoding='UTF8') as f:
                    json_file = json.load(f)

                # print(json_file)

                file_name = json_file["FILE NAME"]
                form = json_file["FORM"]
                resolution = json_file["RESOLUTION"]
                boundnum = json_file["BoundingCount"]
                int_boundnum = int(boundnum)
                for idx in range(int_boundnum):
                    classes = json_file["Bounding"][idx]["CLASS"]
                    details = json_file["Bounding"][idx]["DETAILS"]
                    x1 = 0
                    y1 = 0
                    x2 = 0
                    y2 = 0
                    try:
                        x1 = int(json_file["Bounding"][idx]["x1"])
                        y1 = int(json_file["Bounding"][idx]["y1"])
                        x2 = int(json_file["Bounding"][idx]["x2"])
                        y2 = int(json_file["Bounding"][idx]["y2"])

                    except:
                        try:
                            polynum = json_file["Bounding"][idx]["PolygonCount"]
                            int_polynum = int(polynum)
                            poly = json_file["Bounding"][idx]["PolygonPoint"]
                            min_x = 10000
                            min_y = 10000
                            max_x = 0
                            max_y = 0

                            for count in range(0, int_polynum):
                                point = poly[count]["Point%s" % str(count + 1)]
                                x = int(point.split(",")[0])
                                y = int(point.split(",")[1])
                                min_x = min(min_x, x)
                                min_y = min(min_y, y)

                                max_x = max(max_x, x)
                                max_y = max(max_y, y)

                            x1 = min_x
                            y1 = min_y
                            x2 = max_x
                            y2 = max_y
                        except:
                            passnum = passnum + 1
                            pass
                    df = pd.concat([df, pd.DataFrame(
                        {"filename": [file_name], "form": [form], "resolution": [resolution], "class": [classes],
                         "details": [details], "x1": [x1], "y1": [y1], "x2": [x2], "y2": [y2]})], ignore_index=True)

                    # df = df.append({"filename": file_name, "form": form, "resolution": resolution, "class": classes,
                    #               "details": details, "x1": x1, "y1": y1, "x2": x2, "y2": y2}, ignore_index=True)
    return df, passnum


if __name__ == "__main__":
    # 데이터 디렉토리 설정
    DATA_DIR = "C:/Users/rlaej/Desktop/label_data/"

    # json파일 train 데이터셋 디렉토리 설정
    trainlabel_dir = os.path.join(DATA_DIR, "trainlabel/")

    # json파일 val 데이터셋 디렉토리 설정
    vallabel_dir = os.path.join(DATA_DIR, "vallabel/")

    loop_dir(trainlabel_dir, "train_label.xlsx")
    # analze_json(vallabel_dir, "val_label.xlsx")
