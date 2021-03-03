import json
import logging
import os

import random
import sys
from zipfile import ZipFile

from transformers import cached_path

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# 比赛各阶段数据集，CAIL2019-SCM-big 是第二阶段数据集
DATASET_ARCHIVE_MAP = {
    "CAIL2019-SCM-big": "https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip"
}


def download_data(dataset_name):
    """
    下载数据集

    :param dataset_name: 数据集名称。
    :return:
    """
    url = DATASET_ARCHIVE_MAP[dataset_name]
    try:
        resolved_archive_file = cached_path(url)
        logger.info("Dataset is ready!")
    except EnvironmentError:
        logger.error("Dataset Download failed!")
        return None

    data_dir = os.path.join("data/raw", dataset_name)
    with ZipFile(resolved_archive_file, "r") as zipObj:
        """
        更新：主办方更新了数据集下载链接里的文件内容（由一个data.json分为了train.json、dev.json、test.json三个文件），
        其中的train.json为比赛第二阶段的5000条数据集，dev.json、test.json 在比赛中没有给出。
        本项目仅取train.json使用，保持和比赛时一致的数据和环境。
        """
        # 通过文件大小筛选出第二阶段数据集
        zipinfo_dataset_stage2 = max(
            zipObj.infolist(), key=lambda zipinfo: zipinfo.compress_size
        )
        legacy_filename = "SCM_5k.json"
        zipinfo_dataset_stage2.filename = legacy_filename
        zipObj.extract(zipinfo_dataset_stage2, data_dir)
        return os.path.join(data_dir, legacy_filename)


def generate_fix_test_data(raw_input_file):
    """
    生成固定的测试集数据。该数据仅用于基本的模型可用性测试。抽取20%数据，打乱BC顺序。

    :param raw_input_file: 原始的数据集文件
    :return:
    """
    test_input_file = "data/test/input.txt"
    train_input_file = "data/train/input.txt"
    label_output_file = "data/test/ground_truth.txt"
    lines = []
    with open(raw_input_file, encoding="utf-8") as raw_input:
        for line in raw_input:
            lines.append(line.strip())
    random.seed(42)
    random.shuffle(lines)
    n_test = int(len(lines) * 0.2)
    test_lines = lines[:n_test]
    train_lines = lines[n_test:]

    os.makedirs("data/train", exist_ok=True)
    with open(train_input_file, mode="w", encoding="utf-8") as train_input:
        for line in train_lines:
            train_input.write(line)
            train_input.write("\n")

    os.makedirs("data/test", exist_ok=True)
    with open(test_input_file, mode="w", encoding="utf-8") as test_input, open(
        label_output_file, encoding="utf-8", mode="w"
    ) as label_output:
        for line in test_lines:
            item = json.loads(line, encoding="utf-8")
            a, b, c, origin_label = (
                item["A"],
                item["B"],
                item["C"],
                item.get("label", "B"),
            )

            # 对测试数据随机调整BC顺序和label
            choice = int(random.getrandbits(1))
            label = "B" if choice == 0 else "C"

            label_output.write(label)
            label_output.write("\n")
            if label != origin_label:
                line = json.dumps({"A": a, "B": c, "C": b}, ensure_ascii=False).strip()

            test_input.write(line)
            test_input.write("\n")


if __name__ == "__main__":
    data_file = download_data("CAIL2019-SCM-big")
    generate_fix_test_data(data_file)
