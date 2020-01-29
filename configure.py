import argparse
import sys


def parse_args():

    parser = argparse.ArgumentParser()
    # 对于函数add_argumen()第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明

    # data loading
    parser.add_argument("--train_path", default="data/train_data.txt",
                        type=str, help="Path of train data")
    parser.add_argument("--test_path",
                        default="data/test_data.txt",
                        type=str, help="Path of test data")
    parser.add_argument("--max_sentence_length", default=90,
                        type=int, help="Max sentence length in data")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args

FLAGS = parse_args()


