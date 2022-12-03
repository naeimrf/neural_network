"""
* The purpose of this module is to predict the name of one flower with
    the probability of that name.
* Basic input: the path to a single image and a checkpoint file
    * Options:
        * python predict.py /path/to/image checkpoint
        * python predict.py input checkpoint --top_k 3
        * python predict.py input checkpoint --category_names cat_to_name.json
        * python predict.py input checkpoint --gpu
* Output: the flower name and a class probability
"""
from utils import parse_predict_arguments, load_json_file

input_args = parse_predict_arguments()

if input_args.category_names:
    category_names = load_json_file(input_args.category_names)

if __name__ == "__main__":
    print(input_args)
