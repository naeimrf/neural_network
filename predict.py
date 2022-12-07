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
from utils import parse_predict_arguments, predict
from utils import process_image, load_json_file
from model import rebuild_simple
from utils import result

# 1. Read command line arguments (options) for prediction part
input_args = parse_predict_arguments()

# 2. If any other json file provided, read it here
if input_args.category_names:
    category_names = load_json_file(input_args.category_names)
    print(f"-> Loading json file: {input_args.category_names}!")

# 3. Prepare imported image and rebuild the model for prediction
img = process_image(input_args.input)
model = rebuild_simple(input_args.checkpoint, input_args.gpu)

# 4. predict flower classes with the highest probabilities
k = input_args.top_k if input_args.top_k else 1

probs, probs_rounded, classes = predict(img, model, k=k)

# 5. Print and/or plot the result
result(
    input_args.input,
    probs,
    probs_rounded,
    classes,
    input_args.category_names,
    plot=False,
)
