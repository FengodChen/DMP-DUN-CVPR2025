from argparse import ArgumentParser

from func_train import train
from func_test import test

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--cs-ratios", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--testset-names", type=str)
    parser.add_argument("--test-out-file-path", type=str, default="test_ans.txt")
    args = parser.parse_args()

    model_name = args.model_name
    cs_ratios:list[str] = args.cs_ratios.split(",")
    gpu_id:int = args.gpu_id

    # Make sure select and only select one of train or test
    assert (args.train and args.test) == False
    assert (args.train or args.test) == True

    if args.train:
        train(model_name, gpu_id, cs_ratios)
    elif args.test:
        assert args.testset_names is not None

        testset_names:list[str] = args.testset_names.split(",")
        test_out_file_path:str = args.test_out_file_path
        test_ans = test(model_name, gpu_id, cs_ratios, testset_names)

        test_ans = f">>>>>> {model_name} <<<<<<\n{test_ans}"

        with open(test_out_file_path, "a", encoding='utf-8') as fp:
            fp.write(test_ans)
