import torch

from api import api
from imageCS_utils.base_utils.status import QueueStateManager
from imageCS_utils.file_writer import Log_Writer
from imageCS_utils.base_utils.info import Info

def test(cfg_name:str, gpu_num:int, cs_ratios:list[str], testset_names:list[str]):
    # Init Logger and Queue
    log_writer = Log_Writer("log", "finished_work.log")
    task_name = f"test-{cfg_name}-{testset_names}-{cs_ratios}"
    qma = QueueStateManager(
        name=task_name,
        gpu=gpu_num
    )
    qma.start()

    # Test Func
    ## Using pretrained weight to test
    test_ans = {}
    for cs_ratio in cs_ratios:
        Info.WARN('If you are testing instead of training, you can ignore the warn infomations about "Fail to load the state dict of optimizer", "Fail to load the state dict of learning rate scheduler" or "Fail to load pretrained guided-diffusion weights", for which will not affect the test results.')
        base = api(
            cfg_name=cfg_name,
            gpu_num=gpu_num,
            img_channels=1,
            cs_ratio = cs_ratio,
            if_train = False,
        )

        this_test_ans_list = []
        for testset_name in testset_names:
            try:
                torch.cuda.empty_cache()
                this_test_ans_list.append(base.test_image(testset_name))
            except Exception as e:
                e = str(e)
                log_writer.write(f"[Warn] {task_name}: {e}")
                Info.error(e)
        test_ans[cs_ratio] = this_test_ans_list

    ## Generate test answers
    ans_str = ""
    for cs_ratio, ans_list in test_ans.items():
        split_str = f"[CS ratio = {cs_ratio}]\n"
        ans_str += split_str
        for ans in ans_list:
            ans_str += f"\t{ans}\n"
    ans_str += "\n"
    
    # Release Logger and Queue
    qma.stop()
    log_writer.write(task_name)

    # Return
    return ans_str
