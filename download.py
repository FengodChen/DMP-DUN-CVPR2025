from argparse import ArgumentParser

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download

class DownloadHelper:
    def __init__(self, model_id='FengodChen/DMP-DUN', local_dir='./'):
        self.api = HubApi()
        self.model_id = model_id
        self.local_dir = local_dir

    def _download(self, file_path:str):
        model_file_download(
            model_id=self.model_id,
            file_path=file_path,
            local_dir=self.local_dir
        )

    def _get_file_path_list(self, remote_folder:str):
        file_path_list = []
        all_file_info = self.api.get_model_files(
            model_id=self.model_id,
            recursive=True,
            revision='master',
            root=remote_folder
        )

        for file_info in all_file_info:
            if file_info['Type'] == 'blob':
                file_path_list.append(file_info['Path'])

        return file_path_list

class ModelDownloadHelper(DownloadHelper):
    def _get_remote_folder(self, model_name:str, cs_ratio:str):
        assert isinstance(model_name, str)
        assert isinstance(cs_ratio, str)
        assert model_name in ["DMP_DUN_10step", "DMP_DUN_plus_2step", "DMP_DUN_plus_4step"]
        assert cs_ratio in ["0.5", "0.25", "0.1", "0.04", "0.01"]

        remote_folder = f'save/{model_name}/C1R{cs_ratio}'
        return remote_folder

    def download(self, model_name:str, cs_ratio:str):
        remote_folder = self._get_remote_folder(model_name, cs_ratio)
        file_path_list = self._get_file_path_list(remote_folder)
        for file_path in file_path_list:
            self._download(file_path)

class TestsetDownloadHelper(DownloadHelper):
    def download(self, testset_name:str):
        assert testset_name in ["Set11", "Urban100"]
        remote_folder = f"datasets/{testset_name}"
        file_path_list = self._get_file_path_list(remote_folder)
        for file_path in file_path_list:
            self._download(file_path)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", action='store_true')
    parser.add_argument("--testset", action='store_true')
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--cs-ratios", type=str)
    parser.add_argument("--testset-name", type=str)
    args = parser.parse_args()

    # Make sure select and only select one of model or testset
    assert (args.model and args.testset) == False
    assert (args.model or args.testset) == True

    if args.model:
        download_helper = ModelDownloadHelper()
        model_name = args.model_name
        cs_ratios:list[str] = args.cs_ratios.split(",")

        for cs_ratio in cs_ratios:
            download_helper.download(model_name, cs_ratio)
    elif args.testset:
        download_helper = TestsetDownloadHelper()
        testset_name = args.testset_name
        download_helper.download(testset_name)
