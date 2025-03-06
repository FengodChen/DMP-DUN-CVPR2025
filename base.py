from imageCS_utils.base_utils.base import MetricsBase
from utils import generate_test_dataloader

class Base(MetricsBase):
    def _test_image(self, dataset_name:str, dataloader):
        metrics_func_dict = self.cfg["train_config"]["metrics_func_dict"]
        (_, avg_metrics_dict, _, _, _, _, _) = super().test(dataloader, metrics_func_dict)
        psnr = avg_metrics_dict["psnr"]
        ssim = avg_metrics_dict["ssim"]

        result_str = f"[{dataset_name}] PSNR(dB)/SSIM = {psnr:.2f}/{ssim:.4f}"
        return result_str
    
    def test_image(self, testset_name:str):
        dataloader = generate_test_dataloader(testset_name)
        result_str = self._test_image(testset_name, dataloader)

        return result_str
        
    
