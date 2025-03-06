from imageCS_utils.base_utils.trainer import UniversalMetricsTrainer

class Trainer(UniversalMetricsTrainer):
    def _get_x(self, data_prepare_output, net_forward_output):
        x_true = data_prepare_output["x_true"]
        x_pred = net_forward_output["x_pred"]
        return (x_pred, x_true)
    
    def _set_metrics_kwargs(self, data_prepare_output, net_forward_output):
        psnr_kwargs = self.other_data_dict['psnr_kwargs']
        ssim_kwargs = self.other_data_dict['ssim_kwargs']
        (x_pred, x_true) = self._get_x(data_prepare_output, net_forward_output)

        psnr_kwargs.update({
            "X": x_pred,
            "Y": x_true
        })

        ssim_kwargs.update({
            "X": x_pred,
            "Y": x_true
        })

        kwargs = {
            "psnr": psnr_kwargs,
            "ssim": ssim_kwargs
        }
        return kwargs

