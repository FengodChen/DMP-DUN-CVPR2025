#! /bin/bash

python main.py --train --model-name DMP_DUN_10step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --gpu-id 0
python main.py --train --model-name DMP_DUN_plus_2step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --gpu-id 0
python main.py --train --model-name DMP_DUN_plus_4step --cs-ratios 0.5,0.25,0.1,0.04,0.01 --gpu-id 0