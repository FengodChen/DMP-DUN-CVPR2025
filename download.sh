# Download Testset
python download.py --testset --testset-name Set11
python download.py --testset --testset-name Urban100

# Download DMP-DUN model weight
python download.py --model --model-name DMP_DUN_10step --cs-ratios 0.5,0.25,0.1,0.04,0.01
python download.py --model --model-name DMP_DUN_plus_2step --cs-ratios 0.5,0.25,0.1,0.04,0.01
python download.py --model --model-name DMP_DUN_plus_4step --cs-ratios 0.5,0.25,0.1,0.04,0.01
