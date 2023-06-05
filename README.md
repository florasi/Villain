# Backdoor Attacks Against Vertical Split Learning
Code for the paper "Backdoor Attacks Against Vertical Split Learning". Now we are refining the journal editions and will open source soon.

![avatar](/samples/sleuth_whole.png)

# Our method of label inference and backdoor attack

usage: label_inference_and_backdoor_attack.py [-h] [--dataset DATASET]

                                              [--target_label TARGET_LABEL]
                                              
                                              [--target_label_candidates_number TARGET_LABEL_CANDIDATES_NUMBER]
                                              
                                              [--test_parameter TEST_PARAMETER]
                                              
                                              [--test_noise TEST_NOISE]
                                              
                                              [--test_trigger_fabrication TEST_TRIGGER_FABRICATION]
                                              
                                              [--test_sample_n TEST_SAMPLE_N]
                                              
                                              [--test_server_layer TEST_SERVER_LAYER]
                                              
                                              [--test_lr TEST_LR]
                                              
                                              [--noise_range NOISE_RANGE]
                                              
                                              [--noise_p NOISE_P]
                                              
                                              [--mask_size MASK_SIZE]
                                              
                                              [--poisoning_rate POISONING_RATE]
                                              
                                              [--wave_multiple WAVE_MULTIPLE]
                                              
                                              [--backdoor_epochs BACKDOOR_EPOCHS]
                                              
                                              [--upload_method UPLOAD_METHOD]
                                              

You can run the code as follow:  
python -u label_inference_and_backdoor_attack.py --dataset MNIST --backdoor_epochs 10  
python -u label_inference_and_backdoor_attack.py --dataset imagenette --backdoor_epochs 100  
python -u label_inference_and_backdoor_attack.py --dataset Cifar10  
python -u label_inference_and_backdoor_attack.py --dataset cinic10  
python -u label_inference_and_backdoor_attack.py --dataset bank  
python -u label_inference_and_backdoor_attack.py --dataset givemesomecredit  

# Experiments of impact of Client Numbers

usage: data_split_backdoor.py [-h] [--dataset DATASET]

                              [--target_label TARGET_LABEL]
                              
                              [--target_label_candidates_number TARGET_LABEL_CANDIDATES_NUMBER]
                              
                              [--noise_range NOISE_RANGE] [--noise_p NOISE_P]
                              
                              [--mask_size MASK_SIZE]
                              
                              [--poisoning_rate POISONING_RATE]
                              
                              [--wave_multiple WAVE_MULTIPLE]
                              
                              [--backdoor_epochs BACKDOOR_EPOCHS]
                              
                              [--n_workers N_WORKERS]
                              
                              [--id_attacker ID_ATTACKER]
                                               
You can run the code as follow:  
python -u data_split_backdoor.py --dataset Cifar10 --n_workers 4 --id_attacker 1   
