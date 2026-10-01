#modality
modal_dict={
    'T1w':'T1-weighted MRI',
    'T2w':'T2-weighted MRI',
    'FLAIR':'Fluid-attenuated inversion recovery MRI',
    'CT':'CT',
    'PET':'PET',
    'Optical':'Optical Imaging',
    'Fluorescent':'Fluorescent staining'
}
modal_map={
    'T1-weighted MRI': 0,
    'T2-weighted MRI': 1,
    'Fluid-attenuated inversion recovery MRI': 2,
    'CT': 3,
    'PET': 4,
    'Optical Imaging': 5,
    'Fluorescent staining':6

}
modal_map_inv = {v:k for k,v in modal_map.items()}
modal_map_idx = {k:i for i,k in enumerate(modal_map_inv.keys())}

brain_level_1_dict = {
    'smooth':['mouseFlunorescent','batBrainExtraction','MouseSag','MouseCor','MouseAxi',],
    'primary':['rabbitBrainExtraction','monkeyBrainExtraction','monkeyBrainParaffin','RabbitSag','RabbitCor','RabbitAxi','MonkeySag','MonkeyCor','MonkeyAxi','Monkey'],
    'complicate':['humanBrainExtraction','HumanSag','HumanCor','HumanAxi'],
}
brain_level_1_map = {
    'smooth': 0,
    'primary': 1,
    'complicate':2,
}
brain_level_1_map_inv = {v:k for k,v in brain_level_1_map.items()}
brain_level_1_map_idx = {k:i for i,k in enumerate(brain_level_1_map_inv.keys())}

brain_level_2_dict = {
    'mouse':['mouseFlunorescent','MouseSag','MouseCor','MouseAxi',],
    'bat':['batBrainExtraction'],
    'monkey':['monkeyBrainExtraction','monkeyBrainParaffin','Monkey','MonkeySag','MonkeyCor','MonkeyAxi'],
    'human':['humanBrainExtraction','HumanSag','HumanCor','HumanAxi',],
    'rabbit':['rabbitBrainExtraction','RabbitSag','RabbitCor','RabbitAxi',],
}
brain_level_2_map = {
    'mouse': 0,
    'bat': 1,
    'monkey': 2,
    'human': 3,
    'rabbit': 4,
}

brain_level_2_map_inv = {v:k for k,v in brain_level_2_map.items()}
brain_level_2_map_idx = {k:i for i,k in enumerate(brain_level_2_map_inv.keys())}

task_list = [
    'T1_mouse_Flunorescent','batBrainExtraction','monkeyBrainExtraction','monkeyBrainParaffin',
    'humanBrainExtraction','T1_HumanSag','T1_HumanCor','T1_HumanAxi','rabbitBrainExtraction','T2_RabbitSag',
    'T2_RabbitCor','T2_RabbitAxi','T2_MouseAxi','T2_MouseCor','T2_MouseSag','Optical_Monkey','T1_MonkeySag', 
    'T1_MonkeyCor', 'T1_MonkeyAxi',
]
task_idx = {k :i for i,k in enumerate(task_list)}

