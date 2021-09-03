# voc 2007 dataset의 ImageJPEG Folder에서 Segmentation-Semantic 으로 사용되는 이미지만 골라서 따로 옮기는 작업

import shutil

file_list = []
file_path = "./Segmentation-Semantic/ImageSets/Segmentation-Semantic/trainval.txt"

f = open(file_path, 'r')

# file list 읽어오기 
line = None
while True:
    # readline() -> file에서 한 줄씩 계속 읽어옴
    line = f.readline().replace('\n', '') # 개행 문자 공백으로 대체 
    if line is None or len(line) == 0:
        break
    file_list.append(line)
    
f.close()    

# 해당 이름을 가진 file 옮기기 

root = "./"
to_path = root + "SegImg"

for filename in file_list:
    from_path = root + 'Segmentation-Semantic/JPEGImages/' + filename + '.jpg'
    shutil.copy2(from_path, to_path)
