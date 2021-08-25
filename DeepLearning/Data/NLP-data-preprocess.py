# NLP의 data augmentation에 쓸 데이터를 Shuffle해서 유사질문을 만드는 과정 
# 하나의 질문을 Shuffle과 Random Crop을 통해 20개 생성한 후 처리

import random

# create sample text file 
f = open("newfile.txt", 'w')
f.close()

# write shuffled sample text 
f = open("newfile.txt", 'a')

f.write("애완동물 키울 때 보호자의 책임에 대해 알려주세요\n\n")

for _ in range(20):
    random.shuffle(arr)
    result = " ".join(arr)
    result += str("\n")
    f.write(str(result))
    f.write(str("\n"))
    
f.close()    
