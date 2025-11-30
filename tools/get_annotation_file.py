import os
import sys
sys.path.insert(0,os.getcwd())


def main():
    datasets_path = 'data_3/train_47'
    class_name_list = os.listdir(datasets_path)
    class_name_list.sort()
    print(class_name_list)
    f = open('data_3/annotations.txt', 'w+')
    for i, class_name in enumerate(class_name_list):
        if i == len(class_name_list) - 1:
            text = class_name + ' ' + str(i)
        else:
            text = class_name + ' ' + str(i) + '\n'
        f.write(text)
    f.close
    
    
if __name__ == "__main__":
    main()
    
    