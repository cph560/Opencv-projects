import os 
import cv2

mainpath = 'Images'
folders = os.listdir(mainpath)

for folder in folders:
    path = mainpath+'/'+folder
    img_list = []
    listdir = os.listdir(path)
    print(f'# of images detected: {len(listdir)}')

    for img_no in listdir:
        img = cv2.imread(f'{path}/{img_no}')
        img = cv2.resize(img, (0,0), None, 0.2, 0.2)
        img_list.append(img)
    
    stitcher = cv2.Stitcher.create()
    (status, res) = stitcher.stitch(img_list)
    
    if (status == cv2.STITCHER_OK):
        print('Panorama Generated')
        respath = os.getcwd()+'/'+path
 
        cv2.imwrite(respath+f'/{folder}result.jpg', res)
        cv2.imshow(folder, res)
        cv2.waitKey(1)
    else:
        print('Panorama Generating Failed')

cv2.waitKey(0)